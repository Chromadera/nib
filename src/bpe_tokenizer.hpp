#pragma once

#include "inkling_engine.h"
#include "unicode_cat_table.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace inkling {

// ============================================================================
// Lightweight GPT-2 Byte-Level BPE Tokenizer (Zero External Dependencies)
// ============================================================================
// Uses the TokenizerData (vocab + merges) already parsed from GGUF by gguf_loader.cpp.
// Supports exact inkling2 / o200k_base (LLAMA_VOCAB_PRE_TYPE_GPT4O) UTF-8 pre-tokenization,
// optional special-token splitting, BPE encoding, and streaming UTF-8 decoding.
class BpeTokenizer {
public:
    void init(const TokenizerData & tok) {
        tok_ = &tok;
        build_byte_tables();

        tok_to_id_.clear();
        tok_to_id_.reserve(tok.vocab.size() * 2);
        special_toks_.clear();
        for (size_t i = 0; i < tok.vocab.size(); ++i) {
            tok_to_id_[tok.vocab[i]] = (int32_t)i;
            // Control/special tokens have token_type == 3 (CONTROL) or 4 (USER_DEFINED) and look like <|...|>
            const bool is_ctrl = (i < tok.token_types.size() && (tok.token_types[i] == 3 || tok.token_types[i] == 4));
            if (is_ctrl && tok.vocab[i].size() >= 4 && tok.vocab[i].rfind("<|", 0) == 0) {
                special_toks_.emplace_back(tok.vocab[i], (int32_t)i);
            }
        }
        // Longest-first match for special tokens
        std::sort(special_toks_.begin(), special_toks_.end(),
                  [](const auto & a, const auto & b) { return a.first.size() > b.first.size(); });

        merge_rank_.clear();
        merge_rank_.reserve(tok.merges.size() * 2);
        for (size_t i = 0; i < tok.merges.size(); ++i) {
            merge_rank_[tok.merges[i]] = (int32_t)i;
        }
    }

    int32_t token_id(const std::string & s) const {
        auto it = tok_to_id_.find(s);
        return (it != tok_to_id_.end()) ? it->second : -1;
    }

    // Encode plain text without scanning for <|...|> control tokens (safe for untrusted user/tool content).
    void append_plain(const std::string & text, std::vector<int32_t> & out) const {
        if (!text.empty()) encode_plain_segment(text, out);
    }

    std::vector<int32_t> encode_plain(const std::string & text) const {
        std::vector<int32_t> out;
        append_plain(text, out);
        return out;
    }

    // Emit a single registered special token by exact name (falls back to plain BPE if absent).
    void append_special(const std::string & tok, std::vector<int32_t> & out) const {
        const int32_t id = token_id(tok);
        if (id >= 0) {
            out.push_back(id);
        } else {
            append_plain(tok, out);
        }
    }

    // Encode text; when parse_special is true, registered <|...|> control tokens are matched verbatim.
    std::vector<int32_t> encode(const std::string & text, bool parse_special = true) const {
        std::vector<int32_t> out;
        if (!parse_special) {
            encode_plain_segment(text, out);
            return out;
        }
        size_t pos = 0;
        const size_t n = text.size();
        while (pos < n) {
            // Find the next position >= pos where a registered special token matches verbatim.
            size_t search = pos;
            size_t match_pos = n;
            int32_t matched_id = -1;
            size_t matched_len = 0;
            while (search < n) {
                size_t cand = text.find("<|", search);
                if (cand == std::string::npos) break;
                for (const auto & sp : special_toks_) {
                    const size_t sl = sp.first.size();
                    if (cand + sl <= n && text.compare(cand, sl, sp.first) == 0) {
                        match_pos = cand;
                        matched_id = sp.second;
                        matched_len = sl;
                        break;
                    }
                }
                if (matched_id >= 0) break;
                search = cand + 2;
            }
            if (match_pos > pos) {
                encode_plain_segment(text.substr(pos, match_pos - pos), out);
            }
            if (matched_id >= 0) {
                out.push_back(matched_id);
                pos = match_pos + matched_len;
            } else {
                break;
            }
        }
        return out;
    }

    // Decode a single token ID to raw UTF-8 text (returns "" for control tokens if skip_special)
    std::string decode_token(int32_t id, bool skip_special = false) const {
        if (!tok_ || id < 0 || (size_t)id >= tok_->vocab.size()) return "";
        const bool is_ctrl = ((size_t)id < tok_->token_types.size() &&
                              (tok_->token_types[id] == 3 || tok_->token_types[id] == 4));
        if (is_ctrl) {
            return skip_special ? "" : tok_->vocab[id];
        }
        return gpt2_str_to_bytes(tok_->vocab[id]);
    }

    std::string decode(const std::vector<int32_t> & ids, bool skip_special = true) const {
        std::string s;
        for (int32_t id : ids) s += decode_token(id, skip_special);
        return s;
    }

private:
    static std::string cp_to_utf8(uint32_t cp) {
        std::string r;
        if (cp < 0x80) {
            r.push_back((char)cp);
        } else if (cp < 0x800) {
            r.push_back((char)(0xC0 | (cp >> 6)));
            r.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            r.push_back((char)(0xE0 | (cp >> 12)));
            r.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            r.push_back((char)(0x80 | (cp & 0x3F)));
        }
        return r;
    }

    void build_byte_tables() {
        bool direct[256] = {};
        for (int b = '!'; b <= '~'; ++b) direct[b] = true;
        for (int b = 0xA1; b <= 0xAC; ++b) direct[b] = true;
        for (int b = 0xAE; b <= 0xFF; ++b) direct[b] = true;

        uint32_t n = 0;
        cp_to_byte_.clear();
        for (int b = 0; b < 256; ++b) {
            uint32_t cp = direct[b] ? (uint32_t)b : (256u + n++);
            byte_to_u8_[b] = cp_to_utf8(cp);
            cp_to_byte_[cp] = (uint8_t)b;
        }
    }

    std::string gpt2_str_to_bytes(const std::string & s) const {
        std::string out;
        out.reserve(s.size());
        size_t i = 0;
        while (i < s.size()) {
            uint8_t c = (uint8_t)s[i];
            uint32_t cp = 0;
            size_t len = 1;
            if ((c & 0x80) == 0) {
                cp = c;
            } else if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
                cp = ((uint32_t)(c & 0x1F) << 6) | ((uint8_t)s[i + 1] & 0x3F);
                len = 2;
            } else if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
                cp = ((uint32_t)(c & 0x0F) << 12) | (((uint8_t)s[i + 1] & 0x3F) << 6) | ((uint8_t)s[i + 2] & 0x3F);
                len = 3;
            } else {
                out.push_back((char)c);
                i += 1;
                continue;
            }
            auto it = cp_to_byte_.find(cp);
            if (it != cp_to_byte_.end()) {
                out.push_back((char)it->second);
            } else {
                out.append(s, i, len);
            }
            i += len;
        }
        return out;
    }

    struct CpSpan {
        uint32_t   cp;
        size_t     off;
        UnicodeCat cat;
    };

    static void decode_utf8_spans(const std::string & text, std::vector<CpSpan> & out) {
        out.clear();
        out.reserve(text.size() + 1);
        const size_t n = text.size();
        size_t i = 0;
        while (i < n) {
            const uint8_t b0 = (uint8_t)text[i];
            uint32_t cp = 0xFFFDu;
            size_t len = 1;
            if (b0 < 0x80u) {
                cp = b0;
                len = 1;
            } else if ((b0 & 0xE0u) == 0xC0u && i + 1 < n) {
                const uint8_t b1 = (uint8_t)text[i + 1];
                if ((b1 & 0xC0u) == 0x80u) {
                    const uint32_t v = ((uint32_t)(b0 & 0x1Fu) << 6) | (uint32_t)(b1 & 0x3Fu);
                    if (v >= 0x80u) { cp = v; len = 2; }
                }
            } else if ((b0 & 0xF0u) == 0xE0u && i + 2 < n) {
                const uint8_t b1 = (uint8_t)text[i + 1];
                const uint8_t b2 = (uint8_t)text[i + 2];
                if ((b1 & 0xC0u) == 0x80u && (b2 & 0xC0u) == 0x80u) {
                    const uint32_t v = ((uint32_t)(b0 & 0x0Fu) << 12) |
                                       ((uint32_t)(b1 & 0x3Fu) << 6) |
                                       (uint32_t)(b2 & 0x3Fu);
                    if (v >= 0x800u && !(v >= 0xD800u && v <= 0xDFFFu)) { cp = v; len = 3; }
                }
            } else if ((b0 & 0xF8u) == 0xF0u && i + 3 < n) {
                const uint8_t b1 = (uint8_t)text[i + 1];
                const uint8_t b2 = (uint8_t)text[i + 2];
                const uint8_t b3 = (uint8_t)text[i + 3];
                if ((b1 & 0xC0u) == 0x80u && (b2 & 0xC0u) == 0x80u && (b3 & 0xC0u) == 0x80u) {
                    const uint32_t v = ((uint32_t)(b0 & 0x07u) << 18) |
                                       ((uint32_t)(b1 & 0x3Fu) << 12) |
                                       ((uint32_t)(b2 & 0x3Fu) << 6) |
                                       (uint32_t)(b3 & 0x3Fu);
                    if (v >= 0x10000u && v <= 0x10FFFFu) { cp = v; len = 4; }
                }
            }
            out.push_back({cp, i, unicode_cat(cp)});
            i += len;
        }
        out.push_back({0u, n, UC_OTHER});
    }

    static size_t match_contraction(const std::vector<CpSpan> & cps, size_t n, size_t pos) {
        if (pos + 1 >= n || cps[pos].cp != '\'') return 0;
        const uint32_t c1 = cps[pos + 1].cp;
        if (c1 == 's' || c1 == 'S' || c1 == 't' || c1 == 'T' ||
            c1 == 'm' || c1 == 'M' || c1 == 'd' || c1 == 'D') {
            return 2;
        }
        if (pos + 2 < n) {
            const uint32_t c2 = cps[pos + 2].cp;
            if (((c1 == 'r' || c1 == 'R') && (c2 == 'e' || c2 == 'E')) ||
                ((c1 == 'v' || c1 == 'V') && (c2 == 'e' || c2 == 'E')) ||
                ((c1 == 'l' || c1 == 'L') && (c2 == 'l' || c2 == 'L'))) {
                return 3;
            }
        }
        return 0;
    }

    // Exact inkling2 / o200k_base (LLAMA_VOCAB_PRE_TYPE_GPT4O) pre-tokenizer:
    //   [^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?
    // | [^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?
    // | \p{N}{1,3}
    // |  ?[^\s\p{L}\p{N}]+[\r\n/]*
    // | \s*[\r\n]+
    // | \s+(?!\S)
    // | \s+
    static std::vector<std::string> pretokenize(const std::string & text) {
        std::vector<std::string> pieces;
        if (text.empty()) return pieces;

        std::vector<CpSpan> cps;
        decode_utf8_spans(text, cps);
        const size_t n = cps.size() - 1; // exclude sentinel

        auto is_L  = [](const CpSpan & c) { return c.cat == UC_LETTER; };
        auto is_N  = [](const CpSpan & c) { return c.cat == UC_NUMBER; };
        auto is_S  = [](const CpSpan & c) { return c.cat == UC_SPACE; };
        auto is_LM = [](const CpSpan & c) { return c.cat == UC_LETTER || c.cat == UC_MARK; };
        auto is_U  = [&](const CpSpan & c) { return is_LM(c) && !(c.cp >= 'a' && c.cp <= 'z'); };
        auto is_Lo = [&](const CpSpan & c) { return is_LM(c) && !(c.cp >= 'A' && c.cp <= 'Z'); };
        auto is_P  = [](const CpSpan & c) {
            return c.cp != '\r' && c.cp != '\n' && c.cat != UC_LETTER && c.cat != UC_NUMBER;
        };

        auto emit = [&](size_t a, size_t b) {
            pieces.push_back(text.substr(cps[a].off, cps[b].off - cps[a].off));
        };

        auto try_branch1 = [&](size_t s) -> size_t {
            // U* Lo+ C?
            size_t j = s;
            while (j < n && is_U(cps[j])) ++j;
            size_t end = 0;
            if (j < n && is_Lo(cps[j])) {
                end = j + 1;
                while (end < n && is_Lo(cps[end])) ++end;
            } else {
                while (j > s && !is_Lo(cps[j - 1])) --j;
                if (j > s) end = j;
                else return 0;
            }
            end += match_contraction(cps, n, end);
            return end;
        };

        auto try_branch2 = [&](size_t s) -> size_t {
            // U+ Lo* C?
            if (s >= n || !is_U(cps[s])) return 0;
            size_t j = s + 1;
            while (j < n && is_U(cps[j])) ++j;
            size_t end = j;
            while (end < n && is_Lo(cps[end])) ++end;
            end += match_contraction(cps, n, end);
            return end;
        };

        size_t i = 0;
        while (i < n) {
            const bool p0 = is_P(cps[i]);

            // Branch 1: P? U* Lo+ C?
            size_t end = p0 ? try_branch1(i + 1) : 0;
            if (end == 0) end = try_branch1(i);
            if (end > i) {
                emit(i, end);
                i = end;
                continue;
            }

            // Branch 2: P? U+ Lo* C?
            end = p0 ? try_branch2(i + 1) : 0;
            if (end == 0) end = try_branch2(i);
            if (end > i) {
                emit(i, end);
                i = end;
                continue;
            }

            // Branch 3: \p{N}{1,3}
            if (is_N(cps[i])) {
                end = i + 1;
                while (end < n && (end - i) < 3 && is_N(cps[end])) ++end;
                emit(i, end);
                i = end;
                continue;
            }

            // Branch 4:  ?[^\s\p{L}\p{N}]+[\r\n/]*
            {
                const size_t s = (cps[i].cp == ' ') ? (i + 1) : i;
                if (s < n && !is_S(cps[s]) && !is_L(cps[s]) && !is_N(cps[s])) {
                    end = s + 1;
                    while (end < n && !is_S(cps[end]) && !is_L(cps[end]) && !is_N(cps[end])) ++end;
                    while (end < n && (cps[end].cp == '\r' || cps[end].cp == '\n' || cps[end].cp == '/')) ++end;
                    emit(i, end);
                    i = end;
                    continue;
                }
            }

            // Branches 5, 6, 7: \s*[\r\n]+ | \s+(?!\S) | \s+
            if (is_S(cps[i])) {
                size_t j = i + 1;
                while (j < n && is_S(cps[j])) ++j;
                // Branch 5: \s*[\r\n]+
                size_t k = j;
                while (k > i && cps[k - 1].cp != '\r' && cps[k - 1].cp != '\n') --k;
                if (k > i) {
                    emit(i, k);
                    i = k;
                    continue;
                }
                // Branch 6: \s+(?!\S)
                if (j == n) {
                    emit(i, j);
                    i = j;
                    continue;
                }
                if (j - i >= 2) {
                    emit(i, j - 1);
                    i = j - 1;
                    continue;
                }
                // Branch 7: \s+
                emit(i, j);
                i = j;
                continue;
            }

            // Unreachable fallback: guarantee forward progress
            emit(i, i + 1);
            ++i;
        }
        return pieces;
    }

    void encode_plain_segment(const std::string & seg, std::vector<int32_t> & out) const {
        auto pieces = pretokenize(seg);
        for (const auto & piece : pieces) {
            std::vector<std::string> syms;
            syms.reserve(piece.size());
            for (unsigned char b : piece) syms.push_back(byte_to_u8_[b]);

            while (syms.size() > 1) {
                int best_rank = INT_MAX;
                size_t best_idx = 0;
                for (size_t i = 0; i + 1 < syms.size(); ++i) {
                    auto it = merge_rank_.find(syms[i] + " " + syms[i + 1]);
                    if (it != merge_rank_.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best_idx = i;
                    }
                }
                if (best_rank == INT_MAX) break;
                syms[best_idx] += syms[best_idx + 1];
                syms.erase(syms.begin() + (ptrdiff_t)(best_idx + 1));
            }

            for (const auto & sym : syms) {
                auto it = tok_to_id_.find(sym);
                if (it != tok_to_id_.end()) {
                    out.push_back(it->second);
                }
            }
        }
    }

    const TokenizerData *                      tok_ = nullptr;
    std::string                                byte_to_u8_[256];
    std::unordered_map<uint32_t, uint8_t>      cp_to_byte_;
    std::unordered_map<std::string, int32_t>   tok_to_id_;
    std::unordered_map<std::string, int32_t>   merge_rank_;
    std::vector<std::pair<std::string, int32_t>> special_toks_;
};

} // namespace inkling
