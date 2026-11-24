#pragma once

#include "inkling_engine.h"

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
// Supports exact special-token splitting, BPE encoding, and streaming UTF-8 decoding.
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
            // Control/special tokens have token_type == 3 (CONTROL) or 4 (USER_DEFINED) or look like <|...|>
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

    // Encode text (splitting special tokens like <|message_user|> verbatim)
    std::vector<int32_t> encode(const std::string & text) const {
        std::vector<int32_t> out;
        size_t pos = 0;
        const size_t n = text.size();
        while (pos < n) {
            // Check special token at current position
            int32_t matched_id = -1;
            size_t matched_len = 0;
            if (text[pos] == '<' && pos + 1 < n && text[pos + 1] == '|') {
                for (const auto & sp : special_toks_) {
                    const size_t sl = sp.first.size();
                    if (pos + sl <= n && text.compare(pos, sl, sp.first) == 0) {
                        matched_id = sp.second;
                        matched_len = sl;
                        break;
                    }
                }
            }
            if (matched_id >= 0) {
                out.push_back(matched_id);
                pos += matched_len;
                continue;
            }
            // Find next special token boundary
            size_t next_sp = text.find("<|", pos + 1);
            size_t seg_end = (next_sp == std::string::npos) ? n : next_sp;
            encode_plain_segment(text.substr(pos, seg_end - pos), out);
            pos = seg_end;
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

    // Simple GPT-2/cl100k-style word pre-tokenizer (splits whitespace+word, numbers, punctuation, newlines)
    static std::vector<std::string> pretokenize(const std::string & text) {
        std::vector<std::string> pieces;
        size_t i = 0;
        const size_t n = text.size();
        auto is_letter = [](unsigned char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c >= 0x80; };
        auto is_digit  = [](unsigned char c) { return (c >= '0' && c <= '9'); };
        while (i < n) {
            unsigned char c = (unsigned char)text[i];
            if (c == '\r' || c == '\n') {
                size_t j = i;
                while (j < n && (text[j] == '\r' || text[j] == '\n')) ++j;
                pieces.push_back(text.substr(i, j - i));
                i = j;
                continue;
            }
            // Optional single leading space attached to letters or punctuation
            if (c == ' ' && i + 1 < n && text[i + 1] != ' ' && text[i + 1] != '\n' && text[i + 1] != '\r') {
                unsigned char nx = (unsigned char)text[i + 1];
                if (is_letter(nx)) {
                    size_t j = i + 1;
                    while (j < n && is_letter((unsigned char)text[j])) ++j;
                    pieces.push_back(text.substr(i, j - i));
                    i = j;
                    continue;
                }
                if (is_digit(nx)) {
                    pieces.push_back(" ");
                    i += 1;
                    continue;
                }
                size_t j = i + 1;
                while (j < n && !is_letter((unsigned char)text[j]) && !is_digit((unsigned char)text[j]) &&
                       text[j] != ' ' && text[j] != '\n' && text[j] != '\r') ++j;
                pieces.push_back(text.substr(i, j - i));
                i = j;
                continue;
            }
            if (c == ' ') {
                size_t j = i;
                while (j < n && text[j] == ' ') ++j;
                pieces.push_back(text.substr(i, j - i));
                i = j;
                continue;
            }
            if (is_letter(c)) {
                size_t j = i;
                while (j < n && is_letter((unsigned char)text[j])) ++j;
                pieces.push_back(text.substr(i, j - i));
                i = j;
                continue;
            }
            if (is_digit(c)) {
                size_t j = i;
                while (j < n && is_digit((unsigned char)text[j]) && (j - i) < 3) ++j;
                pieces.push_back(text.substr(i, j - i));
                i = j;
                continue;
            }
            size_t j = i;
            while (j < n && !is_letter((unsigned char)text[j]) && !is_digit((unsigned char)text[j]) &&
                   text[j] != ' ' && text[j] != '\n' && text[j] != '\r') ++j;
            pieces.push_back(text.substr(i, j - i));
            i = j;
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
