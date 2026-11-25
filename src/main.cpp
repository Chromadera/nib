#include "bpe_tokenizer.hpp"
#include "dspark_mtp.h"
#include "engine.hpp"
#include "server.hpp"
#include "vision_encoder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace inkling;

// Step 7F A/B switch parser: off/-1 = shipped kernel, auto/0 = MOE_V3_AUTO,
// alt/1 = MOE_V3_ALT, alt2/2 = MOE_V3_ALT2 (wo only).
static bool parse_v3_variant(const std::string & v, bool allow_alt2, bool allow_ship, int32_t * out) {
    if (v == "off"   || v == "-1") { *out = -1; return true; }
    if (v == "auto"  || v == "0")  { *out =  0; return true; }
    if (v == "alt"   || v == "1")  { *out =  1; return true; }
    if (allow_alt2 && (v == "alt2" || v == "2")) { *out = 2; return true; }
    if (v == "cache" || v == "3")  { *out =  3; return true; }
    if (allow_ship && (v == "ship"  || v == "4")) { *out = 4; return true; }
    if (allow_ship && (v == "ship3" || v == "5")) { *out = 5; return true; }
    return false;
}

static bool parse_v1v2v3(const std::string & v, int32_t * out) {
    if (v == "v1" || v == "0") { *out = 0; return true; }
    if (v == "v2" || v == "1") { *out = 1; return true; }
    if (v == "v3" || v == "2") { *out = 2; return true; }
    if (v == "v4" || v == "3") { *out = 3; return true; }
    if (v == "def" || v == "-1") { *out = -1; return true; }
    return false;
}

static bool parse_meta_fmt(const std::string & v, int32_t * out) {
    if (v == "q8") { *out = 0; return true; }
    if (v == "q4") { *out = 1; return true; }
    return false;
}

static std::vector<int32_t> load_token_ids_file(const std::string & path) {
    std::ifstream f(path);
    std::vector<int32_t> ids;
    int32_t x;
    while (f >> x) ids.push_back(x);
    return ids;
}

static double compute_nll_from_logits(
        const float * logits, int n_tokens, const int32_t * target_ids, int start_pos, int count) {
    double nll = 0.0;
    for (int i = start_pos; i < start_pos + count; ++i) {
        const float * row = logits + (size_t)i * VOCAB_UNPADDED;
        float maxv = row[0];
        for (int v = 1; v < VOCAB_UNPADDED; ++v) {
            if (row[v] > maxv) maxv = row[v];
        }
        double sum_exp = 0.0;
        for (int v = 0; v < VOCAB_UNPADDED; ++v) {
            sum_exp += std::exp((double)(row[v] - maxv));
        }
        const int32_t tgt = target_ids[i + 1];
        double log_prob = (double)(row[tgt] - maxv) - std::log(sum_exp);
        nll -= log_prob;
    }
    return nll;
}

#ifndef NIB_VERSION
#define NIB_VERSION "0.1.0"
#endif
#ifndef NIB_GIT_SHA
#define NIB_GIT_SHA "unknown"
#endif
#ifndef NIB_GPU_TARGET
#define NIB_GPU_TARGET "gfx1100"
#endif

static void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "Nib v%s (%s)\n"
        "Usage: %s <mode> [options]\n"
        "Modes:\n"
        "  bench    -m <gguf> [-c <max_ctx=262144>] [-d 0,8192,32768] [-p 512] [-n 128] [-r 3] [--kv-type q8_0|q4_1|q4_0] [--sparse-pages 0|112|192|224] [--sparse-meta q8|q4] [--no-graph]\n"
        "           [--wo-v3 off|auto|alt2|cache] [--moe-gu-v3 off|auto|alt|cache|ship|ship3] [--moe-dn-v3 off|auto|alt|cache] [--moe-fused-v3 off|auto|cache]\n"
        "  ppl      -m <gguf> -t <token_ids.txt> [-c 512] [--kv-type q8_0|q4_1|q4_0] [--sparse-pages 0|112|192|224] [--sparse-meta q8|q4] [--decode-ub1] [--no-graph]\n"
        "  oracle   -m <gguf> -t <token_ids.txt> -o <out_logits.bin> [--kv-type q8_0|q4_1|q4_0] [--sparse-pages 0|112|192|224] [--sparse-meta q8|q4] [--decode-ub1]\n"
        "  gen      -m <gguf> --prompt-file <text.txt> [-n 32] [-c 262144] [--kv-type q8_0|q4_1|q4_0] [--sparse-pages 0|112|192|224] [--sparse-meta q8|q4] [--no-graph]\n"
        "           [--mtp models/mtp.safetensors] [--spec-threshold 0.8] [--draft-ctx 8192] [--no-verify-batch]\n"
        "  vision   -m <gguf> --mmproj <mmproj.gguf> --image <img.png> [--prompt <text>] [-c 262144] [-d 0] [-n 256] [--kv-type q8_0|q4_1|q4_0] [--sparse-pages 0|112|192|224] [--sparse-meta q8|q4] [--bench-vision 5] [--dump-embeds <path>] [--no-graph]\n"
        "  serve    -m <gguf> [--host 127.0.0.1] [--port 4444] [--model-name inkling-2-mini] [--slots 4] [-c 262848] [--strict-slot-ctx] [--max-slot-ctx N] [--kv-type q8_0|q4_1|q4_0] [--sparse-pages 0|112|192|224] [--sparse-meta q8|q4]\n"
        "           [--default-temp 0.6] [--default-top-p 0.95] [--default-top-k 40] [--default-min-p 0.0] [--mtp models/mtp.safetensors] [--spec-threshold 0.8] [--draft-ctx 8192] [--mmproj <mmproj.gguf> | --no-vision] [--no-graph]\n"
        "  selftest -m <gguf> [--kv-type q8_0|q4_1|q4_0] [--sparse-pages 0|112|192|224] [--sparse-meta q8|q4] [--mtp models/mtp.safetensors] [--bench-mtp]\n"
        "  --version  Print Nib version and target GPU architecture\n",
        NIB_VERSION, NIB_GPU_TARGET, argv0);
}


// ============================================================================
// Step 8A: DSpark MTP speculative generation loop
// ============================================================================
namespace {

struct MtpSession {
    InklingEngine * eng = nullptr;
    DSparkDrafter * drafter = nullptr;
    float           threshold = 0.8f;
    bool            active = false;

    // telemetry
    long   rounds = 0;
    long   accepted = 0;
    long   m_sum = 0;
    long   tokens = 0;
    double draft_ms = 0.0;
    double verify_ms = 0.0;
    long   draft_calls = 0;
    long   verify_calls = 0;

    double tok_per_round() const {
        return rounds > 0 ? (1.0 + (double)accepted / (double)rounds) : 0.0;
    }
    double avg_accepted() const { return rounds > 0 ? (double)accepted / (double)rounds : 0.0; }
    double avg_m() const { return rounds > 0 ? (double)m_sum / (double)rounds : 0.0; }
};

bool mtp_session_init(MtpSession & s, const std::string & mtp_path, InklingEngine & eng,
                      float threshold, bool verbose,
                      int draft_ctx = DSparkDrafter::DEFAULT_DRAFT_CTX) {
    static DSparkDrafter holder;  // one drafter per process (see mtp_session_init callers)
    s.eng = &eng;
    s.threshold = threshold;
    s.drafter = &holder;
    int vocab = VOCAB_UNPADDED;
    if (!s.drafter->init(mtp_path, eng.weights(), eng.max_ctx(), vocab, eng.stream(), draft_ctx)) {
        return false;
    }
    if (!eng.enable_mtp_io()) return false;
    eng.set_tap_hook([&s](const uint16_t * taps, const int32_t * toks, int n, int start) {
        s.drafter->commit(taps, n, start);
        if (toks != nullptr) s.drafter->note_tokens(toks, n);
    });
    s.active = true;
    if (verbose) {
        std::printf("[mtp] drafter online (threshold=%.2f, ctx=%d, shortlist=%d rows, vram=%.2f GiB total)\n",
                    threshold, s.drafter->ctx(), s.drafter->short_rows(),
                    (double)eng.total_vram_bytes() / (1024.0 * 1024.0 * 1024.0));
        std::fflush(stdout);
    }
    return true;
}

// Greedy speculative generation of `n_gen` tokens from `anchor` at position
// eng.current_pos().  Appends emissions to `out` (and streams them via `emit`).
bool mtp_generate(MtpSession & s, int32_t anchor, int n_gen, std::vector<int32_t> & out,
                  const std::function<void(int32_t)> & emit, double * gen_ms) {
    InklingEngine & eng = *s.eng;
    hipEvent_t ev0 = nullptr, ev1 = nullptr, ev2 = nullptr;
    (void)hipEventCreate(&ev0);
    (void)hipEventCreate(&ev1);
    (void)hipEventCreate(&ev2);
    int emitted = 0;
    std::vector<int32_t> round_tokens(8);
    auto t_start = std::chrono::steady_clock::now();
    while (emitted < n_gen) {
        const int32_t cur_anchor = anchor;
        if (emit) emit(cur_anchor);
        out.push_back(cur_anchor);
        ++emitted;
        if (emitted >= n_gen) break;
        s.drafter->note_tokens(&cur_anchor, 1);

        const int P = eng.current_pos();
        // ---- draft --------------------------------------------------------
        (void)hipEventRecord(ev0, eng.stream());
        DSparkDraft d = s.drafter->draft(cur_anchor, P);
        (void)hipEventRecord(ev1, eng.stream());
        (void)hipStreamSynchronize(eng.stream());
        float draft_ms = 0.0f;
        (void)hipEventElapsedTime(&draft_ms, ev0, ev1);
        s.draft_ms += draft_ms;
        ++s.draft_calls;

        int32_t drafts[8] = {0};
        float conf_f[7] = {0.0f};
        (void)hipMemcpy(conf_f, d.conf, sizeof(float) * 7, hipMemcpyDeviceToHost);
        (void)hipMemcpy(drafts, d.tokens, sizeof(int32_t) * 7, hipMemcpyDeviceToHost);

        // ---- confidence-gated exact verify width (m in 2..8) --------------
        int lead = 0;
        while (lead < 7 && conf_f[lead] > s.threshold) ++lead;
        const int m = std::max(2, std::min(8, lead + 1));
        round_tokens[0] = cur_anchor;
        for (int r = 1; r < m; ++r) round_tokens[(size_t)r] = drafts[r - 1];

        // ---- target verify ------------------------------------------------
        (void)hipEventRecord(ev1, eng.stream());
        if (!eng.verify_step(round_tokens.data(), m, P)) {
            (void)hipEventDestroy(ev0);
            (void)hipEventDestroy(ev1);
            (void)hipEventDestroy(ev2);
            return false;
        }
        (void)hipEventRecord(ev2, eng.stream());
        (void)hipStreamSynchronize(eng.stream());
        float verify_ms = 0.0f;
        (void)hipEventElapsedTime(&verify_ms, ev1, ev2);
        s.verify_ms += verify_ms;
        ++s.verify_calls;

        const int32_t * target = eng.host_target_ids();
        if (std::getenv("INKLING_MTP_DEBUG") != nullptr && s.rounds < 4) {
            {
                static std::vector<int32_t> cand(7 * 1024);
                static std::vector<float> cval(7 * 1024);
                (void)hipMemcpy(cand.data(), s.drafter->debug_cand_idx(), cand.size() * 4, hipMemcpyDeviceToHost);
                (void)hipMemcpy(cval.data(), s.drafter->debug_cand_val(), cval.size() * 4, hipMemcpyDeviceToHost);
                int in0 = -1, in1 = -1;
                for (int k = 0; k < 1024; ++k) {
                    if (cand[(size_t)k] == target[0]) in0 = k;
                    if (cand[(size_t)k] == target[1]) in1 = k;
                }
                // rank of target[0] by base value among candidates (like the Markov head would see)
                std::printf("  [mtp-dbg] target0=%d in_cands=%d (val %.3f) | target1=%d in_cands=%d | base_max0=%.3f\n",
                            target[0], in0, in0 >= 0 ? cval[(size_t)in0] : 0.0f, target[1], in1,
                            *std::max_element(cval.begin(), cval.begin() + 1024));
            }
            std::printf("  [mtp-dbg] P=%d m=%d conf=[", P, m);
            for (int i = 0; i < 7; ++i) std::printf("%.3f%s", conf_f[i], i < 6 ? " " : "");
            std::printf("] drafts=[");
            for (int i = 0; i < 7; ++i) std::printf("%d%s", drafts[i], i < 6 ? " " : "");
            std::printf("] target=[");
            for (int i = 0; i < m; ++i) std::printf("%d%s", target[i], i < m - 1 ? " " : "");
            std::printf("]\n");
            std::fflush(stdout);
        }
        int acc = 0;
        while (acc < m - 1 && target[acc] == drafts[acc]) ++acc;
        const int32_t next = target[acc];
        const int commit = acc + 1;
        eng.verify_rollback(P, commit, m);
        s.drafter->commit(eng.taps(), commit, P);

        for (int r = 0; r < acc; ++r) {
            if (emitted >= n_gen) break;
            if (emit) emit(drafts[r]);
            out.push_back(drafts[r]);
            ++emitted;
            s.drafter->note_tokens(&drafts[r], 1);
        }
        ++s.rounds;
        s.accepted += acc;
        s.m_sum += m;
        s.tokens += commit;
        anchor = next;
    }
    auto t_end = std::chrono::steady_clock::now();
    if (gen_ms) *gen_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    (void)hipEventDestroy(ev0);
    (void)hipEventDestroy(ev1);
    (void)hipEventDestroy(ev2);
    return true;
}

void mtp_print_stats(const MtpSession & s, int n_tokens, double ms) {
    if (s.rounds == 0) return;
    const double tps = (ms > 0.0) ? (double)n_tokens / (ms / 1000.0) : 0.0;
    std::printf("[mtp] %d tokens in %.2f ms = %.2f tok/s | rounds=%ld avg_accepted=%.2f tok/round=%.2f avg_m=%.2f\n",
                n_tokens, ms, tps, s.rounds, s.avg_accepted(), s.tok_per_round(), s.avg_m());
    std::printf("[mtp] per round: draft=%.2f ms verify=%.2f ms (%.2f ms/tok wall) | verify m: %ld calls\n",
                s.draft_ms / (double)s.draft_calls, s.verify_ms / (double)s.verify_calls,
                (ms > 0.0 && n_tokens > 0) ? ms / (double)n_tokens : 0.0, s.verify_calls);
    std::fflush(stdout);
}

} // namespace


// ============================================================================
// Step 8A selftest: exactness of the speculative verify / rollback / generation
// ============================================================================
namespace {

struct SelftestResult {
    int failures = 0;
};

void st_fail(SelftestResult & r, const char * what) {
    std::printf("  [FAIL] %s\n", what);
    ++r.failures;
}

int32_t st_argmax(const float * logits, int n) {
    int32_t best = 0;
    for (int i = 1; i < n; ++i) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

bool st_logits_equal(const std::vector<float> & a, const std::vector<float> & b, double * maxdiff) {
    if (a.size() != b.size()) return false;
    double md = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > md) md = d;
    }
    if (maxdiff) *maxdiff = md;
    return md == 0.0;
}

// Prefill the fixture prompt on a fresh state (and a fresh drafter context).
bool st_prefill_fixture(InklingEngine & eng, DSparkDrafter * dr, const std::vector<int32_t> & prompt,
                        std::vector<float> * logits) {
    eng.reset_state();
    if (dr) dr->reset();
    return eng.prefill(prompt, false, logits);
}

// Issue #4: the exact GPU full-vocab sampler against the analytic distribution
// of the same logit row (row 0 of the engine's logits buffer = `logits`, the
// softcapped row the last decode step produced).  Fixed seeds -> deterministic.
//   * Z, p(excl) and the kept set must match the host's double-precision softmax.
//   * N bonus draws (no exclusion) must follow p:            chi-square gate.
//   * N residual draws (excl = argmax) must never return the excluded token and
//     must follow p(x) / (1 - p(argmax)), x != argmax:     chi-square gate.
//   * a min_p run must never leave { x : p(x) >= min_p * p_max }.
int run_selftest_sampler(InklingEngine & eng, const std::vector<float> & logits) {
    const int V = VOCAB_UNPADDED;
    if ((int)logits.size() < V) return 1;
    int failures = 0;
    const int32_t amax = (int32_t)(std::max_element(logits.begin(), logits.begin() + V) - logits.begin());

    // T = 1.5 flattens the row so the tail actually gets exercised.
    const float temp = 1.5f;
    const float inv_t = 1.0f / temp;
    const double mx = (double)logits[(size_t)amax];
    std::vector<double> p((size_t)V);
    double z = 0.0;
    for (int x = 0; x < V; ++x) {
        p[(size_t)x] = std::exp(((double)logits[(size_t)x] - mx) * (double)inv_t);
        z += p[(size_t)x];
    }
    for (double & v : p) v /= z;

    auto chi_square = [&](const std::vector<int> & cnt, int n, int32_t excl, double & chi, int & dof) {
        // Bins: every token with expected count >= 5, plus one pooled tail bin.
        const double pe = (excl >= 0) ? p[(size_t)excl] : 0.0;
        const double norm = 1.0 / (1.0 - pe);
        chi = 0.0;
        dof = -1;
        double tail_e = 0.0;
        long long tail_o = 0;
        for (int x = 0; x < V; ++x) {
            if (x == excl) continue;
            const double e = (double)n * p[(size_t)x] * norm;
            if (e >= 5.0) {
                const double d = (double)cnt[(size_t)x] - e;
                chi += d * d / e;
                ++dof;
            } else {
                tail_e += e;
                tail_o += cnt[(size_t)x];
            }
        }
        if (tail_e >= 5.0) {
            const double d = (double)tail_o - tail_e;
            chi += d * d / tail_e;
            ++dof;
        }
    };

    const int n_draw = 20000;
    for (int mode = 0; mode < 2; ++mode) {
        const int32_t excl = (mode == 0) ? -1 : amax;
        std::vector<int> cnt((size_t)V, 0);
        int bad = 0;
        double z_rel = 0.0, pd_err = 0.0;
        for (int i = 0; i < n_draw; ++i) {
            int32_t tok = -1;
            float st[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            const uint64_t seed = 0x9E3779B97F4A7C15ull * (uint64_t)(i + 1) + (uint64_t)mode;
            if (!eng.sample_rows_full_to_host(1, inv_t, 0.0f, &excl, seed, &tok, st)) { ++bad; break; }
            if (tok < 0 || tok >= V || tok == excl) { ++bad; continue; }
            ++cnt[(size_t)tok];
            if (i == 0) {
                z_rel = std::fabs((double)st[1] / z - 1.0);
                if (excl >= 0) pd_err = std::fabs((double)st[2] / (double)st[1] - p[(size_t)excl]);
            }
        }
        double chi = 0.0;
        int dof = 0;
        chi_square(cnt, n_draw, excl, chi, dof);
        // ~6 sigma above the chi-square mean (dof) for a deterministic, non-flaky gate.
        const double lim = (double)dof + 6.0 * std::sqrt(2.0 * (double)std::max(dof, 1));
        const bool ok = (bad == 0) && (chi <= lim) && (z_rel < 1e-4) && (pd_err < 1e-5);
        std::printf("[selftest] sampler %-8s T=%.1f: %d draws, bad=%d, chi2=%.1f (dof=%d, gate %.1f), |Z/Z_ref-1|=%.2e, |p(excl)-ref|=%.2e, p(argmax)=%.4f -> %s\n",
                    (mode == 0) ? "bonus" : "residual", temp, n_draw, bad, chi, dof, lim, z_rel, pd_err,
                    p[(size_t)amax], ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }

    // min_p: no draw may fall outside the kept set.
    {
        const float min_p = 0.05f;
        int out_of_set = 0, bad = 0;
        for (int i = 0; i < 4000; ++i) {
            int32_t tok = -1;
            float st[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            const int32_t excl = -1;
            if (!eng.sample_rows_full_to_host(1, inv_t, min_p, &excl, 0xA5A5ull + (uint64_t)i, &tok, st) ||
                tok < 0 || tok >= V) { ++bad; continue; }
            if (p[(size_t)tok] < (double)min_p * p[(size_t)amax] * (1.0 - 1e-5)) ++out_of_set;
        }
        const bool ok = (bad == 0 && out_of_set == 0);
        std::printf("[selftest] sampler min_p=%.2f: 4000 draws, bad=%d, outside kept set=%d -> %s\n",
                    min_p, bad, out_of_set, ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }

    // Heterogeneous batch sampler: 4 items with distinct (inv_t, min_p, seed) must
    // match 4 independent solo draws bit-for-bit (batch-invariance of seeded sampling).
    {
        const int32_t  rows[4]  = { 0, 0, 0, 0 };
        const float    invts[4] = { 1.0f / 0.6f, 1.0f / 1.0f, 1.0f / 1.5f, 1.0f / 2.0f };
        const float    minps[4] = { 0.0f, 0.05f, 0.0f, 0.10f };
        const uint64_t seeds[4] = { 0x1111222233334444ull, 0x5555666677778888ull,
                                    0x9999AAAABBBBCCCCull, 0xDDDDEEEEFFFF0001ull };
        int32_t b_tok[4] = { -1, -1, -1, -1 };
        float   b_st[16] = {};
        int mismatches = 0;
        bool ok = eng.sample_batch_rows_full_to_host(4, rows, invts, minps, seeds, b_tok, b_st);
        if (ok) {
            for (int k = 0; k < 4; ++k) {
                int32_t s_tok = -1;
                float   s_st[4] = {};
                const int32_t excl = -1;
                if (!eng.sample_rows_full_to_host(1, invts[k], minps[k], &excl, seeds[k], &s_tok, s_st) ||
                    s_tok != b_tok[k] || s_st[1] != b_st[4 * k + 1]) {
                    ++mismatches;
                }
            }
        }
        ok = ok && (mismatches == 0);
        std::printf("[selftest] sampler batch(4x heterogeneous): mismatches=%d -> %s\n",
                    mismatches, ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }
    std::fflush(stdout);
    return failures;
}

SelftestResult run_selftest_mtp(InklingEngine & eng, DSparkDrafter & dr, float threshold, bool bench) {
    SelftestResult res;
    const std::vector<int32_t> prompt = { 1, 100, 200, 300, 400 };
    const int32_t probe = 77;
    const int32_t tok[8] = { 11, 22, 33, 44, 55, 66, 77, 88 };

    // ---- (0) drafter GEMV differential ------------------------------------
    (void)dr.debug_gemv_check();
    (void)dr.debug_shortlist_check();
    {
        // anchor sensitivity: the block input must depend on the anchor token
        std::vector<float> l0(N_EMBD);
        std::vector<float> b0(7 * 1024), b1(7 * 1024);
        (void)hipStreamSynchronize(eng.stream());
        (void)dr.draft(100, 5);
        (void)dr.debug_base_copy(b0.data());
        (void)dr.draft(101, 5);
        (void)dr.debug_base_copy(b1.data());
        double md = 0.0;
        for (size_t i = 0; i < b0.size(); ++i) md = std::max(md, (double)std::fabs((double)b0[i] - (double)b1[i]));
        (void)l0;
        std::printf("[selftest] draft anchor sensitivity (100 vs 101): max |d(logit)| = %.6g\n", md);
        // context sensitivity: same anchor, empty vs populated draft context
        (void)dr.draft(100, 5);
        (void)dr.debug_base_copy(b0.data());
        dr.reset();
        (void)dr.draft(100, 5);
        (void)dr.debug_base_copy(b1.data());
        double mc = 0.0;
        for (size_t i = 0; i < b0.size(); ++i) mc = std::max(mc, (double)std::fabs((double)b0[i] - (double)b1[i]));
        std::printf("[selftest] draft context sensitivity (populated vs reset): max |d(logit)| = %.6g\n", mc);
    }
    {
        std::vector<float> tn(N_EMBD), on(N_EMBD);
        (void)hipMemcpy(tn.data(), eng.weights().tok_norm, tn.size() * 4, hipMemcpyDeviceToHost);
        (void)hipMemcpy(on.data(), eng.weights().output_norm, on.size() * 4, hipMemcpyDeviceToHost);
        double a = 0.0, b = 0.0;
        for (int i = 0; i < N_EMBD; ++i) { a += (double)tn[(size_t)i] * tn[(size_t)i]; b += (double)on[(size_t)i] * on[(size_t)i]; }
        std::printf("[selftest] tok_norm rms=%.4f  output_norm rms=%.4f\n",
                    std::sqrt(a / N_EMBD), std::sqrt(b / N_EMBD));
    }

    // ---- (A) verify(m) == m sequential decode steps ------------------------
    if (std::getenv("INKLING_ST_A1") != nullptr) {
        // debug: does the batched verify path reproduce a single decode step?
        std::vector<float> ref_logits;
        if (!st_prefill_fixture(eng, nullptr, prompt, nullptr)) { st_fail(res, "prefill"); return res; }
        const int P1 = eng.current_pos();
        const int32_t ra = eng.decode_step(tok[0], &ref_logits);
        if (!st_prefill_fixture(eng, nullptr, prompt, nullptr)) { st_fail(res, "prefill"); return res; }
        if (!eng.verify_step(tok, 1, P1)) { st_fail(res, "verify_step"); return res; }
        std::vector<float> vlog;
        eng.verify_logits_to_host(1, vlog);
        double md = 0.0;
        const bool eq = st_logits_equal(vlog, ref_logits, &md);
        std::printf("  [A1] m=1: argmax %d vs %d, max |d(logit)| = %.6g\n", eng.host_target_ids()[0], ra, md);
        (void)eq;
    }
    {
        std::printf("[selftest] A: verify(m) vs sequential decode (argmax + bit-exact logits)\n");
        for (int m : {2, 4, 8}) {
            std::vector<std::vector<float>> ref_logits((size_t)m);
            std::vector<int32_t> ref_arg((size_t)m, -1);
            if (!st_prefill_fixture(eng, nullptr, prompt, nullptr)) { st_fail(res, "prefill"); return res; }
            const int P = eng.current_pos();
            for (int r = 0; r < m; ++r) {
                ref_arg[(size_t)r] = eng.decode_step(tok[r], &ref_logits[(size_t)r]);
            }
            if (!st_prefill_fixture(eng, nullptr, prompt, nullptr)) { st_fail(res, "prefill"); return res; }
            if (!eng.verify_step(tok, m, P)) { st_fail(res, "verify_step"); return res; }
            const int32_t * tgt = eng.host_target_ids();
            std::vector<float> vlog;
            eng.verify_logits_to_host(m, vlog);
            bool ok = true;
            for (int r = 0; r < m; ++r) {
                if (tgt[r] != ref_arg[(size_t)r]) {
                    std::printf("  [FAIL] m=%d row %d: verify argmax %d != decode %d\n", m, r, tgt[r], ref_arg[(size_t)r]);
                    ok = false;
                }
                std::vector<float> row(vlog.begin() + (size_t)r * VOCAB_UNPADDED,
                                       vlog.begin() + (size_t)(r + 1) * VOCAB_UNPADDED);
                double md = 0.0;
                if (!st_logits_equal(row, ref_logits[(size_t)r], &md)) {
                    std::printf("  [FAIL] m=%d row %d: logits differ (max |d| = %.6g)\n", m, r, md);
                    ok = false;
                }
            }
            if (ok) std::printf("  [ok]   m=%d: %d rows bit-identical to sequential decode\n", m, m);
            if (!ok) ++res.failures;
        }
    }

    // ---- (B) verify_rollback(commit) == decode after `commit` tokens -------
    {
        std::printf("[selftest] B: verify_rollback(commit) vs decode (bit-exact next-step logits)\n");
        for (int m : {2, 4, 8}) {
            bool all_ok = true;
            for (int commit = 1; commit <= m; ++commit) {
                std::vector<float> ref_logits;
                std::vector<std::vector<float>> ref_hists;
                if (!st_prefill_fixture(eng, nullptr, prompt, nullptr)) { st_fail(res, "prefill"); return res; }
                const int P = eng.current_pos();
                for (int r = 0; r < commit; ++r) (void)eng.decode_step(tok[r], nullptr);
                ref_hists.resize((size_t)N_LAYER_SLOTS);
                for (int sl = 0; sl < N_LAYER_SLOTS; ++sl) eng.debug_dump_hist(sl, ref_hists[(size_t)sl]);
                (void)eng.decode_step(probe, &ref_logits);
                if (!st_prefill_fixture(eng, nullptr, prompt, nullptr)) { st_fail(res, "prefill"); return res; }
                if (!eng.verify_step(tok, m, P)) { st_fail(res, "verify_step"); return res; }
                eng.verify_rollback(P, commit, m);
                if (eng.current_pos() != P + commit) {
                    std::printf("  [FAIL] m=%d commit=%d: cur_pos %d != %d\n", m, commit, eng.current_pos(), P + commit);
                    all_ok = false;
                    continue;
                }
                std::vector<float> test_logits;
                if (std::getenv("INKLING_ST_DEBUG") != nullptr) {
                    // compare the probe's short-conv history with the reference's
                    std::vector<float> h_test;
                    int wslot = -1, wch = -1;
                    double worst = 0.0;
                    for (int sl = 0; sl < N_LAYER_SLOTS; ++sl) {
                        eng.debug_dump_hist(sl, h_test);
                        const std::vector<float> & h_ref = ref_hists[(size_t)sl];
                        for (size_t i = 0; i < h_ref.size(); ++i) {
                            const double d = std::fabs((double)h_ref[i] - (double)h_test[i]);
                            if (d > worst) { worst = d; wslot = sl; wch = (int)i; }
                        }
                    }
                    {
                        std::vector<float> ht;
                        std::string bad_slots;
                        for (int sl = 0; sl < N_LAYER_SLOTS; ++sl) {
                            eng.debug_dump_hist(sl, ht);
                            double w2 = 0.0;
                            for (size_t i = 0; i < ht.size(); ++i) {
                                w2 = std::max(w2, (double)std::fabs((double)ref_hists[(size_t)sl][i] - (double)ht[i]));
                            }
                            if (w2 > 0.0) bad_slots += " " + std::to_string(sl) + "(" + std::to_string((int)w2) + ")";
                        }
                        std::printf("           differing slots:%s\n", bad_slots.c_str());
                    }
                    const int c = wch / N_CONV_HIST;
                    std::printf("    [dbg] m=%d commit=%d PRE-PROBE hist max|d|=%.6g slot %d ch %d (tap ffn=%d)\n",
                                m, commit, worst, wslot, c,
                                c >= 2 * N_KV + N_EMBD);
                    if (wslot >= 0) {
                        eng.debug_dump_hist(wslot, h_test);
                        std::printf("           ref=[%.6f %.6f %.6f] test=[%.6f %.6f %.6f]\n",
                                    ref_hists[(size_t)wslot][c * 3 + 0], ref_hists[(size_t)wslot][c * 3 + 1],
                                    ref_hists[(size_t)wslot][c * 3 + 2], h_test[c * 3 + 0],
                                    h_test[c * 3 + 1], h_test[c * 3 + 2]);
                    }
                }
                (void)eng.decode_step(probe, &test_logits);
                double md = 0.0;
                if (!st_logits_equal(ref_logits, test_logits, &md)) {
                    std::printf("  [FAIL] m=%d commit=%d: next-step logits differ (max |d| = %.6g)\n", m, commit, md);
                    all_ok = false;
                }
            }
            if (all_ok) std::printf("  [ok]   m=%d: rollback exact for commit = 1..%d\n", m, m);
            if (!all_ok) ++res.failures;
        }
    }

    // ---- (C) 64-token greedy parity: plain decode vs MTP ------------------
    {
        std::printf("[selftest] C: 64-token greedy parity (non-MTP vs --mtp)\n");
        std::vector<float> logits;
        if (!st_prefill_fixture(eng, nullptr, prompt, &logits)) { st_fail(res, "prefill"); return res; }
        int32_t cur = st_argmax(logits.data(), VOCAB_UNPADDED);
        std::vector<int32_t> ref;
        for (int i = 0; i < 64; ++i) {
            ref.push_back(cur);
            cur = eng.decode_step(cur, nullptr);
            if (cur < 0) break;
        }
        // MTP run (the tap hook fills the drafter context during the prefill)
        MtpSession mtp;
        mtp.eng = &eng;
        mtp.drafter = &dr;
        mtp.threshold = threshold;
        mtp.active = true;
        if (!st_prefill_fixture(eng, &dr, prompt, &logits)) { st_fail(res, "prefill"); return res; }
        cur = st_argmax(logits.data(), VOCAB_UNPADDED);
        dr.reset();
        std::vector<int32_t> got;
        double gen_ms = 0.0;
        if (!mtp_generate(mtp, cur, (int)ref.size(), got, nullptr, &gen_ms)) { st_fail(res, "mtp_generate"); return res; }
        int mismatch = -1;
        for (size_t i = 0; i < ref.size() && i < got.size(); ++i) {
            if (ref[i] != got[i]) { mismatch = (int)i; break; }
        }
        if (got.size() != ref.size()) {
            std::printf("  [FAIL] token count %zu != %zu\n", got.size(), ref.size());
            ++res.failures;
        } else if (mismatch >= 0) {
            std::printf("  [FAIL] mismatch at token %d: non-MTP %d vs MTP %d\n", mismatch, ref[(size_t)mismatch],
                        got[(size_t)mismatch]);
            ++res.failures;
        } else {
            std::printf("  [ok]   all %zu tokens identical\n", ref.size());
        }
        mtp_print_stats(mtp, (int)got.size(), gen_ms);
    }

    // ---- (D) timing breakdown for Step 8B ---------------------------------
    if (bench) {
        std::printf("[selftest] D: verify(m) / draft timing breakdown\n");
        std::vector<float> logits;
        if (!st_prefill_fixture(eng, nullptr, prompt, &logits)) { st_fail(res, "prefill"); return res; }
        const int32_t tokv[8] = { 11, 22, 33, 44, 55, 66, 77, 88 };
        for (int m : {1, 2, 4, 8}) {
            const int reps = (m == 1) ? 16 : (m == 2 ? 8 : 4);
            (void)eng.verify_step(tokv, m, eng.current_pos()); // warm
            hipEvent_t e0, e1;
            (void)hipEventCreate(&e0);
            (void)hipEventCreate(&e1);
            (void)hipEventRecord(e0, eng.stream());
            for (int i = 0; i < reps; ++i) {
                (void)eng.verify_step(tokv, m, eng.current_pos());
            }
            (void)hipEventRecord(e1, eng.stream());
            (void)hipStreamSynchronize(eng.stream());
            float ms = 0.0f;
            (void)hipEventElapsedTime(&ms, e0, e1);
            std::printf("  verify m=%d: %.3f ms/call (%.3f ms per row)\n", m, ms / reps, ms / reps / m);
            (void)hipEventDestroy(e0);
            (void)hipEventDestroy(e1);
        }
        {
            const int reps = 8;
            (void)dr.draft(1000, eng.current_pos());
            hipEvent_t e0, e1;
            (void)hipEventCreate(&e0);
            (void)hipEventCreate(&e1);
            (void)hipEventRecord(e0, eng.stream());
            for (int i = 0; i < reps; ++i) (void)dr.draft(1000 + i, eng.current_pos());
            (void)hipEventRecord(e1, eng.stream());
            (void)hipStreamSynchronize(eng.stream());
            float ms = 0.0f;
            (void)hipEventElapsedTime(&ms, e0, e1);
            std::printf("  draft: %.3f ms/call (ctx=%d)\n", ms / reps, dr.ctx());
            (void)hipEventDestroy(e0);
            (void)hipEventDestroy(e1);
        }
    }
    return res;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    std::string mode = argv[1];
    if (mode == "--version" || mode == "-V" || mode == "-v" || mode == "version") {
        if (std::strcmp(NIB_GIT_SHA, "unknown") != 0 && NIB_GIT_SHA[0] != '\0') {
            std::printf("nib %s (%s, git %s)\n", NIB_VERSION, NIB_GPU_TARGET, NIB_GIT_SHA);
        } else {
            std::printf("nib %s (%s)\n", NIB_VERSION, NIB_GPU_TARGET);
        }
        return 0;
    }
    if (mode == "--help" || mode == "-h" || mode == "help") {
        print_usage(argv[0]);
        return 0;
    }
    const char * env_model  = std::getenv("NIB_MODEL");
    const char * env_mmproj = std::getenv("NIB_MMPROJ");
    std::string model_path  = (env_model && env_model[0])   ? env_model  : "models/Inkling-2-Mini-NVFP4-Mixed.gguf";
    std::string mmproj_path = (env_mmproj && env_mmproj[0]) ? env_mmproj : "models/mmproj-Inkling-2-Mini-Q8_0.gguf";
    std::string image_path;
    std::string prompt_text = "Describe this image in detail.";
    std::string prompt_file;
    std::string dump_embeds_path;
    std::string tokens_path;
    std::string out_path;
    std::string mtp_path;
    std::string serve_host = "127.0.0.1";
    int serve_port = 4444;
    std::string serve_model_name = "inkling-2-mini";
    bool no_vision = false;
    bool serve_strict_slot_ctx = false;
    int32_t serve_max_slot_ctx = 0;
    float serve_default_temp = 0.6f;
    float serve_default_top_p = 0.95f;
    float serve_default_min_p = 0.0f;
    int serve_default_top_k = 40;
    float spec_threshold = 0.8f;
    int draft_ctx = DSparkDrafter::DEFAULT_DRAFT_CTX;
    bool bench_mtp = false;
    bool explicit_no_verify_batch = false;
    bool explicit_slots = false;
    EngineConfig cfg;
    std::vector<int> depths = { 0 };
    int n_prompt     = 512;
    int n_gen        = 128;
    int n_reps       = 3;
    int bench_vision = 0;
    bool decode_ub1  = false;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--mmproj" && i + 1 < argc) mmproj_path = argv[++i];
        else if (a == "--image" && i + 1 < argc) image_path = argv[++i];
        else if (a == "--prompt" && i + 1 < argc) prompt_text = argv[++i];
        else if (a == "--prompt-file" && i + 1 < argc) prompt_file = argv[++i];
        else if (a == "--sparse-pages" && i + 1 < argc) cfg.sparse_pages = std::atoi(argv[++i]);
        else if (a == "--dump-embeds" && i + 1 < argc) dump_embeds_path = argv[++i];
        else if (a == "--bench-vision" && i + 1 < argc) bench_vision = std::atoi(argv[++i]);
        else if (a == "-t" && i + 1 < argc) tokens_path = argv[++i];
        else if (a == "-o" && i + 1 < argc) out_path = argv[++i];
        else if (a == "-c" && i + 1 < argc) cfg.max_ctx = std::atoi(argv[++i]);
        else if (a == "--kv-type" && i + 1 < argc) {
            std::string kt = argv[++i];
            if (kt == "q8_0") cfg.kv_type = KV_Q8_0;
            else if (kt == "q4_1") cfg.kv_type = KV_Q4_1;
            else if (kt == "q4_0") cfg.kv_type = KV_Q4_0;
            else {
                std::fprintf(stderr, "Unknown --kv-type '%s' (expected q8_0, q4_1, or q4_0)\n", kt.c_str());
                return 1;
            }
        }
        else if (a == "-p" && i + 1 < argc) {
            if (mode == "vision") prompt_text = argv[++i];
            else n_prompt = std::atoi(argv[++i]);
        }
        else if (a == "-n" && i + 1 < argc) n_gen = std::atoi(argv[++i]);
        else if (a == "-r" && i + 1 < argc) n_reps = std::atoi(argv[++i]);
        else if (a == "--no-graph") cfg.use_hip_graph = false;
        else if (a == "--verify-batch") cfg.verify_batch = 1;
        else if (a == "--no-verify-batch") { cfg.verify_batch = 0; explicit_no_verify_batch = true; }
        else if (a == "--decode-ub1") decode_ub1 = true;
        else if (a == "--wo-v3" && i + 1 < argc) {
            if (!parse_v3_variant(argv[++i], /*allow_alt2=*/true, /*allow_ship=*/false, &cfg.wo_v3)) {
                std::fprintf(stderr, "Unknown --wo-v3 value '%s' (off|auto|alt2)\n", argv[i]);
                return 1;
            }
        }
        else if (a == "--moe-gu-v3" && i + 1 < argc) {
            if (!parse_v3_variant(argv[++i], /*allow_alt2=*/false, /*allow_ship=*/true, &cfg.moe_gu_v3)) {
                std::fprintf(stderr, "Unknown --moe-gu-v3 value '%s' (off|auto|alt)\n", argv[i]);
                return 1;
            }
        }
        else if (a == "--moe-dn-v3" && i + 1 < argc) {
            if (!parse_v3_variant(argv[++i], /*allow_alt2=*/false, /*allow_ship=*/false, &cfg.moe_dn_v3)) {
                std::fprintf(stderr, "Unknown --moe-dn-v3 value '%s' (off|auto|alt)\n", argv[i]);
                return 1;
            }
        }
        else if (a == "--moe-fused-v3" && i + 1 < argc) {
            if (!parse_v3_variant(argv[++i], /*allow_alt2=*/false, /*allow_ship=*/false, &cfg.moe_fused_v3)) {
                std::fprintf(stderr, "Unknown --moe-fused-v3 value '%s' (off|auto)\n", argv[i]);
                return 1;
            }
        }
        else if (a == "-d" && i + 1 < argc) {
            depths.clear();
            std::stringstream ss(argv[++i]);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) depths.push_back(std::atoi(item.c_str()));
            }
        }
        else if (a == "--fa-swa" && i + 1 < argc) {
            if (!parse_v1v2v3(argv[++i], &cfg.fa_swa_v3)) {
                std::fprintf(stderr, "Unknown --fa-swa value '%s' (v1|v2|v3|v4)\n", argv[i]);
                return 1;
            }
        }
        else if (a == "--fa-select" && i + 1 < argc) {
            if (!parse_v1v2v3(argv[++i], &cfg.fa_select_v3)) {
                std::fprintf(stderr, "Unknown --fa-select value '%s' (v1|v2|v3|v4)\n", argv[i]);
                return 1;
            }
        }
        else if (a == "--fa-global" && i + 1 < argc) {
            if (!parse_v1v2v3(argv[++i], &cfg.fa_global_v3)) {
                std::fprintf(stderr, "Unknown --fa-global value '%s' (v1|v2|v3|v4)\n", argv[i]);
                return 1;
            }
        }
        else if (a == "--sparse-meta" && i + 1 < argc) {
            if (!parse_meta_fmt(argv[++i], &cfg.sparse_meta)) {
                std::fprintf(stderr, "Unknown --sparse-meta value '%s' (q8|q4)\n", argv[i]);
                return 1;
            }
        }
        else if (a == "--mtp" && i + 1 < argc) mtp_path = argv[++i];
        else if (a == "--spec-threshold" && i + 1 < argc) spec_threshold = (float)std::atof(argv[++i]);
        else if (a == "--draft-ctx" && i + 1 < argc) draft_ctx = std::atoi(argv[++i]);
        else if (a == "--bench-mtp") bench_mtp = true;
        else if (a == "--fa-q-lds" && i + 1 < argc) cfg.fa_q_lds = std::atoi(argv[++i]);
        else if (a == "--fa-cached" && i + 1 < argc) cfg.fa_cached = std::atoi(argv[++i]);
        else if (a == "--fa-swa-splits" && i + 1 < argc) cfg.fa_swa_splits = std::atoi(argv[++i]);
        else if (a == "--fa-sel-chunks" && i + 1 < argc) cfg.fa_sel_chunks = std::atoi(argv[++i]);
        else if (a == "--fa-glob-splits" && i + 1 < argc) cfg.fa_glob_splits = std::atoi(argv[++i]);
        else if (a == "--moe-ship-fused" && i + 1 < argc) cfg.moe_ship_fused = std::atoi(argv[++i]);
        else if (a == "--moe-q8-fused" && i + 1 < argc) cfg.moe_q8_fused = std::atoi(argv[++i]);
        else if (a == "--b-v3" && i + 1 < argc) cfg.b_v3 = std::atoi(argv[++i]);
        else if (a == "--mix-fast" && i + 1 < argc) cfg.mix_fast = std::atoi(argv[++i]);
        else if (a == "--qkv-grid" && i + 1 < argc) cfg.qkv_grid = std::atoi(argv[++i]);
        else if (a == "--c-v3" && i + 1 < argc) cfg.c_v3 = std::atoi(argv[++i]);
        // Step 7I-1 super-fusions (0 = shipped path, 1 = fused).
        else if (a == "--fuse-wob" && i + 1 < argc) cfg.fuse_wob = std::atoi(argv[++i]);
        else if (a == "--fuse-woc" && i + 1 < argc) cfg.fuse_woc = std::atoi(argv[++i]);
        else if (a == "--fuse-mix-qkv" && i + 1 < argc) cfg.fuse_mix_qkv = std::atoi(argv[++i]);
        else if (a == "--fuse-a-qkv" && i + 1 < argc) cfg.fuse_a_qkv = std::atoi(argv[++i]);
        else if (a == "--fuse-minmax" && i + 1 < argc) cfg.fuse_minmax = std::atoi(argv[++i]);
        // Step 7J: v_dot2_f32_f16 + __half2 LDS weight-stream kernels.
        else if (a == "--fdot2-moe" && i + 1 < argc) cfg.fdot2_moe = std::atoi(argv[++i]);
        else if (a == "--fdot2-qkv" && i + 1 < argc) cfg.fdot2_qkv = std::atoi(argv[++i]);
        else if (a == "--fdot2-wo" && i + 1 < argc) cfg.fdot2_wo = std::atoi(argv[++i]);
        else if (a == "--fdot2-wo-shape" && i + 1 < argc) cfg.fdot2_wo_shape = std::atoi(argv[++i]);
        else if (a == "--fdot2-lm" && i + 1 < argc) cfg.fdot2_lm = std::atoi(argv[++i]);
        else if (a == "--fdot2-q8moe" && i + 1 < argc) cfg.fdot2_q8moe = std::atoi(argv[++i]);
        else if (a == "--fdot2" && i + 1 < argc) {
            // Only lm_head_fdot2 wins *and* passes the PPL gate, so the
            // umbrella toggles exactly that.  --fdot2-wo / -moe / -qkv / -q8moe
            // remain individually selectable for the A/B table.
            const int v = std::atoi(argv[++i]);
            cfg.fdot2_lm = v ? 1 : 0;
            cfg.fdot2_wo = 0;
            cfg.fdot2_moe = 0;
            cfg.fdot2_qkv = 0;
            cfg.fdot2_q8moe = 0;
        }
        else if (a == "--qkv-super" && i + 1 < argc) {
            if (!parse_v1v2v3(argv[++i], &cfg.qkv_super_v3)) {
                std::fprintf(stderr, "Unknown --qkv-super value '%s' (v2|v3)\n", argv[i]);
                return 1;
            }
        }
        else if (a == "--host" && i + 1 < argc) serve_host = argv[++i];
        else if (a == "--port" && i + 1 < argc) serve_port = std::atoi(argv[++i]);
        else if (a == "--model-name" && i + 1 < argc) serve_model_name = argv[++i];
        else if (a == "--slots" && i + 1 < argc) {
            cfg.n_slots = std::atoi(argv[++i]);
            explicit_slots = true;
        }
        else if (a == "--strict-slot-ctx") serve_strict_slot_ctx = true;
        else if (a == "--max-slot-ctx" && i + 1 < argc) serve_max_slot_ctx = std::atoi(argv[++i]);
        else if (a == "--default-temp" && i + 1 < argc) serve_default_temp = (float)std::atof(argv[++i]);
        else if (a == "--default-top-p" && i + 1 < argc) serve_default_top_p = (float)std::atof(argv[++i]);
        else if (a == "--default-min-p" && i + 1 < argc) serve_default_min_p = (float)std::atof(argv[++i]);
        else if (a == "--default-top-k" && i + 1 < argc) serve_default_top_k = std::atoi(argv[++i]);
        else if (a == "--no-vision") no_vision = true;
    }

    if (mode == "serve" && !explicit_slots) {
        cfg.n_slots = 4; // Default: 4 concurrent slots (65,536 ctx/slot when concurrent, 262k solo)
    }

    if (!mtp_path.empty() && cfg.verify_batch == 0 && !explicit_no_verify_batch) {
        cfg.verify_batch = 1;   // Step 8C: batched m-token verify is the default for MTP
    }

    if (mode == "selftest") {
        InklingEngine eng;
        cfg.max_ctx = 4096;
        if (!eng.init(model_path, cfg, /*verify_repack=*/true)) return 1;
        std::vector<int32_t> prompt = { 1, 100, 200, 300, 400 };
        std::vector<float> logits_pf, logits_dec;
        eng.prefill(prompt, false, &logits_pf);
        int32_t next_pf = (int32_t)(std::max_element(logits_pf.begin(), logits_pf.end()) - logits_pf.begin());

        // Compare prefill vs token-by-token decode!
        eng.reset_state();
        int32_t next_dec = 0;
        for (size_t i = 0; i < prompt.size(); ++i) {
            next_dec = eng.decode_step(prompt[i], (i + 1 == prompt.size()) ? &logits_dec : nullptr);
        }
        float max_diff = 0.0f;
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int v = 0; v < VOCAB_UNPADDED; ++v) {
            float d = std::fabs(logits_pf[v] - logits_dec[v]);
            if (d > max_diff) max_diff = d;
            dot += (double)logits_pf[v] * logits_dec[v];
            na  += (double)logits_pf[v] * logits_pf[v];
            nb  += (double)logits_dec[v] * logits_dec[v];
        }
        double cos_sim = dot / (std::sqrt(na * nb) + 1e-12);
        std::printf("[SELFTEST] Prefill vs Decode (5 tokens): next_pf=%d next_dec=%d | max_diff=%.5f | cos_sim=%.7f\n",
                    next_pf, next_dec, max_diff, cos_sim);
        int failures = (cos_sim > 0.999) ? 0 : 1;
        failures += run_selftest_sampler(eng, logits_dec);
        if (!mtp_path.empty()) {
            DSparkDrafter dr;
            if (!eng.enable_mtp_io()) { std::fprintf(stderr, "enable_mtp_io failed\n"); return 1; }
            if (!dr.init(mtp_path, eng.weights(), eng.max_ctx(), VOCAB_UNPADDED, eng.stream(), draft_ctx)) return 1;
            eng.set_tap_hook([&dr](const uint16_t * taps, const int32_t * toks, int n, int start) {
                dr.commit(taps, n, start);
                if (toks != nullptr) dr.note_tokens(toks, n);
            });
            SelftestResult r = run_selftest_mtp(eng, dr, spec_threshold, bench_mtp);
            failures += r.failures;
        }
        return failures > 0 ? (failures == 1 && cos_sim <= 0.999 ? 2 : 3) : 0;
    }

    if (mode == "bench") {
        int max_d = 0;
        for (int d : depths) max_d = std::max(max_d, d);
        if (cfg.max_ctx < max_d + n_prompt + n_gen + 64) {
            cfg.max_ctx = max_d + n_prompt + n_gen + 64;
        }
        InklingEngine eng;
        if (!eng.init(model_path, cfg, false)) return 1;
        MtpSession mtp;
        if (!mtp_path.empty()) {
            if (!mtp_session_init(mtp, mtp_path, eng, spec_threshold, true, draft_ctx)) return 1;
        }
        std::printf("[fuses] wo_v3=%d moe_gu_v3=%d moe_dn_v3=%d moe_fused_v3=%d | 7H: moe_ship=%d q8_fused=%d b_v3=%d c_v3=%d\n",
                    cfg.wo_v3, cfg.moe_gu_v3, cfg.moe_dn_v3, cfg.moe_fused_v3,
                    cfg.moe_ship_fused, cfg.moe_q8_fused, cfg.b_v3, cfg.c_v3);
        std::printf("[fuses] 7I-1: wob=%d woc=%d mix_qkv=%d a_qkv=%d minmax=%d\n",
                    cfg.fuse_wob, cfg.fuse_woc, cfg.fuse_mix_qkv, cfg.fuse_a_qkv, cfg.fuse_minmax);
        std::printf("[fuses] 7K: fa_swa=%d fa_global=%d fa_select=%d meta=%s q_lds=%d cached=%d\n",
                    cfg.fa_swa_v3, cfg.fa_global_v3, cfg.fa_select_v3,
                    (cfg.sparse_meta == 1) ? "q4" : "q8", cfg.fa_q_lds, cfg.fa_cached);

        // Warmup both prefill & decode paths once
        {
            std::vector<int32_t> warm(64, 42);
            eng.prefill(warm, false, nullptr);
            eng.benchmark_decode_steps(42, 8, nullptr);
            eng.reset_state();
        }

        std::printf("\n| depth | pp%d (tok/s) | tg%d (tok/s) |\n", n_prompt, n_gen);
        std::printf("|------:|--------------:|-------------:|\n");
        std::fflush(stdout);

        std::sort(depths.begin(), depths.end());
        eng.reset_state();
        if (mtp.active) {
            // One rep per depth (the drafter's context is a sequential state, so
            // rewinding for repeated reps is not possible without a re-commit).
            std::printf("\n| depth | tg%d MTP (tok/s) | avg_acc | avg_m | draft ms | verify ms |\n", n_gen);
            std::printf("|------:|-----------------:|--------:|------:|---------:|----------:|\n");
            std::fflush(stdout);
            for (int depth : depths) {
                if (depth > eng.current_pos()) {
                    const int need = depth - eng.current_pos();
                    std::vector<int32_t> fill((size_t)need);
                    for (int i = 0; i < need; ++i) fill[(size_t)i] = 100 + ((eng.current_pos() + i) % 10000);
                    if (!eng.prefill(fill, false, nullptr)) return 1;
                }
                std::vector<float> logits;
                std::vector<int32_t> anchor_seed = {300};
                if (!eng.prefill(anchor_seed, false, &logits)) return 1;
                const int32_t anchor = (int32_t)(std::max_element(logits.begin(), logits.end()) - logits.begin());
                std::vector<int32_t> out;
                double gen_ms = 0.0;
                if (!mtp_generate(mtp, anchor, n_gen, out, nullptr, &gen_ms)) return 1;
                std::printf("| %5d | %16.2f | %7.2f | %5.2f | %8.2f | %9.2f |\n",
                            depth, (double)out.size() / (gen_ms / 1000.0),
                            mtp.avg_accepted(), mtp.avg_m(),
                            mtp.draft_ms / (double)mtp.draft_calls,
                            mtp.verify_ms / (double)mtp.verify_calls);
                std::fflush(stdout);
                mtp.rounds = 0; mtp.accepted = 0; mtp.m_sum = 0;
                mtp.draft_ms = 0.0; mtp.verify_ms = 0.0;
                mtp.draft_calls = 0; mtp.verify_calls = 0;
            }
            return 0;
        }
        for (int depth : depths) {
            if (depth > eng.current_pos()) {
                const int need = depth - eng.current_pos();
                std::vector<int32_t> fill(need);
                for (int i = 0; i < need; ++i) fill[i] = 100 + ((eng.current_pos() + i) % 10000);
                eng.prefill(fill, false, nullptr);
            }
            std::vector<double> pp_tps, tg_tps;
            for (int rep = 0; rep < n_reps; ++rep) {
                eng.rewind_pos_for_bench(depth);
                std::vector<int32_t> pchunk(n_prompt);
                for (int i = 0; i < n_prompt; ++i) pchunk[i] = 200 + (i % 10000);
                double pp_ms = (n_prompt > 0) ? eng.benchmark_prefill_chunk(pchunk) : 0.0;
                double tg_ms = eng.benchmark_decode_steps(300, n_gen, nullptr);
                if (n_prompt > 0) pp_tps.push_back((double)n_prompt / (pp_ms / 1000.0));
                tg_tps.push_back((double)n_gen    / (tg_ms / 1000.0));
            }
            eng.rewind_pos_for_bench(depth);
            auto mean_std = [](const std::vector<double> & v) {
                if (v.empty()) return std::make_pair(0.0, 0.0);
                double m = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
                double s = 0.0;
                for (double x : v) s += (x - m) * (x - m);
                s = (v.size() > 1) ? std::sqrt(s / (v.size() - 1)) : 0.0;
                return std::make_pair(m, s);
            };
            auto pp = mean_std(pp_tps);
            auto tg = mean_std(tg_tps);
            std::printf("| %5d | %8.2f ± %5.2f | %6.2f ± %4.2f |\n",
                        depth, pp.first, pp.second, tg.first, tg.second);
            std::fflush(stdout);
        }
        return 0;
    }

    if (mode == "gen") {
        const std::string & pf = prompt_file.empty() ? tokens_path : prompt_file;
        if (pf.empty()) {
            std::fprintf(stderr, "gen mode requires --prompt-file <text.txt>\n");
            return 1;
        }
        std::ifstream f(pf, std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "Cannot open prompt file '%s'\n", pf.c_str());
            return 1;
        }
        const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

        InklingEngine eng;
        if (!eng.init(model_path, cfg, false)) return 1;

        BpeTokenizer bpe;
        bpe.init(eng.tokenizer());
        const auto toks = bpe.encode(text);
        std::printf("[gen] prompt: %zu bytes -> %zu tokens | n_gen=%d | kv=%s | sparse_pages=%d | max_ctx=%d\n",
                    text.size(), toks.size(), n_gen,
                    (cfg.kv_type == KV_Q4_1) ? "q4_1" : (cfg.kv_type == KV_Q4_0) ? "q4_0" : "q8_0",
                    cfg.sparse_pages, cfg.max_ctx);
        std::fflush(stdout);
        if (toks.size() < 1 || (int)toks.size() + n_gen + 8 > cfg.max_ctx) {
            std::fprintf(stderr, "[gen] prompt+generation (%zu + %d + 8) exceeds max_ctx=%d\n",
                         toks.size(), n_gen, cfg.max_ctx);
            return 1;
        }

        MtpSession mtp;
        if (!mtp_path.empty()) {
            if (!mtp_session_init(mtp, mtp_path, eng, spec_threshold, true, draft_ctx)) return 1;
        }

        std::vector<float> logits;
        auto tp0 = std::chrono::steady_clock::now();
        if (!eng.prefill(toks, false, &logits)) return 1;
        const double pf_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp0).count();
        int32_t cur_tok = (int32_t)(std::max_element(logits.begin(), logits.end()) - logits.begin());
        std::printf("[gen] prefill: %zu tokens in %.2f s (%.1f tok/s) | first next token id=%d\n",
                    toks.size(), pf_ms / 1000.0, (double)toks.size() / (pf_ms / 1000.0), cur_tok);
        std::printf("--- GEN OUTPUT ---\n");
        std::fflush(stdout);

        std::vector<int32_t> produced;
        auto td0 = std::chrono::steady_clock::now();
        auto emit = [&bpe](int32_t t) {
            const std::string piece = bpe.decode_token(t, /*skip_special=*/false);
            std::fwrite(piece.data(), 1, piece.size(), stdout);
            std::fflush(stdout);
        };
        if (mtp.active) {
            double gen_ms = 0.0;
            if (!mtp_generate(mtp, cur_tok, n_gen, produced, emit, &gen_ms)) return 1;
            const double dec_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
            std::printf("\n--- END (%zu tokens generated in %.2f ms | final pos=%d) ---\n",
                        produced.size(), dec_ms, eng.current_pos());
            mtp_print_stats(mtp, (int)produced.size(), gen_ms);
            std::printf("[gen] token ids:");
            for (int32_t t : produced) std::printf(" %d", t);
            std::printf("\n");
            return 0;
        }
        for (int step = 0; step < n_gen; ++step) {
            produced.push_back(cur_tok);
            emit(cur_tok);
            if (step + 1 < n_gen) {
                cur_tok = eng.decode_step(cur_tok, nullptr);
                if (cur_tok < 0) break;
            }
        }
        const double dec_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
        std::printf("\n--- END (%zu tokens generated in %.2f ms | final pos=%d) ---\n",
                    produced.size(), dec_ms, eng.current_pos());
        std::printf("[gen] token ids:");
        for (int32_t t : produced) std::printf(" %d", t);
        std::printf("\n");
        return 0;
    }

    if (mode == "ppl" || mode == "oracle") {
        cfg.need_all_logits_buf = true; // every row's logits are scored here
        auto toks = load_token_ids_file(tokens_path);
        if (toks.size() < 2) {
            std::fprintf(stderr, "Need >=2 token IDs in '%s'\n", tokens_path.c_str());
            return 1;
        }
        if (cfg.max_ctx < (int)toks.size() + 64) cfg.max_ctx = (int)toks.size() + 64;
        InklingEngine eng;
        if (!eng.init(model_path, cfg, false)) return 1;

        if (mode == "oracle") {
            std::vector<float> logits;
            if (decode_ub1) {
                for (size_t i = 0; i < toks.size(); ++i) {
                    eng.decode_step(toks[i], (i + 1 == toks.size()) ? &logits : nullptr);
                }
            } else {
                eng.prefill(toks, false, &logits);
            }
            if (!out_path.empty()) {
                std::ofstream of(out_path, std::ios::binary);
                of.write((const char *)logits.data(), logits.size() * sizeof(float));
            }
            // Print top-5 tokens & logits
            std::vector<int> idx(VOCAB_UNPADDED);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                              [&](int a, int b) { return logits[a] > logits[b]; });
            std::printf("Top-5 logits after %zu tokens (%s):\n", toks.size(), decode_ub1 ? "decode_ub1" : "prefill");
            for (int k = 0; k < 5; ++k) {
                std::printf("  #%d: id=%d logit=%.6f\n", k + 1, idx[k], logits[idx[k]]);
            }
            return 0;
        }

        // Exact llama-perplexity chunked PPL evaluation (c=512, second half of each chunk scored!)
        const int n_ctx_chunk = 512;
        const int n_chunks = (int)toks.size() / n_ctx_chunk;
        double total_nll = 0.0;
        int total_count = 0;
        for (int c = 0; c < n_chunks; ++c) {
            eng.reset_state();
            std::vector<int32_t> chunk(toks.begin() + c * n_ctx_chunk, toks.begin() + (c + 1) * n_ctx_chunk);
            std::vector<float> logits;
            if (decode_ub1) {
                logits.resize((size_t)n_ctx_chunk * VOCAB_UNPADDED);
                std::vector<float> row;
                for (int i = 0; i < n_ctx_chunk; ++i) {
                    eng.decode_step(chunk[i], &row);
                    std::memcpy(logits.data() + (size_t)i * VOCAB_UNPADDED, row.data(), (size_t)VOCAB_UNPADDED * sizeof(float));
                }
            } else {
                eng.prefill(chunk, /*compute_all_logits=*/true, &logits);
            }
            // Match llama-perplexity: evaluate positions [n_ctx_chunk/2 .. n_ctx_chunk - 2] predicting [n_ctx_chunk/2 + 1 .. n_ctx_chunk - 1]
            const int first = n_ctx_chunk / 2;
            const int cnt   = n_ctx_chunk - 1 - first;
            double chunk_nll = compute_nll_from_logits(logits.data(), n_ctx_chunk, chunk.data(), first, cnt);
            total_nll   += chunk_nll;
            total_count += cnt;
            std::printf("[%d] PPL = %.4f (running = %.4f)\n",
                        c + 1, std::exp(chunk_nll / cnt), std::exp(total_nll / total_count));
            std::fflush(stdout);
        }
        std::printf("Final PPL (%d chunks, %d tokens, %s) = %.4f\n",
                    n_chunks, total_count, decode_ub1 ? "decode_ub1" : "prefill_ub512",
                    std::exp(total_nll / total_count));
        return 0;
    }

    if (mode == "vision") {
        if (image_path.empty()) {
            std::fprintf(stderr, "vision mode requires --image <path>\n");
            return 1;
        }
        const int prefill_depth = depths.empty() ? 0 : depths[0];
        if (cfg.max_ctx < prefill_depth + n_gen + 2048) {
            cfg.max_ctx = prefill_depth + n_gen + 2048;
        }

        InklingEngine eng;
        if (!eng.init(model_path, cfg, false)) return 1;

        MtpSession mtp;
        if (!mtp_path.empty()) {
            if (!mtp_session_init(mtp, mtp_path, eng, spec_threshold, true, draft_ctx)) return 1;
        }

        VisionEncoder vision;
        if (!vision.init(mmproj_path, eng.stream())) return 1;

        BpeTokenizer bpe;
        bpe.init(eng.tokenizer());

        // Encode image (with optional benchmark repetitions)
        VisionEncodeResult vres;
        std::vector<float> host_embeds;
        const bool want_host = !dump_embeds_path.empty();

        if (!vision.encode_image_file(image_path, vres, want_host ? &host_embeds : nullptr)) {
            return 1;
        }
        const double total_vram_gib = (double)(eng.total_vram_bytes() + vision.vram_bytes()) / (1024.0 * 1024.0 * 1024.0);
        std::printf("[vision] encoded %s (%dx%d -> grid %dx%d -> %d merged tokens) | preproc=%.2f ms | gpu=%.2f ms | total_vram=%.2f GiB (llm=%.2f + vision=%.2f)\n",
                    image_path.c_str(), vres.img_w, vres.img_h, vres.grid_w, vres.grid_h, vres.n_tokens,
                    vres.preproc_ms, vres.gpu_ms, total_vram_gib,
                    (double)eng.total_vram_bytes() / (1024.0 * 1024.0 * 1024.0),
                    (double)vision.vram_bytes() / (1024.0 * 1024.0 * 1024.0));

        if (bench_vision > 0) {
            std::vector<double> gpu_times;
            for (int r = 0; r < bench_vision; ++r) {
                VisionEncodeResult tmp;
                if (!vision.encode_image_file(image_path, tmp, nullptr)) return 1;
                gpu_times.push_back(tmp.gpu_ms);
            }
            double mean = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / gpu_times.size();
            double var  = 0.0;
            for (double x : gpu_times) var += (x - mean) * (x - mean);
            double sd = (gpu_times.size() > 1) ? std::sqrt(var / (gpu_times.size() - 1)) : 0.0;
            std::printf("[vision-bench] %d reps (warmed): gpu_encode = %.2f ± %.2f ms (%.1f img/s)\n",
                        bench_vision, mean, sd, 1000.0 / mean);
        }

        if (!dump_embeds_path.empty()) {
            std::ofstream of(dump_embeds_path, std::ios::binary);
            of.write(reinterpret_cast<const char *>(host_embeds.data()), host_embeds.size() * sizeof(float));
            std::printf("[vision] dumped %zu floats ([%d, %d]) to %s\n",
                        host_embeds.size(), vres.n_tokens, N_EMBD, dump_embeds_path.c_str());
        }

        // Optional long-context depth fill BEFORE the multimodal user turn
        if (prefill_depth > 0) {
            std::vector<int32_t> fill(prefill_depth);
            for (int i = 0; i < prefill_depth; ++i) fill[i] = 100 + (i % 10000);
            auto tf0 = std::chrono::steady_clock::now();
            if (!eng.prefill(fill, false, nullptr)) return 1;
            double fill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tf0).count();
            std::printf("[vision] prefilled context depth=%d tokens in %.2f s (%.1f tok/s)\n",
                        prefill_depth, fill_s, (double)prefill_depth / fill_s);
        }

        // Exact Inkling 2 Mini multimodal chat sequence (thinking effort = 0):
        //   <|message_system|><|content_text|>Thinking effort level: 0<|end_message|>
        //   <|message_user|><|content_image|>[IMAGE_EMBEDS]<|end_message|>
        //   <|message_user|><|content_text|>{prompt}<|end_message|>
        //   <|message_model|><|content_text|>
        const std::string pre_str  = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_image|>";
        const std::string post_str = "<|end_message|><|message_user|><|content_text|>" + prompt_text + "<|end_message|><|message_model|><|content_text|>";
        auto pre_toks  = bpe.encode(pre_str);
        auto post_toks = bpe.encode(post_str);

        const int32_t id_end_sampling = bpe.token_id("<|content_model_end_sampling|>");
        const int32_t id_end_message  = bpe.token_id("<|end_message|>");
        const int32_t id_eot          = bpe.token_id("<|endoftext|>");

        auto tp0 = std::chrono::steady_clock::now();
        if (!eng.prefill(pre_toks, false, nullptr)) return 1;
        if (!eng.prefill_embeds(vres.d_embeds, vres.n_tokens, false, nullptr)) return 1;
        std::vector<float> logits;
        if (!eng.prefill(post_toks, false, &logits)) return 1;
        double mm_prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp0).count();
        const int total_mm_toks = (int)pre_toks.size() + vres.n_tokens + (int)post_toks.size();

        int32_t cur_tok = (int32_t)(std::max_element(logits.begin(), logits.end()) - logits.begin());

        std::printf("\n--- MODEL RESPONSE (pos=%d, mm_prefill=%d toks in %.2f ms = %.1f tok/s) ---\n",
                    eng.current_pos(), total_mm_toks, mm_prefill_ms, (double)total_mm_toks / (mm_prefill_ms / 1000.0));
        std::fflush(stdout);

        auto td0 = std::chrono::steady_clock::now();
        if (mtp.active) {
            std::vector<int32_t> produced;
            auto emit = [&](int32_t t) {
                if (t == id_end_sampling || t == id_end_message || t == id_eot) return;
                const std::string piece = bpe.decode_token(t, /*skip_special=*/false);
                std::fwrite(piece.data(), 1, piece.size(), stdout);
                std::fflush(stdout);
            };
            double gen_ms = 0.0;
            if (!mtp_generate(mtp, cur_tok, n_gen, produced, emit, &gen_ms)) return 1;
            const double dec_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
            std::printf("\n--- END (%zu tokens generated in %.2f ms | final pos=%d) ---\n",
                        produced.size(), dec_ms, eng.current_pos());
            mtp_print_stats(mtp, (int)produced.size(), gen_ms);
            return 0;
        }
        int gen_count = 0;
        for (int step = 0; step < n_gen; ++step) {
            if (cur_tok == id_end_sampling || cur_tok == id_end_message || cur_tok == id_eot) {
                break;
            }
            std::string piece = bpe.decode_token(cur_tok, /*skip_special=*/false);
            std::fwrite(piece.data(), 1, piece.size(), stdout);
            std::fflush(stdout);
            gen_count++;
            if (step + 1 < n_gen) {
                cur_tok = eng.decode_step(cur_tok, nullptr);
                if (cur_tok < 0) break;
            }
        }
        double dec_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
        std::printf("\n--- END (%d tokens generated in %.2f ms = %.2f tok/s | final pos=%d) ---\n",
                    gen_count, dec_ms, (gen_count > 1) ? ((double)(gen_count - 1) / (dec_ms / 1000.0)) : 0.0,
                    eng.current_pos());
        return 0;
    }

    if (mode == "serve") {
        InklingEngine eng;
        if (!eng.init(model_path, cfg, false)) return 1;

        BpeTokenizer bpe;
        bpe.init(eng.tokenizer());

        MtpSession mtp;
        DSparkDrafter * drafter_ptr = nullptr;
        if (!mtp_path.empty()) {
            if (!mtp_session_init(mtp, mtp_path, eng, spec_threshold, true, draft_ctx)) return 1;
            drafter_ptr = mtp.drafter;
        }

        VisionEncoder vision;
        VisionEncoder * vision_ptr = nullptr;
        if (!no_vision && !mmproj_path.empty()) {
            std::ifstream vf(mmproj_path);
            if (vf.good()) {
                vf.close();
                if (!vision.init(mmproj_path, eng.stream())) {
                    std::fprintf(stderr, "[serve] warning: failed to init vision encoder from '%s'; continuing text-only\n",
                                 mmproj_path.c_str());
                } else {
                    vision_ptr = &vision;
                }
            } else {
                std::fprintf(stderr, "[serve] note: mmproj '%s' not found; starting text-only\n",
                             mmproj_path.c_str());
            }
        }

        ServerConfig scfg;
        scfg.host = serve_host;
        scfg.port = serve_port;
        scfg.model_name = serve_model_name;
        scfg.spec_threshold = spec_threshold;
        scfg.mtp_enabled = (drafter_ptr != nullptr);
        scfg.vision_enabled = (vision_ptr != nullptr);
        scfg.kv_type_name = (cfg.kv_type == KV_Q4_1) ? "q4_1" : (cfg.kv_type == KV_Q4_0) ? "q4_0" : "q8_0";
        scfg.sparse_pages = cfg.sparse_pages;
        scfg.n_slots = eng.num_slots();
        scfg.strict_slot_ctx = serve_strict_slot_ctx;
        scfg.max_slot_ctx = serve_max_slot_ctx;
        scfg.default_temperature = serve_default_temp;
        scfg.default_top_p = serve_default_top_p;
        scfg.default_min_p = serve_default_min_p;
        scfg.default_top_k = serve_default_top_k;

        return run_http_server(eng, bpe, drafter_ptr, vision_ptr, scfg);
    }

    print_usage(argv[0]);
    return 1;
}
