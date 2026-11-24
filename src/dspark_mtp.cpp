// Step 8A: DSpark MTP drafter host side — safetensors loader (tiny JSON header
// parser), GPU weight repacking, shortlist head, draft-context commit and the
// greedy draft block.

#include "dspark_mtp.h"
#include "dspark_mtp_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace inkling {
namespace {

// ---------------------------------------------------------------------------
// Minimal safetensors header reader: the JSON is a flat object of
//   "name": {"dtype": "BF16", "shape": [..], "data_offsets": [a, b]}
// ---------------------------------------------------------------------------
struct STTensor {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t a = 0, b = 0;
    uint64_t bytes() const { return b - a; }
};

struct STReader {
    std::string json;
    std::vector<STTensor> tensors;
    size_t pos = 0;

    void ws() {
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\t' || json[pos] == '\r')) {
            ++pos;
        }
    }
    bool eat(char c) {
        ws();
        if (pos < json.size() && json[pos] == c) { ++pos; return true; }
        return false;
    }
    std::string str() {
        ws();
        if (pos >= json.size() || json[pos] != '"') throw std::runtime_error("mtp: expected string");
        ++pos;
        std::string out;
        while (pos < json.size() && json[pos] != '"') out.push_back(json[pos++]);
        if (pos >= json.size()) throw std::runtime_error("mtp: unterminated string");
        ++pos;
        return out;
    }
    int64_t num() {
        ws();
        bool neg = false;
        if (pos < json.size() && json[pos] == '-') { neg = true; ++pos; }
        int64_t v = 0;
        while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') v = v * 10 + (json[pos++] - '0');
        return neg ? -v : v;
    }
    std::vector<int64_t> iarray() {
        std::vector<int64_t> out;
        if (!eat('[')) throw std::runtime_error("mtp: expected array");
        ws();
        if (eat(']')) return out;
        for (;;) {
            out.push_back(num());
            ws();
            if (eat(']')) break;
            if (!eat(',')) throw std::runtime_error("mtp: expected , or ] in array");
        }
        return out;
    }
    void parse() {
        if (!eat('{')) throw std::runtime_error("mtp: header is not an object");
        ws();
        if (eat('}')) return;
        for (;;) {
            const std::string name = str();
            if (!eat(':')) throw std::runtime_error("mtp: expected :");
            if (!eat('{')) throw std::runtime_error("mtp: expected tensor object");
            if (name == "__metadata__") {
                // {"format": "pt", ...}: skip a flat object of string values
                ws();
                if (!eat('}')) {
                    for (;;) {
                        (void)str();
                        if (!eat(':')) throw std::runtime_error("mtp: expected : in metadata");
                        (void)str();
                        ws();
                        if (eat('}')) break;
                        if (!eat(',')) throw std::runtime_error("mtp: expected , in metadata");
                    }
                }
                ws();
                if (eat('}')) break;
                if (!eat(',')) throw std::runtime_error("mtp: expected , or } after metadata");
                continue;
            }
            STTensor t;
            t.name = name;
            ws();
            if (!eat('}')) {
                for (;;) {
                    const std::string key = str();
                    if (!eat(':')) throw std::runtime_error("mtp: expected : after key");
                    if (key == "dtype") t.dtype = str();
                    else if (key == "shape") t.shape = iarray();
                    else if (key == "data_offsets") {
                        std::vector<int64_t> v = iarray();
                        if (v.size() != 2) throw std::runtime_error("mtp: bad data_offsets");
                        t.a = (uint64_t)v[0];
                        t.b = (uint64_t)v[1];
                    } else {
                        throw std::runtime_error("mtp: unknown tensor key " + key);
                    }
                    ws();
                    if (eat('}')) break;
                    if (!eat(',')) throw std::runtime_error("mtp: expected , or }");
                }
            }
            tensors.push_back(std::move(t));
            ws();
            if (eat('}')) break;
            if (!eat(',')) throw std::runtime_error("mtp: expected , or } between tensors");
        }
    }
    const STTensor & at(const std::string & name) const {
        for (const auto & t : tensors) {
            if (t.name == name) return t;
        }
        throw std::runtime_error("mtp.safetensors: missing tensor " + name);
    }
};

struct MappedFile {
    const uint8_t * data = nullptr;
    size_t size = 0;
    void * base = nullptr;

    bool open(const std::string & path) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size <= 16) { ::close(fd); return false; }
        size = (size_t)st.st_size;
        void * p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (p == MAP_FAILED) return false;
        base = p;
        data = static_cast<const uint8_t *>(p);
        return true;
    }
    ~MappedFile() {
        if (base) munmap(base, size);
    }
};

// bf16 host helpers
inline float h_bf2f(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

} // namespace

DSparkDrafter::~DSparkDrafter() {
    for (hipGraphExec_t ex : draft_execs_) {
        if (ex) (void)hipGraphExecDestroy(ex);
    }
    draft_execs_.clear();
    if (commit_graph_exec_) {
        (void)hipGraphExecDestroy(commit_graph_exec_);
        commit_graph_exec_ = nullptr;
    }
    if (h_dp_) {
        (void)hipHostFree(h_dp_);
        h_dp_ = nullptr;
    }
    for (void * p : allocs_) {
        if (p) (void)hipFree(p);
    }
    allocs_.clear();
}

static void * dsg_alloc(std::vector<void *> & allocs, size_t bytes, size_t & total) {
    void * p = nullptr;
    const size_t b = (bytes + 255) & ~size_t(255);
    if (hipMalloc(&p, b) != hipSuccess) {
        throw std::runtime_error("DSparkDrafter: hipMalloc failed");
    }
    allocs.push_back(p);
    total += b;
    return p;
}

bool DSparkDrafter::init(const std::string & mtp_path, const ModelWeights & w, int max_ctx,
                         int vocab, hipStream_t stream, int draft_ctx_cap) {
    w_ = &w;
    stream_ = stream;
    vocab_ = vocab;
    if (draft_ctx_cap <= 0 || draft_ctx_cap > DRAFT_CTX) draft_ctx_cap = DRAFT_CTX;
    ctx_ = std::max(BLOCK, std::min(max_ctx, draft_ctx_cap));

    MappedFile f;
    if (!f.open(mtp_path)) {
        std::fprintf(stderr, "[dspark] cannot open %s\n", mtp_path.c_str());
        return false;
    }
    uint64_t hlen = 0;
    std::memcpy(&hlen, f.data, 8);
    if (hlen > f.size - 8 || hlen < 2) {
        std::fprintf(stderr, "[dspark] bad safetensors header length\n");
        return false;
    }
    STReader hdr;
    hdr.json.assign(reinterpret_cast<const char *>(f.data + 8), (size_t)hlen);
    try {
        hdr.parse();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[dspark] safetensors header parse failed: %s\n", e.what());
        return false;
    }
    const uint8_t * data = f.data + 8 + hlen;
    auto src = [&](const std::string & name, std::initializer_list<int64_t> shape) -> const uint8_t * {
        const STTensor & t = hdr.at(name);
        if (t.dtype != "BF16") throw std::runtime_error("mtp: " + name + " is not BF16");
        if (!std::equal(t.shape.begin(), t.shape.end(), shape.begin(), shape.end())) {
            throw std::runtime_error("mtp: unexpected shape of " + name);
        }
        return data + t.a;
    };

    inter_ = (int)hdr.at("layers.0.mlp.gate_proj.weight").shape.at(0);
    const int64_t H = N_EMBD;

    try {
        // ---- alpha = bf16(softmax(fuse_weight, -1)) ------------------------
        {
            const uint8_t * fw = src("fuse_weight", {4, 4});
            const uint16_t * fwh = reinterpret_cast<const uint16_t *>(fw);
            uint16_t host[16];
            for (int r = 0; r < 4; ++r) {
                float m = -INFINITY;
                float fv[4];
                for (int c = 0; c < 4; ++c) { fv[c] = h_bf2f(fwh[r * 4 + c]); m = std::max(m, fv[c]); }
                float e[4], s = 0.0f;
                for (int c = 0; c < 4; ++c) s += (e[c] = std::exp(fv[c] - m));
                for (int c = 0; c < 4; ++c) {
                    const float v = e[c] / s;
                    uint32_t u;
                    std::memcpy(&u, &v, 4);
                    u += 0x7fffu + ((u >> 16) & 1u);
                    host[r * 4 + c] = (uint16_t)(u >> 16);
                }
            }
            uint16_t * d = static_cast<uint16_t *>(dsg_alloc(allocs_, sizeof(host), vram_bytes_));
            if (hipMemcpyAsync(d, host, sizeof(host), hipMemcpyHostToDevice, stream_) != hipSuccess) {
                throw std::runtime_error("mtp: alpha upload failed");
            }
            alpha_ = d;
        }

        // ---- one reusable BF16 staging buffer for the row-concatenations ----
        const size_t max_rows = (size_t)2 * inter_;
        uint16_t * stage = static_cast<uint16_t *>(dsg_alloc(allocs_, max_rows * H * 2, vram_bytes_));

        auto upload_bf16 = [&](const std::string & name, std::initializer_list<int64_t> shape) {
            const STTensor & t = hdr.at(name);
            if (t.dtype != "BF16") throw std::runtime_error("mtp: " + name + " is not BF16");
            const size_t bytes = (size_t)t.bytes();
            uint16_t * d = static_cast<uint16_t *>(dsg_alloc(allocs_, bytes, vram_bytes_));
            if (hipMemcpyAsync(d, data + t.a, bytes, hipMemcpyHostToDevice, stream_) != hipSuccess) {
                throw std::runtime_error("mtp: upload of " + name + " failed");
            }
            return d;
        };

        // rows of {name, nrows} concatenated along rows at `stage`, then repacked
        auto repack = [&](std::initializer_list<std::pair<const char *, int64_t>> parts, int64_t K,
                          SoaQ8Matrix & out) {
            int64_t rows = 0;
            for (const auto & [name, nr] : parts) {
                const uint8_t * s = src(name, {nr, K});
                if (hipMemcpyAsync(stage + rows * K, s, (size_t)(nr * K * 2),
                                   hipMemcpyHostToDevice, stream_) != hipSuccess) {
                    throw std::runtime_error(std::string("mtp: stage failed for ") + name);
                }
                rows += nr;
            }
            out.rows = (int)rows;
            out.K = (int)K;
            out.qs = static_cast<int8_t *>(dsg_alloc(allocs_, (size_t)rows * K, vram_bytes_));
            out.d  = static_cast<__half *>(dsg_alloc(allocs_, (size_t)rows * (K / 32) * 2, vram_bytes_));
            launch_dspark_repack_q8(stage, (int)rows, (int)K, out.qs, out.d, stream_);
        };

        auto gemv_ref = [](const char * l, const char * x) {
            static thread_local std::string s;
            s = std::string("layers.") + l + ".self_attn." + x;
            return s;
        };
        auto owned_name = [](const std::string & s) {
            static std::vector<std::string> keep;
            keep.push_back(s);
            return keep.back().c_str();
        };
        // (string lifetimes: `repack` reads the names immediately, so a
        // thread_local scratch name is enough for the per-layer loop below)
        std::string names[3][8];
        for (int l = 0; l < 3; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            const std::string a = p + "self_attn.";
            Layer & L = layers_[l];
            L.local = (l != 1);
            L.in_norm     = upload_bf16(p + "input_layernorm.weight", {H});
            L.post_norm   = upload_bf16(p + "post_attention_layernorm.weight", {H});
            L.ctx_norm    = upload_bf16(p + "dflare_norm.weight", {H});
            L.q_norm      = upload_bf16(a + "q_norm.weight", {HEAD_DIM});
            L.k_norm      = upload_bf16(a + "k_norm.weight", {HEAD_DIM});
            L.k_norm_ctx  = upload_bf16(a + "k_norm_ctx.weight", {HEAD_DIM});
            names[l][0] = a + "q_proj.weight";
            names[l][1] = a + "k_proj.weight";
            names[l][2] = a + "v_proj.weight";
            names[l][3] = a + "o_proj.weight";
            names[l][4] = p + "mlp.gate_proj.weight";
            names[l][5] = p + "mlp.up_proj.weight";
            names[l][6] = p + "mlp.down_proj.weight";
            names[l][7] = a + "k_proj_ctx.weight";
            repack({{names[l][0].c_str(), N_Q}, {names[l][1].c_str(), N_KV}, {names[l][2].c_str(), N_KV}},
                   H, L.qkv);
            repack({{names[l][3].c_str(), H}}, N_Q, L.wo);
            repack({{names[l][4].c_str(), inter_}, {names[l][5].c_str(), inter_}}, H, L.gate_up);
            repack({{names[l][6].c_str(), H}}, inter_, L.w_down);
            names[l][1] = a + "k_proj_ctx.weight";
            names[l][2] = a + "v_proj_ctx.weight";
            repack({{names[l][1].c_str(), N_KV}, {names[l][2].c_str(), N_KV}}, H, L.wkv_ctx);
        }
        (void)gemv_ref;
        (void)owned_name;

        final_norm_ = upload_bf16("norm.weight", {H});
        mask_embed_ = upload_bf16("mask_embed", {H});
        f_norm_     = upload_bf16("markov_head.dflare_norm.weight", {H});
        rnn_w_      = upload_bf16("markov_head.rnn_proj.weight", {3 * RANK, 2 * RANK + 2 * H});
        rnn_b_      = upload_bf16("markov_head.rnn_proj.bias", {3 * RANK});
        w1_         = upload_bf16("markov_head.markov_w1.weight",
                                  {hdr.at("markov_head.markov_w1.weight").shape.at(0), RANK});
        const uint16_t * hw = upload_bf16("markov_head.hidden_out.weight", {H, RANK});
        const uint16_t * hb = upload_bf16("markov_head.hidden_out.bias", {H});

        // ---- fold_w / fold_b (FP64 on the GPU) -----------------------------
        fold_w_ = static_cast<uint16_t *>(dsg_alloc(allocs_, (size_t)vocab_ * RANK * 2, vram_bytes_));
        fold_b_ = static_cast<uint16_t *>(dsg_alloc(allocs_, (size_t)vocab_ * 2, vram_bytes_));
        launch_dspark_fold_head(w.lm_head, hw, hb, vocab_, fold_w_, fold_b_, stream_);
        if (std::getenv("INKLING_MTP_NO_FOLD") != nullptr) {  // A/B: is the fold or the hidden wrong?
            (void)hipMemsetAsync(fold_b_, 0, (size_t)vocab_ * 2, stream_);
            (void)hipMemsetAsync(fold_w_, 0, (size_t)vocab_ * RANK * 2, stream_);
        }

        // ---- draft context caches ------------------------------------------
        for (int l = 0; l < 3; ++l) {
            Layer & L = layers_[l];
            if (L.local) {
                const size_t slots = (size_t)WINDOW + BLOCK;
                L.k_ctx = static_cast<int8_t *>(dsg_alloc(allocs_, slots * N_KV * 2, vram_bytes_));
                L.v_ctx = static_cast<int8_t *>(dsg_alloc(allocs_, slots * N_KV * 2, vram_bytes_));
                L.ctx_pos = static_cast<int32_t *>(dsg_alloc(allocs_, (size_t)WINDOW * 4, vram_bytes_));
            } else {
                const size_t slots = (size_t)ctx_;
                L.k_ctx = static_cast<int8_t *>(dsg_alloc(allocs_, slots * N_KV, vram_bytes_));
                L.d_ctx = static_cast<__half *>(dsg_alloc(allocs_, slots * N_HEAD_KV * 4 * 2, vram_bytes_));
                L.v_ctx = static_cast<int8_t *>(dsg_alloc(allocs_, slots * N_KV, vram_bytes_));
                L.vd_ctx = static_cast<__half *>(dsg_alloc(allocs_, slots * N_HEAD_KV * 4 * 2, vram_bytes_));
            }
        }

        // ---- shortlist head -------------------------------------------------
        {
            std::vector<uint32_t> ranks;
            {
                const std::string dir = mtp_path.substr(0, mtp_path.find_last_of('/') + 1);
                std::string sl = dir + "draft_shortlist.bin";
                FILE * fp = std::fopen(sl.c_str(), "rb");
                if (!fp) {
                    sl = "models/draft_shortlist.bin";
                    fp = std::fopen(sl.c_str(), "rb");
                }
                if (!fp) throw std::runtime_error("mtp: cannot open draft_shortlist.bin");
                ranks.resize(65536);
                const size_t got = std::fread(ranks.data(), 4, ranks.size(), fp);
                std::fclose(fp);
                if (got != ranks.size()) throw std::runtime_error("mtp: short list is not 65536 entries");
            }
            short_in_.assign((size_t)vocab_, 0);
            std::vector<int32_t> list;
            int taken = 0;
            for (uint32_t t : ranks) {
                if (taken < 32768 && t < (uint32_t)vocab_ && !short_in_[t]) {
                    short_in_[t] = 1;
                    ++taken;
                }
            }
            for (int t = 199990; t < vocab_; ++t) short_in_[t] = 1; // <|unused|>, <|endoftext|>, added
            for (int t = 0; t < vocab_; ++t) {
                if (short_in_[t]) list.push_back(t);
            }
            short_K_ = (int)list.size();
            short_base_list_ = list;
            const int rows = short_K_ + MAX_CTX_ROWS;
            short_head_.rows = rows;
            short_head_.K = N_EMBD;
            short_head_.qs = static_cast<int8_t *>(dsg_alloc(allocs_, (size_t)rows * N_EMBD, vram_bytes_));
            short_head_.d  = static_cast<__half *>(dsg_alloc(allocs_, (size_t)rows * (N_EMBD / 32) * 2, vram_bytes_));
            short_map_ = static_cast<int32_t *>(dsg_alloc(allocs_, (size_t)rows * 4, vram_bytes_));
            short_ids_ = static_cast<int32_t *>(dsg_alloc(allocs_, (size_t)rows * 4, vram_bytes_));
            if (hipMemsetAsync(short_map_, 0, (size_t)rows * 4, stream_) != hipSuccess) {
                throw std::runtime_error("mtp: short map memset failed");
            }
            if (hipMemcpyAsync(short_ids_, list.data(), list.size() * 4, hipMemcpyHostToDevice, stream_) != hipSuccess) {
                throw std::runtime_error("mtp: short ids upload failed");
            }
            launch_dspark_gather_q8(w.lm_head, short_ids_, short_K_, short_head_.qs, short_head_.d,
                                    0, short_map_, stream_);
            base_ = static_cast<float *>(dsg_alloc(allocs_, (size_t)GAMMA * rows * 4, vram_bytes_));
            launch_dspark_fill(base_, (int64_t)GAMMA * rows, -INFINITY, stream_);
        }

        // ---- per-call scratch ----------------------------------------------
        auto raw = [&](size_t bytes) { return dsg_alloc(allocs_, bytes, vram_bytes_); };
        res_      = static_cast<uint16_t *>(raw((size_t)BLOCK * H * 2));
        xn_       = static_cast<uint16_t *>(raw((size_t)BLOCK * H * 2));
        q_        = static_cast<uint16_t *>(raw((size_t)BLOCK * N_Q * 2));
        attn_o_   = static_cast<uint16_t *>(raw((size_t)BLOCK * N_Q * 2));
        delta_    = static_cast<float *>(raw((size_t)BLOCK * H * 4));
        act_      = static_cast<uint16_t *>(raw((size_t)BLOCK * inter_ * 2));
        qkv_out_  = static_cast<float *>(raw((size_t)BLOCK * (N_Q + 2 * N_KV) * 4));
        gu_out_   = static_cast<float *>(raw((size_t)BLOCK * 2 * inter_ * 4));
        f_rnn_    = static_cast<uint16_t *>(raw((size_t)H * 2));
        xin_      = static_cast<uint16_t *>(raw((size_t)BLOCK * H * 2));
        kv_commit_= static_cast<float *>(raw((size_t)BLOCK * 2 * N_KV * 4));
        cand_val_ = static_cast<float *>(raw((size_t)GAMMA * 1024 * 4));
        cand_idx_ = static_cast<int32_t *>(raw((size_t)GAMMA * 1024 * 4));
        cand_logits_ = static_cast<float *>(raw((size_t)GAMMA * 1024 * 4));
        state_    = static_cast<uint16_t *>(raw((size_t)(GAMMA + 1) * RANK * 2));
        latent_   = static_cast<uint16_t *>(raw((size_t)RANK * 2));
        picks_    = static_cast<uint64_t *>(raw((size_t)GAMMA * 8));
        part_     = static_cast<float *>(raw((size_t)GAMMA * 3 * RANK * 4));
        tokens_   = static_cast<int32_t *>(raw((size_t)GAMMA * 4));
        conf_     = static_cast<float *>(raw((size_t)GAMMA * 4));
        const int64_t ws_local = dspark_attention_ws_floats(WINDOW + BLOCK);
        const int64_t ws_global = dspark_attention_ws_floats(ctx_);
        attn_ws_  = static_cast<float *>(raw((size_t)std::max(ws_local, ws_global) * 4));
        topk_ws_  = static_cast<float *>(raw(256));

        if (hipMemsetAsync(state_, 0, (size_t)(GAMMA + 1) * RANK * 2, stream_) != hipSuccess) {
            throw std::runtime_error("mtp: state init failed");
        }
        if (hipMemsetAsync(f_rnn_, 0, (size_t)H * 2, stream_) != hipSuccess) {
            throw std::runtime_error("mtp: f_rnn init failed");
        }
        // Step 8B: pinned + device param struct for the captured graphs
        if (hipHostMalloc(reinterpret_cast<void **>(&h_dp_), sizeof(DSparkDraftParams)) != hipSuccess) {
            throw std::runtime_error("mtp: dp host alloc failed");
        }
        std::memset(h_dp_, 0, sizeof(DSparkDraftParams));
        d_dp_ = static_cast<DSparkDraftParams *>(dsg_alloc(allocs_, sizeof(DSparkDraftParams), vram_bytes_));
        if (hipStreamSynchronize(stream_) != hipSuccess) {
            throw std::runtime_error("mtp: init stream sync failed");
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[dspark] init failed: %s\n", e.what());
        return false;
    }

    reset();
    if (hipStreamSynchronize(stream_) != hipSuccess) return false;
    if (std::getenv("INKLING_MTP_NO_GRAPH") == nullptr) {
        capture_draft_graphs();
    }
    std::printf("[dspark] drafter ready: layers=3 inter=%d ctx=%d short_K=%d vocab=%d vram=%.0f MiB\n",
                inter_, ctx_, short_K_, vocab_, (double)vram_bytes_ / 1048576.0);
    std::fflush(stdout);
    return true;
}

void DSparkDrafter::reset() {
    for (int l = 0; l < 3; ++l) {
        if (layers_[l].ctx_pos) {
            (void)hipMemsetAsync(layers_[l].ctx_pos, 0xFF, (size_t)WINDOW * 4, stream_);
        }
    }
    length_ = 0;
    set_context_rows(0);
}

void DSparkDrafter::set_context_rows(int rows) {
    if (rows >= short_E_) return;
    for (size_t i = (size_t)rows; i < short_extra_.size(); ++i) {
        short_in_[(size_t)short_extra_[i]] = 0;
    }
    short_extra_.resize((size_t)rows);
    short_E_ = rows;
    // the dropped rows' base logits back to -inf (the GEMV only writes the rows in use)
    for (int i = 0; i < GAMMA; ++i) {
        launch_dspark_fill(base_ + (size_t)i * short_head_.rows + short_K_ + rows,
                           MAX_CTX_ROWS - rows, -INFINITY, stream_);
    }
}

void DSparkDrafter::note_tokens(const int32_t * ids, int n) {
    if (n <= 0) return;
    std::vector<int32_t> add;
    for (int i = 0; i < n; ++i) {
        const int32_t t = ids[i];
        if (t >= 0 && t < vocab_ && !short_in_[(size_t)t] && short_E_ + (int)add.size() < MAX_CTX_ROWS) {
            short_in_[(size_t)t] = 1;
            add.push_back(t);
        }
    }
    if (add.empty()) return;
    const int row = short_K_ + short_E_;
    if (hipMemcpyAsync(short_ids_, add.data(), add.size() * 4, hipMemcpyHostToDevice, stream_) != hipSuccess) {
        std::fprintf(stderr, "[dspark] note_tokens: id upload failed\n");
        return;
    }
    launch_dspark_gather_q8(w_->lm_head, short_ids_, (int)add.size(), short_head_.qs,
                            short_head_.d, row, short_map_, stream_);
    short_extra_.insert(short_extra_.end(), add.begin(), add.end());
    short_E_ += (int)add.size();
}

void DSparkDrafter::run_commit_piece(const uint16_t * feats, int n, int start, bool last,
                                     const DSparkDraftParams * dp) {
    for (int l = 0; l < 3; ++l) {
        Layer & L = layers_[l];
        launch_dspark_mix_rmsnorm(feats, n, alpha_ + 4 * l, L.ctx_norm, xin_, stream_,
                                  dp, dp ? 1 : 0);
        launch_dspark_gemv(L.wkv_ctx, xin_, N_EMBD, /*x_bf16=*/true, n, kv_commit_, 2 * N_KV,
                           stream_, dp, dp ? 2 : 0);
        launch_dspark_ctx_kv_store(kv_commit_, n, start, L.local ? WINDOW : ctx_, L.k_norm_ctx,
                                   L.local, L.k_ctx, L.d_ctx, L.v_ctx, L.vd_ctx, L.ctx_pos,
                                   stream_, dp);
    }
    if (last) {
        if (dp != nullptr) {
            // dp mode 2: the single row ncommit - 1 (graph-static source offset)
            launch_dspark_mix_rmsnorm(feats, 1, alpha_ + 12,
                                      f_norm_, f_rnn_, stream_, dp, 2);
        } else {
            launch_dspark_mix_rmsnorm(feats + (size_t)(n - 1) * 4 * N_EMBD, 1, alpha_ + 12,
                                      f_norm_, f_rnn_, stream_, nullptr, 0);
        }
    }
}

void DSparkDrafter::commit(const uint16_t * d_feats, int n, int start) {
    if (n <= 0 || d_feats == nullptr) return;
    if (n <= BLOCK) {
        if (d_feats != d_taps_base_) {
            if (commit_graph_exec_ != nullptr) {
                (void)hipGraphExecDestroy(commit_graph_exec_);
                commit_graph_exec_ = nullptr;
            }
            d_taps_base_ = d_feats;
        }
        if (commit_graph_exec_ == nullptr && std::getenv("INKLING_MTP_NO_GRAPH") == nullptr &&
            d_dp_ != nullptr) {
            // Lazy one-time capture: needs an idle stream (this is the tap hook,
            // i.e. right after a prefill chunk, which already syncs).
            if (hipStreamSynchronize(stream_) == hipSuccess) {
                hipGraph_t g = nullptr;
                if (hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal) == hipSuccess) {
                    run_commit_piece(d_feats, BLOCK, start, /*last=*/true, d_dp_);
                    if (hipStreamEndCapture(stream_, &g) == hipSuccess) {
                        if (hipGraphInstantiate(&commit_graph_exec_, g, nullptr, nullptr, 0) != hipSuccess) {
                            commit_graph_exec_ = nullptr;
                        }
                        (void)hipGraphDestroy(g);
                    }
                }
            }
        }
        if (commit_graph_exec_ != nullptr) {
            // Step 8B: one captured graph; n/start ride in the 32-byte param struct
            // uploaded by value so back-to-back commit() + draft() never race.
            DSparkDraftParams p{};
            p.ncommit = n;
            p.pos     = start;
            launch_dspark_set_params(d_dp_, p, stream_);
            {
                hipError_t ge = hipGraphLaunch(commit_graph_exec_, stream_);
                if (std::getenv("INKLING_MTP_GRAPH_SYNC") != nullptr) {
                    hipError_t se = hipStreamSynchronize(stream_);
                    std::fprintf(stderr, "[dspark] commit graph n=%d start=%d dp(ncommit=%d pos=%d) launch=%d sync=%d (%s)\n",
                                 n, start, p.ncommit, p.pos, (int)ge, (int)se, hipGetErrorString(se));
                    std::fflush(stderr);
                }
                if (ge != hipSuccess) {
                    std::fprintf(stderr, "[dspark] commit graph launch failed: %s\n", hipGetErrorString(ge));
                    std::abort();
                }
            }
            length_ = start + n;
            return;
        }
    }
    // pieces of BLOCK rows (the m-row GEMV holds BLOCK accumulators)
    for (int r0 = 0; r0 < n; r0 += BLOCK) {
        const int k = std::min(BLOCK, n - r0);
        run_commit_piece(d_feats + (size_t)r0 * 4 * N_EMBD, k, start + r0, r0 + k == n, nullptr);
    }
    length_ = start + n;
}

static void dsg_report(const char * tag, hipStream_t stream, const uint16_t * p, int n) {
    std::vector<uint16_t> h((size_t)n);
    (void)hipStreamSynchronize(stream);
    (void)hipMemcpy(h.data(), p, (size_t)n * 2, hipMemcpyDeviceToHost);
    double ss = 0.0, mx = 0.0;
    for (int i = 0; i < n; ++i) {
        const float v = h_bf2f(h[(size_t)i]);
        ss += (double)v * v;
        mx = std::max(mx, (double)std::fabs(v));
    }
    std::printf("  [mtp-norm] %-10s rms=%.4f max=%.4f (first=%.4f %.4f %.4f)\n", tag,
                std::sqrt(ss / n), mx, h_bf2f(h[0]), h_bf2f(h[1]), h_bf2f(h[2]));
}

void DSparkDrafter::debug_base_copy(float * host_out) const {
    (void)hipStreamSynchronize(stream_);
    for (int r = 0; r < GAMMA; ++r) {
        (void)hipMemcpy(host_out + (size_t)r * 1024, cand_val_ + (size_t)r * 1024, 1024 * 4,
                        hipMemcpyDeviceToHost);
    }
}

double DSparkDrafter::debug_gemv_check() {
    const int m = 8;
    std::vector<uint16_t> hx((size_t)m * N_EMBD);
    for (int r = 0; r < m; ++r) {
        for (int i = 0; i < N_EMBD; ++i) {
            const float v = std::sin((float)(i + 13 * r)) * 1.7f + 0.3f * (float)((i * 7 + r) % 5);
            uint32_t u;
            std::memcpy(&u, &v, 4);
            u += 0x7fffu + ((u >> 16) & 1u);
            hx[(size_t)r * N_EMBD + i] = (uint16_t)(u >> 16);
        }
    }
    uint16_t * dx = static_cast<uint16_t *>(dsg_alloc(allocs_, (size_t)m * N_EMBD * 2, vram_bytes_));
    float * ym = static_cast<float *>(dsg_alloc(allocs_, (size_t)m * VOCAB_UNPADDED * 4, vram_bytes_));
    float * y1 = static_cast<float *>(dsg_alloc(allocs_, (size_t)VOCAB_UNPADDED * 4, vram_bytes_));
    (void)hipMemcpyAsync(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice, stream_);
    SoaQ8Matrix head = w_->lm_head;
    launch_dspark_gemv(head, dx, N_EMBD, true, m, ym, VOCAB_UNPADDED, stream_);
    double worst = 0.0;
    for (int r = 0; r < m; ++r) {
        std::vector<float> xf((size_t)N_EMBD);
        for (int i = 0; i < N_EMBD; ++i) xf[(size_t)i] = h_bf2f(hx[(size_t)r * N_EMBD + i]);
        float * dxf = static_cast<float *>(dsg_alloc(allocs_, (size_t)N_EMBD * 4, vram_bytes_));
        (void)hipMemcpyAsync(dxf, xf.data(), xf.size() * 4, hipMemcpyHostToDevice, stream_);
        launch_soa_q8_gemv(head, dxf, y1, stream_);
        std::vector<float> hm((size_t)VOCAB_UNPADDED), h1((size_t)VOCAB_UNPADDED);
        (void)hipMemcpy(hm.data(), ym + (size_t)r * VOCAB_UNPADDED, hm.size() * 4, hipMemcpyDeviceToHost);
        (void)hipMemcpy(h1.data(), y1, h1.size() * 4, hipMemcpyDeviceToHost);
        for (size_t i = 0; i < hm.size(); ++i) {
            worst = std::max(worst, (double)std::fabs((double)hm[i] - (double)h1[i]));
        }
    }
    std::printf("[dspark] gemv differential (m=8 rows vs engine 1-row GEMV): max |d| = %.6g\n", worst);
    std::fflush(stdout);
    return worst;
}

// Verify that the shortlist head's row i really is lm_head row short_map[i].
double DSparkDrafter::debug_shortlist_check() {
    const int m = 1;
    std::vector<uint16_t> hx((size_t)N_EMBD);
    for (int i = 0; i < N_EMBD; ++i) {
        const float v = std::sin((float)i * 0.37f) * 1.1f;
        uint32_t u;
        std::memcpy(&u, &v, 4);
        u += 0x7fffu + ((u >> 16) & 1u);
        hx[(size_t)i] = (uint16_t)(u >> 16);
    }
    uint16_t * dx = static_cast<uint16_t *>(dsg_alloc(allocs_, (size_t)N_EMBD * 2, vram_bytes_));
    float * ys = static_cast<float *>(dsg_alloc(allocs_, (size_t)short_head_.rows * 4, vram_bytes_));
    float * yl = static_cast<float *>(dsg_alloc(allocs_, (size_t)VOCAB_UNPADDED * 4, vram_bytes_));
    std::vector<float> xf((size_t)N_EMBD);
    for (int i = 0; i < N_EMBD; ++i) xf[(size_t)i] = h_bf2f(hx[(size_t)i]);
    float * dxf = static_cast<float *>(dsg_alloc(allocs_, (size_t)N_EMBD * 4, vram_bytes_));
    (void)hipMemcpyAsync(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice, stream_);
    (void)hipMemcpyAsync(dxf, xf.data(), xf.size() * 4, hipMemcpyHostToDevice, stream_);
    SoaQ8Matrix hv = short_head_;
    hv.rows = short_K_;
    launch_dspark_gemv(hv, dx, N_EMBD, true, m, ys, short_head_.rows, stream_);
    launch_soa_q8_gemv(w_->lm_head, dxf, yl, stream_);
    (void)hipStreamSynchronize(stream_);
    std::vector<float> hs((size_t)short_head_.rows), hl((size_t)VOCAB_UNPADDED);
    std::vector<int32_t> map((size_t)short_head_.rows);
    (void)hipMemcpy(hs.data(), ys, hs.size() * 4, hipMemcpyDeviceToHost);
    (void)hipMemcpy(hl.data(), yl, hl.size() * 4, hipMemcpyDeviceToHost);
    (void)hipMemcpy(map.data(), short_map_, map.size() * 4, hipMemcpyDeviceToHost);
    double worst = 0.0;
    int bad_row = -1;
    for (int i = 0; i < short_K_; ++i) {
        const int id = map[(size_t)i];
        const double d = std::fabs((double)hs[(size_t)i] - (double)hl[(size_t)id]);
        if (d > worst) { worst = d; bad_row = i; }
    }
    std::printf("[dspark] shortlist head differential vs lm_head: max |d| = %.6g (row %d, id %d, short_K=%d)\n",
                worst, bad_row, bad_row >= 0 ? map[(size_t)bad_row] : -1, short_K_);
    std::fflush(stdout);
    return worst;
}

void DSparkDrafter::run_draft_body(const DSparkDraftParams * dp, int anchor, int P) {
    launch_dspark_embed_block(w_->tok_embd, w_->tok_norm, mask_embed_, layers_[0].in_norm,
                              anchor, res_, xn_, picks_, stream_, dp);
    for (int l = 0; l < 3; ++l) {
        Layer & L = layers_[l];
        launch_dspark_gemv(L.qkv, xn_, N_EMBD, /*x_bf16=*/true, BLOCK, qkv_out_, N_Q + 2 * N_KV,
                           stream_);
        launch_dspark_qkv_rope(qkv_out_, L.q_norm, L.k_norm, P, q_, L.local, ctx_,
                               L.k_ctx, L.d_ctx, L.v_ctx, L.vd_ctx, stream_, dp);
        const int nkeys = L.local ? (WINDOW + BLOCK) : std::min(P + BLOCK, ctx_);
        const int n_chunks = dspark_attention_chunks(nkeys);
        const int chunk = dspark_attention_chunk_len(nkeys);
        launch_dspark_attention(q_, nkeys, L.local, L.local ? WINDOW : ctx_, L.k_ctx, L.d_ctx,
                                L.v_ctx, L.vd_ctx, L.ctx_pos, P, attn_ws_, attn_o_, stream_,
                                dp, n_chunks, chunk);
        launch_dspark_gemv(L.wo, attn_o_, N_Q, /*x_bf16=*/true, BLOCK, delta_, N_EMBD, stream_);
        launch_dspark_add_rmsnorm(res_, delta_, L.post_norm, xn_, BLOCK, stream_);
        launch_dspark_gemv(L.gate_up, xn_, N_EMBD, /*x_bf16=*/true, BLOCK, gu_out_, 2 * inter_,
                           stream_);
        launch_dspark_silu_mul(gu_out_, BLOCK, inter_, act_, stream_);
        launch_dspark_gemv(L.w_down, act_, inter_, /*x_bf16=*/true, BLOCK, delta_, N_EMBD, stream_);
        const uint16_t * nxt = (l < 2) ? layers_[l + 1].in_norm : final_norm_;
        launch_dspark_add_rmsnorm(res_, delta_, nxt, xn_, BLOCK, stream_);
    }
    // base logits over the shortlist head for rows 1..7 (row 0 is the anchor's mask).
    // The captured graph keeps the MAX row stride so base_ never moves; the active
    // row count rides in dp->head_rows.
    const int head_rows = short_K_ + (dp ? MAX_CTX_ROWS : short_E_);
    launch_dspark_gemv(short_head_, xn_ + (size_t)N_EMBD, N_EMBD, /*x_bf16=*/true, GAMMA,
                       base_, short_head_.rows, stream_, dp, 1, head_rows);
    launch_dspark_topk_candidates(base_, GAMMA, short_K_ + short_E_, short_map_,
                                  cand_val_, cand_idx_, topk_ws_, stream_, dp,
                                  short_head_.rows);
    launch_dspark_markov_pre(rnn_w_, xn_ + (size_t)N_EMBD, f_rnn_, GAMMA, part_, stream_);
    for (int i = 0; i < GAMMA; ++i) {
        launch_dspark_markov_step(i, rnn_w_, rnn_b_, w1_, part_, picks_, cand_idx_, anchor,
                                  state_, latent_, stream_, dp);
        launch_dspark_markov_logits(i, fold_w_, fold_b_, cand_val_, cand_idx_, latent_, GAMMA,
                                    cand_logits_, picks_, stream_);
    }
    launch_dspark_markov_finish(cand_idx_, cand_logits_, picks_, tokens_, conf_, stream_);
}

// Step 8B: capture the whole draft() pipeline as one static hipGraphExec_t per
// nkeys bucket.  Every dynamic per-call value (anchor, P, nkeys_global,
// head_rows) is read from d_dp_ by the kernels, so a graph launch + one 32-byte
// by-value param kernel is the entire per-call host cost.
void DSparkDrafter::capture_draft_graphs() {
    if (d_dp_ == nullptr) return;
    int nb = 1024;
    while (nb < ctx_) nb <<= 1;
    std::vector<int> buckets;
    for (int b = 1024; b < nb; b <<= 1) buckets.push_back(b);
    buckets.push_back(nb);
    if (buckets.back() != ctx_) buckets.push_back(ctx_);
    if (hipStreamSynchronize(stream_) != hipSuccess) return;
    for (int b : buckets) {
        const int cap = std::min(b, ctx_);
        hipGraph_t g = nullptr;
        if (hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal) != hipSuccess) {
            std::fprintf(stderr, "[dspark] graph capture begin failed (bucket %d)\n", b);
            return;
        }
        // capture with the bucket's maximum nkeys so the captured split grid and
        // chunk length match every launch inside the bucket
        run_draft_body(d_dp_, 0, std::max(0, cap - BLOCK));
        if (hipStreamEndCapture(stream_, &g) != hipSuccess) {
            std::fprintf(stderr, "[dspark] graph capture end failed (bucket %d)\n", b);
            return;
        }
        hipGraphExec_t ex = nullptr;
        if (hipGraphInstantiate(&ex, g, nullptr, nullptr, 0) != hipSuccess) {
            std::fprintf(stderr, "[dspark] graph instantiate failed (bucket %d)\n", b);
            (void)hipGraphDestroy(g);
            return;
        }
        (void)hipGraphDestroy(g);
        draft_buckets_.push_back(cap);
        draft_execs_.push_back(ex);
    }
    std::printf("[dspark] captured %zu static draft HIP graphs\n",
                draft_execs_.size());
    std::fflush(stdout);
}

DSparkDraft DSparkDrafter::draft(int32_t anchor, int P) {
    if (anchor < 0 || anchor >= vocab_) {
        std::fprintf(stderr, "[dspark] anchor %d out of range\n", anchor);
        return {};
    }
    const bool dbg = std::getenv("INKLING_MTP_DEBUG_K") != nullptr;
    DSparkDraftParams p{};
    p.anchor       = anchor;
    p.pos          = P;
    p.nkeys_global = std::min(P + BLOCK, ctx_);
    p.head_rows    = short_K_ + short_E_;
    p.ncommit      = BLOCK;
    if (!dbg && !draft_execs_.empty()) {
        int idx = (int)draft_execs_.size() - 1;
        for (size_t i = 0; i < draft_buckets_.size(); ++i) {
            if (draft_buckets_[i] >= p.nkeys_global) { idx = (int)i; break; }
        }
        launch_dspark_set_params(d_dp_, p, stream_);
        {
            hipError_t ge = hipGraphLaunch(draft_execs_[(size_t)idx], stream_);
            if (std::getenv("INKLING_MTP_GRAPH_SYNC") != nullptr) {
                hipError_t se = hipStreamSynchronize(stream_);
                std::fprintf(stderr, "[dspark] draft graph bucket=%d idx=%d launch=%d sync=%d (%s)\n",
                             p.nkeys_global, idx, (int)ge, (int)se, hipGetErrorString(se));
                std::fflush(stderr);
            }
            if (ge != hipSuccess) {
                std::fprintf(stderr, "[dspark] draft graph launch failed: %s\n", hipGetErrorString(ge));
                std::abort();
            }
        }
        DSparkDraft d;
        d.tokens = tokens_;
        d.conf = conf_;
        return d;
    }
    run_draft_body(nullptr, anchor, P);
    static bool dbg_once = dbg;
    if (dbg_once) {
        dbg_once = false;
        dsg_report("res[0]", stream_, res_, N_EMBD);
        dsg_report("attn_o[0]", stream_, attn_o_, N_Q);
        dsg_report("act[0]", stream_, act_, inter_);
        dsg_report("xn[0]", stream_, xn_, N_EMBD);
        dsg_report("xn[1]", stream_, xn_ + N_EMBD, N_EMBD);
        (void)hipStreamSynchronize(stream_);
        float cv[8];
        int32_t ci[8];
        float base_row[8];
        (void)hipMemcpy(cv, cand_val_, sizeof(cv), hipMemcpyDeviceToHost);
        (void)hipMemcpy(ci, cand_idx_, sizeof(ci), hipMemcpyDeviceToHost);
        (void)hipMemcpy(base_row, base_, sizeof(base_row), hipMemcpyDeviceToHost);
        std::printf("[mtp-dbg] rows=%d head0: base[0..7]=", short_K_ + short_E_);
        for (int i = 0; i < 8; ++i) std::printf("%.3f ", base_row[i]);
        std::printf("| top cands (val,id):");
        for (int i = 0; i < 8; ++i) std::printf(" (%.3f,%d)", cv[i], ci[i]);
        std::printf("\n");
        std::fflush(stdout);
    }
    DSparkDraft d;
    d.tokens = tokens_;
    d.conf = conf_;
    return d;
}

} // namespace inkling
