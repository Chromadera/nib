// Native RDNA3/RDNA4 DFlash 2 speculative drafter host implementation
// (`incoai/Inkling-2-Mini-DFlash2`) and unified auto-detecting SpecDrafter facade.

#include "dflash2_mtp.h"
#include "dflash2_mtp_kernels.h"
#include "kernels_step7j_fdot2.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace inkling {
namespace {

struct Df2STTensor {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t a = 0;
    uint64_t b = 0;
    uint64_t bytes() const { return b - a; }
};

struct Df2STReader {
    std::string json;
    std::vector<Df2STTensor> tensors;
    size_t pos = 0;

    void ws() {
        while (pos < json.size() &&
               (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\t' || json[pos] == '\r')) {
            ++pos;
        }
    }
    bool eat(char c) {
        ws();
        if (pos < json.size() && json[pos] == c) {
            ++pos;
            return true;
        }
        return false;
    }
    std::string str() {
        ws();
        if (pos >= json.size() || json[pos] != '"') {
            throw std::runtime_error("dflash2: expected string in safetensors header");
        }
        ++pos;
        std::string out;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                ++pos;
                out.push_back(json[pos++]);
            } else {
                out.push_back(json[pos++]);
            }
        }
        if (pos >= json.size()) {
            throw std::runtime_error("dflash2: unterminated string in safetensors header");
        }
        ++pos;
        return out;
    }
    int64_t num() {
        ws();
        bool neg = false;
        if (pos < json.size() && json[pos] == '-') {
            neg = true;
            ++pos;
        }
        int64_t v = 0;
        while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
            v = v * 10 + (json[pos++] - '0');
        }
        return neg ? -v : v;
    }
    std::vector<int64_t> iarray() {
        std::vector<int64_t> out;
        if (!eat('[')) throw std::runtime_error("dflash2: expected '['");
        ws();
        if (eat(']')) return out;
        for (;;) {
            out.push_back(num());
            ws();
            if (eat(']')) break;
            if (!eat(',')) throw std::runtime_error("dflash2: expected ',' or ']'");
        }
        return out;
    }
    void skip_value() {
        ws();
        if (pos >= json.size()) throw std::runtime_error("dflash2: unexpected EOF");
        const char c = json[pos];
        if (c == '"') {
            (void)str();
        } else if (c == '{') {
            ++pos;
            ws();
            if (!eat('}')) {
                for (;;) {
                    (void)str();
                    if (!eat(':')) throw std::runtime_error("dflash2: expected ':' in object");
                    skip_value();
                    ws();
                    if (eat('}')) break;
                    if (!eat(',')) throw std::runtime_error("dflash2: expected ',' or '}' in object");
                }
            }
        } else if (c == '[') {
            ++pos;
            ws();
            if (!eat(']')) {
                for (;;) {
                    skip_value();
                    ws();
                    if (eat(']')) break;
                    if (!eat(',')) throw std::runtime_error("dflash2: expected ',' or ']' in array");
                }
            }
        } else {
            while (pos < json.size() && json[pos] != ',' && json[pos] != '}' &&
                   json[pos] != ']' && json[pos] != ' ' && json[pos] != '\n' &&
                   json[pos] != '\r' && json[pos] != '\t') {
                ++pos;
            }
        }
    }
    void parse() {
        if (!eat('{')) throw std::runtime_error("dflash2: header is not a JSON object");
        ws();
        if (eat('}')) return;
        for (;;) {
            const std::string name = str();
            if (!eat(':')) throw std::runtime_error("dflash2: expected ':'");
            if (name == "__metadata__") {
                skip_value();
                ws();
                if (eat('}')) break;
                if (!eat(',')) throw std::runtime_error("dflash2: expected ',' or '}' after metadata");
                continue;
            }
            if (!eat('{')) throw std::runtime_error("dflash2: expected tensor object");
            Df2STTensor t;
            t.name = name;
            ws();
            if (!eat('}')) {
                for (;;) {
                    const std::string key = str();
                    if (!eat(':')) throw std::runtime_error("dflash2: expected ':' after key");
                    if (key == "dtype") {
                        t.dtype = str();
                    } else if (key == "shape") {
                        t.shape = iarray();
                    } else if (key == "data_offsets") {
                        std::vector<int64_t> v = iarray();
                        if (v.size() != 2) throw std::runtime_error("dflash2: bad data_offsets");
                        t.a = (uint64_t)v[0];
                        t.b = (uint64_t)v[1];
                    } else {
                        skip_value();
                    }
                    ws();
                    if (eat('}')) break;
                    if (!eat(',')) throw std::runtime_error("dflash2: expected ',' or '}'");
                }
            }
            tensors.push_back(std::move(t));
            ws();
            if (eat('}')) break;
            if (!eat(',')) throw std::runtime_error("dflash2: expected ',' or '}' between tensors");
        }
    }
    const Df2STTensor * find(const std::string & name) const {
        for (const auto & t : tensors) {
            if (t.name == name) return &t;
        }
        return nullptr;
    }
    const Df2STTensor & at(const std::string & name) const {
        const Df2STTensor * t = find(name);
        if (!t) throw std::runtime_error("dflash2 safetensors: missing tensor " + name);
        return *t;
    }
};

struct Df2MappedFile {
    const uint8_t * data = nullptr;
    size_t size = 0;
    void * base = nullptr;

    bool open(const std::string & path) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 16) {
            ::close(fd);
            return false;
        }
        size = (size_t)st.st_size;
        void * p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (p == MAP_FAILED) return false;
        base = p;
        data = static_cast<const uint8_t *>(p);
        return true;
    }
    ~Df2MappedFile() {
        if (base) munmap(base, size);
    }
};

inline float df2_h_bf2f(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

static void * df2_alloc(std::vector<void *> & allocs, size_t bytes, size_t & total, hipStream_t stream) {
    void * p = nullptr;
    const size_t b = (bytes + 255) & ~size_t(255);
    if (hipMalloc(&p, b) != hipSuccess) {
        throw std::runtime_error("DFlash2Drafter: hipMalloc failed");
    }
    if (hipMemsetAsync(p, 0, b, stream) != hipSuccess) {
        (void)hipFree(p);
        throw std::runtime_error("DFlash2Drafter: hipMemsetAsync failed");
    }
    allocs.push_back(p);
    total += b;
    return p;
}

static std::string resolve_safetensors_path(const std::string & path) {
    struct stat st;
    if (::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
        std::string cand = path;
        if (!cand.empty() && cand.back() != '/') cand.push_back('/');
        cand += "model.safetensors";
        return cand;
    }
    return path;
}

} // namespace

// ---------------------------------------------------------------------------
// DFlash2Drafter
// ---------------------------------------------------------------------------
bool DFlash2Drafter::is_dflash2_checkpoint(const std::string & path,
                                           std::string * resolved_safetensors) {
    if (path.empty()) return false;
    const std::string resolved = resolve_safetensors_path(path);
    const int fd = ::open(resolved.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 16) {
        ::close(fd);
        return false;
    }
    uint64_t hlen = 0;
    if (::read(fd, &hlen, 8) != 8 || hlen < 2 || hlen > 16ull * 1024 * 1024 ||
        hlen > (uint64_t)st.st_size - 8) {
        ::close(fd);
        return false;
    }
    std::string header((size_t)hlen, '\0');
    size_t got = 0;
    while (got < (size_t)hlen) {
        const ssize_t n = ::read(fd, &header[got], (size_t)hlen - got);
        if (n <= 0) {
            ::close(fd);
            return false;
        }
        got += (size_t)n;
    }
    ::close(fd);

    const bool has_selector = header.find("\"candidate_selector.hidden_projection.weight\"") != std::string::npos;
    const bool has_conv     = header.find("\"layers.0.attention_conv.base_kernel\"") != std::string::npos;
    if (has_selector && has_conv) {
        if (resolved_safetensors) *resolved_safetensors = resolved;
        return true;
    }
    return false;
}

size_t DFlash2Drafter::estimate_vram_bytes() {
    auto q8_bytes = [](size_t rows, size_t K) -> size_t {
        return rows * K + rows * (K / 32) * 2;
    };
    size_t bytes = 0;
    // fc [2304, 9216]
    bytes += q8_bytes(N_EMBD, 4 * N_EMBD);
    // 5 layers: qkv [6144, 2304], wo [2304, 4096], gate_up [13824, 2304],
    // w_down [2304, 6912], attn_conv_proj & mlp_conv_proj [576, 2304]
    const size_t per_layer_w =
        q8_bytes(QKV_DIM_D, N_EMBD) +
        q8_bytes(N_EMBD, Q_DIM_D) +
        q8_bytes(2 * INTER_D, N_EMBD) +
        q8_bytes(N_EMBD, INTER_D) +
        2 * q8_bytes(CONV_PROJ, N_EMBD) +
        64 * 1024; // norms + conv base kernels
    bytes += (size_t)N_LAYERS * per_layer_w;
    // Candidate selector: hidden_proj [256, 2304] + 2 codebooks [200058, 256]
    bytes += q8_bytes(SEL_RANK, N_EMBD);
    bytes += 2 * q8_bytes(VOCAB_UNPADDED, SEL_RANK);
    // 5-layer SWA-2048 BF16 KV rings
    bytes += (size_t)N_LAYERS * (2 * (size_t)(WINDOW + BLOCK) * KV_DIM_D * 2 + (size_t)WINDOW * 4);
    // Scratch + logits [7, 200058] + workspace + small alignment margin
    bytes += 16ull * 1024 * 1024;
    return bytes;
}

DFlash2Drafter::~DFlash2Drafter() {
    if (draft_graph_exec_) {
        (void)hipGraphExecDestroy(draft_graph_exec_);
        draft_graph_exec_ = nullptr;
    }
    if (commit_graph_exec_) {
        (void)hipGraphExecDestroy(commit_graph_exec_);
        commit_graph_exec_ = nullptr;
    }
    for (void * p : allocs_) {
        if (p) (void)hipFree(p);
    }
    allocs_.clear();
}

bool DFlash2Drafter::init(const std::string & mtp_path, const ModelWeights & w, int max_ctx,
                          int vocab, hipStream_t stream, int draft_ctx_cap) {
    (void)max_ctx;
    (void)draft_ctx_cap;
    w_ = &w;
    stream_ = stream;
    vocab_ = std::min(vocab, w.lm_head.rows);
    if (vocab_ <= MASK_TOK_ID || w.tok_embd.rows <= MASK_TOK_ID) {
        std::fprintf(stderr, "[dflash2] vocab (%d) or tok_embd (%d) too small for mask token %d\n",
                     vocab_, w.tok_embd.rows, MASK_TOK_ID);
        return false;
    }

    const std::string resolved = resolve_safetensors_path(mtp_path);
    Df2MappedFile f;
    if (!f.open(resolved)) {
        std::fprintf(stderr, "[dflash2] cannot open %s\n", resolved.c_str());
        return false;
    }
    uint64_t hlen = 0;
    std::memcpy(&hlen, f.data, 8);
    if (hlen < 2 || hlen > f.size - 8) {
        std::fprintf(stderr, "[dflash2] bad safetensors header length\n");
        return false;
    }
    Df2STReader hdr;
    hdr.json.assign(reinterpret_cast<const char *>(f.data + 8), (size_t)hlen);
    try {
        hdr.parse();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[dflash2] safetensors header parse failed: %s\n", e.what());
        return false;
    }

    const uint8_t * data = f.data + 8 + hlen;
    auto src = [&](const std::string & name, std::initializer_list<int64_t> shape) -> const uint8_t * {
        const Df2STTensor & t = hdr.at(name);
        if (t.dtype != "BF16") throw std::runtime_error("dflash2: " + name + " is not BF16");
        if (t.shape.size() != shape.size() ||
            !std::equal(t.shape.begin(), t.shape.end(), shape.begin(), shape.end())) {
            throw std::runtime_error("dflash2: unexpected shape of " + name);
        }
        return data + t.a;
    };
    auto src_vocab = [&](const std::string & name, int64_t cols) -> const uint8_t * {
        const Df2STTensor & t = hdr.at(name);
        if (t.dtype != "BF16") throw std::runtime_error("dflash2: " + name + " is not BF16");
        if (t.shape.size() != 2 || t.shape[0] < vocab_ || t.shape[1] != cols) {
            throw std::runtime_error("dflash2: unexpected shape of " + name);
        }
        return data + t.a;
    };

    uint16_t * stage_bf16 = nullptr;
    constexpr size_t STAGE_BYTES = 64ull * 1024 * 1024; // 64 MiB transient staging (freed after init)

    try {
        if (hipMalloc(&stage_bf16, STAGE_BYTES) != hipSuccess) {
            throw std::runtime_error("DFlash2Drafter: transient staging hipMalloc failed");
        }

        auto alloc_q8 = [&](int rows, int K) -> SoaQ8Matrix {
            SoaQ8Matrix M;
            M.rows = rows;
            M.K = K;
            M.qs = static_cast<int8_t *>(df2_alloc(allocs_, (size_t)rows * K, vram_bytes_, stream_));
            M.d  = static_cast<__half *>(df2_alloc(allocs_, (size_t)rows * (K / 32) * 2, vram_bytes_, stream_));
            return M;
        };

        auto upload_bf16 = [&](const uint8_t * p, size_t n_elems) -> uint16_t * {
            uint16_t * d = static_cast<uint16_t *>(df2_alloc(allocs_, n_elems * 2, vram_bytes_, stream_));
            if (hipMemcpyAsync(d, p, n_elems * 2, hipMemcpyHostToDevice, stream_) != hipSuccess) {
                throw std::runtime_error("DFlash2Drafter: upload_bf16 failed");
            }
            return d;
        };

        // Repack BF16 [rows, K] into dst SoA Q8_0 starting at row_dst0 using chunked staging.
        auto repack_into = [&](const uint8_t * p, int rows, int K,
                               SoaQ8Matrix & dst, int row_dst0) {
            const size_t row_bytes = (size_t)K * 2;
            const int max_chunk_rows = std::max(1, (int)(STAGE_BYTES / row_bytes));
            const int nb = K / 32;
            for (int r0 = 0; r0 < rows; r0 += max_chunk_rows) {
                const int rc = std::min(max_chunk_rows, rows - r0);
                if (hipMemcpyAsync(stage_bf16, p + (size_t)r0 * row_bytes,
                                   (size_t)rc * row_bytes, hipMemcpyHostToDevice, stream_) != hipSuccess) {
                    throw std::runtime_error("DFlash2Drafter: staging memcpy failed");
                }
                launch_dflash2_repack_q8(
                    stage_bf16, rc, K,
                    dst.qs + (size_t)(row_dst0 + r0) * K,
                    dst.d  + (size_t)(row_dst0 + r0) * nb,
                    stream_);
            }
        };

        auto repack_matrix = [&](const uint8_t * p, int rows, int K) -> SoaQ8Matrix {
            SoaQ8Matrix M = alloc_q8(rows, K);
            repack_into(p, rows, K, M, 0);
            return M;
        };

        // 1. Target feature projection fc + hidden_norm + final norm
        fc_ = repack_matrix(src("fc.weight", {N_EMBD, 4 * N_EMBD}), N_EMBD, 4 * N_EMBD);
        hidden_norm_ = upload_bf16(src("hidden_norm.weight", {N_EMBD}), N_EMBD);
        final_norm_  = upload_bf16(src("norm.weight", {N_EMBD}), N_EMBD);

        // Precompute mask token embedding once from target tok_embd + tok_norm
        mask_embed_ = static_cast<uint16_t *>(df2_alloc(allocs_, (size_t)N_EMBD * 2, vram_bytes_, stream_));
        launch_dflash2_precompute_mask_embed(w_->tok_embd, w_->tok_norm, MASK_TOK_ID, mask_embed_, stream_);

        // 2. 5 decoder layers
        for (int l = 0; l < N_LAYERS; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            Layer & L = layers_[l];
            L.in_norm   = upload_bf16(src(p + "input_layernorm.weight", {N_EMBD}), N_EMBD);
            L.post_norm = upload_bf16(src(p + "post_attention_layernorm.weight", {N_EMBD}), N_EMBD);
            L.q_norm    = upload_bf16(src(p + "self_attn.q_norm.weight", {HEAD_DIM}), HEAD_DIM);
            L.k_norm    = upload_bf16(src(p + "self_attn.k_norm.weight", {HEAD_DIM}), HEAD_DIM);

            L.attn_conv_base = upload_bf16(src(p + "attention_conv.base_kernel", {2, 2, N_EMBD}),
                                           (size_t)2 * 2 * N_EMBD);
            L.mlp_conv_base  = upload_bf16(src(p + "mlp_conv.base_kernel", {2, 2, N_EMBD}),
                                           (size_t)2 * 2 * N_EMBD);
            L.attn_conv_proj = repack_matrix(
                src(p + "attention_conv.kernel_projection.weight", {CONV_PROJ, N_EMBD}),
                CONV_PROJ, N_EMBD);
            L.mlp_conv_proj  = repack_matrix(
                src(p + "mlp_conv.kernel_projection.weight", {CONV_PROJ, N_EMBD}),
                CONV_PROJ, N_EMBD);

            L.qkv = alloc_q8(QKV_DIM_D, N_EMBD);
            repack_into(src(p + "self_attn.q_proj.weight", {Q_DIM_D, N_EMBD}), Q_DIM_D, N_EMBD, L.qkv, 0);
            repack_into(src(p + "self_attn.k_proj.weight", {KV_DIM_D, N_EMBD}), KV_DIM_D, N_EMBD, L.qkv, Q_DIM_D);
            repack_into(src(p + "self_attn.v_proj.weight", {KV_DIM_D, N_EMBD}), KV_DIM_D, N_EMBD, L.qkv, Q_DIM_D + KV_DIM_D);

            L.wo = repack_matrix(src(p + "self_attn.o_proj.weight", {N_EMBD, Q_DIM_D}), N_EMBD, Q_DIM_D);

            L.gate_up = alloc_q8(2 * INTER_D, N_EMBD);
            repack_into(src(p + "mlp.gate_proj.weight", {INTER_D, N_EMBD}), INTER_D, N_EMBD, L.gate_up, 0);
            repack_into(src(p + "mlp.up_proj.weight", {INTER_D, N_EMBD}), INTER_D, N_EMBD, L.gate_up, INTER_D);

            L.w_down = repack_matrix(src(p + "mlp.down_proj.weight", {N_EMBD, INTER_D}), N_EMBD, INTER_D);

            // Sliding-window BF16 KV ring: [WINDOW + BLOCK, 8, 128]
            const size_t kv_bytes = (size_t)(WINDOW + BLOCK) * KV_DIM_D * 2;
            L.k_ctx   = static_cast<uint16_t *>(df2_alloc(allocs_, kv_bytes, vram_bytes_, stream_));
            L.v_ctx   = static_cast<uint16_t *>(df2_alloc(allocs_, kv_bytes, vram_bytes_, stream_));
            L.ctx_pos = static_cast<int32_t *>(df2_alloc(allocs_, (size_t)WINDOW * 4, vram_bytes_, stream_));
        }

        // 3. Candidate Path Selector weights
        sel_hidden_proj_ = repack_matrix(
            src("candidate_selector.hidden_projection.weight", {SEL_RANK, N_EMBD}),
            SEL_RANK, N_EMBD);
        sel_pred_cb_ = repack_matrix(
            src_vocab("candidate_selector.predecessor_codebook", SEL_RANK),
            vocab_, SEL_RANK);
        sel_succ_cb_ = repack_matrix(
            src_vocab("candidate_selector.successor_codebook", SEL_RANK),
            vocab_, SEL_RANK);

        // Synchronize and free transient BF16 staging buffer right away.
        if (hipStreamSynchronize(stream_) != hipSuccess) {
            throw std::runtime_error("DFlash2Drafter: sync after repack failed");
        }
        (void)hipFree(stage_bf16);
        stage_bf16 = nullptr;

        // 4. Per-call scratch buffers
        res_        = static_cast<uint16_t *>(df2_alloc(allocs_, (size_t)BLOCK * N_EMBD * 2, vram_bytes_, stream_));
        xn_         = static_cast<uint16_t *>(df2_alloc(allocs_, (size_t)BLOCK * N_EMBD * 2, vram_bytes_, stream_));
        xconv_      = static_cast<uint16_t *>(df2_alloc(allocs_, (size_t)BLOCK * N_EMBD * 2, vram_bytes_, stream_));
        conv_coeff_ = static_cast<float *>(df2_alloc(allocs_, (size_t)BLOCK * CONV_PROJ * 4, vram_bytes_, stream_));
        qkv_out_    = static_cast<float *>(df2_alloc(allocs_, (size_t)BLOCK * QKV_DIM_D * 4, vram_bytes_, stream_));
        q_          = static_cast<uint16_t *>(df2_alloc(allocs_, (size_t)BLOCK * Q_DIM_D * 2, vram_bytes_, stream_));
        attn_o_     = static_cast<uint16_t *>(df2_alloc(allocs_, (size_t)BLOCK * Q_DIM_D * 2, vram_bytes_, stream_));
        delta_      = static_cast<float *>(df2_alloc(allocs_, (size_t)BLOCK * N_EMBD * 4, vram_bytes_, stream_));
        gu_out_     = static_cast<float *>(df2_alloc(allocs_, (size_t)BLOCK * 2 * INTER_D * 4, vram_bytes_, stream_));
        act_        = static_cast<uint16_t *>(df2_alloc(allocs_, (size_t)BLOCK * INTER_D * 2, vram_bytes_, stream_));
        h_final_bf_ = static_cast<uint16_t *>(df2_alloc(allocs_, (size_t)GAMMA * N_EMBD * 2, vram_bytes_, stream_));
        h_final_f32_= static_cast<float *>(df2_alloc(allocs_, (size_t)GAMMA * N_EMBD * 4, vram_bytes_, stream_));
        logits_     = static_cast<float *>(df2_alloc(allocs_, (size_t)GAMMA * vocab_ * 4, vram_bytes_, stream_));
        sel_h_      = static_cast<float *>(df2_alloc(allocs_, (size_t)GAMMA * SEL_RANK * 4, vram_bytes_, stream_));
        topk_pv_    = static_cast<float *>(df2_alloc(allocs_, (size_t)dflash2_topk16_part_elems() * 4, vram_bytes_, stream_));
        topk_pi_    = static_cast<int32_t *>(df2_alloc(allocs_, (size_t)dflash2_topk16_part_elems() * 4, vram_bytes_, stream_));
        cand_val_   = static_cast<float *>(df2_alloc(allocs_, (size_t)GAMMA * SEL_TOP_K * 4, vram_bytes_, stream_));
        cand_idx_   = static_cast<int32_t *>(df2_alloc(allocs_, (size_t)GAMMA * SEL_TOP_K * 4, vram_bytes_, stream_));
        tokens_     = static_cast<int32_t *>(df2_alloc(allocs_, (size_t)GAMMA * 4, vram_bytes_, stream_));
        conf_       = static_cast<float *>(df2_alloc(allocs_, (size_t)GAMMA * 4, vram_bytes_, stream_));
        attn_ws_    = static_cast<float *>(df2_alloc(allocs_, (size_t)dflash2_attention_ws_floats() * 4, vram_bytes_, stream_));
        ctx_proj_   = static_cast<float *>(df2_alloc(allocs_, (size_t)BLOCK * N_EMBD * 4, vram_bytes_, stream_));
        ctx_h_      = static_cast<uint16_t *>(df2_alloc(allocs_, (size_t)BLOCK * N_EMBD * 2, vram_bytes_, stream_));
        kv_commit_  = static_cast<float *>(df2_alloc(allocs_, (size_t)BLOCK * 2 * KV_DIM_D * 4, vram_bytes_, stream_));
        d_dp_       = static_cast<DFlash2DraftParams *>(df2_alloc(allocs_, sizeof(DFlash2DraftParams), vram_bytes_, stream_));

        reset();
        step7j_prime_occupancy();
        if (hipStreamSynchronize(stream_) != hipSuccess) {
            throw std::runtime_error("DFlash2Drafter: final init sync failed");
        }
        if (std::getenv("INKLING_MTP_NO_GRAPH") == nullptr) {
            capture_draft_graph();
        }
    } catch (const std::exception & e) {
        if (stage_bf16) (void)hipFree(stage_bf16);
        std::fprintf(stderr, "[dflash2] init failed: %s\n", e.what());
        return false;
    }

    std::printf("[dflash2] loaded %s (5 layers, SWA=%d, vocab=%d, selector rank=%d top_k=%d, %.1f MiB VRAM)\n",
                resolved.c_str(), WINDOW, vocab_, SEL_RANK, SEL_TOP_K,
                (double)vram_bytes_ / (1024.0 * 1024.0));
    std::fflush(stdout);
    return true;
}

void DFlash2Drafter::reset() {
    length_ = 0;
    for (int l = 0; l < N_LAYERS; ++l) {
        if (layers_[l].ctx_pos != nullptr) {
            (void)hipMemsetAsync(layers_[l].ctx_pos, 0xff, (size_t)WINDOW * 4, stream_);
        }
    }
}

bool DFlash2Drafter::save_checkpoint(Checkpoint & out) const {
    if (length_ <= 0 || layers_[0].k_ctx == nullptr) return false;
    const size_t kv_bytes  = (size_t)(WINDOW + BLOCK) * KV_DIM_D * 2;
    const size_t pos_bytes = (size_t)WINDOW * 4;
    const size_t total_bytes = (size_t)N_LAYERS * (2 * kv_bytes + pos_bytes);
    out.layers_blob.resize(total_bytes);

    size_t off = 0;
    for (int l = 0; l < N_LAYERS; ++l) {
        const Layer & L = layers_[l];
        if (hipMemcpyAsync(out.layers_blob.data() + off, L.k_ctx, kv_bytes, hipMemcpyDeviceToHost, stream_) != hipSuccess) return false;
        off += kv_bytes;
        if (hipMemcpyAsync(out.layers_blob.data() + off, L.v_ctx, kv_bytes, hipMemcpyDeviceToHost, stream_) != hipSuccess) return false;
        off += kv_bytes;
        if (hipMemcpyAsync(out.layers_blob.data() + off, L.ctx_pos, pos_bytes, hipMemcpyDeviceToHost, stream_) != hipSuccess) return false;
        off += pos_bytes;
    }
    if (hipStreamSynchronize(stream_) != hipSuccess) return false;
    out.length = length_;
    return true;
}

bool DFlash2Drafter::restore_checkpoint(const Checkpoint & ckpt) {
    if (ckpt.length <= 0 || layers_[0].k_ctx == nullptr) return false;
    const size_t kv_bytes  = (size_t)(WINDOW + BLOCK) * KV_DIM_D * 2;
    const size_t pos_bytes = (size_t)WINDOW * 4;
    const size_t total_bytes = (size_t)N_LAYERS * (2 * kv_bytes + pos_bytes);
    if (ckpt.layers_blob.size() != total_bytes) return false;

    size_t off = 0;
    for (int l = 0; l < N_LAYERS; ++l) {
        const Layer & L = layers_[l];
        if (hipMemcpyAsync(L.k_ctx, ckpt.layers_blob.data() + off, kv_bytes, hipMemcpyHostToDevice, stream_) != hipSuccess) return false;
        off += kv_bytes;
        if (hipMemcpyAsync(L.v_ctx, ckpt.layers_blob.data() + off, kv_bytes, hipMemcpyHostToDevice, stream_) != hipSuccess) return false;
        off += kv_bytes;
        if (hipMemcpyAsync(L.ctx_pos, ckpt.layers_blob.data() + off, pos_bytes, hipMemcpyHostToDevice, stream_) != hipSuccess) return false;
        off += pos_bytes;
    }
    if (hipStreamSynchronize(stream_) != hipSuccess) return false;
    length_ = ckpt.length;
    return true;
}

void DFlash2Drafter::run_commit_piece(const uint16_t * feats, int n, int start,
                                      const DFlash2DraftParams * dp) {
    const int mode_gemv = dp ? 2 : 0;
    const int mode_norm = dp ? 1 : 0;
    // 1. Project target taps [n, 9216] -> [n, 2304] -> hidden_norm (BF16)
    launch_dflash2_gemv(fc_, 0, N_EMBD, feats, 4 * N_EMBD, n, ctx_proj_, N_EMBD,
                        stream_, dp, mode_gemv);
    launch_dflash2_rmsnorm_f32_to_bf16(ctx_proj_, n, hidden_norm_, ctx_h_,
                                       stream_, dp, mode_norm);

    // 2. For all 5 layers: directly project ctx_h -> K, V (NO input_layernorm, NO attention_conv)
    for (int l = 0; l < N_LAYERS; ++l) {
        Layer & L = layers_[l];
        launch_dflash2_gemv(L.qkv, Q_DIM_D, 2 * KV_DIM_D, ctx_h_, N_EMBD, n,
                            kv_commit_, 2 * KV_DIM_D, stream_, dp, mode_gemv);
        launch_dflash2_ctx_kv_store(kv_commit_, n, start, L.k_norm,
                                    L.k_ctx, L.v_ctx, L.ctx_pos, stream_, dp);
    }
}

void DFlash2Drafter::commit(const uint16_t * d_feats, int n, int start) {
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
            if (hipStreamSynchronize(stream_) == hipSuccess) {
                hipGraph_t g = nullptr;
                if (hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal) == hipSuccess) {
                    run_commit_piece(d_feats, BLOCK, start, d_dp_);
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
            DFlash2DraftParams p{};
            p.pos     = start;
            p.ncommit = n;
            launch_dflash2_set_params(d_dp_, p, stream_);
            hipError_t ge = hipGraphLaunch(commit_graph_exec_, stream_);
            if (ge != hipSuccess) {
                std::fprintf(stderr, "[dflash2] commit graph launch failed: %s\n", hipGetErrorString(ge));
                std::abort();
            }
            length_ = start + n;
            return;
        }
    }

    // Chunked path for prefill (n > BLOCK) or when graphs are disabled
    for (int r0 = 0; r0 < n; r0 += BLOCK) {
        const int k = std::min(BLOCK, n - r0);
        run_commit_piece(d_feats + (size_t)r0 * 4 * N_EMBD, k, start + r0, nullptr);
    }
    length_ = start + n;
}

void DFlash2Drafter::run_draft_body(const DFlash2DraftParams * dp, int anchor, int P) {
    // 1. Block input embeddings: res[0]=embed_norm(anchor), res[1..7]=mask_embed_, xn=in_norm(res)
    launch_dflash2_embed_block(w_->tok_embd, w_->tok_norm, mask_embed_, layers_[0].in_norm,
                               anchor, res_, xn_, stream_, dp);

    // 2. 5-layer backbone with dynamic 2-tap grouped convolutions + bidirectional SWA-2048 attention
    for (int l = 0; l < N_LAYERS; ++l) {
        Layer & L = layers_[l];

        // Attention conv prepare
        launch_dflash2_gemv(L.attn_conv_proj, 0, CONV_PROJ, xn_, N_EMBD, BLOCK,
                            conv_coeff_, CONV_PROJ, stream_);
        launch_dflash2_conv_in(xn_, conv_coeff_, L.attn_conv_base, xconv_, stream_);

        // Attention QKV + QK-norm + NeoX RoPE (theta=1e4) + bidirectional SWA attention + O proj
        launch_dflash2_gemv(L.qkv, 0, QKV_DIM_D, xconv_, N_EMBD, BLOCK,
                            qkv_out_, QKV_DIM_D, stream_);
        launch_dflash2_qkv_rope(qkv_out_, L.q_norm, L.k_norm, P, q_,
                                L.k_ctx, L.v_ctx, stream_, dp);
        launch_dflash2_attention(q_, L.k_ctx, L.v_ctx, L.ctx_pos, P,
                                 attn_ws_, attn_o_, stream_, dp);
        launch_dflash2_gemv(L.wo, 0, N_EMBD, attn_o_, Q_DIM_D, BLOCK,
                            delta_, N_EMBD, stream_);

        // Attention conv finish + residual add + post_attention_layernorm
        launch_dflash2_conv_out_add_rmsnorm(delta_, conv_coeff_, L.attn_conv_base,
                                            res_, L.post_norm, xn_, stream_);

        // MLP conv prepare + SwiGLU MLP + conv finish + residual add + next input_layernorm
        launch_dflash2_gemv(L.mlp_conv_proj, 0, CONV_PROJ, xn_, N_EMBD, BLOCK,
                            conv_coeff_, CONV_PROJ, stream_);
        launch_dflash2_conv_in(xn_, conv_coeff_, L.mlp_conv_base, xconv_, stream_);
        launch_dflash2_gemv(L.gate_up, 0, 2 * INTER_D, xconv_, N_EMBD, BLOCK,
                            gu_out_, 2 * INTER_D, stream_);
        launch_dflash2_silu_mul(gu_out_, BLOCK, INTER_D, act_, stream_);
        launch_dflash2_gemv(L.w_down, 0, N_EMBD, act_, INTER_D, BLOCK,
                            delta_, N_EMBD, stream_);

        const uint16_t * next_norm = (l + 1 < N_LAYERS) ? layers_[l + 1].in_norm : nullptr;
        launch_dflash2_conv_out_add_rmsnorm(delta_, conv_coeff_, L.mlp_conv_base,
                                            res_, next_norm, xn_, stream_);
    }

    // 3. Final RMSNorm on prediction rows 1..7 -> BF16 (selector) + FP32 (target lm_head)
    launch_dflash2_final_norm_pred(res_, final_norm_, h_final_bf_, h_final_f32_, stream_);

    // 4. Full-vocabulary raw logits [7, VOCAB_UNPADDED] via engine's audited fdot2 m-row lm_head GEMV
    SoaQ8Matrix head = w_->lm_head;
    head.rows = vocab_;
    launch_soa_q8_gemv_fdot2_mrows(head, h_final_f32_, logits_, GAMMA, stream_, 0);

    // 5. Exact 2-stage Top-16 over [7, vocab_]
    launch_dflash2_topk16(logits_, vocab_, topk_pv_, topk_pi_, cand_val_, cand_idx_, stream_);

    // 6. Candidate Path Selector: hidden_projection [7, 256] + single-CTA greedy walk
    launch_dflash2_gemv(sel_hidden_proj_, 0, SEL_RANK, h_final_bf_, N_EMBD, GAMMA,
                        sel_h_, SEL_RANK, stream_);
    launch_dflash2_selector_walk(sel_pred_cb_, sel_succ_cb_, sel_h_,
                                 cand_val_, cand_idx_, anchor,
                                 tokens_, conf_, stream_, dp);
}

void DFlash2Drafter::capture_draft_graph() {
    if (d_dp_ == nullptr) return;
    if (hipStreamSynchronize(stream_) != hipSuccess) return;
    hipGraph_t g = nullptr;
    if (hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal) != hipSuccess) {
        std::fprintf(stderr, "[dflash2] draft graph capture begin failed\n");
        return;
    }
    run_draft_body(d_dp_, 0, 0);
    if (hipStreamEndCapture(stream_, &g) != hipSuccess) {
        std::fprintf(stderr, "[dflash2] draft graph capture end failed\n");
        return;
    }
    if (hipGraphInstantiate(&draft_graph_exec_, g, nullptr, nullptr, 0) != hipSuccess) {
        std::fprintf(stderr, "[dflash2] draft graph instantiate failed\n");
        draft_graph_exec_ = nullptr;
    }
    (void)hipGraphDestroy(g);
    if (draft_graph_exec_ != nullptr) {
        std::printf("[dflash2] captured 1 static draft HIP graph (O(1) SWA=%d)\n", WINDOW);
        std::fflush(stdout);
    }
}

DSparkDraft DFlash2Drafter::draft(int32_t anchor, int P) {
    if (anchor < 0 || anchor >= vocab_) {
        std::fprintf(stderr, "[dflash2] anchor %d out of range [0, %d)\n", anchor, vocab_);
        return {};
    }
    const bool dbg = std::getenv("INKLING_MTP_DEBUG_K") != nullptr;
    if (!dbg && draft_graph_exec_ != nullptr) {
        DFlash2DraftParams p{};
        p.anchor  = anchor;
        p.pos     = P;
        p.ncommit = BLOCK;
        launch_dflash2_set_params(d_dp_, p, stream_);
        hipError_t ge = hipGraphLaunch(draft_graph_exec_, stream_);
        if (ge != hipSuccess) {
            std::fprintf(stderr, "[dflash2] draft graph launch failed: %s\n", hipGetErrorString(ge));
            std::abort();
        }
        DSparkDraft d;
        d.tokens = tokens_;
        d.conf   = conf_;
        return d;
    }

    run_draft_body(nullptr, anchor, P);
    static bool dbg_once = dbg;
    if (dbg_once) {
        dbg_once = false;
        (void)hipStreamSynchronize(stream_);
        float cv[8];
        int32_t ci[8];
        int32_t tk[GAMMA];
        float cf[GAMMA];
        (void)hipMemcpy(cv, cand_val_, sizeof(cv), hipMemcpyDeviceToHost);
        (void)hipMemcpy(ci, cand_idx_, sizeof(ci), hipMemcpyDeviceToHost);
        (void)hipMemcpy(tk, tokens_, sizeof(tk), hipMemcpyDeviceToHost);
        (void)hipMemcpy(cf, conf_, sizeof(cf), hipMemcpyDeviceToHost);
        std::printf("[dflash2-dbg] P=%d anchor=%d | top8 row0 (val,id):", P, anchor);
        for (int i = 0; i < 8; ++i) std::printf(" (%.3f,%d)", cv[i], ci[i]);
        std::printf(" | picked:");
        for (int i = 0; i < GAMMA; ++i) std::printf(" %d(%.2f)", tk[i], cf[i]);
        std::printf("\n");
        std::fflush(stdout);
    }
    DSparkDraft d;
    d.tokens = tokens_;
    d.conf   = conf_;
    return d;
}

void DFlash2Drafter::debug_base_copy(float * host_out) const {
    (void)hipStreamSynchronize(stream_);
    (void)hipMemcpy(host_out, cand_val_, (size_t)GAMMA * SEL_TOP_K * 4, hipMemcpyDeviceToHost);
}

double DFlash2Drafter::debug_gemv_check() {
    const int m = 8;
    const int test_rows = std::min(4096, w_->lm_head.rows);
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
    uint16_t * dx = nullptr;
    float * ym = nullptr;
    float * y1 = nullptr;
    float * dxf = nullptr;
    if (hipMalloc(&dx, (size_t)m * N_EMBD * 2) != hipSuccess ||
        hipMalloc(&ym, (size_t)m * test_rows * 4) != hipSuccess ||
        hipMalloc(&y1, (size_t)test_rows * 4) != hipSuccess ||
        hipMalloc(&dxf, (size_t)N_EMBD * 4) != hipSuccess) {
        if (dx) (void)hipFree(dx);
        if (ym) (void)hipFree(ym);
        if (y1) (void)hipFree(y1);
        if (dxf) (void)hipFree(dxf);
        return -1.0;
    }

    (void)hipMemcpyAsync(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice, stream_);
    SoaQ8Matrix head = w_->lm_head;
    head.rows = test_rows;
    launch_dflash2_gemv(head, 0, test_rows, dx, N_EMBD, m, ym, test_rows, stream_);

    double worst = 0.0;
    std::vector<float> xf((size_t)N_EMBD);
    std::vector<float> hm((size_t)test_rows);
    std::vector<float> h1((size_t)test_rows);
    for (int r = 0; r < m; ++r) {
        for (int i = 0; i < N_EMBD; ++i) xf[(size_t)i] = df2_h_bf2f(hx[(size_t)r * N_EMBD + i]);
        (void)hipMemcpyAsync(dxf, xf.data(), xf.size() * 4, hipMemcpyHostToDevice, stream_);
        launch_soa_q8_gemv_fdot2(head, dxf, y1, stream_, 0);
        (void)hipStreamSynchronize(stream_);
        (void)hipMemcpy(hm.data(), ym + (size_t)r * test_rows, hm.size() * 4, hipMemcpyDeviceToHost);
        (void)hipMemcpy(h1.data(), y1, h1.size() * 4, hipMemcpyDeviceToHost);
        for (size_t i = 0; i < hm.size(); ++i) {
            worst = std::max(worst, (double)std::fabs((double)hm[i] - (double)h1[i]));
        }
    }

    (void)hipFree(dx);
    (void)hipFree(ym);
    (void)hipFree(y1);
    (void)hipFree(dxf);
    std::printf("[dflash2] gemv differential (m=8 fdot2 vs engine 1-row fdot2): max |d| = %.6g\n", worst);
    std::fflush(stdout);
    return worst;
}

// ---------------------------------------------------------------------------
// SpecDrafter unified auto-detecting facade
// ---------------------------------------------------------------------------
size_t SpecDrafter::estimate_vram_bytes(const std::string & mtp_path) {
    if (DFlash2Drafter::is_dflash2_checkpoint(mtp_path)) {
        return DFlash2Drafter::estimate_vram_bytes();
    }
    return 1536ull * 1024 * 1024; // DSpark ~1.5 GiB
}

bool SpecDrafter::init(const std::string & mtp_path, const ModelWeights & w, int max_ctx,
                       int vocab, hipStream_t stream, int draft_ctx_cap) {
    dspark_.reset();
    dflash2_.reset();
    kind_ = NONE;

    if (DFlash2Drafter::is_dflash2_checkpoint(mtp_path)) {
        auto df = std::make_unique<DFlash2Drafter>();
        if (!df->init(mtp_path, w, max_ctx, vocab, stream, draft_ctx_cap)) {
            return false;
        }
        dflash2_ = std::move(df);
        kind_ = DFLASH2;
        return true;
    }

    auto ds = std::make_unique<DSparkDrafter>();
    if (!ds->init(mtp_path, w, max_ctx, vocab, stream, draft_ctx_cap)) {
        return false;
    }
    dspark_ = std::move(ds);
    kind_ = DSPARK;
    return true;
}

void SpecDrafter::reset() {
    if (kind_ == DFLASH2) dflash2_->reset();
    else if (kind_ == DSPARK) dspark_->reset();
}

void SpecDrafter::commit(const uint16_t * d_feats, int n, int start) {
    if (kind_ == DFLASH2) dflash2_->commit(d_feats, n, start);
    else if (kind_ == DSPARK) dspark_->commit(d_feats, n, start);
}

void SpecDrafter::note_tokens(const int32_t * ids, int n) {
    if (kind_ == DFLASH2) dflash2_->note_tokens(ids, n);
    else if (kind_ == DSPARK) dspark_->note_tokens(ids, n);
}

DSparkDraft SpecDrafter::draft(int32_t anchor, int P) {
    if (kind_ == DFLASH2) return dflash2_->draft(anchor, P);
    if (kind_ == DSPARK)  return dspark_->draft(anchor, P);
    return {};
}

int SpecDrafter::length() const {
    if (kind_ == DFLASH2) return dflash2_->length();
    if (kind_ == DSPARK)  return dspark_->length();
    return 0;
}

int SpecDrafter::ctx() const {
    if (kind_ == DFLASH2) return dflash2_->ctx();
    if (kind_ == DSPARK)  return dspark_->ctx();
    return 0;
}

int SpecDrafter::vocab() const {
    if (kind_ == DFLASH2) return dflash2_->vocab();
    if (kind_ == DSPARK)  return dspark_->vocab();
    return 0;
}

size_t SpecDrafter::vram_bytes() const {
    if (kind_ == DFLASH2) return dflash2_->vram_bytes();
    if (kind_ == DSPARK)  return dspark_->vram_bytes();
    return 0;
}

int SpecDrafter::short_rows() const {
    if (kind_ == DFLASH2) return dflash2_->short_rows();
    if (kind_ == DSPARK)  return dspark_->short_rows();
    return 0;
}

bool SpecDrafter::save_checkpoint(Checkpoint & out) const {
    out.kind = kind_;
    if (kind_ == DFLASH2) return dflash2_->save_checkpoint(out.dflash2);
    if (kind_ == DSPARK)  return dspark_->save_checkpoint(out.dspark);
    return false;
}

bool SpecDrafter::restore_checkpoint(const Checkpoint & ckpt) {
    if (ckpt.kind != kind_) return false;
    if (kind_ == DFLASH2) return dflash2_->restore_checkpoint(ckpt.dflash2);
    if (kind_ == DSPARK)  return dspark_->restore_checkpoint(ckpt.dspark);
    return false;
}

double SpecDrafter::debug_gemv_check() {
    if (kind_ == DFLASH2) return dflash2_->debug_gemv_check();
    if (kind_ == DSPARK)  return dspark_->debug_gemv_check();
    return 0.0;
}

double SpecDrafter::debug_shortlist_check() {
    if (kind_ == DFLASH2) return dflash2_->debug_shortlist_check();
    if (kind_ == DSPARK)  return dspark_->debug_shortlist_check();
    return 0.0;
}

const int32_t * SpecDrafter::debug_cand_idx() const {
    if (kind_ == DFLASH2) return dflash2_->debug_cand_idx();
    if (kind_ == DSPARK)  return dspark_->debug_cand_idx();
    return nullptr;
}

const float * SpecDrafter::debug_cand_val() const {
    if (kind_ == DFLASH2) return dflash2_->debug_cand_val();
    if (kind_ == DSPARK)  return dspark_->debug_cand_val();
    return nullptr;
}

void SpecDrafter::debug_base_copy(float * host_out) const {
    if (kind_ == DFLASH2) dflash2_->debug_base_copy(host_out);
    else if (kind_ == DSPARK) dspark_->debug_base_copy(host_out);
}

} // namespace inkling
