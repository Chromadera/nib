// ============================================================================
// vision_encoder.cpp — Native RDNA3 Vision Encoder (inkling2v / mmproj-Q8_0)
// ============================================================================

#include "vision_encoder.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#define STBI_ONLY_TGA
#include "../third_party/stb_image.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace inkling {

namespace detail {
void repack_q8_0_to_soa(
    const void * d_raw, int8_t * d_qs, __half * d_d,
    int rows, int K, int dst_row_offset, hipStream_t stream);
} // namespace detail

namespace {

static inline void v_hip_check(hipError_t err, const char * expr, int line) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "HIP error %d (%s) at vision_encoder.cpp:%d: %s\n",
                     (int)err, hipGetErrorString(err), line, expr);
        std::abort();
    }
}
#define V_HIP_CHECK(x) v_hip_check((x), #x, __LINE__)

enum : uint32_t {
    GGUF_T_UINT8   = 0,
    GGUF_T_INT8    = 1,
    GGUF_T_UINT16  = 2,
    GGUF_T_INT16   = 3,
    GGUF_T_UINT32  = 4,
    GGUF_T_INT32   = 5,
    GGUF_T_FLOAT32 = 6,
    GGUF_T_BOOL    = 7,
    GGUF_T_STRING  = 8,
    GGUF_T_ARRAY   = 9,
    GGUF_T_UINT64  = 10,
    GGUF_T_INT64   = 11,
    GGUF_T_FLOAT64 = 12,
};

enum : uint32_t {
    GGML_T_F32  = 0,
    GGML_T_Q8_0 = 8,
};

struct VReader {
    const uint8_t * p;
    const uint8_t * end;
    void need(size_t n) const {
        if ((size_t)(end - p) < n) throw std::runtime_error("mmproj gguf: truncated");
    }
    uint8_t  u8()  { need(1); return *p++; }
    uint16_t u16() { need(2); uint16_t v; std::memcpy(&v, p, 2); p += 2; return v; }
    uint32_t u32() { need(4); uint32_t v; std::memcpy(&v, p, 4); p += 4; return v; }
    uint64_t u64() { need(8); uint64_t v; std::memcpy(&v, p, 8); p += 8; return v; }
    float    f32() { uint32_t b = u32(); float v; std::memcpy(&v, &b, 4); return v; }
    std::string str() {
        uint64_t n = u64();
        need((size_t)n);
        std::string s(reinterpret_cast<const char *>(p), (size_t)n);
        p += n;
        return s;
    }
    void skip(size_t n) { need(n); p += n; }
};

static int64_t read_int_any(VReader & r, uint32_t t) {
    switch (t) {
        case GGUF_T_UINT8:  return (int64_t)r.u8();
        case GGUF_T_INT8:   return (int64_t)(int8_t)r.u8();
        case GGUF_T_UINT16: return (int64_t)r.u16();
        case GGUF_T_INT16:  return (int64_t)(int16_t)r.u16();
        case GGUF_T_UINT32: return (int64_t)r.u32();
        case GGUF_T_INT32:  return (int64_t)(int32_t)r.u32();
        case GGUF_T_UINT64: return (int64_t)r.u64();
        case GGUF_T_INT64:  return (int64_t)r.u64();
        default: throw std::runtime_error("mmproj gguf: bad int type");
    }
}

static void skip_val(VReader & r, uint32_t t) {
    switch (t) {
        case GGUF_T_UINT8:
        case GGUF_T_INT8:
        case GGUF_T_BOOL:    r.skip(1); break;
        case GGUF_T_UINT16:
        case GGUF_T_INT16:   r.skip(2); break;
        case GGUF_T_UINT32:
        case GGUF_T_INT32:
        case GGUF_T_FLOAT32: r.skip(4); break;
        case GGUF_T_UINT64:
        case GGUF_T_INT64:
        case GGUF_T_FLOAT64: r.skip(8); break;
        case GGUF_T_STRING:  (void)r.str(); break;
        case GGUF_T_ARRAY: {
            uint32_t et = r.u32();
            uint64_t cnt = r.u64();
            for (uint64_t i = 0; i < cnt; ++i) skip_val(r, et);
            break;
        }
        default: throw std::runtime_error("mmproj gguf: unknown kv type");
    }
}

struct VTensorInfo {
    std::string name;
    uint32_t    n_dims = 0;
    uint64_t    dims[4] = { 1, 1, 1, 1 };
    uint32_t    type = 0;
    uint64_t    offset = 0;
    size_t n_elems() const {
        return (size_t)dims[0] * dims[1] * dims[2] * dims[3];
    }
    size_t n_bytes() const {
        if (type == GGML_T_F32)  return n_elems() * 4;
        if (type == GGML_T_Q8_0) return (n_elems() / 32) * 34;
        throw std::runtime_error("mmproj gguf: unsupported tensor type for " + name);
    }
};

static inline int align_up(int x, int a) {
    return ((x + a - 1) / a) * a;
}

// Simple bilinear RGB resize (only triggered if image exceeds max_pixels ~ 6.55 MP)
static void resize_rgb_bilinear(
    const uint8_t * src, int sw, int sh,
    std::vector<uint8_t> & dst, int dw, int dh)
{
    dst.resize((size_t)dw * dh * 3);
    const float sx = (float)sw / (float)dw;
    const float sy = (float)sh / (float)dh;
    for (int y = 0; y < dh; ++y) {
        const float yf = ((float)y + 0.5f) * sy - 0.5f;
        int y0 = std::max(0, std::min((int)std::floor(yf), sh - 1));
        int y1 = std::max(0, std::min(y0 + 1, sh - 1));
        const float dy = std::max(0.0f, std::min(1.0f, yf - (float)y0));
        for (int x = 0; x < dw; ++x) {
            const float xf = ((float)x + 0.5f) * sx - 0.5f;
            int x0 = std::max(0, std::min((int)std::floor(xf), sw - 1));
            int x1 = std::max(0, std::min(x0 + 1, sw - 1));
            const float dx = std::max(0.0f, std::min(1.0f, xf - (float)x0));
            for (int c = 0; c < 3; ++c) {
                float a = src[((size_t)y0 * sw + x0) * 3 + c];
                float b = src[((size_t)y0 * sw + x1) * 3 + c];
                float d = src[((size_t)y1 * sw + x0) * 3 + c];
                float e = src[((size_t)y1 * sw + x1) * 3 + c];
                float v = a * (1.0f - dx) * (1.0f - dy)
                        + b * dx * (1.0f - dy)
                        + d * (1.0f - dx) * dy
                        + e * dx * dy;
                int iv = (int)std::lround(v);
                dst[((size_t)y * dw + x) * 3 + c] = (uint8_t)std::max(0, std::min(255, iv));
            }
        }
    }
}

} // namespace

VisionEncoder::~VisionEncoder() {
    destroy();
}

void VisionEncoder::destroy() {
    release_workspace();
    if (d_proj_out_) (void)hipFree(d_proj_out_);
    d_proj_out_         = nullptr;
    proj_out_bytes_     = 0;
    proj_out_cap_       = 0;
    scratch_vram_bytes_ = 0;
    last_arena_bytes_   = 0;
    peak_arena_bytes_   = 0;

    for (void * p : weight_allocs_) {
        if (p) (void)hipFree(p);
    }
    weight_allocs_.clear();
    weight_vram_bytes_ = 0;
    total_vram_bytes_  = 0;
    layers_.clear();
}

bool VisionEncoder::init(const std::string & mmproj_path, hipStream_t stream) {
    destroy();
    stream_ = stream;
    const auto t0 = std::chrono::steady_clock::now();

    int fd = ::open(mmproj_path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "[vision] failed to open %s\n", mmproj_path.c_str());
        return false;
    }
    struct stat st {};
    if (::fstat(fd, &st) != 0) { ::close(fd); return false; }
    const size_t file_size = (size_t)st.st_size;
    void * map = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (map == MAP_FAILED) {
        std::fprintf(stderr, "[vision] mmap failed for %s\n", mmproj_path.c_str());
        return false;
    }

    auto unmap_guard = [&]() { ::munmap(map, file_size); };

    try {
        VReader r{ static_cast<const uint8_t *>(map), static_cast<const uint8_t *>(map) + file_size };
        uint8_t magic[4] = { r.u8(), r.u8(), r.u8(), r.u8() };
        if (std::memcmp(magic, "GGUF", 4) != 0) {
            throw std::runtime_error("mmproj: not a GGUF file");
        }
        const uint32_t ver = r.u32();
        if (ver < 2 || ver > 3) {
            throw std::runtime_error("mmproj: unsupported GGUF version");
        }
        const uint64_t n_tensors = r.u64();
        const uint64_t n_kv      = r.u64();
        uint64_t alignment = 32;

        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key = r.str();
            uint32_t t = r.u32();
            if (key == "general.alignment") {
                alignment = (uint64_t)read_int_any(r, t);
            } else if (key == "clip.vision.block_count") {
                hp_.n_layer = (int32_t)read_int_any(r, t);
            } else if (key == "clip.vision.embedding_length") {
                hp_.n_embd = (int32_t)read_int_any(r, t);
            } else if (key == "clip.vision.attention.head_count") {
                hp_.n_head = (int32_t)read_int_any(r, t);
            } else if (key == "clip.vision.feed_forward_length") {
                hp_.n_ff = (int32_t)read_int_any(r, t);
            } else if (key == "clip.vision.patch_size") {
                hp_.patch_size = (int32_t)read_int_any(r, t);
            } else if (key == "clip.vision.image_size") {
                hp_.image_size = (int32_t)read_int_any(r, t);
            } else if (key == "clip.vision.spatial_merge_size") {
                hp_.n_merge = (int32_t)read_int_any(r, t);
            } else if (key == "clip.vision.projection_dim") {
                hp_.proj_dim = (int32_t)read_int_any(r, t);
            } else if (key == "clip.vision.attention.layer_norm_epsilon") {
                hp_.ln_eps = r.f32();
            } else if (key == "clip.vision.image_mean" && t == GGUF_T_ARRAY) {
                uint32_t et = r.u32();
                uint64_t cnt = r.u64();
                for (uint64_t c = 0; c < cnt; ++c) {
                    float v = (et == GGUF_T_FLOAT32) ? r.f32() : (skip_val(r, et), 0.5f);
                    if (c < 3) hp_.image_mean[c] = v;
                }
            } else if (key == "clip.vision.image_std" && t == GGUF_T_ARRAY) {
                uint32_t et = r.u32();
                uint64_t cnt = r.u64();
                for (uint64_t c = 0; c < cnt; ++c) {
                    float v = (et == GGUF_T_FLOAT32) ? r.f32() : (skip_val(r, et), 0.5f);
                    if (c < 3) hp_.image_std[c] = v;
                }
            } else {
                skip_val(r, t);
            }
        }
        hp_.head_dim = hp_.n_embd / hp_.n_head;

        std::unordered_map<std::string, VTensorInfo> tmap;
        tmap.reserve((size_t)n_tensors * 2);
        for (uint64_t i = 0; i < n_tensors; ++i) {
            VTensorInfo ti;
            ti.name   = r.str();
            ti.n_dims = r.u32();
            for (uint32_t d = 0; d < ti.n_dims; ++d) ti.dims[d] = r.u64();
            ti.type   = r.u32();
            ti.offset = r.u64();
            tmap[ti.name] = ti;
        }

        const size_t header_bytes = (size_t)(r.p - static_cast<const uint8_t *>(map));
        const size_t data_base    = ((header_bytes + alignment - 1) / alignment) * alignment;
        const uint8_t * data_ptr  = static_cast<const uint8_t *>(map) + data_base;

        auto get_tensor = [&](const std::string & name) -> const VTensorInfo & {
            auto it = tmap.find(name);
            if (it == tmap.end()) throw std::runtime_error("mmproj: missing tensor " + name);
            return it->second;
        };

        auto alloc_w = [&](size_t bytes) -> void * {
            void * p = nullptr;
            V_HIP_CHECK(hipMalloc(&p, bytes));
            V_HIP_CHECK(hipMemsetAsync(p, 0, bytes, stream_));   // never read uninitialised VRAM (issue #1)
            weight_allocs_.push_back(p);
            weight_vram_bytes_ += bytes;
            return p;
        };

        auto upload_f32 = [&](const std::string & name) -> float * {
            const auto & ti = get_tensor(name);
            const size_t bytes = ti.n_bytes();
            float * d = static_cast<float *>(alloc_w(bytes));
            V_HIP_CHECK(hipMemcpyAsync(d, data_ptr + ti.offset, bytes, hipMemcpyHostToDevice, stream_));
            return d;
        };

        // Reusable staging buffer for Q8_0 repacking (max tensor is mm.1.weight: 4096*4096/32*34 = 17.8 MB)
        size_t max_q8_bytes = 0;
        for (const auto & kv : tmap) {
            if (kv.second.type == GGML_T_Q8_0) {
                max_q8_bytes = std::max(max_q8_bytes, kv.second.n_bytes());
            }
        }
        void * d_stage = nullptr;
        V_HIP_CHECK(hipMalloc(&d_stage, max_q8_bytes));

        auto load_q8_soa = [&](const std::string & name) -> SoaQ8Matrix {
            const auto & ti = get_tensor(name);
            const int K    = (int)ti.dims[0];
            const int rows = (int)ti.dims[1];
            SoaQ8Matrix m;
            m.rows = rows;
            m.K    = K;
            m.qs   = static_cast<int8_t *>(alloc_w((size_t)rows * K));
            m.d    = static_cast<__half *>(alloc_w((size_t)rows * (K / 32) * sizeof(__half)));
            V_HIP_CHECK(hipMemcpyAsync(d_stage, data_ptr + ti.offset, ti.n_bytes(), hipMemcpyHostToDevice, stream_));
            detail::repack_q8_0_to_soa(d_stage, m.qs, m.d, rows, K, 0, stream_);
            return m;
        };

        // Precompute CLS token = class_embd + class_position_embd on host and upload
        {
            const auto & t_cls  = get_tensor("v.class_embd");
            const auto & t_cpos = get_tensor("v.class_position_embd.weight");
            const float * p_cls  = reinterpret_cast<const float *>(data_ptr + t_cls.offset);
            const float * p_cpos = reinterpret_cast<const float *>(data_ptr + t_cpos.offset);
            std::vector<float> cls_sum(hp_.n_embd);
            for (int i = 0; i < hp_.n_embd; ++i) cls_sum[i] = p_cls[i] + p_cpos[i];
            d_cls_tok_ = static_cast<float *>(alloc_w((size_t)hp_.n_embd * sizeof(float)));
            V_HIP_CHECK(hipMemcpyAsync(d_cls_tok_, cls_sum.data(), cls_sum.size() * sizeof(float),
                                       hipMemcpyHostToDevice, stream_));
        }

        d_patch_embd_w_ = upload_f32("v.patch_embd.weight");
        d_pos_embd_w_   = upload_f32("v.position_embd.weight");
        d_pre_ln_w_     = upload_f32("v.pre_ln.weight");
        d_pre_ln_b_     = upload_f32("v.pre_ln.bias");

        layers_.resize(hp_.n_layer);
        for (int il = 0; il < hp_.n_layer; ++il) {
            auto & L = layers_[il];
            const std::string pfx = "v.blk." + std::to_string(il) + ".";

            L.ln1_w = upload_f32(pfx + "ln1.weight");
            L.ln1_b = upload_f32(pfx + "ln1.bias");

            // Fuse Q, K, V into one contiguous [3*D, D] SoA Q8_0 matrix for 1 single WMMA call!
            const int D = hp_.n_embd;
            L.qkv_w.rows = 3 * D;
            L.qkv_w.K    = D;
            L.qkv_w.qs   = static_cast<int8_t *>(alloc_w((size_t)(3 * D) * D));
            L.qkv_w.d    = static_cast<__half *>(alloc_w((size_t)(3 * D) * (D / 32) * sizeof(__half)));

            const char * qkv_names[3] = { "attn_q.weight", "attn_k.weight", "attn_v.weight" };
            const char * qkv_bnames[3] = { "attn_q.bias", "attn_k.bias", "attn_v.bias" };
            L.qkv_b = static_cast<float *>(alloc_w((size_t)(3 * D) * sizeof(float)));
            for (int j = 0; j < 3; ++j) {
                const auto & tw = get_tensor(pfx + qkv_names[j]);
                V_HIP_CHECK(hipMemcpyAsync(d_stage, data_ptr + tw.offset, tw.n_bytes(), hipMemcpyHostToDevice, stream_));
                detail::repack_q8_0_to_soa(d_stage, L.qkv_w.qs, L.qkv_w.d, D, D, j * D, stream_);

                const auto & tb = get_tensor(pfx + qkv_bnames[j]);
                V_HIP_CHECK(hipMemcpyAsync(L.qkv_b + j * D, data_ptr + tb.offset, tb.n_bytes(),
                                           hipMemcpyHostToDevice, stream_));
            }

            L.attn_out_w = load_q8_soa(pfx + "attn_out.weight");
            L.attn_out_b = upload_f32(pfx + "attn_out.bias");
            L.ls1_w      = upload_f32(pfx + "ls1.weight");

            L.ln2_w      = upload_f32(pfx + "ln2.weight");
            L.ln2_b      = upload_f32(pfx + "ln2.bias");
            L.ffn_up_w   = load_q8_soa(pfx + "ffn_up.weight");
            L.ffn_up_b   = upload_f32(pfx + "ffn_up.bias");
            L.ffn_down_w = load_q8_soa(pfx + "ffn_down.weight");
            L.ffn_down_b = upload_f32(pfx + "ffn_down.bias");
            L.ls2_w      = upload_f32(pfx + "ls2.weight");
        }

        mm1_w_ = load_q8_soa("mm.1.weight");
        mm2_w_ = load_q8_soa("mm.2.weight");

        V_HIP_CHECK(hipStreamSynchronize(stream_));
        V_HIP_CHECK(hipFree(d_stage));
        total_vram_bytes_ = weight_vram_bytes_;

        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        std::printf("[vision] loaded %s in %.1f ms | %d layers, D=%d, H=%d, patch=%d, merge=%d | VRAM=%.1f MiB\n",
                    mmproj_path.c_str(), ms, hp_.n_layer, hp_.n_embd, hp_.n_head, hp_.patch_size, hp_.n_merge,
                    (double)weight_vram_bytes_ / (1024.0 * 1024.0));
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[vision] init error: %s\n", e.what());
        unmap_guard();
        destroy();
        return false;
    }

    unmap_guard();
    return true;
}

namespace {

static inline size_t v_align256(size_t b) { return (b + 255) & ~(size_t)255; }

// Byte layout of the single aliased ViT workspace arena (see vision_encoder.hpp).
// Liveness (single in-order stream, every kernel writes its outputs for all
// n_pos rows before any reader runs; no kernel assumes zero-init):
//   hidden, norm : live for the whole encode          -> dedicated regions
//   img          : patch_embed only                   -> pool phase A
//   qkv/q/k/v    : qkv GEMM -> split -> FA            -> pool phase B
//   attn_out     : FA -> attn_out GEMM; qkv is dead after the split and FA only
//                  reads q/k/v, so attn_out aliases qkv
//   ffn_mid      : ffn_up -> gelu -> ffn_down          -> pool phase C
//   merged/mid   : patch_merge -> mm1 -> gelu -> mm2   -> pool phase D
struct VisionArenaLayout {
    size_t off_hidden = 0, off_norm = 0, off_pool = 0;
    size_t off_q = 0, off_k = 0, off_v = 0, off_proj_mid = 0;
    size_t total = 0;
};

static VisionArenaLayout vision_arena_layout(size_t n_pos, size_t pixels, size_t n_merged,
                                             size_t D, size_t FF, size_t MD) {
    VisionArenaLayout L;
    const size_t act  = v_align256(n_pos * D * sizeof(float));
    L.off_hidden = 0;
    L.off_norm   = act;
    L.off_pool   = 2 * act;

    const size_t img_b  = v_align256(pixels * 3 * sizeof(float));
    const size_t qkv_b  = v_align256(n_pos * 3 * D * sizeof(float));
    const size_t head_b = v_align256(n_pos * D * sizeof(__half));
    const size_t attn_b = qkv_b + 3 * head_b;                 // attn_out lives inside qkv_b
    const size_t ffn_b  = v_align256(n_pos * FF * sizeof(float));
    const size_t mrg_b  = v_align256(n_merged * MD * sizeof(float));
    const size_t proj_b = 2 * mrg_b;                          // merged | proj_mid

    L.off_q        = qkv_b;
    L.off_k        = qkv_b + head_b;
    L.off_v        = qkv_b + 2 * head_b;
    L.off_proj_mid = mrg_b;

    const size_t pool = std::max(std::max(img_b, attn_b), std::max(ffn_b, proj_b));
    L.total = L.off_pool + pool;
    return L;
}

} // namespace

size_t VisionEncoder::estimate_encode_bytes(int n_merged) const {
    const size_t nm     = (size_t)std::max(1, n_merged);
    const size_t n_pos  = nm * hp_.n_merge * hp_.n_merge + 1;
    const size_t pixels = (n_pos - 1) * hp_.patch_size * hp_.patch_size;
    const size_t MD     = (size_t)hp_.n_merge * hp_.n_merge * hp_.n_embd;
    const VisionArenaLayout L = vision_arena_layout(n_pos, pixels, nm, hp_.n_embd, hp_.n_ff, MD);
    return L.total + v_align256(nm * hp_.proj_dim * sizeof(float));
}

void VisionEncoder::release_workspace() {
    if (d_arena_) (void)hipFree(d_arena_);   // hipFree synchronises the device first
    d_arena_            = nullptr;
    arena_bytes_        = 0;
    scratch_max_pos_    = 0;
    scratch_max_pixels_ = 0;
    scratch_max_merged_ = 0;
    d_img_f32_ = d_hidden_ = d_norm_ = d_qkv_ = d_attn_out_ = d_ffn_mid_ = d_merged_ = d_proj_mid_ = nullptr;
    d_q_h_ = d_k_h_ = d_v_h_ = nullptr;
    scratch_vram_bytes_ = proj_out_bytes_;
    total_vram_bytes_   = weight_vram_bytes_ + scratch_vram_bytes_;
}

void VisionEncoder::set_keep_scratch(bool keep) {
    keep_scratch_ = keep;
    if (!keep) release_workspace();
}

bool VisionEncoder::ensure_output(int n_merged) {
    if (n_merged <= proj_out_cap_ && d_proj_out_ != nullptr) return true;
    if (d_proj_out_) (void)hipFree(d_proj_out_);
    d_proj_out_     = nullptr;
    proj_out_bytes_ = 0;
    proj_out_cap_   = 0;
    const size_t bytes = v_align256((size_t)n_merged * hp_.proj_dim * sizeof(float));
    void * p = nullptr;
    if (hipMalloc(&p, bytes) != hipSuccess) {
        (void)hipGetLastError();
        std::fprintf(stderr, "[vision] hipMalloc(embeds output, %.1f MiB) failed\n",
                     (double)bytes / (1024.0 * 1024.0));
        scratch_vram_bytes_ = arena_bytes_;
        total_vram_bytes_   = weight_vram_bytes_ + scratch_vram_bytes_;
        return false;
    }
    V_HIP_CHECK(hipMemsetAsync(p, 0, bytes, stream_));   // never read uninitialised VRAM (issue #1)
    d_proj_out_     = static_cast<float *>(p);
    proj_out_bytes_ = bytes;
    proj_out_cap_   = n_merged;
    scratch_vram_bytes_ = arena_bytes_ + proj_out_bytes_;
    total_vram_bytes_   = weight_vram_bytes_ + scratch_vram_bytes_;
    return true;
}

bool VisionEncoder::ensure_scratch(int n_patches, int n_merged, int w_pad, int h_pad) {
    const int n_pos  = n_patches + 1;
    const int pixels = w_pad * h_pad;
    if (!ensure_output(n_merged)) return false;
    if (d_arena_ != nullptr && n_pos <= scratch_max_pos_ && pixels <= scratch_max_pixels_ &&
        n_merged <= scratch_max_merged_) {
        last_arena_bytes_ = arena_bytes_;
        return true;
    }

    // Grow (keep mode) or size exactly for this image (release mode: caps are 0 here).
    if (d_arena_) (void)hipFree(d_arena_);
    d_arena_     = nullptr;
    arena_bytes_ = 0;
    scratch_max_pos_    = std::max(scratch_max_pos_, n_pos);
    scratch_max_pixels_ = std::max(scratch_max_pixels_, pixels);
    scratch_max_merged_ = std::max(scratch_max_merged_, n_merged);

    const size_t MD = (size_t)hp_.n_merge * hp_.n_merge * hp_.n_embd;
    const VisionArenaLayout L = vision_arena_layout(
        (size_t)scratch_max_pos_, (size_t)scratch_max_pixels_, (size_t)scratch_max_merged_,
        (size_t)hp_.n_embd, (size_t)hp_.n_ff, MD);

    void * p = nullptr;
    if (hipMalloc(&p, L.total) != hipSuccess) {
        (void)hipGetLastError();
        size_t fr = 0, tot = 0;
        (void)hipMemGetInfo(&fr, &tot);
        std::fprintf(stderr, "[vision] hipMalloc(ViT workspace, %.1f MiB) failed (%.1f MiB free)\n",
                     (double)L.total / (1024.0 * 1024.0), (double)fr / (1024.0 * 1024.0));
        scratch_max_pos_ = scratch_max_pixels_ = scratch_max_merged_ = 0;
        scratch_vram_bytes_ = proj_out_bytes_;
        total_vram_bytes_   = weight_vram_bytes_ + scratch_vram_bytes_;
        return false;
    }
    V_HIP_CHECK(hipMemsetAsync(p, 0, L.total, stream_));   // never read uninitialised VRAM (issue #1)
    d_arena_     = p;
    arena_bytes_ = L.total;

    char * base = static_cast<char *>(p);
    char * pool = base + L.off_pool;
    d_hidden_   = reinterpret_cast<float *>(base + L.off_hidden);
    d_norm_     = reinterpret_cast<float *>(base + L.off_norm);
    d_img_f32_  = reinterpret_cast<float *>(pool);
    d_qkv_      = reinterpret_cast<float *>(pool);
    d_attn_out_ = reinterpret_cast<float *>(pool);             // aliases d_qkv_ (dead after split)
    d_q_h_      = reinterpret_cast<__half *>(pool + L.off_q);
    d_k_h_      = reinterpret_cast<__half *>(pool + L.off_k);
    d_v_h_      = reinterpret_cast<__half *>(pool + L.off_v);
    d_ffn_mid_  = reinterpret_cast<float *>(pool);
    d_merged_   = reinterpret_cast<float *>(pool);
    d_proj_mid_ = reinterpret_cast<float *>(pool + L.off_proj_mid);

    last_arena_bytes_   = arena_bytes_;
    peak_arena_bytes_   = std::max(peak_arena_bytes_, arena_bytes_);
    scratch_vram_bytes_ = arena_bytes_ + proj_out_bytes_;
    total_vram_bytes_   = weight_vram_bytes_ + scratch_vram_bytes_;
    return true;
}

namespace {

static bool is_webp_bytes(const uint8_t * data, size_t len) {
    return data != nullptr && len >= 12 &&
           std::memcmp(data, "RIFF", 4) == 0 &&
           std::memcmp(data + 8, "WEBP", 4) == 0;
}

static bool decode_webp_rgb(const uint8_t * data, size_t len,
                            std::vector<uint8_t> & out_rgb, int & out_w, int & out_h) {
    using WebPDecodeRGBFn = uint8_t * (*)(const uint8_t *, size_t, int *, int *);
    using WebPFreeFn      = void (*)(void *);
    static WebPDecodeRGBFn fn_decode = nullptr;
    static WebPFreeFn      fn_free   = nullptr;
    static bool            tried     = false;
    static std::mutex      mu;
    {
        std::lock_guard<std::mutex> lk(mu);
        if (!tried) {
            tried = true;
            const char * names[] = { "libwebp.so.7", "libwebp.so.6", "libwebp.so" };
            for (const char * nm : names) {
                void * h = dlopen(nm, RTLD_LAZY | RTLD_LOCAL);
                if (!h) continue;
                auto dec = reinterpret_cast<WebPDecodeRGBFn>(dlsym(h, "WebPDecodeRGB"));
                auto fr  = reinterpret_cast<WebPFreeFn>(dlsym(h, "WebPFree"));
                if (dec && fr) {
                    fn_decode = dec;
                    fn_free   = fr;
                    break;
                }
                dlclose(h);
            }
        }
    }
    if (!fn_decode || !fn_free) {
        std::fprintf(stderr, "[vision] WebP image received, but libwebp runtime was not found\n");
        return false;
    }
    int w = 0, h = 0;
    uint8_t * rgb = fn_decode(data, len, &w, &h);
    if (!rgb || w <= 0 || h <= 0) {
        if (rgb) fn_free(rgb);
        std::fprintf(stderr, "[vision] WebPDecodeRGB failed\n");
        return false;
    }
    out_w = w;
    out_h = h;
    out_rgb.assign(rgb, rgb + (size_t)w * (size_t)h * 3);
    fn_free(rgb);
    return true;
}

} // namespace

bool VisionEncoder::encode_image_file(
    const std::string & image_path,
    VisionEncodeResult & out_res,
    std::vector<float> * out_host_embeds)
{
    std::ifstream ifs(image_path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        std::fprintf(stderr, "[vision] failed to open image file '%s'\n", image_path.c_str());
        return false;
    }
    const std::streamsize fsz = ifs.tellg();
    if (fsz <= 0) {
        std::fprintf(stderr, "[vision] image file '%s' is empty\n", image_path.c_str());
        return false;
    }
    std::vector<uint8_t> bytes((size_t)fsz);
    ifs.seekg(0, std::ios::beg);
    if (!ifs.read(reinterpret_cast<char *>(bytes.data()), fsz)) {
        std::fprintf(stderr, "[vision] failed to read image file '%s'\n", image_path.c_str());
        return false;
    }
    return encode_image_bytes(bytes.data(), bytes.size(), out_res, out_host_embeds);
}

bool VisionEncoder::encode_image_bytes(
    const uint8_t * data, size_t len,
    VisionEncodeResult & out_res,
    std::vector<float> * out_host_embeds)
{
    if (!data || len == 0) return false;
    const auto t0 = std::chrono::steady_clock::now();
    int w = 0, h = 0, ch = 0;
    bool ok = false;
    if (is_webp_bytes(data, len)) {
        std::vector<uint8_t> webp_rgb;
        if (!decode_webp_rgb(data, len, webp_rgb, w, h)) {
            return false;
        }
        ok = encode_rgb_u8(webp_rgb.data(), w, h, out_res, out_host_embeds);
    } else {
        uint8_t * rgb = stbi_load_from_memory(data, (int)len, &w, &h, &ch, 3);
        if (!rgb) {
            std::fprintf(stderr, "[vision] stbi_load_from_memory failed: %s\n",
                         stbi_failure_reason());
            return false;
        }
        ok = encode_rgb_u8(rgb, w, h, out_res, out_host_embeds);
        stbi_image_free(rgb);
    }
    if (ok) {
        const double total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        out_res.preproc_ms = std::max(0.0, total_ms - out_res.gpu_ms);
    }
    return ok;
}

bool VisionEncoder::encode_rgb_u8(
    const uint8_t * rgb, int width, int height,
    VisionEncodeResult & out_res,
    std::vector<float> * out_host_embeds)
{
    const auto t_pre0 = std::chrono::steady_clock::now();
    const int align = hp_.patch_size * hp_.n_merge; // 40
    const int64_t max_pixels = (int64_t)hp_.max_tokens * align * align;

    int cur_w = width;
    int cur_h = height;
    std::vector<uint8_t> resized_buf;
    const uint8_t * src_rgb = rgb;

    if ((int64_t)align_up(cur_w, align) * align_up(cur_h, align) > max_pixels) {
        const double scale = std::sqrt((double)max_pixels / ((double)cur_w * cur_h));
        cur_w = std::max(align, (int)std::floor(cur_w * scale / align) * align);
        cur_h = std::max(align, (int)std::floor(cur_h * scale / align) * align);
        resize_rgb_bilinear(rgb, width, height, resized_buf, cur_w, cur_h);
        src_rgb = resized_buf.data();
    }

    const int w_pad     = align_up(cur_w, align);
    const int h_pad     = align_up(cur_h, align);
    const int grid_w    = w_pad / hp_.patch_size;
    const int grid_h    = h_pad / hp_.patch_size;
    const int n_patches = grid_w * grid_h;
    const int n_pos     = n_patches + 1;
    const int n_merged  = (grid_w / hp_.n_merge) * (grid_h / hp_.n_merge);

    // Exact mtmd_image_preprocessor_inkling2v: pad right/bottom with -1.0f, then (pix - mean*255) / (std*255)
    const float mean255[3] = { hp_.image_mean[0] * 255.0f, hp_.image_mean[1] * 255.0f, hp_.image_mean[2] * 255.0f };
    const float inv_std[3] = { 1.0f / (hp_.image_std[0] * 255.0f),
                               1.0f / (hp_.image_std[1] * 255.0f),
                               1.0f / (hp_.image_std[2] * 255.0f) };
    const float pad_val[3] = { (-1.0f - mean255[0]) * inv_std[0],
                               (-1.0f - mean255[1]) * inv_std[1],
                               (-1.0f - mean255[2]) * inv_std[2] };

    std::vector<float> h_img((size_t)w_pad * h_pad * 3);
    for (int y = 0; y < h_pad; ++y) {
        float * dst_row = h_img.data() + (size_t)y * w_pad * 3;
        if (y < cur_h) {
            const uint8_t * srow = src_rgb + (size_t)y * cur_w * 3;
            for (int x = 0; x < cur_w; ++x) {
                dst_row[x * 3 + 0] = ((float)srow[x * 3 + 0] - mean255[0]) * inv_std[0];
                dst_row[x * 3 + 1] = ((float)srow[x * 3 + 1] - mean255[1]) * inv_std[1];
                dst_row[x * 3 + 2] = ((float)srow[x * 3 + 2] - mean255[2]) * inv_std[2];
            }
            for (int x = cur_w; x < w_pad; ++x) {
                dst_row[x * 3 + 0] = pad_val[0];
                dst_row[x * 3 + 1] = pad_val[1];
                dst_row[x * 3 + 2] = pad_val[2];
            }
        } else {
            for (int x = 0; x < w_pad; ++x) {
                dst_row[x * 3 + 0] = pad_val[0];
                dst_row[x * 3 + 1] = pad_val[1];
                dst_row[x * 3 + 2] = pad_val[2];
            }
        }
    }

    if (!ensure_scratch(n_patches, n_merged, w_pad, h_pad)) {
        return false;
    }

    const double pre_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_pre0).count();

    // Upload normalized image & time pure GPU forward pass with HIP events
    V_HIP_CHECK(hipMemcpyAsync(d_img_f32_, h_img.data(), h_img.size() * sizeof(float),
                               hipMemcpyHostToDevice, stream_));

    hipEvent_t ev0, ev1;
    V_HIP_CHECK(hipEventCreate(&ev0));
    V_HIP_CHECK(hipEventCreate(&ev1));
    V_HIP_CHECK(hipEventRecord(ev0, stream_));

    const int D        = hp_.n_embd;
    const int pos_grid = hp_.image_size / hp_.patch_size; // 32
    const float kq_scale = 1.0f / std::sqrt((float)hp_.head_dim);

    // 1. Patch Conv2D + Bilinear Pos Embed + Prepend CLS -> d_norm_ (temp), then pre_ln -> d_hidden_
    launch_vision_patch_embed_pos_cls(
        d_img_f32_, d_patch_embd_w_, d_pos_embd_w_, d_cls_tok_, d_norm_,
        grid_w, grid_h, hp_.patch_size, pos_grid, stream_);
    launch_vision_layernorm(d_norm_, d_pre_ln_w_, d_pre_ln_b_, d_hidden_, n_pos, D, hp_.ln_eps, stream_);

    // 2. 23 ViT Blocks
    for (int il = 0; il < hp_.n_layer; ++il) {
        const auto & L = layers_[il];

        // Attention sub-block
        launch_vision_layernorm(d_hidden_, L.ln1_w, L.ln1_b, d_norm_, n_pos, D, hp_.ln_eps, stream_);
        launch_soa_q8_gemm_wmma(L.qkv_w, d_norm_, d_qkv_, n_pos, stream_);
        launch_vision_qkv_bias_rope2d_split(
            d_qkv_, L.qkv_b, d_q_h_, d_k_h_, d_v_h_, n_pos, grid_w, hp_.rope_theta, stream_);
        launch_vision_fa_wmma_d64(
            d_q_h_, d_k_h_, d_v_h_, d_attn_out_, n_pos, hp_.n_head, kq_scale, stream_);
        launch_soa_q8_gemm_wmma(L.attn_out_w, d_attn_out_, d_norm_, n_pos, stream_);
        launch_vision_bias_ls_res(d_norm_, L.attn_out_b, L.ls1_w, d_hidden_, n_pos, D, stream_);

        // FFN sub-block
        launch_vision_layernorm(d_hidden_, L.ln2_w, L.ln2_b, d_norm_, n_pos, D, hp_.ln_eps, stream_);
        launch_soa_q8_gemm_wmma(L.ffn_up_w, d_norm_, d_ffn_mid_, n_pos, stream_);
        launch_vision_bias_gelu_erf(d_ffn_mid_, L.ffn_up_b, n_pos, hp_.n_ff, stream_);
        launch_soa_q8_gemm_wmma(L.ffn_down_w, d_ffn_mid_, d_norm_, n_pos, stream_);
        launch_vision_bias_ls_res(d_norm_, L.ffn_down_b, L.ls2_w, d_hidden_, n_pos, D, stream_);
    }

    // 3. Drop CLS + 2x2 Spatial Merge -> [n_merged, 4096]
    launch_vision_patch_merge(d_hidden_, d_merged_, grid_w, grid_h, D, hp_.n_merge, stream_);

    // 4. 2-Layer MLP Projector: mm.1 (4096->4096) -> GELU_ERF -> mm.2 (4096->2304)
    launch_soa_q8_gemm_wmma(mm1_w_, d_merged_, d_proj_mid_, n_merged, stream_);
    launch_vision_bias_gelu_erf(d_proj_mid_, nullptr, n_merged, hp_.n_merge * hp_.n_merge * D, stream_);
    launch_soa_q8_gemm_wmma(mm2_w_, d_proj_mid_, d_proj_out_, n_merged, stream_);

    V_HIP_CHECK(hipEventRecord(ev1, stream_));
    V_HIP_CHECK(hipEventSynchronize(ev1));

    float gpu_ms = 0.0f;
    V_HIP_CHECK(hipEventElapsedTime(&gpu_ms, ev0, ev1));
    V_HIP_CHECK(hipEventDestroy(ev0));
    V_HIP_CHECK(hipEventDestroy(ev1));

    // The transient arena is dead now (ev1 covers every kernel that touches it);
    // only d_proj_out_ (separate allocation) is read after this point.
    if (!keep_scratch_) {
        release_workspace();
    }

    if (out_host_embeds != nullptr) {
        out_host_embeds->resize((size_t)n_merged * hp_.proj_dim);
        V_HIP_CHECK(hipMemcpy(out_host_embeds->data(), d_proj_out_,
                              out_host_embeds->size() * sizeof(float), hipMemcpyDeviceToHost));
    }

    out_res.n_tokens   = n_merged;
    out_res.grid_w     = grid_w;
    out_res.grid_h     = grid_h;
    out_res.img_w      = cur_w;
    out_res.img_h      = cur_h;
    out_res.preproc_ms = pre_ms;
    out_res.gpu_ms     = (double)gpu_ms;
    out_res.d_embeds   = d_proj_out_;
    return true;
}

} // namespace inkling
