// ============================================================================
// gguf_loader.cpp -- Fast mmap GGUF v3 loader + GPU SoA repacker driver
// ============================================================================
//
// Parses Inkling-2-Mini-NVFP4-Mixed.gguf (GGUF v3) directly out of a read-only
// mmap, stages each quantised tensor into a single reusable device buffer and
// repacks it into the 16-byte-aligned SoA layouts from include/inkling_engine.h
// via the gfx1100 kernels in src/repack_soa.hip.
//
// Only the two source files gguf_loader.cpp + repack_soa.hip define this
// subsystem; the shared detail:: declarations live below and are defined in
// repack_soa.hip.
// ============================================================================

#include "inkling_engine.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace inkling {

// ---------------------------------------------------------------------------
// Prototypes for the gfx1100 repack kernels / self-test (defined in repack_soa.hip)
// ---------------------------------------------------------------------------
namespace detail {

void repack_q8_0_to_soa(
    const void * d_raw, int8_t * d_qs, __half * d_d,
    int rows, int K, int dst_row_offset, hipStream_t stream);

void repack_nvfp4_to_soa(
    const void * d_raw, uint8_t * d_qs, uint8_t * d_scales,
    int n_exp, int rows, int K, hipStream_t stream);

void convert_f32_to_f16(
    const float * d_src, __half * d_dst, size_t n, hipStream_t stream);

float verify_q8_0_repack(
    const void * raw_host, const int8_t * d_qs, const __half * d_d,
    int rows, int K);

float verify_nvfp4_repack(
    const void * raw_host, const uint8_t * d_qs, const uint8_t * d_scales,
    const float * d_scale2, int n_exp, int rows, int K);

} // namespace detail

// ============================================================================
// Destroy
// ============================================================================
void ModelWeights::destroy()
{
    for (void * p : gpu_allocs) {
        if (p != nullptr) {
            (void)hipFree(p);
        }
    }
    gpu_allocs.clear();
    total_vram_bytes = 0;
    for (void * h : host_allocs) {
        if (h != nullptr) {
            (void)hipHostFree(h);
        }
    }
    host_allocs.clear();
    host_pinned_bytes = 0;
    tok_embd_on_host  = false;
}

// ============================================================================
// GGUF parsing internals
// ============================================================================
namespace {

// GGUF metadata value types.
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

// GGML tensor types we consume.
enum : uint32_t {
    GGML_T_F32   = 0,
    GGML_T_F16   = 1,
    GGML_T_Q8_0  = 8,
    GGML_T_NVFP4 = 40,
};

struct Reader {
    const uint8_t * p;
    const uint8_t * end;

    void need(size_t n) const {
        if (static_cast<size_t>(end - p) < n) {
            throw std::runtime_error("gguf: truncated file");
        }
    }
    uint8_t u8()   { need(1); return *p++; }
    uint16_t u16() { need(2); uint16_t v; std::memcpy(&v, p, 2); p += 2; return v; }
    uint32_t u32() { need(4); uint32_t v; std::memcpy(&v, p, 4); p += 4; return v; }
    uint64_t u64() { need(8); uint64_t v; std::memcpy(&v, p, 8); p += 8; return v; }
    std::string str() {
        uint64_t n = u64();
        need(static_cast<size_t>(n));
        std::string s(reinterpret_cast<const char *>(p), static_cast<size_t>(n));
        p += n;
        return s;
    }
    void skip(size_t n) { need(n); p += n; }
};

int64_t read_int_any(Reader & r, uint32_t t)
{
    switch (t) {
        case GGUF_T_UINT8:  return static_cast<int64_t>(r.u8());
        case GGUF_T_INT8:   return static_cast<int64_t>(static_cast<int8_t>(r.u8()));
        case GGUF_T_UINT16: return static_cast<int64_t>(r.u16());
        case GGUF_T_INT16:  return static_cast<int64_t>(static_cast<int16_t>(r.u16()));
        case GGUF_T_UINT32: return static_cast<int64_t>(r.u32());
        case GGUF_T_INT32:  return static_cast<int64_t>(static_cast<int32_t>(r.u32()));
        case GGUF_T_UINT64: return static_cast<int64_t>(r.u64());
        case GGUF_T_INT64:  return static_cast<int64_t>(r.u64());
        default:
            throw std::runtime_error("gguf: unexpected integer scalar type");
    }
}

float read_f32(Reader & r)
{
    uint32_t bits = r.u32();
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

void skip_value(Reader & r, uint32_t t)
{
    switch (t) {
        case GGUF_T_UINT8:
        case GGUF_T_INT8:
        case GGUF_T_BOOL:
            r.skip(1);
            break;
        case GGUF_T_UINT16:
        case GGUF_T_INT16:
            r.skip(2);
            break;
        case GGUF_T_UINT32:
        case GGUF_T_INT32:
        case GGUF_T_FLOAT32:
            r.skip(4);
            break;
        case GGUF_T_UINT64:
        case GGUF_T_INT64:
        case GGUF_T_FLOAT64:
            r.skip(8);
            break;
        case GGUF_T_STRING:
            r.str();
            break;
        case GGUF_T_ARRAY: {
            uint32_t et = r.u32();
            uint64_t n  = r.u64();
            if (et == GGUF_T_STRING) {
                for (uint64_t i = 0; i < n; ++i) {
                    r.str();
                }
            } else {
                static const size_t sz[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
                if (et >= 13 || sz[et] == 0) {
                    throw std::runtime_error("gguf: unsupported array element type");
                }
                r.skip(static_cast<size_t>(n) * sz[et]);
            }
            break;
        }
        default:
            throw std::runtime_error("gguf: unsupported metadata value type");
    }
}

struct TensorInfo {
    std::string           name;
    std::vector<uint64_t> dims;   // ne[0..ndim-1]  (ne0 == K / innermost)
    uint32_t              type   = 0;
    uint64_t              offset = 0;   // relative to tensor-data section
    size_t                nbytes = 0;   // raw GGML bytes
    const uint8_t *       data   = nullptr; // resolved host pointer
};

using TensorMap = std::unordered_map<std::string, TensorInfo>;

size_t tensor_raw_bytes(const std::vector<uint64_t> & dims, uint32_t type)
{
    size_t n = 1;
    for (uint64_t d : dims) {
        n *= static_cast<size_t>(d);
    }
    switch (type) {
        case GGML_T_F32:   return n * 4;
        case GGML_T_F16:   return n * 2;
        case GGML_T_Q8_0:  return (n / 32) * 34;
        case GGML_T_NVFP4: return (n / 64) * 36;
        default:
            throw std::runtime_error("gguf: unsupported tensor ggml type " + std::to_string(type));
    }
}

// GCC 15 false-positives this accessor under -Wdangling-reference: it returns a
// reference into the map, while the lookup key is often a temporary string. The
// returned element outlives every call, so mark the attribute to state that.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 14
#  define INKLING_REQUIRE_ATTR [[gnu::no_dangling]]
#else
#  define INKLING_REQUIRE_ATTR
#endif

INKLING_REQUIRE_ATTR const TensorInfo & require(const TensorMap & m, const std::string & name)
{
    auto it = m.find(name);
    if (it == m.end()) {
        throw std::runtime_error("missing required tensor: " + name);
    }
    return it->second;
}

void expect_type(const TensorInfo & t, uint32_t ty)
{
    if (t.type != ty) {
        throw std::runtime_error(
            "tensor " + t.name + " has ggml type " + std::to_string(t.type) +
            ", expected " + std::to_string(ty));
    }
}

int K_of(const TensorInfo & t)
{
    if (t.dims.size() < 2) {
        throw std::runtime_error("tensor " + t.name + " is not >= 2D");
    }
    return static_cast<int>(t.dims[0]);
}

int rows_of(const TensorInfo & t)
{
    if (t.dims.size() < 2) {
        throw std::runtime_error("tensor " + t.name + " is not 2D");
    }
    return static_cast<int>(t.dims[1]);
}

int nexp_of(const TensorInfo & t)
{
    if (t.dims.size() != 3) {
        throw std::runtime_error("tensor " + t.name + " is not 3D");
    }
    return static_cast<int>(t.dims[2]);
}

size_t nelem_of(const TensorInfo & t)
{
    size_t n = 1;
    for (uint64_t d : t.dims) {
        n *= static_cast<size_t>(d);
    }
    return n;
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
// Stream the loader works on; alloc_tracked zero-fills on it (stream-ordered
// before every upload / repack that writes the allocation).
static hipStream_t g_alloc_zero_stream = nullptr;

void * alloc_tracked(ModelWeights & w, size_t bytes)
{
    if (bytes == 0) {
        return nullptr;
    }
    void * p = nullptr;
    hipError_t err = hipMalloc(&p, bytes);
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("hipMalloc failed: ") + hipGetErrorString(err) +
                                 " (" + std::to_string(bytes) + " bytes)");
    }
    // Zero the whole allocation (incl. any SoA padding the repack never
    // writes): hipMalloc'd VRAM is not guaranteed to be clean (issue #1).
    // Ordered on the loader stream; never the legacy null stream (an extra HW
    // queue for the rest of the process costs ~3.5% prefill throughput).
    if (g_alloc_zero_stream == nullptr) {
        (void)hipFree(p);
        throw std::runtime_error("alloc_tracked: loader stream not set");
    }
    err = hipMemsetAsync(p, 0, bytes, g_alloc_zero_stream);
    if (err != hipSuccess) {
        (void)hipFree(p);
        throw std::runtime_error(std::string("hipMemset failed: ") + hipGetErrorString(err));
    }
    w.gpu_allocs.push_back(p);
    w.total_vram_bytes += bytes;
    return p;
}

// --cpu-embd: pinned host RAM mapped into the GPU address space.  Returns the
// device-side alias; the host pointer is tracked in w.host_allocs (hipHostFree).
void * alloc_host_mapped(ModelWeights & w, size_t bytes)
{
    if (bytes == 0) {
        return nullptr;
    }
    void * h = nullptr;
    hipError_t err = hipHostMalloc(&h, bytes, hipHostMallocMapped);
    if (err != hipSuccess || h == nullptr) {
        throw std::runtime_error(std::string("hipHostMalloc(Mapped) failed: ") + hipGetErrorString(err) +
                                 " (" + std::to_string(bytes) + " bytes)");
    }
    void * d = nullptr;
    err = hipHostGetDevicePointer(&d, h, 0);
    if (err != hipSuccess || d == nullptr) {
        (void)hipHostFree(h);
        throw std::runtime_error(std::string("hipHostGetDevicePointer failed: ") + hipGetErrorString(err));
    }
    w.host_allocs.push_back(h);
    w.host_pinned_bytes += bytes;
    // Same stream-ordered zero fill as alloc_tracked (issue #1), through the
    // device alias; fall back to a host memset if the runtime refuses it.
    err = (g_alloc_zero_stream != nullptr) ? hipMemsetAsync(d, 0, bytes, g_alloc_zero_stream)
                                           : hipErrorInvalidValue;
    if (err != hipSuccess) {
        (void)hipGetLastError();
        if (g_alloc_zero_stream != nullptr) {
            (void)hipStreamSynchronize(g_alloc_zero_stream);
        }
        std::memset(h, 0, bytes);
    }
    return d;
}

SoaQ8Matrix make_q8_matrix_host(ModelWeights & w, int rows, int K)
{
    SoaQ8Matrix m;
    m.rows = rows;
    m.K    = K;
    m.qs   = static_cast<int8_t *>(alloc_host_mapped(w, static_cast<size_t>(rows) * K));
    m.d    = static_cast<__half *>(
        alloc_host_mapped(w, static_cast<size_t>(rows) * (K / 32) * sizeof(__half)));
    return m;
}

SoaQ8Matrix make_q8_matrix(ModelWeights & w, int rows, int K)
{
    SoaQ8Matrix m;
    m.rows = rows;
    m.K    = K;
    m.qs   = static_cast<int8_t *>(alloc_tracked(w, static_cast<size_t>(rows) * K));
    m.d    = static_cast<__half *>(
        alloc_tracked(w, static_cast<size_t>(rows) * (K / 32) * sizeof(__half)));
    return m;
}

SoaQ8Experts make_q8_experts(ModelWeights & w, int n_exp, int rows, int K)
{
    SoaQ8Experts e;
    e.n_exp = n_exp;
    e.rows  = rows;
    e.K     = K;
    e.qs    = static_cast<int8_t *>(
        alloc_tracked(w, static_cast<size_t>(n_exp) * rows * K));
    e.d     = static_cast<__half *>(
        alloc_tracked(w, static_cast<size_t>(n_exp) * rows * (K / 32) * sizeof(__half)));
    return e;
}

SoaNVFP4Experts make_nvfp4_experts(ModelWeights & w, int n_exp, int rows, int K)
{
    SoaNVFP4Experts e;
    e.n_exp  = n_exp;
    e.rows   = rows;
    e.K      = K;
    e.qs     = static_cast<uint8_t *>(
        alloc_tracked(w, static_cast<size_t>(n_exp) * rows * (K / 2)));
    e.scales = static_cast<uint8_t *>(
        alloc_tracked(w, static_cast<size_t>(n_exp) * rows * (K / 16)));
    e.scale2 = static_cast<float *>(
        alloc_tracked(w, static_cast<size_t>(n_exp) * sizeof(float)));
    return e;
}

void stage_and_repack_q8(
    hipStream_t stream, void * staging, size_t staging_cap,
    const TensorInfo & t, int rows, int K,
    int8_t * d_qs, __half * d_d, int dst_row_offset)
{
    expect_type(t, GGML_T_Q8_0);
    const size_t bytes = static_cast<size_t>(rows) * (K / 32) * 34;
    if (bytes > staging_cap) {
        throw std::runtime_error("staging buffer too small for " + t.name);
    }
    hipError_t err = hipMemcpyAsync(staging, t.data, bytes, hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("hipMemcpyAsync(staging) failed: ") + hipGetErrorString(err));
    }
    detail::repack_q8_0_to_soa(staging, d_qs, d_d, rows, K, dst_row_offset, stream);
}

void stage_and_repack_nvfp4(
    hipStream_t stream, void * staging, size_t staging_cap,
    const TensorInfo & t, int n_exp, int rows, int K,
    uint8_t * d_qs, uint8_t * d_scales)
{
    expect_type(t, GGML_T_NVFP4);
    const size_t bytes = static_cast<size_t>(n_exp) * rows * (K / 64) * 36;
    if (bytes > staging_cap) {
        throw std::runtime_error("staging buffer too small for " + t.name);
    }
    hipError_t err = hipMemcpyAsync(staging, t.data, bytes, hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("hipMemcpyAsync(staging) failed: ") + hipGetErrorString(err));
    }
    detail::repack_nvfp4_to_soa(staging, d_qs, d_scales, n_exp, rows, K, stream);
}

float * upload_f32(
    ModelWeights & w, hipStream_t stream, const TensorInfo & t)
{
    expect_type(t, GGML_T_F32);
    const size_t n = nelem_of(t);
    const size_t bytes = n * sizeof(float);
    if (bytes != t.nbytes) {
        throw std::runtime_error("F32 size mismatch for " + t.name);
    }
    float * p = static_cast<float *>(alloc_tracked(w, bytes));
    hipError_t err = hipMemcpyAsync(p, t.data, bytes, hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("hipMemcpyAsync(F32) failed: ") + hipGetErrorString(err));
    }
    return p;
}

// Upload a tensor that lives as FP16 on the GPU. GGML F16 is copied directly;
// GGML F32 is staged into `staging` (size-checked) and converted on device.
__half * upload_f16(
    ModelWeights & w, hipStream_t stream, void * staging, size_t staging_bytes,
    const TensorInfo & t)
{
    const size_t n     = nelem_of(t);
    const size_t bytes = n * sizeof(__half);
    __half * p = static_cast<__half *>(alloc_tracked(w, bytes));

    if (t.type == GGML_T_F16) {
        if (t.nbytes != bytes) {
            throw std::runtime_error("F16 size mismatch for " + t.name);
        }
        hipError_t err = hipMemcpyAsync(p, t.data, bytes, hipMemcpyHostToDevice, stream);
        if (err != hipSuccess) {
            throw std::runtime_error(std::string("hipMemcpyAsync(F16) failed: ") +
                                     hipGetErrorString(err));
        }
    } else if (t.type == GGML_T_F32) {
        const size_t src_bytes = n * sizeof(float);
        if (t.nbytes != src_bytes) {
            throw std::runtime_error("F32 size mismatch for " + t.name);
        }
        if (t.nbytes > staging_bytes) {
            throw std::runtime_error("staging buffer too small for " + t.name);
        }
        hipError_t err = hipMemcpyAsync(staging, t.data, src_bytes,
                                        hipMemcpyHostToDevice, stream);
        if (err != hipSuccess) {
            throw std::runtime_error(std::string("hipMemcpyAsync(F32->staging) failed: ") +
                                     hipGetErrorString(err));
        }
        detail::convert_f32_to_f16(reinterpret_cast<const float *>(staging), p, n, stream);
    } else {
        throw std::runtime_error(
            "upload_f16: tensor " + t.name + " has unsupported ggml type " +
            std::to_string(t.type) + " (expected F16=1 or F32=0)");
    }
    return p;
}

} // namespace

// ============================================================================
// load_inkling_gguf
// ============================================================================
bool load_inkling_gguf(
    const std::string & path,
    ModelWeights & out_weights,
    TokenizerData & out_tok,
    bool verify_repack_self_test,
    bool tok_embd_host)
{
    const auto t_start = std::chrono::steady_clock::now();

    // ---- open + mmap --------------------------------------------------------
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("cannot open model: " + path);
    }
    struct stat st {};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        throw std::runtime_error("cannot stat model: " + path);
    }
    const size_t file_size = static_cast<size_t>(st.st_size);
    void * map = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        ::close(fd);
        throw std::runtime_error("mmap failed for " + path);
    }

    const uint8_t * base = static_cast<const uint8_t *>(map);

    auto munmap_guard = [&]() {
        ::munmap(map, file_size);
        ::close(fd);
    };

    // Declared outside the try so the catch path can release them on failure.
    hipStream_t stream  = nullptr;
    void *      staging = nullptr;

    try {
        Reader r{base, base + file_size};

        // ---- header --------------------------------------------------------
        r.need(4);
        if (std::memcmp(r.p, "GGUF", 4) != 0) {
            throw std::runtime_error("not a GGUF file");
        }
        r.p += 4;
        const uint32_t version = r.u32();
        if (version != 3) {
            throw std::runtime_error("unsupported GGUF version " + std::to_string(version) +
                                     " (need v3)");
        }
        const uint64_t n_tensors = r.u64();
        const uint64_t n_kv      = r.u64();

        uint32_t alignment = 32;

        // ---- metadata KVs --------------------------------------------------
        for (uint64_t i = 0; i < n_kv; ++i) {
            const std::string key = r.str();
            const uint32_t type = r.u32();

            if (key == "tokenizer.ggml.tokens") {
                if (type != GGUF_T_ARRAY) {
                    throw std::runtime_error("tokenizer.ggml.tokens is not an array");
                }
                const uint32_t et = r.u32();
                const uint64_t n  = r.u64();
                if (et != GGUF_T_STRING) {
                    throw std::runtime_error("tokenizer.ggml.tokens is not a string array");
                }
                out_tok.vocab.clear();
                out_tok.vocab.reserve(static_cast<size_t>(n));
                for (uint64_t j = 0; j < n; ++j) {
                    out_tok.vocab.push_back(r.str());
                }
            } else if (key == "tokenizer.ggml.merges") {
                if (type != GGUF_T_ARRAY) {
                    throw std::runtime_error("tokenizer.ggml.merges is not an array");
                }
                const uint32_t et = r.u32();
                const uint64_t n  = r.u64();
                if (et != GGUF_T_STRING) {
                    throw std::runtime_error("tokenizer.ggml.merges is not a string array");
                }
                out_tok.merges.clear();
                out_tok.merges.reserve(static_cast<size_t>(n));
                for (uint64_t j = 0; j < n; ++j) {
                    out_tok.merges.push_back(r.str());
                }
            } else if (key == "tokenizer.ggml.token_type") {
                if (type != GGUF_T_ARRAY) {
                    throw std::runtime_error("tokenizer.ggml.token_type is not an array");
                }
                const uint32_t et = r.u32();
                const uint64_t n  = r.u64();
                out_tok.token_types.clear();
                out_tok.token_types.reserve(static_cast<size_t>(n));
                for (uint64_t j = 0; j < n; ++j) {
                    out_tok.token_types.push_back(static_cast<int32_t>(read_int_any(r, et)));
                }
            } else if (key == "tokenizer.ggml.scores") {
                if (type != GGUF_T_ARRAY) {
                    throw std::runtime_error("tokenizer.ggml.scores is not an array");
                }
                const uint32_t et = r.u32();
                const uint64_t n  = r.u64();
                if (et != GGUF_T_FLOAT32) {
                    throw std::runtime_error("tokenizer.ggml.scores is not a float array");
                }
                out_tok.scores.clear();
                out_tok.scores.reserve(static_cast<size_t>(n));
                for (uint64_t j = 0; j < n; ++j) {
                    out_tok.scores.push_back(read_f32(r));
                }
            } else if (key == "tokenizer.ggml.bos_token_id") {
                out_tok.bos_id = static_cast<int32_t>(read_int_any(r, type));
            } else if (key == "tokenizer.ggml.eos_token_id") {
                out_tok.eos_id = static_cast<int32_t>(read_int_any(r, type));
            } else if (key == "tokenizer.ggml.padding_token_id") {
                out_tok.pad_id = static_cast<int32_t>(read_int_any(r, type));
            } else if (key == "tokenizer.ggml.add_bos_token") {
                if (type != GGUF_T_BOOL && type != GGUF_T_UINT8 && type != GGUF_T_INT8) {
                    throw std::runtime_error("tokenizer.ggml.add_bos_token has bad type");
                }
                out_tok.add_bos = (r.u8() != 0);
            } else if (key == "tokenizer.chat_template") {
                out_tok.chat_template = r.str();
            } else if (key == "tokenizer.ggml.model") {
                out_tok.model_type = r.str();
            } else if (key == "general.alignment") {
                alignment = r.u32();
            } else {
                skip_value(r, type);
            }
        }

        // ---- tensor infos --------------------------------------------------
        TensorMap tensors;
        tensors.reserve(static_cast<size_t>(n_tensors));
        for (uint64_t i = 0; i < n_tensors; ++i) {
            TensorInfo t;
            t.name = r.str();
            const uint32_t nd = r.u32();
            t.dims.resize(nd);
            for (uint32_t d = 0; d < nd; ++d) {
                t.dims[d] = r.u64();
            }
            t.type   = r.u32();
            t.offset = r.u64();
            t.nbytes = tensor_raw_bytes(t.dims, t.type);
            tensors.emplace(t.name, std::move(t));
        }

        // ---- resolve tensor data pointers ---------------------------------
        const size_t pos = static_cast<size_t>(r.p - base);
        const size_t data_start =
            (pos + static_cast<size_t>(alignment) - 1) & ~(static_cast<size_t>(alignment) - 1);
        for (auto & kv : tensors) {
            TensorInfo & t = kv.second;
            const size_t abs = data_start + static_cast<size_t>(t.offset);
            if (abs + t.nbytes > file_size) {
                throw std::runtime_error("tensor out of bounds: " + t.name);
            }
            t.data = base + abs;
        }

        // ---- choose staging size (largest quantised tensor) ----------------
        size_t max_quant_bytes = 0;
        for (const auto & kv : tensors) {
            if (kv.second.type == GGML_T_Q8_0 || kv.second.type == GGML_T_NVFP4) {
                max_quant_bytes = std::max(max_quant_bytes, kv.second.nbytes);
            }
        }
        if (max_quant_bytes == 0) {
            throw std::runtime_error("gguf has no quantised tensors");
        }

        if (hipStreamCreate(&stream) != hipSuccess) {
            throw std::runtime_error("hipStreamCreate failed");
        }
        g_alloc_zero_stream = stream;
        auto stream_guard = [&]() {
            if (stream != nullptr) {
                (void)hipStreamDestroy(stream);
                stream = nullptr;
            }
            g_alloc_zero_stream = nullptr;
        };

        {
            hipError_t err = hipMalloc(&staging, max_quant_bytes);
            if (err != hipSuccess) {
                stream_guard();
                throw std::runtime_error(std::string("hipMalloc(staging) failed: ") +
                                         hipGetErrorString(err));
            }
        }
        auto staging_guard = [&]() {
            if (staging != nullptr) {
                (void)hipFree(staging);
                staging = nullptr;
            }
        };

        // ====================================================================
        // Global tensors
        // ====================================================================
        {
            // token_embd: [2304, 201024] Q8_0 -> [201024, 2304]
            const TensorInfo & te = require(tensors, "token_embd.weight");
            expect_type(te, GGML_T_Q8_0);
            if (K_of(te) != N_EMBD || rows_of(te) != VOCAB_PADDED) {
                throw std::runtime_error("token_embd.weight unexpected shape");
            }
            out_weights.tok_embd = tok_embd_host
                ? make_q8_matrix_host(out_weights, VOCAB_PADDED, N_EMBD)
                : make_q8_matrix(out_weights, VOCAB_PADDED, N_EMBD);
            out_weights.tok_embd_on_host = tok_embd_host;
            stage_and_repack_q8(stream, staging, max_quant_bytes, te,
                                VOCAB_PADDED, N_EMBD,
                                out_weights.tok_embd.qs, out_weights.tok_embd.d, 0);

            out_weights.tok_norm = upload_f32(
                out_weights, stream, require(tensors, "token_embd_norm.weight"));

            // output.weight: [2304, 201024] Q8_0 -> repack ONLY first
            // VOCAB_UNPADDED rows into lm_head [200058, 2304].
            const TensorInfo & ow = require(tensors, "output.weight");
            expect_type(ow, GGML_T_Q8_0);
            if (K_of(ow) != N_EMBD || rows_of(ow) < VOCAB_UNPADDED) {
                throw std::runtime_error("output.weight unexpected shape");
            }
            out_weights.lm_head = make_q8_matrix(out_weights, VOCAB_UNPADDED, N_EMBD);
            stage_and_repack_q8(stream, staging, max_quant_bytes, ow,
                                VOCAB_UNPADDED, N_EMBD,
                                out_weights.lm_head.qs, out_weights.lm_head.d, 0);

            out_weights.output_norm  = upload_f32(
                out_weights, stream, require(tensors, "output_norm.weight"));
            out_weights.output_res_q = upload_f32(
                out_weights, stream, require(tensors, "output_res_query.weight"));
        }

        // ====================================================================
        // Physical layers 0..23
        // ====================================================================
        for (int i = 0; i < N_LAYER_PHYS; ++i) {
            PhysLayerWeights & L = out_weights.layers[i];
            L.is_swa    = is_swa_phys(i);

            const std::string p = "blk." + std::to_string(i) + ".";
            const TensorInfo & tgu_probe = require(tensors, p + "ffn_gate_up_exps.weight");
            L.is_nvfp4  = (tgu_probe.type == GGML_T_NVFP4);

            // ---- attention projections ------------------------------------
            const TensorInfo & tq    = require(tensors, p + "attn_q.weight");
            const TensorInfo & tg    = require(tensors, p + "attn_gate.weight");
            const TensorInfo & trel  = require(tensors, p + "attn_rel.weight");
            const TensorInfo & tsink = require(tensors, p + "attn_sink_proj.weight");
            const TensorInfo & tk    = require(tensors, p + "attn_k.weight");
            const TensorInfo & tv    = require(tensors, p + "attn_v.weight");

            if (L.is_swa) {
                // proj_qkv_all: [wq | wgate | wrel | wsink | wk | wv] = 8600 rows
                static constexpr int OFF_Q     = 0;
                static constexpr int OFF_GATE  = 3072;
                static constexpr int OFF_REL   = 6144;
                static constexpr int OFF_SINK  = 6528;
                static constexpr int OFF_K     = 6552;
                static constexpr int OFF_V     = 7576;
                static constexpr int ROWS_ALL  = 8600;

                L.proj_qkv_all = make_q8_matrix(out_weights, ROWS_ALL, N_EMBD);
                int8_t * qs = L.proj_qkv_all.qs;
                __half * dd = L.proj_qkv_all.d;

                stage_and_repack_q8(stream, staging, max_quant_bytes, tq,
                                    rows_of(tq), N_EMBD, qs, dd, OFF_Q);
                stage_and_repack_q8(stream, staging, max_quant_bytes, tg,
                                    rows_of(tg), N_EMBD, qs, dd, OFF_GATE);
                stage_and_repack_q8(stream, staging, max_quant_bytes, trel,
                                    rows_of(trel), N_EMBD, qs, dd, OFF_REL);
                stage_and_repack_q8(stream, staging, max_quant_bytes, tsink,
                                    rows_of(tsink), N_EMBD, qs, dd, OFF_SINK);
                stage_and_repack_q8(stream, staging, max_quant_bytes, tk,
                                    rows_of(tk), N_EMBD, qs, dd, OFF_K);
                stage_and_repack_q8(stream, staging, max_quant_bytes, tv,
                                    rows_of(tv), N_EMBD, qs, dd, OFF_V);

                // zero-copy views into the packed parent
                L.proj_q4.rows = 6552;  L.proj_q4.K = N_EMBD;
                L.proj_q4.qs = qs;
                L.proj_q4.d  = dd;

                L.proj_k.rows = 1024;   L.proj_k.K = N_EMBD;
                L.proj_k.qs = qs + static_cast<size_t>(OFF_K) * N_EMBD;
                L.proj_k.d  = dd + static_cast<size_t>(OFF_K) * (N_EMBD / 32);

                L.proj_v.rows = 1024;   L.proj_v.K = N_EMBD;
                L.proj_v.qs = qs + static_cast<size_t>(OFF_V) * N_EMBD;
                L.proj_v.d  = dd + static_cast<size_t>(OFF_V) * (N_EMBD / 32);
            } else {
                // Global: proj_q4 = [wq|wgate|wrel|wsink] (6552), proj_k = [wk|wk_dyn],
                // proj_v = [wv|wv_dyn].
                const TensorInfo & tkc = require(tensors, p + "attn_k_conv_proj.weight");
                const TensorInfo & tvc = require(tensors, p + "attn_v_conv_proj.weight");

                L.proj_q4 = make_q8_matrix(out_weights, 6552, N_EMBD);
                stage_and_repack_q8(stream, staging, max_quant_bytes, tq,
                                    rows_of(tq), N_EMBD, L.proj_q4.qs, L.proj_q4.d, 0);
                stage_and_repack_q8(stream, staging, max_quant_bytes, tg,
                                    rows_of(tg), N_EMBD, L.proj_q4.qs, L.proj_q4.d, 3072);
                stage_and_repack_q8(stream, staging, max_quant_bytes, trel,
                                    rows_of(trel), N_EMBD, L.proj_q4.qs, L.proj_q4.d, 6144);
                stage_and_repack_q8(stream, staging, max_quant_bytes, tsink,
                                    rows_of(tsink), N_EMBD, L.proj_q4.qs, L.proj_q4.d, 6528);

                L.proj_k = make_q8_matrix(out_weights, 1056, N_EMBD);
                stage_and_repack_q8(stream, staging, max_quant_bytes, tk,
                                    rows_of(tk), N_EMBD, L.proj_k.qs, L.proj_k.d, 0);
                stage_and_repack_q8(stream, staging, max_quant_bytes, tkc,
                                    rows_of(tkc), N_EMBD, L.proj_k.qs, L.proj_k.d, 1024);

                L.proj_v = make_q8_matrix(out_weights, 1056, N_EMBD);
                stage_and_repack_q8(stream, staging, max_quant_bytes, tv,
                                    rows_of(tv), N_EMBD, L.proj_v.qs, L.proj_v.d, 0);
                stage_and_repack_q8(stream, staging, max_quant_bytes, tvc,
                                    rows_of(tvc), N_EMBD, L.proj_v.qs, L.proj_v.d, 1024);
            }

            // ---- output projection (wo) -----------------------------------
            {
                const TensorInfo & two = require(tensors, p + "attn_output.weight");
                expect_type(two, GGML_T_Q8_0);
                if (K_of(two) != N_Q || rows_of(two) != N_EMBD) {
                    throw std::runtime_error(two.name + " unexpected shape");
                }
                L.wo = make_q8_matrix(out_weights, N_EMBD, N_Q);
                stage_and_repack_q8(stream, staging, max_quant_bytes, two,
                                    N_EMBD, N_Q, L.wo.qs, L.wo.d, 0);
            }

            // ---- F32 norms / clips / convs / router -----------------------
            if (L.is_swa) {
                L.attn_norm = upload_f32(out_weights, stream, require(tensors, p + "attn_norm.weight"));
            } else {
                L.attn_q_inp_norm = upload_f32(out_weights, stream, require(tensors, p + "attn_q_inp_norm.weight"));
                L.attn_k_inp_norm = upload_f32(out_weights, stream, require(tensors, p + "attn_k_inp_norm.weight"));
                L.attn_v_inp_norm = upload_f32(out_weights, stream, require(tensors, p + "attn_v_inp_norm.weight"));
            }

            L.attn_q_clip    = upload_f32(out_weights, stream, require(tensors, p + "attn_q_clip.weight"));
            L.attn_rel_clip  = upload_f32(out_weights, stream, require(tensors, p + "attn_rel_clip.weight"));
            L.attn_rel_bias  = upload_f32(out_weights, stream, require(tensors, p + "attn_rel_bias.weight"));
            L.attn_k_conv    = upload_f32(out_weights, stream, require(tensors, p + "attn_k_conv.weight"));
            L.attn_v_conv    = upload_f32(out_weights, stream, require(tensors, p + "attn_v_conv.weight"));
            L.attn_post_conv = upload_f32(out_weights, stream, require(tensors, p + "attn_post_conv.weight"));
            L.ffn_post_conv  = upload_f32(out_weights, stream, require(tensors, p + "ffn_post_conv.weight"));

            if (i != 0) {
                L.attn_res_query = upload_f32(out_weights, stream, require(tensors, p + "attn_res_query.weight"));
            }
            L.ffn_res_query = upload_f32(out_weights, stream, require(tensors, p + "ffn_res_query.weight"));
            L.ffn_norm      = upload_f32(out_weights, stream, require(tensors, p + "ffn_norm.weight"));
            L.ffn_gate_inp  = upload_f16(out_weights, stream, staging, max_quant_bytes,
                                         require(tensors, p + "ffn_gate_inp.weight"));
            L.ffn_exp_probs_b = upload_f32(out_weights, stream, require(tensors, p + "exp_probs_b.bias"));

            // ---- MoE experts ----------------------------------------------
            if (L.is_nvfp4) {
                const TensorInfo & tgu = require(tensors, p + "ffn_gate_up_exps.weight");
                const TensorInfo & tgu_s = require(tensors, p + "ffn_gate_up_exps.scale");
                const TensorInfo & tdn = require(tensors, p + "ffn_down_exps.weight");
                const TensorInfo & tdn_s = require(tensors, p + "ffn_down_exps.scale");

                L.gate_up_nvfp4 = make_nvfp4_experts(out_weights, N_EXPERT, N_GATE_UP_EXP, N_EMBD);
                // gate_up: rows = 2304, K = 2304
                if (nexp_of(tgu) != N_EXPERT || rows_of(tgu) != N_GATE_UP_EXP || K_of(tgu) != N_EMBD) {
                    throw std::runtime_error(tgu.name + " unexpected shape");
                }
                stage_and_repack_nvfp4(stream, staging, max_quant_bytes, tgu,
                                       N_EXPERT, N_GATE_UP_EXP, N_EMBD,
                                       L.gate_up_nvfp4.qs, L.gate_up_nvfp4.scales);
                if (nelem_of(tgu_s) != static_cast<size_t>(N_EXPERT)) {
                    throw std::runtime_error(tgu_s.name + " unexpected shape");
                }
                (void)hipMemcpyAsync(L.gate_up_nvfp4.scale2, tgu_s.data, N_EXPERT * sizeof(float),
                               hipMemcpyHostToDevice, stream);

                L.down_nvfp4 = make_nvfp4_experts(out_weights, N_EXPERT, N_EMBD, N_FF_EXP);
                // down: rows = 2304, K = 1152
                if (nexp_of(tdn) != N_EXPERT || rows_of(tdn) != N_EMBD || K_of(tdn) != N_FF_EXP) {
                    throw std::runtime_error(tdn.name + " unexpected shape");
                }
                stage_and_repack_nvfp4(stream, staging, max_quant_bytes, tdn,
                                       N_EXPERT, N_EMBD, N_FF_EXP,
                                       L.down_nvfp4.qs, L.down_nvfp4.scales);
                if (nelem_of(tdn_s) != static_cast<size_t>(N_EXPERT)) {
                    throw std::runtime_error(tdn_s.name + " unexpected shape");
                }
                (void)hipMemcpyAsync(L.down_nvfp4.scale2, tdn_s.data, N_EXPERT * sizeof(float),
                               hipMemcpyHostToDevice, stream);
            } else {
                const TensorInfo & tgu = require(tensors, p + "ffn_gate_up_exps.weight");
                const TensorInfo & tdn = require(tensors, p + "ffn_down_exps.weight");

                // gate_up: [128, 2304, 2304] Q8_0
                if (nexp_of(tgu) != N_EXPERT || rows_of(tgu) != N_GATE_UP_EXP || K_of(tgu) != N_EMBD) {
                    throw std::runtime_error(tgu.name + " unexpected shape");
                }
                L.gate_up_q8 = make_q8_experts(out_weights, N_EXPERT, N_GATE_UP_EXP, N_EMBD);
                stage_and_repack_q8(stream, staging, max_quant_bytes, tgu,
                                    N_EXPERT * (N_GATE_UP_EXP), N_EMBD,
                                    L.gate_up_q8.qs, L.gate_up_q8.d, 0);

                // down: [128, 2304, 1152] Q8_0
                if (nexp_of(tdn) != N_EXPERT || rows_of(tdn) != N_EMBD || K_of(tdn) != N_FF_EXP) {
                    throw std::runtime_error(tdn.name + " unexpected shape");
                }
                L.down_q8 = make_q8_experts(out_weights, N_EXPERT, N_EMBD, N_FF_EXP);
                stage_and_repack_q8(stream, staging, max_quant_bytes, tdn,
                                    N_EXPERT * N_EMBD, N_FF_EXP,
                                    L.down_q8.qs, L.down_q8.d, 0);
            }
        }

        // ====================================================================
        // Loop controllers (physical layers 6..17)
        // ====================================================================
        static const char * const kCtrlSuffix[LOOP_CTRL_COUNT] = {"q", "k", "v", "ffn"};
        for (int l = LOOP_START; l < LOOP_END; ++l) {
            LoopLayerWeights & LL = out_weights.loop[l - LOOP_START];
            const std::string p = "blk." + std::to_string(l) + ".";

            LL.msg_query = upload_f32(out_weights, stream, require(tensors, p + "loop_msg_query.weight"));
            for (int c = 0; c < LOOP_CTRL_COUNT; ++c) {
                const std::string s = std::string("loop_") + kCtrlSuffix[c] + "_ctrl_";
                LL.ctrl_down[c]   = upload_f16(out_weights, stream, staging, max_quant_bytes,
                                               require(tensors, p + s + "down.weight"));
                LL.ctrl_gain[c]   = upload_f16(out_weights, stream, staging, max_quant_bytes,
                                               require(tensors, p + s + "gain.weight"));
                LL.ctrl_forget[c] = upload_f16(out_weights, stream, staging, max_quant_bytes,
                                               require(tensors, p + s + "forget.weight"));
            }
        }

        // ---- settle the stream ---------------------------------------------
        {
            hipError_t err = hipStreamSynchronize(stream);
            if (err != hipSuccess) {
                throw std::runtime_error(std::string("hipStreamSynchronize failed: ") +
                                         hipGetErrorString(err));
            }
        }

        // ---- self-test (must be exactly 0.0f) ------------------------------
        if (verify_repack_self_test) {
            const TensorInfo & te = require(tensors, "token_embd.weight");
            const float e_q8 = detail::verify_q8_0_repack(
                te.data, out_weights.tok_embd.qs, out_weights.tok_embd.d,
                out_weights.tok_embd.rows, out_weights.tok_embd.K);

            const TensorInfo & tgu = require(tensors, "blk.0.ffn_gate_up_exps.weight");
            const float e_nv = detail::verify_nvfp4_repack(
                tgu.data,
                out_weights.layers[0].gate_up_nvfp4.qs,
                out_weights.layers[0].gate_up_nvfp4.scales,
                out_weights.layers[0].gate_up_nvfp4.scale2,
                N_EXPERT, N_GATE_UP_EXP, N_EMBD);

            std::printf("[verify] Q8_0  token_embd             max abs err = %.6f\n", e_q8);
            std::printf("[verify] NVFP4 blk.0.ffn_gate_up_exps  max abs err = %.6f\n", e_nv);
            if (e_q8 != 0.0f || e_nv != 0.0f) {
                throw std::runtime_error("repack self-test FAILED (expected 0.000000)");
            }
        }

        // ---- free staging + stream (keep VRAM minimal) --------------------
        staging_guard();
        stream_guard();

        // ---- final completeness check -------------------------------------
        auto yes = [](const void * p, const std::string & what) {
            if (p == nullptr) {
                throw std::runtime_error("null weight pointer: " + what);
            }
        };
        yes(out_weights.tok_embd.qs,  "tok_embd.qs");
        yes(out_weights.tok_embd.d,   "tok_embd.d");
        yes(out_weights.tok_norm,     "tok_norm");
        yes(out_weights.lm_head.qs,   "lm_head.qs");
        yes(out_weights.lm_head.d,    "lm_head.d");
        yes(out_weights.output_norm,  "output_norm");
        yes(out_weights.output_res_q, "output_res_q");

        for (int i = 0; i < N_LAYER_PHYS; ++i) {
            const PhysLayerWeights & L = out_weights.layers[i];
            const std::string lp = "layer " + std::to_string(i) + ": ";
            yes(L.proj_q4.qs, lp + "proj_q4.qs");
            yes(L.proj_q4.d,  lp + "proj_q4.d");
            yes(L.proj_k.qs,  lp + "proj_k.qs");
            yes(L.proj_k.d,   lp + "proj_k.d");
            yes(L.proj_v.qs,  lp + "proj_v.qs");
            yes(L.proj_v.d,   lp + "proj_v.d");
            yes(L.wo.qs,      lp + "wo.qs");
            yes(L.wo.d,       lp + "wo.d");
            if (L.is_swa) {
                yes(L.proj_qkv_all.qs, lp + "proj_qkv_all.qs");
                yes(L.proj_qkv_all.d,  lp + "proj_qkv_all.d");
                yes(L.attn_norm,       lp + "attn_norm");
            } else {
                yes(L.attn_q_inp_norm, lp + "attn_q_inp_norm");
                yes(L.attn_k_inp_norm, lp + "attn_k_inp_norm");
                yes(L.attn_v_inp_norm, lp + "attn_v_inp_norm");
            }
            yes(L.attn_q_clip,    lp + "attn_q_clip");
            yes(L.attn_rel_clip,  lp + "attn_rel_clip");
            yes(L.attn_rel_bias,  lp + "attn_rel_bias");
            yes(L.attn_k_conv,    lp + "attn_k_conv");
            yes(L.attn_v_conv,    lp + "attn_v_conv");
            yes(L.attn_post_conv, lp + "attn_post_conv");
            yes(L.ffn_post_conv,  lp + "ffn_post_conv");
            if (i != 0) {
                yes(L.attn_res_query, lp + "attn_res_query");
            }
            yes(L.ffn_res_query,  lp + "ffn_res_query");
            yes(L.ffn_norm,       lp + "ffn_norm");
            yes(L.ffn_gate_inp,   lp + "ffn_gate_inp");
            yes(L.ffn_exp_probs_b, lp + "exp_probs_b");
            if (L.is_nvfp4) {
                yes(L.gate_up_nvfp4.qs,     lp + "gate_up_nvfp4.qs");
                yes(L.gate_up_nvfp4.scales, lp + "gate_up_nvfp4.scales");
                yes(L.gate_up_nvfp4.scale2, lp + "gate_up_nvfp4.scale2");
                yes(L.down_nvfp4.qs,        lp + "down_nvfp4.qs");
                yes(L.down_nvfp4.scales,    lp + "down_nvfp4.scales");
                yes(L.down_nvfp4.scale2,    lp + "down_nvfp4.scale2");
            } else {
                yes(L.gate_up_q8.qs, lp + "gate_up_q8.qs");
                yes(L.gate_up_q8.d,  lp + "gate_up_q8.d");
                yes(L.down_q8.qs,    lp + "down_q8.qs");
                yes(L.down_q8.d,     lp + "down_q8.d");
            }
        }

        for (int l = 0; l < LOOP_CORE_LEN; ++l) {
            const LoopLayerWeights & LL = out_weights.loop[l];
            const std::string lp = "loop " + std::to_string(LOOP_START + l) + ": ";
            yes(LL.msg_query, lp + "msg_query");
            for (int c = 0; c < LOOP_CTRL_COUNT; ++c) {
                yes(LL.ctrl_down[c],   lp + "ctrl_down[" + std::to_string(c) + "]");
                yes(LL.ctrl_gain[c],   lp + "ctrl_gain[" + std::to_string(c) + "]");
                yes(LL.ctrl_forget[c], lp + "ctrl_forget[" + std::to_string(c) + "]");
            }
        }

        // ---- report --------------------------------------------------------
        const auto t_end = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t_end - t_start).count();
        std::printf("[gguf] loaded %s in %.3f s\n", path.c_str(), secs);
        std::printf("[gguf] total VRAM: %zu bytes (%.1f MiB)\n",
                    out_weights.total_vram_bytes,
                    static_cast<double>(out_weights.total_vram_bytes) / (1024.0 * 1024.0));
        if (out_weights.tok_embd_on_host) {
            std::printf("[gguf] tok_embd=CPU pinned (%.1f MiB saved; GPU-mapped host RAM, lm_head stays in VRAM)\n",
                        static_cast<double>(out_weights.host_pinned_bytes) / (1024.0 * 1024.0));
        }

    } catch (...) {
        if (staging != nullptr) {
            (void)hipFree(staging);
            staging = nullptr;
        }
        if (stream != nullptr) {
            (void)hipStreamDestroy(stream);
            stream = nullptr;
        }
        g_alloc_zero_stream = nullptr;
        munmap_guard();
        throw;
    }

    munmap_guard();
    return true;
}

} // namespace inkling
