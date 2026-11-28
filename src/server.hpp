#pragma once

#include "bpe_tokenizer.hpp"
#include "dflash2_mtp.h"
#include "engine.hpp"
#include "vision_encoder.hpp"

#include <string>

namespace inkling {

struct ServerConfig {
    std::string host = "127.0.0.1";
    int         port = 4444;
    std::string model_name = "inkling-2-mini";
    float       spec_threshold = 0.8f;
    bool        mtp_enabled = false;
    bool        vision_enabled = false;
    std::string kv_type_name = "q8_0";
    int         sparse_pages = 0;
    int         n_slots = 4;
    bool        strict_slot_ctx = false;
    int32_t     max_slot_ctx = 0;
    float       default_temperature = 1.0f;
    float       default_top_p = 1.0f;
    float       default_min_p = 0.0f;
    int         default_top_k = 0;
    std::string output_dir = "outputs";
};

// Runs a blocking, multi-threaded OpenAI-compatible HTTP/1.1 + SSE server
// with a built-in Web Chat UI at GET /. Returns 0 on clean shutdown (SIGINT/SIGTERM).
int run_http_server(
    InklingEngine & eng,
    const BpeTokenizer & bpe,
    SpecDrafter * drafter,        // nullptr if --mtp not loaded
    VisionEncoder * vision,       // nullptr if --mmproj not loaded
    const ServerConfig & cfg);

} // namespace inkling
