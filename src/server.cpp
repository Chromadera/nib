#include "server.hpp"
#include "server_ui.hpp"
#include "nib_logo.hpp"
#include "kernels_mtp_verify.h"

#include <hip/hip_runtime.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace inkling {
namespace {

static std::atomic<bool> g_stop_server{false};
static std::atomic<int>  g_listen_fd{-1};

static void handle_sig(int) {
    g_stop_server.store(true);
    int fd = g_listen_fd.exchange(-1);
    if (fd >= 0) ::close(fd);
}

// ============================================================================
// Minimal JSON DOM + Parser + Serializer (Zero External Dependencies)
// ============================================================================
struct JsonVal {
    enum Type { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } type = J_NULL;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<JsonVal> arr;
    std::map<std::string, JsonVal> obj;

    bool is_null() const { return type == J_NULL; }
    bool is_bool() const { return type == J_BOOL; }
    bool is_num()  const { return type == J_NUM; }
    bool is_str()  const { return type == J_STR; }
    bool is_arr()  const { return type == J_ARR; }
    bool is_obj()  const { return type == J_OBJ; }

    const JsonVal * get(const std::string & k) const {
        if (type != J_OBJ) return nullptr;
        auto it = obj.find(k);
        return (it != obj.end()) ? &it->second : nullptr;
    }
    std::string get_str(const std::string & k, const std::string & def = "") const {
        const JsonVal * v = get(k);
        return (v && v->is_str()) ? v->str : def;
    }
    double get_num(const std::string & k, double def) const {
        const JsonVal * v = get(k);
        return (v && v->is_num()) ? v->num : def;
    }
    bool get_bool(const std::string & k, bool def) const {
        const JsonVal * v = get(k);
        return (v && v->is_bool()) ? v->b : def;
    }
};

struct JsonParser {
    const std::string & s;
    size_t p = 0;

    void ws() {
        while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) ++p;
    }
    static void append_utf8(std::string & out, uint32_t cp) {
        if (cp < 0x80) {
            out.push_back((char)cp);
        } else if (cp < 0x800) {
            out.push_back((char)(0xC0 | (cp >> 6)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back((char)(0xE0 | (cp >> 12)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            out.push_back((char)(0xF0 | (cp >> 18)));
            out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        }
    }
    bool hex4(uint32_t & out) {
        if (p + 4 > s.size()) return false;
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = s[p++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
            else return false;
        }
        out = v;
        return true;
    }
    bool parse_str(std::string & out) {
        ws();
        if (p >= s.size() || s[p] != '"') return false;
        ++p;
        out.clear();
        while (p < s.size()) {
            char c = s[p++];
            if (c == '"') return true;
            if (c == '\\') {
                if (p >= s.size()) return false;
                char e = s[p++];
                switch (e) {
                    case '"':  out.push_back('"');  break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/');  break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    case 'n':  out.push_back('\n'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'u': {
                        uint32_t u1 = 0;
                        if (!hex4(u1)) return false;
                        if (u1 >= 0xD800 && u1 <= 0xDBFF && p + 6 <= s.size() && s[p] == '\\' && s[p + 1] == 'u') {
                            p += 2;
                            uint32_t u2 = 0;
                            if (!hex4(u2)) return false;
                            if (u2 >= 0xDC00 && u2 <= 0xDFFF) {
                                uint32_t cp = 0x10000 + (((u1 - 0xD800) << 10) | (u2 - 0xDC00));
                                append_utf8(out, cp);
                            }
                        } else {
                            append_utf8(out, u1);
                        }
                        break;
                    }
                    default: out.push_back(e); break;
                }
            } else {
                out.push_back(c);
            }
        }
        return false;
    }
    bool parse_val(JsonVal & out, int depth = 0) {
        if (depth > 64) return false;
        ws();
        if (p >= s.size()) return false;
        char c = s[p];
        if (c == '"') {
            out.type = JsonVal::J_STR;
            return parse_str(out.str);
        }
        if (c == '{') {
            ++p;
            out.type = JsonVal::J_OBJ;
            ws();
            if (p < s.size() && s[p] == '}') { ++p; return true; }
            while (p < s.size()) {
                std::string k;
                if (!parse_str(k)) return false;
                ws();
                if (p >= s.size() || s[p] != ':') return false;
                ++p;
                JsonVal v;
                if (!parse_val(v, depth + 1)) return false;
                out.obj[std::move(k)] = std::move(v);
                ws();
                if (p < s.size() && s[p] == ',') { ++p; continue; }
                if (p < s.size() && s[p] == '}') { ++p; return true; }
                return false;
            }
            return false;
        }
        if (c == '[') {
            ++p;
            out.type = JsonVal::J_ARR;
            ws();
            if (p < s.size() && s[p] == ']') { ++p; return true; }
            while (p < s.size()) {
                JsonVal v;
                if (!parse_val(v, depth + 1)) return false;
                out.arr.push_back(std::move(v));
                ws();
                if (p < s.size() && s[p] == ',') { ++p; continue; }
                if (p < s.size() && s[p] == ']') { ++p; return true; }
                return false;
            }
            return false;
        }
        if (s.compare(p, 4, "true") == 0)  { p += 4; out.type = JsonVal::J_BOOL; out.b = true;  return true; }
        if (s.compare(p, 5, "false") == 0) { p += 5; out.type = JsonVal::J_BOOL; out.b = false; return true; }
        if (s.compare(p, 4, "null") == 0)  { p += 4; out.type = JsonVal::J_NULL; return true; }
        if (c == '-' || (c >= '0' && c <= '9')) {
            const char * start = s.c_str() + p;
            char * end = nullptr;
            double v = std::strtod(start, &end);
            if (end == start) return false;
            p = (size_t)(end - s.c_str());
            out.type = JsonVal::J_NUM;
            out.num = v;
            return true;
        }
        return false;
    }
};

static std::string json_escape(const std::string & in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (unsigned char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
                    out += buf;
                } else {
                    out.push_back((char)c);
                }
        }
    }
    return out;
}

static std::string json_str(const std::string & in) {
    return "\"" + json_escape(in) + "\"";
}

static void json_serialize_into(const JsonVal & v, std::string & out) {
    switch (v.type) {
        case JsonVal::J_NULL:
            out += "null";
            break;
        case JsonVal::J_BOOL:
            out += v.b ? "true" : "false";
            break;
        case JsonVal::J_NUM: {
            char buf[64];
            if (std::isfinite(v.num) && std::floor(v.num) == v.num &&
                v.num >= -9007199254740992.0 && v.num <= 9007199254740992.0) {
                std::snprintf(buf, sizeof(buf), "%.0f", v.num);
            } else {
                std::snprintf(buf, sizeof(buf), "%.17g", v.num);
            }
            out += buf;
            break;
        }
        case JsonVal::J_STR:
            out += json_str(v.str);
            break;
        case JsonVal::J_ARR:
            out.push_back('[');
            for (size_t i = 0; i < v.arr.size(); ++i) {
                if (i > 0) out.push_back(',');
                json_serialize_into(v.arr[i], out);
            }
            out.push_back(']');
            break;
        case JsonVal::J_OBJ:
            out.push_back('{');
            {
                bool first = true;
                for (const auto & kv : v.obj) { // std::map is already key-sorted
                    if (!first) out.push_back(',');
                    first = false;
                    out += json_str(kv.first);
                    out.push_back(':');
                    json_serialize_into(kv.second, out);
                }
            }
            out.push_back('}');
            break;
    }
}

static std::string json_serialize(const JsonVal & v) {
    std::string out;
    out.reserve(128);
    json_serialize_into(v, out);
    return out;
}

// ============================================================================
// Base64 Decoder (for data:image/...;base64,... URLs)
// ============================================================================
static std::vector<uint8_t> decode_base64(const std::string & input) {
    static const int8_t T[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
        52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    };
    std::vector<uint8_t> out;
    out.reserve(input.size() * 3 / 4);
    int val = 0, valb = -8;
    for (unsigned char c : input) {
        if (c == '=') break;
        int d = T[c];
        if (d < 0) continue;
        val = (val << 6) + d;
        valb += 6;
        if (valb >= 0) {
            out.push_back((uint8_t)((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

// ============================================================================
// UTF-8 Streaming Boundary Buffer (ensures valid JSON strings per SSE chunk)
// ============================================================================
struct Utf8StreamBuf {
    std::string pending;

    std::string push(const std::string & bytes) {
        pending += bytes;
        if (pending.empty()) return "";
        // Inspect trailing incomplete UTF-8 sequence (at most 3 bytes)
        size_t n = pending.size();
        size_t cut = n;
        for (size_t back = 1; back <= 3 && back <= n; ++back) {
            unsigned char b = (unsigned char)pending[n - back];
            if ((b & 0x80) == 0) break;        // ASCII: complete
            if ((b & 0xC0) == 0xC0) {          // leading byte found
                size_t need = 1;
                if ((b & 0xE0) == 0xC0) need = 2;
                else if ((b & 0xF0) == 0xE0) need = 3;
                else if ((b & 0xF8) == 0xF0) need = 4;
                if (back < need) cut = n - back;
                break;
            }
        }
        std::string ready = pending.substr(0, cut);
        pending.erase(0, cut);
        return ready;
    }

    std::string flush() {
        std::string r = std::move(pending);
        pending.clear();
        return r;
    }
};

// ============================================================================
// Socket Helpers
// ============================================================================
static bool send_all(int fd, const char * data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = ::send(fd, data + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) return false;
        sent += (size_t)n;
    }
    return true;
}

static bool send_all(int fd, const std::string & s) {
    return send_all(fd, s.data(), s.size());
}

static bool client_disconnected(int fd) {
    struct pollfd pfd {};
    pfd.fd = fd;
    pfd.events = POLLIN | POLLHUP | POLLERR;
    int r = ::poll(&pfd, 1, 0);
    if (r <= 0) return false;
    if (pfd.revents & (POLLHUP | POLLERR | POLLNVAL)) return true;
    if (pfd.revents & POLLIN) {
        char ch = 0;
        ssize_t n = ::recv(fd, &ch, 1, MSG_PEEK | MSG_DONTWAIT);
        if (n == 0) return true;
    }
    return false;
}

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    std::map<std::string, std::string> headers;
};

static std::string to_lower(std::string s) {
    for (char & c : s) {
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    }
    return s;
}

static bool read_http_request(int fd, HttpRequest & req) {
    std::string buf;
    buf.reserve(4096);
    char tmp[8192];
    size_t hdr_end = std::string::npos;
    while (hdr_end == std::string::npos) {
        ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) return false;
        buf.append(tmp, (size_t)n);
        hdr_end = buf.find("\r\n\r\n");
        if (buf.size() > 256 * 1024 && hdr_end == std::string::npos) return false;
    }

    std::string hdrs = buf.substr(0, hdr_end);
    std::string rest = buf.substr(hdr_end + 4);

    std::istringstream iss(hdrs);
    std::string line;
    if (!std::getline(iss, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    {
        std::istringstream rl(line);
        rl >> req.method >> req.path;
        size_t q = req.path.find('?');
        if (q != std::string::npos) req.path = req.path.substr(0, q);
    }
    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string k = to_lower(line.substr(0, colon));
        std::string v = line.substr(colon + 1);
        size_t s0 = v.find_first_not_of(" \t");
        if (s0 != std::string::npos) v = v.substr(s0);
        req.headers[k] = v;
    }

    auto it_cl = req.headers.find("content-length");
    auto it_te = req.headers.find("transfer-encoding");
    if (it_cl != req.headers.end()) {
        size_t cl = (size_t)std::strtoull(it_cl->second.c_str(), nullptr, 10);
        if (cl > 32 * 1024 * 1024) return false; // 32 MiB cap
        while (rest.size() < cl) {
            ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
            if (n <= 0) return false;
            rest.append(tmp, (size_t)n);
        }
        req.body = rest.substr(0, cl);
    } else if (it_te != req.headers.end() && to_lower(it_te->second).find("chunked") != std::string::npos) {
        // Decode chunked body
        std::string raw = std::move(rest);
        size_t pos = 0;
        while (true) {
            while (raw.find("\r\n", pos) == std::string::npos) {
                ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
                if (n <= 0) return false;
                raw.append(tmp, (size_t)n);
            }
            size_t crlf = raw.find("\r\n", pos);
            size_t chunk_sz = (size_t)std::strtoull(raw.substr(pos, crlf - pos).c_str(), nullptr, 16);
            pos = crlf + 2;
            if (chunk_sz == 0) break;
            if (req.body.size() + chunk_sz > 32 * 1024 * 1024) return false;
            while (raw.size() < pos + chunk_sz + 2) {
                ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
                if (n <= 0) return false;
                raw.append(tmp, (size_t)n);
            }
            req.body.append(raw, pos, chunk_sz);
            pos += chunk_sz + 2;
        }
    }
    return true;
}

static void send_http_response(int fd, int status, const char * status_text,
                               const char * content_type, const std::string & body) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status << " " << status_text << "\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
        << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        << "Connection: close\r\n\r\n";
    std::string hdr = oss.str();
    (void)send_all(fd, hdr);
    if (!body.empty()) (void)send_all(fd, body);
}

static bool send_sse_headers(int fd) {
    const std::string hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream; charset=utf-8\r\n"
        "Cache-Control: no-cache\r\n"
        "X-Accel-Buffering: no\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Connection: close\r\n\r\n";
    return send_all(fd, hdr);
}

static std::string url_decode_segment(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
        if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
        return -1;
    };
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size()) {
            int hi = hex(s[i + 1]), lo = hex(s[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back((char)((hi << 4) | lo));
                i += 2;
                continue;
            }
        }
        out.push_back(s[i]);
    }
    return out;
}

static bool is_safe_output_filename(const std::string & name) {
    if (name.empty() || name.size() > 128) return false;
    if (name[0] == '.' || name.find("..") != std::string::npos) return false;
    for (unsigned char c : name) {
        bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
        if (!ok) return false;
    }
    return true;
}

static std::string sanitize_output_filename(const std::string & raw) {
    size_t slash = raw.find_last_of("/\\");
    std::string base = (slash == std::string::npos) ? raw : raw.substr(slash + 1);
    std::string clean;
    clean.reserve(base.size());
    for (unsigned char c : base) {
        bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
        if (ok) clean.push_back((char)c);
        else if (c == ' ' || c == '\t') {
            if (!clean.empty() && clean.back() != '-') clean.push_back('-');
        }
    }
    while (!clean.empty() && (clean.front() == '.' || clean.front() == '-')) clean.erase(clean.begin());
    while (clean.find("..") != std::string::npos) {
        clean.replace(clean.find(".."), 2, ".");
    }
    if (clean.size() > 120) clean.resize(120);
    if (clean.empty()) {
        clean = "output-" + std::to_string((long long)std::time(nullptr)) + ".txt";
    }
    return clean;
}

static const char * content_type_for_filename(const std::string & name) {
    size_t dot = name.rfind('.');
    if (dot == std::string::npos) return "text/plain; charset=utf-8";
    std::string ext = to_lower(name.substr(dot));
    if (ext == ".html" || ext == ".htm") return "text/html; charset=utf-8";
    if (ext == ".svg")                   return "image/svg+xml";
    if (ext == ".js" || ext == ".mjs")   return "application/javascript; charset=utf-8";
    if (ext == ".css")                   return "text/css; charset=utf-8";
    if (ext == ".json")                  return "application/json; charset=utf-8";
    if (ext == ".md")                    return "text/markdown; charset=utf-8";
    if (ext == ".xml")                   return "application/xml; charset=utf-8";
    return "text/plain; charset=utf-8";
}

static bool is_loopback_host(const std::string & host) {
    return host == "127.0.0.1" || host == "localhost" || host == "::1";
}

static bool path_is_within(const std::filesystem::path & child, const std::filesystem::path & root) {
    auto ci = child.begin();
    auto ri = root.begin();
    for (; ri != root.end(); ++ri, ++ci) {
        if (ci == child.end() || *ci != *ri) return false;
    }
    return true;
}

static bool resolve_safe_save_dir(const std::string & requested,
                                  const std::filesystem::path & default_dir,
                                  bool allow_home,
                                  std::filesystem::path & out_dir,
                                  std::string & err_msg) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path def_norm = fs::weakly_canonical(fs::absolute(default_dir, ec), ec).lexically_normal();
    if (requested.empty()) {
        out_dir = def_norm;
        return true;
    }
    const char * home_env = std::getenv("HOME");
    fs::path home_norm;
    if (home_env && *home_env) {
        home_norm = fs::weakly_canonical(fs::path(home_env), ec).lexically_normal();
    }
    std::string raw = requested;
    while (!raw.empty() && (raw.front() == ' ' || raw.front() == '\t')) raw.erase(raw.begin());
    while (!raw.empty() && (raw.back() == ' ' || raw.back() == '\t')) raw.pop_back();
    if (raw.empty()) {
        out_dir = def_norm;
        return true;
    }
    fs::path cand;
    if (raw == "~" && !home_norm.empty()) {
        cand = home_norm;
    } else if (raw.rfind("~/", 0) == 0 && !home_norm.empty()) {
        cand = home_norm / raw.substr(2);
    } else {
        fs::path rp(raw);
        cand = rp.is_absolute() ? rp : (def_norm / rp);
    }
    fs::path cand_norm = fs::weakly_canonical(cand, ec).lexically_normal();
    if (allow_home && !home_norm.empty()) {
        if (!path_is_within(cand_norm, home_norm)) {
            err_msg = "Save folder must be inside your home directory (" + home_norm.string() + ")";
            return false;
        }
    } else {
        if (!path_is_within(cand_norm, def_norm)) {
            err_msg = "Save folder must be inside " + def_norm.string();
            return false;
        }
    }
    out_dir = cand_norm;
    return true;
}

// ============================================================================
// Exact CPU Sampler (Greedy, Temperature, Top-K, Top-P, Min-P)
// ============================================================================
struct SampleParams {
    float temperature = 0.0f;
    float top_p       = 1.0f;
    float min_p       = 0.0f;
    int   top_k       = 0;
    uint64_t seed     = 0;
    bool  has_seed    = false;
};

static int32_t sample_logits_ptr(const float * logits, int V, const SampleParams & sp, std::mt19937_64 & rng) {
    if (logits == nullptr || V <= 0) return 0;
    if (sp.temperature <= 1e-5f) {
        return (int32_t)(std::max_element(logits, logits + V) - logits);
    }

    int k = (sp.top_k > 0 && sp.top_k < V) ? sp.top_k : V;
    std::vector<int32_t> idx((size_t)V);
    std::iota(idx.begin(), idx.end(), 0);
    if (k < V) {
        std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                         [&](int32_t a, int32_t b) { return logits[(size_t)a] > logits[(size_t)b]; });
        idx.resize((size_t)k);
    }
    std::sort(idx.begin(), idx.end(),
              [&](int32_t a, int32_t b) { return logits[(size_t)a] > logits[(size_t)b]; });

    const float inv_t = 1.0f / sp.temperature;
    const float max_l = logits[(size_t)idx[0]];
    std::vector<float> probs(idx.size());
    float sum = 0.0f;
    for (size_t i = 0; i < idx.size(); ++i) {
        float p = std::exp((logits[(size_t)idx[i]] - max_l) * inv_t);
        probs[i] = p;
        sum += p;
    }
    if (!(sum > 0.0f)) return idx[0];
    const float inv_s = 1.0f / sum;
    for (float & p : probs) p *= inv_s;

    // Min-P filter
    size_t keep = probs.size();
    if (sp.min_p > 0.0f) {
        const float thresh = probs[0] * sp.min_p;
        size_t cnt = 1;
        while (cnt < keep && probs[cnt] >= thresh) ++cnt;
        keep = cnt;
    }
    // Top-P (nucleus) filter
    if (sp.top_p > 0.0f && sp.top_p < 1.0f) {
        float cum = 0.0f;
        size_t cnt = 0;
        while (cnt < keep) {
            cum += probs[cnt++];
            if (cum >= sp.top_p) break;
        }
        keep = std::max<size_t>(1, cnt);
    }
    float total = 0.0f;
    for (size_t i = 0; i < keep; ++i) total += probs[i];
    std::uniform_real_distribution<float> dist(0.0f, total);
    float r = dist(rng);
    float acc = 0.0f;
    for (size_t i = 0; i < keep; ++i) {
        acc += probs[i];
        if (r <= acc) return idx[i];
    }
    return idx[keep - 1];
}

static int32_t sample_logits(const std::vector<float> & logits, const SampleParams & sp, std::mt19937_64 & rng) {
    return sample_logits_ptr(logits.data(), (int)logits.size(), sp, rng);
}

// Issue #4: settings whose kept set is { x : p(x) >= min_p * p_max } over the
// WHOLE vocabulary (no top-k cut, no nucleus cut) are sampled exactly on the GPU
// (InklingEngine::sample_rows_full_to_host) instead of copying all 200,058
// logits to the host and sorting them.  Same target distribution as
// sample_logits_ptr: softmax(l / T), min-p filter, renormalise.
static bool use_gpu_full_sampler(const SampleParams & sp) {
    return sp.temperature > 1e-5f &&
           (sp.top_k <= 0 || sp.top_k >= VOCAB_UNPADDED) &&
           !(sp.top_p > 0.0f && sp.top_p < 1.0f);
}

// One exact draw from logit row 0 of the engine's logits buffer (the row the
// last decode step wrote).  Falls back to `fallback` if the sampler fails.
static int32_t sample_decode_row_gpu(const InklingEngine & eng, const SampleParams & sp,
                                     std::mt19937_64 & rng, int32_t fallback) {
    const int32_t excl[1] = { -1 };
    int32_t tok[1] = { -1 };
    float stats[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    const uint64_t seed = rng();
    if (!eng.sample_rows_full_to_host(1, 1.0f / sp.temperature, sp.min_p, excl, seed, tok, stats) ||
        tok[0] < 0 || tok[0] >= VOCAB_UNPADDED || !(stats[1] > 0.0f)) {
        return fallback;
    }
    return tok[0];
}

// 100% mathematically lossless speculative rejection sampling for temperature > 0.
// Draft token `draft_tok` is treated as a point-mass proposal q(x) = 1[x == draft_tok]:
//   - Accept `draft_tok` with probability p(draft_tok), where p is the exact target
//     distribution after (temperature, top_k, min_p, top_p).
//   - On reject, sample from the exact residual p'(x) = p(x) / (1 - p(draft_tok)) for x != draft_tok.
//   - When `draft_tok < 0` (bonus position), sample directly from p(x).
// Marginal output distribution is provably identical to `sample_logits_ptr`.
static int32_t sample_speculative_row(
        const InklingEngine & eng, int row_idx,
        const float * topk_vals, const int32_t * topk_ids,
        float row_max_l, float row_sum_exp,
        int32_t draft_tok, const SampleParams & sp,
        std::mt19937_64 & rng, bool & accepted) {
    accepted = false;
    const float inv_t = 1.0f / sp.temperature;

    struct Cand { float val; int32_t id; };
    std::array<Cand, MTP_VERIFY_TOPK> top;
    for (int i = 0; i < MTP_VERIFY_TOPK; ++i) {
        top[(size_t)i] = { topk_vals[i], topk_ids[i] };
    }
    std::sort(top.begin(), top.end(), [](const Cand & a, const Cand & b) {
        return (a.val != b.val) ? (a.val > b.val) : (a.id < b.id);
    });

    std::vector<int32_t> ids;
    std::vector<float>   probs;

    const bool bounded_k = (sp.top_k > 0 && sp.top_k <= MTP_VERIFY_TOPK);
    bool exact_fast = false;
    if (bounded_k) {
        const size_t n = (size_t)sp.top_k;
        ids.resize(n);
        probs.resize(n);
        const float max_l = top[0].val;
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            ids[i] = top[i].id;
            float p = std::exp((top[i].val - max_l) * inv_t);
            probs[i] = p;
            sum += p;
        }
        if (!(sum > 0.0f)) {
            if (draft_tok >= 0 && draft_tok == ids[0]) accepted = true;
            return ids[0];
        }
        const float inv_s = 1.0f / sum;
        for (float & p : probs) p *= inv_s;
        exact_fast = true;
    } else if ((sp.top_k <= 0 || sp.top_k >= VOCAB_UNPADDED) && row_sum_exp > 0.0f) {
        const float inv_s = 1.0f / row_sum_exp;
        std::array<float, MTP_VERIFY_TOPK> fp;
        float cum = 0.0f;
        for (size_t i = 0; i < (size_t)MTP_VERIFY_TOPK; ++i) {
            fp[i] = std::exp((top[i].val - row_max_l) * inv_t) * inv_s;
            cum += fp[i];
        }
        const bool cut_minp = (sp.min_p > 0.0f && fp[MTP_VERIFY_TOPK - 1] < fp[0] * sp.min_p);
        const bool cut_topp = (sp.top_p > 0.0f && sp.top_p < 1.0f && cum >= sp.top_p);
        if (cut_minp || cut_topp) {
            ids.resize(MTP_VERIFY_TOPK);
            probs.resize(MTP_VERIFY_TOPK);
            for (size_t i = 0; i < (size_t)MTP_VERIFY_TOPK; ++i) {
                ids[i] = top[i].id;
                probs[i] = fp[i];
            }
            exact_fast = true;
        }
    }

    if (!exact_fast) {
        // Rare exact fallback (e.g. top_k > 256, or top_k=0 with high-temp tail > 256 tokens)
        std::vector<float> full_logits;
        eng.verify_row_logits_to_host(row_idx, full_logits);
        const int V = (int)full_logits.size();
        const int k = (sp.top_k > 0 && sp.top_k < V) ? sp.top_k : V;
        ids.resize((size_t)V);
        std::iota(ids.begin(), ids.end(), 0);
        if (k < V) {
            std::nth_element(ids.begin(), ids.begin() + k, ids.end(),
                             [&](int32_t a, int32_t b) { return full_logits[(size_t)a] > full_logits[(size_t)b]; });
            ids.resize((size_t)k);
        }
        std::sort(ids.begin(), ids.end(),
                  [&](int32_t a, int32_t b) { return full_logits[(size_t)a] > full_logits[(size_t)b]; });
        const float max_l = full_logits[(size_t)ids[0]];
        probs.resize(ids.size());
        float sum = 0.0f;
        for (size_t i = 0; i < ids.size(); ++i) {
            float p = std::exp((full_logits[(size_t)ids[i]] - max_l) * inv_t);
            probs[i] = p;
            sum += p;
        }
        if (!(sum > 0.0f)) {
            if (draft_tok >= 0 && draft_tok == ids[0]) accepted = true;
            return ids[0];
        }
        const float inv_s = 1.0f / sum;
        for (float & p : probs) p *= inv_s;
    }

    // Apply Min-P and Top-P filters identically to sample_logits_ptr
    size_t keep = probs.size();
    if (sp.min_p > 0.0f) {
        const float thresh = probs[0] * sp.min_p;
        size_t cnt = 1;
        while (cnt < keep && probs[cnt] >= thresh) ++cnt;
        keep = cnt;
    }
    if (sp.top_p > 0.0f && sp.top_p < 1.0f) {
        float cum = 0.0f;
        size_t cnt = 0;
        while (cnt < keep) {
            cum += probs[cnt++];
            if (cum >= sp.top_p) break;
        }
        keep = std::max<size_t>(1, cnt);
    }

    float total = 0.0f;
    for (size_t i = 0; i < keep; ++i) total += probs[i];
    if (!(total > 0.0f)) {
        if (draft_tok >= 0 && draft_tok == ids[0]) accepted = true;
        return ids[0];
    }
    const float inv_tot = 1.0f / total;
    for (size_t i = 0; i < keep; ++i) probs[i] *= inv_tot;

    if (draft_tok >= 0) {
        int idx_d = -1;
        for (size_t i = 0; i < keep; ++i) {
            if (ids[i] == draft_tok) { idx_d = (int)i; break; }
        }
        const float p_d = (idx_d >= 0) ? probs[(size_t)idx_d] : 0.0f;
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        if (u01(rng) < p_d) {
            accepted = true;
            return draft_tok;
        }
        // Rejected: sample from exact residual p'(x) = p(x) / (1 - p_d) for x != draft_tok
        if (idx_d >= 0) probs[(size_t)idx_d] = 0.0f;
        float rem_sum = 0.0f;
        for (size_t i = 0; i < keep; ++i) rem_sum += probs[i];
        if (!(rem_sum > 0.0f)) {
            accepted = true;
            return draft_tok;
        }
        std::uniform_real_distribution<float> rdist(0.0f, rem_sum);
        const float r = rdist(rng);
        float acc = 0.0f;
        int32_t last_nonzero = ids[0];
        for (size_t i = 0; i < keep; ++i) {
            if (probs[i] > 0.0f) last_nonzero = ids[i];
            acc += probs[i];
            if (r <= acc && probs[i] > 0.0f) return ids[i];
        }
        return last_nonzero;
    }

    // Bonus token (draft_tok < 0): direct sample from p(x)
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float r = dist(rng);
    float acc = 0.0f;
    for (size_t i = 0; i < keep; ++i) {
        acc += probs[i];
        if (r <= acc) return ids[i];
    }
    return ids[keep - 1];
}

// ============================================================================
// Inkling 2 Mini Chat Template Builder (supports text + base64 images)
// ============================================================================
struct PromptSegment {
    bool is_image = false;
    std::vector<uint8_t> image_bytes;
    std::vector<int32_t> toks;   // text: pre-tokenized ids (HTTP thread)
    uint64_t hash1 = 0;          // image: 128-bit content hash of image_bytes
    uint64_t hash2 = 0;
    int n_tokens = 0;            // image: ViT token count (set once encoded)
};

// Two independent 64-bit mixes over the image bytes -> 128-bit content identity.
static void image_content_hash(const std::vector<uint8_t> & b, uint64_t & h1, uint64_t & h2) {
    uint64_t a = 1469598103934665603ULL;
    uint64_t x = 0x9E3779B97F4A7C15ULL ^ (uint64_t)b.size();
    for (uint8_t c : b) {
        a ^= c; a *= 1099511628211ULL;
        x = (x + c + 1ULL) * 0xff51afd7ed558ccdULL;
        x ^= x >> 29;
    }
    h1 = a;
    h2 = x * 0xc4ceb9fe1a85ec53ULL;
}

static std::string format_effort_str(const JsonVal * v) {
    double num = 0.9; // default: high (matches official Inkling 2 Mini template)
    if (v != nullptr) {
        if (v->is_str()) {
            std::string s = to_lower(v->str);
            if (s == "none" || s == "off") num = 0.0;
            else if (s == "minimal") num = 0.1;
            else if (s == "low")     num = 0.2;
            else if (s == "medium")  num = 0.7;
            else if (s == "high")    num = 0.9;
            else if (s == "xhigh" || s == "max") num = 0.99;
            else num = std::strtod(s.c_str(), nullptr);
        } else if (v->is_num()) {
            num = v->num;
        }
    }
    if (num < 0.0) num = 0.0;
    if (num > 0.99) num = 0.99;
    if (num == 0.0) return "0";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2g", num);
    if (num >= 0.985) return "0.99";
    return std::string(buf);
}

static bool build_inkling_chat_segments(
        const JsonVal & req_json,
        const BpeTokenizer & bpe,
        std::vector<PromptSegment> & segs,
        bool & has_images,
        bool & primed_text,
        std::string & err) {
    const JsonVal * msgs = req_json.get("messages");
    if (!msgs || !msgs->is_arr() || msgs->arr.empty()) {
        err = "Missing or empty 'messages' array";
        return false;
    }
    const JsonVal * eff_val = req_json.get("reasoning_effort");
    bool enable_thinking_off = false;
    if (!eff_val) {
        if (const JsonVal * ctk = req_json.get("chat_template_kwargs"); ctk && ctk->is_obj()) {
            if (const JsonVal * ctk_eff = ctk->get("reasoning_effort")) {
                eff_val = ctk_eff;
            } else if (const JsonVal * et = ctk->get("enable_thinking"); et && et->is_bool() && !et->b) {
                enable_thinking_off = true;
            }
        }
        if (!eff_val && !enable_thinking_off) {
            if (const JsonVal * et = req_json.get("enable_thinking"); et && et->is_bool() && !et->b) {
                enable_thinking_off = true;
            }
        }
    }
    const std::string effort_str = enable_thinking_off ? "0" : format_effort_str(eff_val);

    auto ensure_text_seg = [&]() -> std::vector<int32_t> & {
        if (segs.empty() || segs.back().is_image) {
            PromptSegment ps;
            ps.is_image = false;
            segs.push_back(std::move(ps));
        }
        return segs.back().toks;
    };
    auto push_special = [&](const std::string & tok) {
        if (tok.empty()) return;
        bpe.append_special(tok, ensure_text_seg());
    };
    auto push_plain = [&](const std::string & t) {
        if (t.empty()) return;
        bpe.append_plain(t, ensure_text_seg());
    };
    auto emit_effort_block = [&]() {
        push_special("<|message_system|>");
        push_special("<|content_text|>");
        push_plain("Thinking effort level: " + effort_str);
        push_special("<|end_message|>");
    };

    // Native Inkling 2 Mini tool declaration block (exact match to GGUF tokenizer.chat_template)
    if (const JsonVal * tools = req_json.get("tools"); tools && tools->is_arr() && !tools->arr.empty()) {
        JsonVal specs;
        specs.type = JsonVal::J_ARR;
        for (const JsonVal & tool : tools->arr) {
            if (!tool.is_obj()) continue;
            const JsonVal * fn = tool.get("function");
            if (!fn || !fn->is_obj()) fn = &tool;
            std::string fname = fn->get_str("name");
            if (fname.empty()) continue;

            JsonVal spec;
            spec.type = JsonVal::J_OBJ;
            JsonVal v_desc; v_desc.type = JsonVal::J_STR; v_desc.str = fn->get_str("description");
            JsonVal v_name; v_name.type = JsonVal::J_STR; v_name.str = fname;
            JsonVal v_type; v_type.type = JsonVal::J_STR; v_type.str = tool.get_str("type", "function");
            JsonVal v_params;
            if (const JsonVal * p = fn->get("parameters"); p && p->is_obj()) {
                v_params = *p;
            } else {
                v_params.type = JsonVal::J_OBJ;
            }
            spec.obj["description"] = std::move(v_desc);
            spec.obj["name"]        = std::move(v_name);
            spec.obj["parameters"]  = std::move(v_params);
            spec.obj["type"]        = std::move(v_type);
            specs.arr.push_back(std::move(spec));
        }
        if (!specs.arr.empty()) {
            push_special("<|message_system|>");
            push_plain("tool_declare");
            push_special("<|content_xml|>");
            push_plain(json_serialize(specs));
            push_special("<|end_message|>");
        }
    }

    auto role_tok = [](const std::string & role) -> std::string {
        if (role == "user")      return "<|message_user|>";
        if (role == "assistant") return "<|message_model|>";
        if (role == "system" || role == "developer") return "<|message_system|>";
        if (role == "tool")      return "<|message_tool|>";
        return "";
    };

    // Map tool_call_id -> function name across conversation history
    std::map<std::string, std::string> tc_id_to_name;
    for (const JsonVal & m : msgs->arr) {
        if (!m.is_obj() || m.get_str("role") != "assistant") continue;
        if (const JsonVal * tcs = m.get("tool_calls"); tcs && tcs->is_arr()) {
            for (const JsonVal & tc : tcs->arr) {
                if (!tc.is_obj()) continue;
                std::string tcid = tc.get_str("id");
                const JsonVal * fn = tc.get("function");
                if (!tcid.empty() && fn && fn->is_obj()) {
                    std::string fname = fn->get_str("name");
                    if (!fname.empty()) tc_id_to_name[tcid] = fname;
                }
            }
        }
    }

    has_images = false;
    primed_text = false;
    bool effort_emitted = false;
    for (const JsonVal & m : msgs->arr) {
        if (!m.is_obj()) continue;
        std::string role = m.get_str("role");
        if (role == "developer") role = "system";
        std::string rtok = role_tok(role);
        if (rtok.empty()) {
            err = "Unknown message role: " + role;
            return false;
        }
        if (!effort_emitted && role != "system") {
            emit_effort_block();
            effort_emitted = true;
        }
        std::string name = m.get_str("name");
        std::string role_name;
        if (role == "tool") {
            if (name.empty()) {
                std::string tcid = m.get_str("tool_call_id");
                auto it = tc_id_to_name.find(tcid);
                if (it != tc_id_to_name.end()) name = it->second;
            }
            role_name = name;
        } else if (role == "assistant" && !name.empty()) {
            role_name = name;
        }
        auto emit_role_header = [&]() {
            push_special(rtok);
            if (!role_name.empty()) push_plain(role_name);
        };

        const std::string rc = (role == "assistant") ? m.get_str("reasoning_content") : "";
        const JsonVal * tcs = (role == "assistant") ? m.get("tool_calls") : nullptr;
        const bool has_tcs = (tcs && tcs->is_arr() && !tcs->arr.empty());
        if (role == "assistant" && !rc.empty()) {
            emit_role_header();
            push_special("<|content_thinking|>");
            push_plain(rc);
            push_special("<|end_message|>");
        }

        const JsonVal * content = m.get("content");
        if (content && content->is_str()) {
            if (!content->str.empty() || !(role == "assistant" && (has_tcs || !rc.empty()))) {
                emit_role_header();
                push_special("<|content_text|>");
                push_plain(content->str);
                push_special("<|end_message|>");
            }
        } else if (content && content->is_arr()) {
            for (const JsonVal & part : content->arr) {
                if (part.is_str()) {
                    emit_role_header();
                    push_special("<|content_text|>");
                    push_plain(part.str);
                    push_special("<|end_message|>");
                    continue;
                }
                if (!part.is_obj()) continue;
                std::string ptype = part.get_str("type", "text");
                if (ptype == "text" || ptype == "input_text") {
                    emit_role_header();
                    push_special("<|content_text|>");
                    push_plain(part.get_str("text"));
                    push_special("<|end_message|>");
                } else if (ptype == "image_url" || ptype == "image" || ptype == "input_image") {
                    std::string url;
                    const JsonVal * iu = part.get("image_url");
                    if (iu && iu->is_obj()) url = iu->get_str("url");
                    else if (iu && iu->is_str()) url = iu->str;
                    else url = part.get_str("url");
                    size_t comma = url.find("base64,");
                    if (comma == std::string::npos) {
                        err = "image_url must be a base64 data URI (data:image/...;base64,...)";
                        return false;
                    }
                    std::vector<uint8_t> bytes = decode_base64(url.substr(comma + 7));
                    if (bytes.empty()) {
                        err = "Invalid base64 image payload";
                        return false;
                    }
                    emit_role_header();
                    push_special("<|content_image|>");
                    PromptSegment img;
                    img.is_image = true;
                    image_content_hash(bytes, img.hash1, img.hash2);
                    img.image_bytes = std::move(bytes);
                    segs.push_back(std::move(img));
                    push_special("<|end_message|>");
                    has_images = true;
                }
            }
        }

        if (role == "assistant" && has_tcs) {
            for (const JsonVal & tc : tcs->arr) {
                if (!tc.is_obj()) continue;
                const JsonVal * fn = tc.get("function");
                if (!fn || !fn->is_obj()) continue;
                std::string fname = fn->get_str("name");
                if (fname.empty()) continue;

                JsonVal args_obj;
                args_obj.type = JsonVal::J_OBJ;
                if (const JsonVal * a = fn->get("arguments")) {
                    if (a->is_obj()) {
                        args_obj = *a;
                    } else if (a->is_str() && !a->str.empty()) {
                        JsonVal parsed;
                        JsonParser ap{a->str, 0};
                        if (ap.parse_val(parsed) && parsed.is_obj()) {
                            args_obj = std::move(parsed);
                        }
                    }
                }
                push_special("<|message_model|>");
                push_plain(fname);
                push_special("<|content_invoke_tool_json|>");
                push_plain("{\"name\":" + json_str(fname) + ",\"args\":" + json_serialize(args_obj) + "}");
                push_special("<|end_message|>");
            }
        }

        if (role == "assistant") {
            push_special("<|content_model_end_sampling|>");
        }
    }
    if (!effort_emitted) {
        emit_effort_block();
    }
    // When reasoning_effort is "none" / 0 and no tools are declared, prime direct text output
    const JsonVal * tools = req_json.get("tools");
    const bool has_tools = (tools && tools->is_arr() && !tools->arr.empty());
    if (effort_str == "0" && !has_tools) {
        push_special("<|message_model|>");
        push_special("<|content_text|>");
        primed_text = true;
    }
    return true;
}


// ============================================================================
// Server State & Generation Engine
// ============================================================================
struct ParsedToolCall {
    std::string id;
    std::string name;
    std::string arguments; // compact JSON object string
};

struct GenResult {
    std::string content;
    std::string reasoning;
    std::vector<ParsedToolCall> tool_calls;
    std::string finish_reason = "length";
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int cache_hit_tokens = 0;
    int slot = 0;
    int slot_span = 1;
    double prompt_ms = 0.0;
    double predicted_ms = 0.0;
    bool mtp_used = false;
    double mtp_tok_per_round = 0.0;
    double mtp_avg_accepted = 0.0;
    double mtp_draft_ms = 0.0;
    double mtp_verify_ms = 0.0;
};

// One segment of a slot's cached multimodal prologue (see slot_mm_prologue_).
struct SlotCacheSeg {
    bool is_image = false;
    int ntok = 0;                 // image: ViT token count
    uint64_t hash1 = 0;           // image: 128-bit content hash
    uint64_t hash2 = 0;
    std::vector<int32_t> toks;    // text: token ids
};

struct ActivityRecord {
    uint64_t id = 0;
    long timestamp = 0;
    std::string model;
    std::string finish_reason;
    bool stream = false;
    bool mtp_used = false;
    bool has_image = false;
    int slot = 0;
    int slot_span = 1;
    int prompt_tokens = 0;
    int cached_tokens = 0;
    int completion_tokens = 0;
    double prompt_ms = 0.0;
    double decode_ms = 0.0;
    double prompt_tps = 0.0;
    double decode_tps = 0.0;
    double mtp_tok_per_round = 0.0;
    double mtp_draft_ms = 0.0;
    double mtp_verify_ms = 0.0;
};

struct LogEntry {
    std::string ts;
    std::string level;
    std::string msg;
};

enum JobState {
    JOB_QUEUED     = 0,
    JOB_PREFILLING = 1,
    JOB_DECODING   = 2,
    JOB_DONE       = 3,
};

struct StreamJob {
    // Immutable request inputs (populated by HTTP thread before enqueue)
    std::string req_id;
    bool is_chat = true;
    bool stream = false;
    bool has_images = false;
    bool primed_text = false;
    bool allow_mtp = true;
    bool need_logits = false;
    int max_tokens = 1024;
    int est_prompt_toks = 0; // exact for text-only; pre-estimated for multimodal
    SampleParams sp;
    std::mt19937_64 rng;
    std::vector<int32_t> prompt_toks;       // populated for text-only
    std::vector<PromptSegment> segs;        // populated for multimodal
    std::chrono::steady_clock::time_point t_enqueued;

    // Live telemetry atomics (read lock-free by /health)
    std::atomic<int>  live_state{JOB_QUEUED};
    std::atomic<int>  live_slot{-1};
    std::atomic<int>  live_span{1};
    std::atomic<int>  live_ctx_limit{0};
    std::atomic<bool> live_waiting_expand{false};
    std::atomic<int>  live_prompt_toks{0};
    std::atomic<int>  live_gen_toks{0};
    std::atomic<bool> live_mtp{false};

    // Cancellation flag (written by HTTP thread on disconnect or stop sequence)
    std::atomic<bool> cancelled{false};

    // Scheduler-owned fields (touched ONLY by GPU scheduler thread)
    JobState state = JOB_QUEUED;
    int slot = -1;
    int span = 1;
    bool waiting_expand = false;
    int32_t expand_needed_pos = 0;
    int prefill_offset = 0;
    size_t seg_idx = 0;
    int seg_offset = 0;
    std::vector<int32_t> cur_seg_toks;
    VisionEncodeResult cur_vis;
    bool cur_vis_loaded = false;
    uint64_t cur_vis_gen = 0;
    int32_t cur_tok = -1;
    int sched_prompt_tokens = 0;
    int sched_cache_hit_tokens = 0;
    int sched_completion_tokens = 0;
    std::chrono::steady_clock::time_point t_prefill_start;
    std::chrono::steady_clock::time_point t_decode_start;
    double sched_prompt_ms = 0.0;
    bool sched_mtp_used = false;
    long mtp_rounds = 0;
    long mtp_accepted = 0;
    double mtp_sum_draft = 0.0;
    double mtp_sum_verify = 0.0;

    // Producer-consumer handoff (GPU scheduler -> HTTP thread)
    std::mutex q_mu;
    std::condition_variable q_cv;
    std::deque<int32_t> out_toks;
    bool prefill_ready = false;
    bool done = false;
    bool failed = false;
    int http_err_code = 500;
    std::string http_err_msg;
    std::string http_err_raw_json;
    std::string finish_reason = "length";
    int final_prompt_tokens = 0;
    int final_cache_hit_tokens = 0;
    int final_slot = 0;
    int final_slot_span = 1;
    double final_prompt_ms = 0.0;
    double final_predicted_ms = 0.0;
    bool final_mtp_used = false;
    double final_mtp_tok_per_round = 0.0;
    double final_mtp_avg_accepted = 0.0;
    double final_mtp_draft_ms = 0.0;
    double final_mtp_verify_ms = 0.0;
};

class InklingServer {
public:
    InklingServer(InklingEngine & eng, const BpeTokenizer & bpe,
                  DSparkDrafter * drafter, VisionEncoder * vision,
                  const ServerConfig & cfg)
        : eng_(eng), bpe_(bpe), drafter_(drafter), vision_(vision), cfg_(cfg),
          start_time_(std::chrono::steady_clock::now()) {
        for (int c = 0; c < MAX_CONCURRENT_SLOTS; ++c) {
            seg_owner_[c] = -1;
            slot_span_[c] = 0;
            slot_span_pub_[c] = 0;
            slot_cache_span_[c] = 0;
            slot_last_used_[c] = 0;
        }
        id_model_            = bpe_.token_id("<|message_model|>");
        id_thinking_         = bpe_.token_id("<|content_thinking|>");
        id_text_             = bpe_.token_id("<|content_text|>");
        id_end_msg_          = bpe_.token_id("<|end_message|>");
        id_end_sampling_     = bpe_.token_id("<|content_model_end_sampling|>");
        id_eot_              = bpe_.token_id("<|endoftext|>");
        id_invoke_tool_json_ = bpe_.token_id("<|content_invoke_tool_json|>");
        id_invoke_tool_text_ = bpe_.token_id("<|content_invoke_tool_text|>");
        char buf[384];
        std::snprintf(buf, sizeof(buf),
                      "Server initialized: model=%s ctx=%d req_max_ctx=%d slots=%d(%d/slot%s) kv=%s sparse_pages=%d mtp=%s(ctx=%d) vision=%s vram=%.2f GiB",
                      cfg_.model_name.c_str(), eng_.max_ctx(), request_max_ctx(), eng_.num_slots(), eng_.slot_ctx(),
                      cfg_.strict_slot_ctx ? ",strict" : ",elastic",
                      cfg_.kv_type_name.c_str(), cfg_.sparse_pages, drafter_ ? "ON" : "OFF",
                      drafter_ ? drafter_->ctx() : 0, vision_ ? "ON" : "OFF", total_vram_gib());
        add_log("INFO", buf);
    }

    ~InklingServer() {
        stop_scheduler();
    }

    void warmup() {
        std::vector<int32_t> warm = { 1, 100, 200, 300 };
        std::vector<float> logits;
        (void)eng_.prefill(warm, false, &logits);
        (void)eng_.decode_step(42, nullptr);
        if (drafter_ != nullptr) {
            int32_t dummy[8] = { 10, 11, 12, 13, 14, 15, 16, 17 };
            for (int m = 2; m <= 8; ++m) {
                const int P = eng_.current_pos();
                (void)eng_.verify_step(dummy, m, P);
                eng_.verify_rollback(P, 1, m);
            }
            (void)drafter_->draft(42, eng_.current_pos());
            (void)hipStreamSynchronize(eng_.stream());
        }
        if (eng_.num_slots() > 1) {
            (void)eng_.prefill_slot_chunk(1, warm.data(), (int)warm.size(), false, nullptr);
            int32_t bslots[2] = { 0, 1 };
            int32_t btoks[2]  = { 42, 43 };
            int32_t bargs[2]  = { 0, 0 };
            (void)eng_.decode_batch(bslots, btoks, 2, bargs, nullptr);
            const int mid = (eng_.num_slots() + 1) / 2;
            if (mid >= 2 && mid < eng_.num_slots()) {
                (void)eng_.prefill_slot_chunk(mid, warm.data(), (int)warm.size(), false, nullptr);
                int32_t bslots_mid[2] = { 0, mid };
                (void)eng_.decode_batch(bslots_mid, btoks, 2, bargs, nullptr);
            }
        }
        reset_all_gpu();
        add_log("INFO", "HIP graphs and multi-slot batch kernels warmed up");
    }

    void start_scheduler() {
        if (sched_running_) return;
        sched_stop_.store(false);
        sched_running_ = true;
        sched_thread_ = std::thread([this]() { scheduler_loop(); });
    }

    void stop_scheduler() {
        if (!sched_running_) return;
        {
            std::lock_guard<std::mutex> lk(sched_mu_);
            sched_stop_.store(true);
        }
        sched_cv_.notify_all();
        if (sched_thread_.joinable()) sched_thread_.join();
        sched_running_ = false;
    }

    void handle_client(int fd) {
        int one = 1;
        (void)::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

        HttpRequest req;
        if (!read_http_request(fd, req)) {
            ::close(fd);
            return;
        }
        if (req.method == "OPTIONS") {
            send_http_response(fd, 204, "No Content", "text/plain", "");
            ::close(fd);
            return;
        }
        if (req.method == "GET" &&
            (req.path == "/" || req.path == "/index.html" ||
             req.path == "/ui" || req.path.rfind("/ui/", 0) == 0)) {
            send_http_response(fd, 200, "OK", "text/html; charset=utf-8", kIndexHtml);
            ::close(fd);
            return;
        }
        if (req.method == "GET" &&
            (req.path == "/logo.svg" || req.path == "/nib-logo.svg")) {
            send_http_response(fd, 200, "OK", "image/svg+xml; charset=utf-8", kNibLogoSvg);
            ::close(fd);
            return;
        }
        if (req.method == "GET" &&
            (req.path == "/favicon.svg" || req.path == "/favicon.ico")) {
            send_http_response(fd, 200, "OK", "image/svg+xml; charset=utf-8", kNibFaviconSvg);
            ::close(fd);
            return;
        }
        if (req.method == "GET" && req.path == "/health") {
            const double llm_vram = (double)eng_.total_vram_bytes() / (1024.0 * 1024.0 * 1024.0);
            const double mtp_vram = drafter_ ? ((double)drafter_->vram_bytes() / (1024.0 * 1024.0 * 1024.0)) : 0.0;
            const double vis_vram = vision_ ? ((double)vision_->vram_bytes() / (1024.0 * 1024.0 * 1024.0)) : 0.0;
            const double vram = llm_vram + mtp_vram + vis_vram;
            size_t dev_free_b = 0, dev_total_b = 0;
            const bool has_dev_mem = (hipMemGetInfo(&dev_free_b, &dev_total_b) == hipSuccess && dev_total_b > 0);
            const double dev_free_gib  = has_dev_mem ? ((double)dev_free_b  / (1024.0 * 1024.0 * 1024.0)) : 0.0;
            const double dev_total_gib = has_dev_mem ? ((double)dev_total_b / (1024.0 * 1024.0 * 1024.0)) : 0.0;
            const int pos = eng_.slot_pos(0);
            const double uptime = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_time_).count();
            uint64_t tot_req = 0, tot_p = 0, tot_c = 0;
            {
                std::lock_guard<std::mutex> tlk(telemetry_mu_);
                tot_req = total_requests_;
                tot_p   = total_prompt_tokens_;
                tot_c   = total_completion_tokens_;
            }

            // Snapshot live in-flight + queued jobs and segment usage
            std::vector<std::shared_ptr<StreamJob>> live_jobs;
            int n_active = 0;
            int n_queued = 0;
            int used_segments = 0;
            int snap_spans[MAX_CONCURRENT_SLOTS] = {};
            {
                std::lock_guard<std::mutex> slk(sched_mu_);
                for (int c = 0; c < eng_.num_slots(); ++c) {
                    snap_spans[c] = slot_span_pub_[c];
                    if (active_slots_pub_[c] != nullptr) {
                        live_jobs.push_back(active_slots_pub_[c]);
                        ++n_active;
                        used_segments += std::max(1, slot_span_pub_[c]);
                    }
                }
                n_queued = (int)pending_.size();
                for (const auto & pj : pending_) {
                    live_jobs.push_back(pj);
                }
            }

#ifndef NIB_VERSION
#define NIB_VERSION "0.2.0"
#endif
            const NibDeviceInfo & dev_info = nib_device_info(0);
            const auto now = std::chrono::steady_clock::now();
            std::ostringstream oss;
            oss << "{\"status\":\"ok\""
                << ",\"version\":" << json_str(NIB_VERSION)
                << ",\"gpu_name\":" << json_str(dev_info.name)
                << ",\"gpu_arch\":" << json_str(dev_info.gcn_arch)
                << ",\"gpu_family\":" << json_str(dev_info.arch_family)
                << ",\"gpu_mps\":" << dev_info.mp_count
                << ",\"model\":" << json_str(cfg_.model_name)
                << ",\"max_ctx\":" << eng_.max_ctx()
                << ",\"request_max_ctx\":" << request_max_ctx()
                << ",\"strict_slot_ctx\":" << (cfg_.strict_slot_ctx ? "true" : "false")
                << ",\"max_slots\":" << eng_.num_slots()
                << ",\"slot_ctx\":" << eng_.slot_ctx()
                << ",\"used_segments\":" << used_segments
                << ",\"slot_spans\":[";
            for (int c = 0; c < eng_.num_slots(); ++c) {
                if (c > 0) oss << ",";
                oss << snap_spans[c];
            }
            oss << "]"
                << ",\"cur_pos\":" << pos
                << ",\"kv_type\":" << json_str(cfg_.kv_type_name)
                << ",\"sparse_pages\":" << cfg_.sparse_pages
                << ",\"mtp\":" << (drafter_ ? "true" : "false")
                << ",\"draft_ctx\":" << (drafter_ ? drafter_->ctx() : 0)
                << ",\"spec_threshold\":" << cfg_.spec_threshold
                << ",\"default_temperature\":" << cfg_.default_temperature
                << ",\"default_top_p\":" << cfg_.default_top_p
                << ",\"default_top_k\":" << cfg_.default_top_k
                << ",\"default_min_p\":" << cfg_.default_min_p
                << ",\"vision\":" << (vision_ ? "true" : "false")
                << ",\"vram_gib\":" << vram
                << ",\"llm_vram_gib\":" << llm_vram
                << ",\"mtp_vram_gib\":" << mtp_vram
                << ",\"vision_vram_gib\":" << vis_vram
                << ",\"device_free_vram_gib\":" << dev_free_gib
                << ",\"device_total_vram_gib\":" << dev_total_gib
                << ",\"cpu_embd\":" << (eng_.cpu_embd() ? "true" : "false")
                << ",\"cpu_embd_host_mib\":" << ((double)eng_.host_pinned_bytes() / (1024.0 * 1024.0))
                << ",\"vision_workspace_mib\":" << (vision_ ? (double)vision_->workspace_bytes() / (1024.0 * 1024.0) : 0.0)
                << ",\"vision_workspace_peak_mib\":" << (vision_ ? (double)vision_->peak_workspace_bytes() / (1024.0 * 1024.0) : 0.0)
                << ",\"uptime_s\":" << (long)uptime
                << ",\"busy\":" << ((n_active > 0 || n_queued > 0) ? "true" : "false")
                << ",\"active_requests\":" << n_active
                << ",\"queued_requests\":" << n_queued
                << ",\"in_flight\":[";
            for (size_t i = 0; i < live_jobs.size(); ++i) {
                const auto & j = live_jobs[i];
                const int st = j->live_state.load(std::memory_order_relaxed);
                const char * st_str = (st == JOB_PREFILLING) ? "prefilling"
                                    : (st == JOB_DECODING)   ? "decoding"
                                    : (st == JOB_DONE)       ? "done"
                                                             : "queued";
                const double elapsed = std::chrono::duration<double, std::milli>(now - j->t_enqueued).count();
                if (i > 0) oss << ",";
                oss << "{\"id\":" << json_str(j->req_id)
                    << ",\"slot\":" << j->live_slot.load(std::memory_order_relaxed)
                    << ",\"span\":" << j->live_span.load(std::memory_order_relaxed)
                    << ",\"ctx_limit\":" << j->live_ctx_limit.load(std::memory_order_relaxed)
                    << ",\"waiting_expand\":" << (j->live_waiting_expand.load(std::memory_order_relaxed) ? "true" : "false")
                    << ",\"state\":" << json_str(st_str)
                    << ",\"stream\":" << (j->stream ? "true" : "false")
                    << ",\"mtp\":" << (j->live_mtp.load(std::memory_order_relaxed) ? "true" : "false")
                    << ",\"prompt_tokens\":" << j->live_prompt_toks.load(std::memory_order_relaxed)
                    << ",\"completion_tokens\":" << j->live_gen_toks.load(std::memory_order_relaxed)
                    << ",\"max_tokens\":" << j->max_tokens
                    << ",\"elapsed_ms\":" << elapsed << "}";
            }
            oss << "]"
                << ",\"total_requests\":" << tot_req
                << ",\"total_prompt_tokens\":" << tot_p
                << ",\"total_completion_tokens\":" << tot_c << "}";
            send_http_response(fd, 200, "OK", "application/json", oss.str());
            ::close(fd);
            return;
        }
        if (req.method == "GET" && req.path == "/v1/models") {
            const NibDeviceInfo & dev_info = nib_device_info(0);
            const int32_t req_ctx = request_max_ctx();
            char name_buf[192];
            if (eng_.num_slots() > 1) {
                std::snprintf(name_buf, sizeof(name_buf),
                              "Inkling 2 Mini NVFP4-Mixed (%s HIP, %dk pool, %dx%dk %s%s%s)",
                              dev_info.arch_family,
                              eng_.max_ctx() / 1024, eng_.num_slots(), eng_.slot_ctx() / 1024,
                              cfg_.strict_slot_ctx ? "strict" : "elastic",
                              drafter_ ? ", MTP" : "", vision_ ? ", Vision" : "");
            } else {
                std::snprintf(name_buf, sizeof(name_buf),
                              "Inkling 2 Mini NVFP4-Mixed (%s HIP, %dk%s%s)",
                              dev_info.arch_family,
                              eng_.max_ctx() / 1024,
                              drafter_ ? ", MTP" : "", vision_ ? ", Vision" : "");
            }
            std::ostringstream oss;
            oss << "{\"object\":\"list\",\"data\":[{\"id\":" << json_str(cfg_.model_name)
                << ",\"object\":\"model\",\"created\":" << (long)std::time(nullptr)
                << ",\"owned_by\":\"nib\""
                << ",\"name\":" << json_str(name_buf)
                << ",\"context_length\":" << req_ctx
                << ",\"max_model_len\":" << req_ctx
                << ",\"max_ctx\":" << eng_.max_ctx()
                << ",\"slot_ctx\":" << eng_.slot_ctx()
                << ",\"parallel_slots\":" << eng_.num_slots()
                << ",\"elastic_long_context\":" << ((!cfg_.strict_slot_ctx && eng_.num_slots() > 1) ? "true" : "false")
                << ",\"architecture\":{\"input_modalities\":"
                << (vision_ ? "[\"text\",\"image\"]" : "[\"text\"]")
                << ",\"output_modalities\":[\"text\"]}"
                << ",\"capabilities\":{\"vision\":" << (vision_ ? "true" : "false")
                << ",\"mtp\":" << (drafter_ ? "true" : "false")
                << ",\"slots\":" << eng_.num_slots()
                << ",\"slot_ctx\":" << eng_.slot_ctx()
                << ",\"max_ctx\":" << eng_.max_ctx() << "}"
                << ",\"status\":{\"value\":" << (busy_.load() ? "\"busy\"" : "\"ready\"") << "}}]}";
            send_http_response(fd, 200, "OK", "application/json", oss.str());
            ::close(fd);
            return;
        }
        if (req.method == "GET" && req.path == "/api/activity") {
            std::ostringstream oss;
            oss << "{\"items\":[";
            std::lock_guard<std::mutex> tlk(telemetry_mu_);
            for (size_t i = 0; i < activity_.size(); ++i) {
                const auto & a = activity_[i];
                if (i > 0) oss << ",";
                oss << "{\"id\":" << a.id
                    << ",\"timestamp\":" << a.timestamp
                    << ",\"model\":" << json_str(a.model)
                    << ",\"slot\":" << a.slot
                    << ",\"slot_span\":" << a.slot_span
                    << ",\"stream\":" << (a.stream ? "true" : "false")
                    << ",\"mtp\":" << (a.mtp_used ? "true" : "false")
                    << ",\"has_image\":" << (a.has_image ? "true" : "false")
                    << ",\"prompt_tokens\":" << a.prompt_tokens
                    << ",\"cached_tokens\":" << a.cached_tokens
                    << ",\"completion_tokens\":" << a.completion_tokens
                    << ",\"prompt_ms\":" << a.prompt_ms
                    << ",\"decode_ms\":" << a.decode_ms
                    << ",\"prompt_tps\":" << a.prompt_tps
                    << ",\"decode_tps\":" << a.decode_tps
                    << ",\"mtp_tok_per_round\":" << a.mtp_tok_per_round
                    << ",\"mtp_draft_ms\":" << a.mtp_draft_ms
                    << ",\"mtp_verify_ms\":" << a.mtp_verify_ms
                    << ",\"finish_reason\":" << json_str(a.finish_reason) << "}";
            }
            oss << "]}";
            send_http_response(fd, 200, "OK", "application/json", oss.str());
            ::close(fd);
            return;
        }
        if (req.method == "GET" && req.path == "/api/logs") {
            std::ostringstream oss;
            oss << "{\"lines\":[";
            std::lock_guard<std::mutex> tlk(telemetry_mu_);
            for (size_t i = 0; i < logs_.size(); ++i) {
                const auto & l = logs_[i];
                if (i > 0) oss << ",";
                oss << "{\"ts\":" << json_str(l.ts)
                    << ",\"level\":" << json_str(l.level)
                    << ",\"msg\":" << json_str(l.msg) << "}";
            }
            oss << "]}";
            send_http_response(fd, 200, "OK", "application/json", oss.str());
            ::close(fd);
            return;
        }
        if (req.method == "POST" && (req.path == "/api/reset-kv" || req.path == "/reset")) {
            request_gpu_reset();
            add_log("INFO", "All slot KV caches and DSpark drafter ring reset (pos=0)");
            send_http_response(fd, 200, "OK", "application/json", "{\"ok\":true,\"cur_pos\":0}");
            ::close(fd);
            return;
        }
        if (req.method == "POST" && req.path == "/api/clear-activity") {
            std::lock_guard<std::mutex> tlk(telemetry_mu_);
            activity_.clear();
            send_http_response(fd, 200, "OK", "application/json", "{\"ok\":true}");
            ::close(fd);
            return;
        }
        if (req.method == "POST" && req.path == "/api/clear-logs") {
            std::lock_guard<std::mutex> tlk(telemetry_mu_);
            logs_.clear();
            send_http_response(fd, 200, "OK", "application/json", "{\"ok\":true}");
            ::close(fd);
            return;
        }
        if (req.method == "GET" && req.path == "/api/outputs") {
            namespace fs = std::filesystem;
            std::error_code ec;
            fs::path dir = fs::weakly_canonical(fs::absolute(cfg_.output_dir.empty() ? "outputs" : cfg_.output_dir, ec), ec);
            struct Entry { std::string name; uint64_t bytes; long long mtime; };
            std::vector<Entry> files;
            if (!ec && fs::exists(dir, ec) && fs::is_directory(dir, ec)) {
                for (const auto & de : fs::directory_iterator(dir, ec)) {
                    if (ec) break;
                    std::error_code fec;
                    if (!de.is_regular_file(fec)) continue;
                    std::string fname = de.path().filename().string();
                    if (!is_safe_output_filename(fname)) continue;
                    uint64_t sz = de.file_size(fec);
                    auto ftime = de.last_write_time(fec);
                    auto sctp = std::chrono::time_point_cast<std::chrono::seconds>(
                        ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
                    files.push_back({fname, sz, (long long)sctp.time_since_epoch().count()});
                }
            }
            std::sort(files.begin(), files.end(), [](const Entry & a, const Entry & b) {
                if (a.mtime != b.mtime) return a.mtime > b.mtime;
                return a.name < b.name;
            });
            const bool allow_home = is_loopback_host(cfg_.host);
            const char * home_env = std::getenv("HOME");
            std::string home_str = (home_env && *home_env) ? std::string(home_env) : "";
            std::vector<std::pair<std::string, std::string>> presets;
            presets.push_back({"Nib outputs", dir.string()});
            if (allow_home && !home_str.empty()) {
                fs::path parent_demos = dir.parent_path().parent_path() / "demos";
                if (fs::exists(parent_demos, ec) && fs::is_directory(parent_demos, ec)) {
                    presets.push_back({"demos", parent_demos.string()});
                }
                fs::path home_p(home_str);
                presets.push_back({"Home (~)", home_p.string()});
                if (fs::exists(home_p / "Desktop", ec))   presets.push_back({"Desktop", (home_p / "Desktop").string()});
                if (fs::exists(home_p / "Downloads", ec)) presets.push_back({"Downloads", (home_p / "Downloads").string()});
            }
            std::ostringstream oss;
            oss << "{\"dir\":" << json_str(dir.string())
                << ",\"home_dir\":" << json_str(home_str)
                << ",\"allow_custom_dir\":" << (allow_home ? "true" : "false")
                << ",\"presets\":[";
            for (size_t i = 0; i < presets.size(); ++i) {
                if (i > 0) oss << ",";
                oss << "{\"label\":" << json_str(presets[i].first)
                    << ",\"path\":" << json_str(presets[i].second) << "}";
            }
            oss << "],\"files\":[";
            for (size_t i = 0; i < files.size(); ++i) {
                if (i > 0) oss << ",";
                oss << "{\"filename\":" << json_str(files[i].name)
                    << ",\"bytes\":" << files[i].bytes
                    << ",\"mtime\":" << files[i].mtime
                    << ",\"url\":" << json_str("/outputs/" + files[i].name) << "}";
            }
            oss << "]}";
            send_http_response(fd, 200, "OK", "application/json", oss.str());
            ::close(fd);
            return;
        }
        if (req.method == "POST" && req.path == "/api/outputs") {
            namespace fs = std::filesystem;
            JsonVal j;
            JsonParser jp{req.body, 0};
            if (!jp.parse_val(j) || !j.is_obj()) {
                send_http_response(fd, 400, "Bad Request", "application/json",
                                   "{\"error\":{\"message\":\"Invalid JSON body\"}}");
                ::close(fd);
                return;
            }
            const JsonVal * cv = j.get("content");
            if (!cv || !cv->is_str()) {
                send_http_response(fd, 400, "Bad Request", "application/json",
                                   "{\"error\":{\"message\":\"Missing string field 'content'\"}}");
                ::close(fd);
                return;
            }
            std::string fname = sanitize_output_filename(j.get_str("filename", ""));
            std::error_code ec;
            fs::path def_dir = fs::weakly_canonical(fs::absolute(cfg_.output_dir.empty() ? "outputs" : cfg_.output_dir, ec), ec);
            fs::path target_dir;
            std::string dir_err;
            const bool allow_home = is_loopback_host(cfg_.host);
            if (!resolve_safe_save_dir(j.get_str("dir", ""), def_dir, allow_home, target_dir, dir_err)) {
                send_http_response(fd, 403, "Forbidden", "application/json",
                                   "{\"error\":{\"message\":" + json_str(dir_err) + "}}");
                ::close(fd);
                return;
            }
            fs::create_directories(target_dir, ec);
            if (ec) {
                send_http_response(fd, 500, "Internal Server Error", "application/json",
                                   "{\"error\":{\"message\":\"Failed to create output directory\"}}");
                ::close(fd);
                return;
            }
            fs::path fpath = target_dir / fname;
            {
                std::ofstream ofs(fpath, std::ios::binary | std::ios::trunc);
                if (!ofs) {
                    send_http_response(fd, 500, "Internal Server Error", "application/json",
                                       "{\"error\":{\"message\":\"Failed to write output file\"}}");
                    ::close(fd);
                    return;
                }
                ofs.write(cv->str.data(), (std::streamsize)cv->str.size());
                if (!ofs) {
                    send_http_response(fd, 500, "Internal Server Error", "application/json",
                                       "{\"error\":{\"message\":\"Failed while writing output file\"}}");
                    ::close(fd);
                    return;
                }
            }
            std::string serve_url;
            if (target_dir == def_dir) {
                serve_url = "/outputs/" + fname;
            } else if (j.get_bool("also_outputs", false)) {
                std::error_code ec2;
                fs::create_directories(def_dir, ec2);
                std::ofstream ofs2(def_dir / fname, std::ios::binary | std::ios::trunc);
                if (ofs2) {
                    ofs2.write(cv->str.data(), (std::streamsize)cv->str.size());
                    if (ofs2) serve_url = "/outputs/" + fname;
                }
            }
            add_log("INFO", "Saved output file: " + fpath.string() + " (" + std::to_string(cv->str.size()) + " bytes)");
            std::ostringstream oss;
            oss << "{\"ok\":true,\"filename\":" << json_str(fname)
                << ",\"dir\":" << json_str(target_dir.string())
                << ",\"path\":" << json_str(fpath.string())
                << ",\"url\":" << json_str(serve_url)
                << ",\"bytes\":" << cv->str.size() << "}";
            send_http_response(fd, 200, "OK", "application/json", oss.str());
            ::close(fd);
            return;
        }
        if (req.method == "DELETE" && req.path.rfind("/api/outputs/", 0) == 0) {
            namespace fs = std::filesystem;
            std::string fname = url_decode_segment(req.path.substr(13));
            if (!is_safe_output_filename(fname)) {
                send_http_response(fd, 400, "Bad Request", "application/json",
                                   "{\"error\":{\"message\":\"Invalid filename\"}}");
                ::close(fd);
                return;
            }
            std::error_code ec;
            fs::path dir = fs::absolute(cfg_.output_dir.empty() ? "outputs" : cfg_.output_dir, ec);
            fs::path fpath = dir / fname;
            bool removed = fs::remove(fpath, ec);
            if (!removed || ec) {
                send_http_response(fd, 404, "Not Found", "application/json",
                                   "{\"error\":{\"message\":\"File not found\"}}");
                ::close(fd);
                return;
            }
            add_log("INFO", "Deleted output file: " + fname);
            send_http_response(fd, 200, "OK", "application/json", "{\"ok\":true}");
            ::close(fd);
            return;
        }
        if (req.method == "GET" && req.path.rfind("/outputs/", 0) == 0) {
            namespace fs = std::filesystem;
            std::string fname = url_decode_segment(req.path.substr(9));
            if (!is_safe_output_filename(fname)) {
                send_http_response(fd, 400, "Bad Request", "application/json",
                                   "{\"error\":{\"message\":\"Invalid filename\"}}");
                ::close(fd);
                return;
            }
            std::error_code ec;
            fs::path dir = fs::absolute(cfg_.output_dir.empty() ? "outputs" : cfg_.output_dir, ec);
            fs::path fpath = dir / fname;
            if (ec || !fs::exists(fpath, ec) || !fs::is_regular_file(fpath, ec)) {
                send_http_response(fd, 404, "Not Found", "application/json",
                                   "{\"error\":{\"message\":\"File not found\"}}");
                ::close(fd);
                return;
            }
            std::ifstream ifs(fpath, std::ios::binary);
            if (!ifs) {
                send_http_response(fd, 500, "Internal Server Error", "application/json",
                                   "{\"error\":{\"message\":\"Failed to read file\"}}");
                ::close(fd);
                return;
            }
            std::ostringstream ss;
            ss << ifs.rdbuf();
            send_http_response(fd, 200, "OK", content_type_for_filename(fname), ss.str());
            ::close(fd);
            return;
        }
        if (req.method == "POST" && (req.path == "/v1/chat/completions" || req.path == "/v1/completions")) {
            handle_generate(fd, req, req.path == "/v1/chat/completions");
            ::close(fd);
            return;
        }
        send_http_response(fd, 404, "Not Found", "application/json",
                           "{\"error\":{\"message\":\"Not found\",\"type\":\"invalid_request_error\"}}");
        ::close(fd);
    }

private:
    void add_log(const char * level, const std::string & msg) {
        std::time_t now = std::time(nullptr);
        std::tm tm_buf {};
        localtime_r(&now, &tm_buf);
        char ts[32];
        std::strftime(ts, sizeof(ts), "%H:%M:%S", &tm_buf);
        std::lock_guard<std::mutex> tlk(telemetry_mu_);
        logs_.push_back(LogEntry{ts, level, msg});
        while (logs_.size() > 500) logs_.pop_front();
    }

    void record_activity(const GenResult & gr, bool stream, bool has_image) {
        ActivityRecord rec;
        rec.timestamp = (long)std::time(nullptr);
        rec.model = cfg_.model_name;
        rec.finish_reason = gr.finish_reason;
        rec.stream = stream;
        rec.mtp_used = gr.mtp_used;
        rec.has_image = has_image;
        rec.slot = gr.slot;
        rec.slot_span = gr.slot_span;
        rec.prompt_tokens = gr.prompt_tokens;
        rec.cached_tokens = gr.cache_hit_tokens;
        rec.completion_tokens = gr.completion_tokens;
        rec.prompt_ms = gr.prompt_ms;
        rec.decode_ms = gr.predicted_ms;
        const int prefilled_tokens = std::max(0, gr.prompt_tokens - gr.cache_hit_tokens);
        rec.prompt_tps = (gr.prompt_ms > 0.0 && prefilled_tokens > 0)
            ? ((double)prefilled_tokens / (gr.prompt_ms / 1000.0)) : 0.0;
        rec.decode_tps = (gr.predicted_ms > 0.0) ? ((double)gr.completion_tokens / (gr.predicted_ms / 1000.0)) : 0.0;
        rec.mtp_tok_per_round = gr.mtp_tok_per_round;
        rec.mtp_draft_ms = gr.mtp_draft_ms;
        rec.mtp_verify_ms = gr.mtp_verify_ms;
        std::lock_guard<std::mutex> tlk(telemetry_mu_);
        rec.id = ++total_requests_;
        total_prompt_tokens_ += (uint64_t)std::max(0, gr.prompt_tokens);
        total_completion_tokens_ += (uint64_t)std::max(0, gr.completion_tokens);
        activity_.push_back(std::move(rec));
        while (activity_.size() > 200) activity_.pop_front();
    }

    double total_vram_gib() const {
        size_t bytes = eng_.total_vram_bytes();
        if (drafter_) bytes += drafter_->vram_bytes();
        if (vision_)  bytes += vision_->vram_bytes();
        return (double)bytes / (1024.0 * 1024.0 * 1024.0);
    }

    int32_t request_max_ctx() const {
        int32_t lim = eng_.max_ctx();
        if (cfg_.strict_slot_ctx && eng_.num_slots() > 1) {
            lim = std::min(lim, eng_.slot_ctx());
        }
        if (cfg_.max_slot_ctx > 0) {
            lim = std::min(lim, cfg_.max_slot_ctx);
        }
        return lim;
    }

    static std::string context_overflow_error_json(int32_t max_ctx, int32_t prompt_toks) {
        std::ostringstream oss;
        oss << "{\"error\":{\"message\":\"This model's maximum context length is " << max_ctx
            << " tokens. However, your messages resulted in " << prompt_toks
            << " tokens. Please reduce the length of the messages.\""
            << ",\"type\":\"invalid_request_error\""
            << ",\"param\":\"messages\""
            << ",\"code\":\"context_length_exceeded\"}}";
        return oss.str();
    }

    void reset_all_gpu() {
        eng_.reset_state();
        if (drafter_) drafter_->reset();
        drafter_in_sync_ = (drafter_ != nullptr);
        clear_slot0_checkpoint();
        for (int c = 0; c < MAX_CONCURRENT_SLOTS; ++c) {
            clear_slot_cache(c);
            slot_last_used_[c] = 0;
            seg_owner_[c] = -1;
            slot_span_[c] = 0;
            slot_span_pub_[c] = 0;
        }
    }

    void request_gpu_reset() {
        std::unique_lock<std::mutex> lk(sched_mu_);
        const uint64_t target = reset_gen_ + 1;
        reset_requested_.store(true);
        sched_cv_.notify_all();
        reset_cv_.wait(lk, [&]() {
            return reset_gen_ >= target || sched_stop_.load();
        });
    }

    static std::string make_id() {
        static std::atomic<uint64_t> seq{1};
        char buf[64];
        std::snprintf(buf, sizeof(buf), "chatcmpl-nib-%llu",
                      (unsigned long long)seq.fetch_add(1));
        return buf;
    }

    static std::string timings_json(const GenResult & gr) {
        const int prefilled_tokens = std::max(0, gr.prompt_tokens - gr.cache_hit_tokens);
        const double pps = (gr.prompt_ms > 0.0 && prefilled_tokens > 0)
            ? ((double)prefilled_tokens / (gr.prompt_ms / 1000.0)) : 0.0;
        const double tps = (gr.predicted_ms > 0.0) ? ((double)gr.completion_tokens / (gr.predicted_ms / 1000.0)) : 0.0;
        char buf[544];
        std::snprintf(buf, sizeof(buf),
            "{\"prompt_ms\":%.2f,\"prompt_per_second\":%.2f,\"predicted_ms\":%.2f,"
            "\"predicted_per_second\":%.2f,\"cache_hit_tokens\":%d,\"slot\":%d,\"slot_span\":%d,"
            "\"mtp_used\":%s,\"mtp_tok_per_round\":%.3f,\"mtp_avg_accepted\":%.3f,"
            "\"mtp_draft_ms\":%.3f,\"mtp_verify_ms\":%.3f}",
            gr.prompt_ms, pps, gr.predicted_ms, tps, gr.cache_hit_tokens, gr.slot, gr.slot_span,
            gr.mtp_used ? "true" : "false", gr.mtp_tok_per_round, gr.mtp_avg_accepted,
            gr.mtp_draft_ms, gr.mtp_verify_ms);
        return buf;
    }

    static std::string usage_json(const GenResult & gr) {
        const int hit  = std::max(0, std::min(gr.cache_hit_tokens, gr.prompt_tokens));
        const int miss = std::max(0, gr.prompt_tokens - hit);
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,"
            "\"prompt_tokens_details\":{\"cached_tokens\":%d},"
            "\"prompt_cache_hit_tokens\":%d,\"prompt_cache_miss_tokens\":%d}",
            gr.prompt_tokens, gr.completion_tokens, gr.prompt_tokens + gr.completion_tokens,
            hit, hit, miss);
        return buf;
    }

    // ---- Scheduler helpers (called ONLY on GPU scheduler thread) -----------
    int32_t span_ctx_limit(int slot, int span) const {
        return std::min(eng_.slot_ctx_limit(slot, span), request_max_ctx());
    }

    int min_span_for_tokens(int32_t n_toks) const {
        const int n_slots = eng_.num_slots();
        if (n_slots <= 1) return 1;
        int sp = 1;
        while (sp < n_slots && n_toks > eng_.slot_ctx_limit(0, sp)) {
            ++sp;
        }
        return sp;
    }

    static int64_t prefix_total_tokens(const std::vector<SlotCacheSeg> & pro,
                                       const std::vector<int32_t> & tail) {
        int64_t n = (int64_t)tail.size();
        for (const auto & s : pro) {
            n += s.is_image ? (int64_t)s.ntok : (int64_t)s.toks.size();
        }
        return n;
    }

    int64_t slot_cached_total(int c) const {
        return prefix_total_tokens(slot_mm_prologue_[c], slot_cached_toks_[c]);
    }

    void clear_slot_cache(int c) {
        slot_mm_prologue_[c].clear();
        slot_cached_toks_[c].clear();
        slot_cache_valid_[c] = false;
        slot_cache_span_[c] = 0;
    }

    void clear_slot0_checkpoint() {
        slot0_ckpt_valid_ = false;
        slot0_ckpt_drafter_valid_ = false;
        slot0_ckpt_span_ = 0;
        slot0_ckpt_mm_prologue_.clear();
        slot0_ckpt_toks_.clear();
    }

    void save_slot0_checkpoint() {
        if (!slot_cache_valid_[0]) return;
        const int64_t n = slot_cached_total(0);
        if (n < SLOT0_CKPT_MIN_TOKENS || (int64_t)eng_.slot_pos(0) != n) return;
        if (!eng_.save_slot0_checkpoint(slot0_ckpt_eng_)) return;
        slot0_ckpt_mm_prologue_ = slot_mm_prologue_[0];
        slot0_ckpt_toks_ = slot_cached_toks_[0];
        slot0_ckpt_span_ = std::max(1, slot_cache_span_[0]);
        slot0_ckpt_drafter_valid_ = false;
        if (drafter_ != nullptr && drafter_in_sync_ && drafter_->length() == (int)n) {
            slot0_ckpt_drafter_valid_ = drafter_->save_checkpoint(slot0_ckpt_drafter_);
        }
        slot0_ckpt_valid_ = true;
    }

    static int match_prefix_against(const std::vector<SlotCacheSeg> & pro,
                                    const std::vector<int32_t> & tail,
                                    int64_t total,
                                    const StreamJob & cand,
                                    int * resume_seg,
                                    int * resume_off) {
        *resume_seg = 0;
        *resume_off = 0;
        if (total <= 0) return 0;

        if (!cand.has_images) {
            if (!pro.empty()) return 0;
            const int n = (int)tail.size();
            if ((int)cand.prompt_toks.size() <= n) return 0;
            if (!std::equal(tail.begin(), tail.end(), cand.prompt_toks.begin())) return 0;
            return n;
        }

        const auto & segs = cand.segs;
        const size_t nseg = segs.size();
        auto skip_empty = [&](size_t & i) {
            while (i < nseg && !segs[i].is_image && segs[i].toks.empty()) ++i;
        };
        size_t si = 0;
        for (const auto & cs : pro) {
            skip_empty(si);
            if (si >= nseg || segs[si].is_image != cs.is_image) return 0;
            if (cs.is_image) {
                if (segs[si].hash1 != cs.hash1 || segs[si].hash2 != cs.hash2) return 0;
            } else {
                if (segs[si].toks != cs.toks) return 0;
            }
            ++si;
        }
        if (!tail.empty()) {
            size_t got = 0;
            while (got < tail.size()) {
                skip_empty(si);
                if (si >= nseg || segs[si].is_image) return 0;
                const auto & st = segs[si].toks;
                const size_t take = std::min(st.size(), tail.size() - got);
                if (std::memcmp(st.data(), tail.data() + got, take * sizeof(int32_t)) != 0) return 0;
                got += take;
                if (take < st.size()) {
                    // Tail ends inside this text segment: resume with its remaining tokens
                    *resume_seg = (int)si;
                    *resume_off = (int)take;
                    return (int)total;
                }
                ++si;
            }
        }
        skip_empty(si);
        if (si >= nseg) return 0; // nothing left to prefill -> no logits
        *resume_seg = (int)si;
        return (int)total;
    }

    int match_slot0_checkpoint(const StreamJob & cand, int * resume_seg, int * resume_off) const {
        *resume_seg = 0;
        *resume_off = 0;
        if (!slot0_ckpt_valid_) return 0;
        const int64_t total = prefix_total_tokens(slot0_ckpt_mm_prologue_, slot0_ckpt_toks_);
        if (total <= 0 || (int64_t)eng_.slot_pos(0) < total) return 0;
        return match_prefix_against(slot0_ckpt_mm_prologue_, slot0_ckpt_toks_, total,
                                    cand, resume_seg, resume_off);
    }

    void update_cache_span(int c) {
        if (!slot_cache_valid_[c] || slot_cached_total(c) <= 0) {
            slot_cache_span_[c] = 0;
            return;
        }
        const int n_slots = eng_.num_slots();
        if (n_slots <= 1) {
            slot_cache_span_[c] = 1;
            return;
        }
        const int32_t sz = (int32_t)slot_cached_total(c);
        int sp = (sz + eng_.slot_ctx() - 1) / eng_.slot_ctx();
        if (sp < 1) sp = 1;
        if (sp > n_slots - c) sp = n_slots - c;
        slot_cache_span_[c] = sp;
    }

    // Longest prefix of `cand` that is already resident in slot c's KV (scheduler thread).
    // Returns the token count (0 = miss). On a hit, *resume_seg / *resume_off give the
    // candidate segment + intra-segment token offset where prefill must continue.
    //   text-only candidate : prologue must be empty; slot_cached_toks_ must equal prompt_toks[0:n].
    //   multimodal candidate : prologue segments must match one-to-one (text ids exact, images by
    //                          128-bit hash), then slot_cached_toks_ must be a prefix of the
    //                          remaining text. Read-only (call apply_mm_prefix_ntokens on commit).
    int match_slot_prefix(int c, const StreamJob & cand, int * resume_seg, int * resume_off) const {
        *resume_seg = 0;
        *resume_off = 0;
        if (!slot_cache_valid_[c]) return 0;
        const int64_t total = slot_cached_total(c);
        if (total <= 0 || (int64_t)eng_.slot_pos(c) != total) return 0;
        return match_prefix_against(slot_mm_prologue_[c], slot_cached_toks_[c], total,
                                    cand, resume_seg, resume_off);
    }

    // Populate matched image token counts from slot c's cached prologue once a hit is chosen.
    void apply_mm_prefix_ntokens(int c, StreamJob & job) const {
        auto & segs = job.segs;
        const size_t nseg = segs.size();
        size_t si = 0;
        for (const auto & cs : slot_mm_prologue_[c]) {
            while (si < nseg && !segs[si].is_image && segs[si].toks.empty()) ++si;
            if (si >= nseg) break;
            if (cs.is_image && segs[si].is_image) {
                segs[si].n_tokens = cs.ntok;
            }
            ++si;
        }
    }

    // Any write into segments [start_seg, start_seg + span) overwrites Global KV
    // and Quest page metadata for that range, so invalidate every other slot whose
    // cached prefix overlaps it (prevents cross-stream KV corruption!).
    void invalidate_overlapping_caches(int start_seg, int span, int except_slot = -1) {
        const int end_seg = start_seg + span;
        for (int k = 0; k < eng_.num_slots(); ++k) {
            if (k == except_slot || !slot_cache_valid_[k]) continue;
            const int k_span = std::max(1, slot_cache_span_[k]);
            const int k_end = k + k_span;
            if (k < end_seg && start_seg < k_end) {
                clear_slot_cache(k);
                if (k == 0 && drafter_ != nullptr) {
                    drafter_in_sync_ = false;
                }
            }
        }
        if (slot0_ckpt_valid_ && except_slot != 0) {
            const int ckpt_end = std::max(1, slot0_ckpt_span_);
            if (start_seg < ckpt_end) {
                clear_slot0_checkpoint();
            }
        }
    }

    void release_slot_segments(int c) {
        for (int s = 0; s < eng_.num_slots(); ++s) {
            if (seg_owner_[s] == c) seg_owner_[s] = -1;
        }
        if (c >= 0 && c < MAX_CONCURRENT_SLOTS) {
            slot_span_[c] = 0;
        }
    }

    // Attempts to expand an active slot c in-place to cover `needed_pos` tokens.
    // Returns true if expanded and ready to proceed.
    // If the adjacent segment (c + span) is currently busy with a higher-indexed slot,
    // marks job->waiting_expand = true and returns false (guaranteed deadlock-free:
    // wait edges are strictly acyclic c -> owner(c+span) > c).
    // If expansion is impossible (at hard cap or end of KV pool), clears waiting_expand
    // and returns false so the caller terminates cleanly.
    bool try_expand_slot(const std::shared_ptr<StreamJob> & job, int32_t needed_pos) {
        const int c = job->slot;
        const int n_slots = eng_.num_slots();
        if (c < 0 || c >= n_slots) return false;

        while (needed_pos > span_ctx_limit(c, job->span)) {
            if (needed_pos > request_max_ctx()) {
                job->waiting_expand = false;
                job->live_waiting_expand.store(false, std::memory_order_relaxed);
                return false;
            }
            const int next_seg = c + job->span;
            if (next_seg >= n_slots) {
                job->waiting_expand = false;
                job->live_waiting_expand.store(false, std::memory_order_relaxed);
                return false;
            }
            if (seg_owner_[next_seg] != -1) {
                job->waiting_expand = true;
                job->expand_needed_pos = needed_pos;
                job->live_waiting_expand.store(true, std::memory_order_relaxed);
                return false;
            }
            invalidate_overlapping_caches(next_seg, 1, /*except_slot=*/c);
            const int new_span = job->span + 1;
            if (!eng_.expand_slot_span(c, new_span)) {
                job->waiting_expand = false;
                job->expand_needed_pos = 0;
                job->live_waiting_expand.store(false, std::memory_order_relaxed);
                return false;
            }
            job->span = new_span;
            slot_span_[c] = new_span;
            seg_owner_[next_seg] = c;
            job->waiting_expand = false;
            job->expand_needed_pos = 0;
            job->live_span.store(new_span, std::memory_order_relaxed);
            job->live_ctx_limit.store(span_ctx_limit(c, new_span), std::memory_order_relaxed);
            job->live_waiting_expand.store(false, std::memory_order_relaxed);
            publish_active_slots();

            char lbuf[160];
            std::snprintf(lbuf, sizeof(lbuf),
                          "Slot s%d expanded in-place to span=%d (segments %d..%d, limit=%d)",
                          c, new_span, c, c + new_span - 1, span_ctx_limit(c, new_span));
            add_log("INFO", lbuf);
        }
        job->waiting_expand = false;
        job->expand_needed_pos = 0;
        job->live_waiting_expand.store(false, std::memory_order_relaxed);
        return true;
    }

    void publish_active_slots() {
        std::lock_guard<std::mutex> lk(sched_mu_);
        int active_cnt = 0;
        for (int c = 0; c < eng_.num_slots(); ++c) {
            active_slots_pub_[c] = active_slots_[c];
            slot_span_pub_[c] = slot_span_[c];
            if (active_slots_[c] != nullptr) ++active_cnt;
        }
        busy_.store(active_cnt > 0 || !pending_.empty(), std::memory_order_relaxed);
    }

    void finish_job(const std::shared_ptr<StreamJob> & job, const char * reason,
                    bool failed = false, int err_code = 500,
                    const std::string & err_msg = "",
                    const std::string & err_raw_json = "") {
        const int c = job->slot;
        const auto now = std::chrono::steady_clock::now();
        double pred_ms = 0.0;
        if (job->state == JOB_DECODING) {
            pred_ms = std::chrono::duration<double, std::milli>(now - job->t_decode_start).count();
        }
        job->state = JOB_DONE;
        job->waiting_expand = false;
        job->live_state.store(JOB_DONE, std::memory_order_relaxed);
        job->live_waiting_expand.store(false, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> qlk(job->q_mu);
            job->prefill_ready = true;
            job->done = true;
            job->failed = failed;
            job->http_err_code = err_code;
            job->http_err_msg = err_msg;
            job->http_err_raw_json = err_raw_json;
            if (reason != nullptr) job->finish_reason = reason;
            job->final_prompt_tokens = job->sched_prompt_tokens;
            job->final_cache_hit_tokens = job->sched_cache_hit_tokens;
            job->final_slot = (c >= 0) ? c : 0;
            job->final_slot_span = std::max(1, job->span);
            job->final_prompt_ms = job->sched_prompt_ms;
            job->final_predicted_ms = pred_ms;
            job->final_mtp_used = job->sched_mtp_used;
            if (job->mtp_rounds > 0) {
                job->final_mtp_avg_accepted  = (double)job->mtp_accepted / (double)job->mtp_rounds;
                job->final_mtp_tok_per_round = 1.0 + job->final_mtp_avg_accepted;
                job->final_mtp_draft_ms      = job->mtp_sum_draft / (double)job->mtp_rounds;
                job->final_mtp_verify_ms     = job->mtp_sum_verify / (double)job->mtp_rounds;
            }
        }
        job->q_cv.notify_all();

        if (c >= 0 && c < eng_.num_slots() && active_slots_[c] == job) {
            update_cache_span(c);
            slot_last_used_[c] = ++sched_lru_seq_;
            active_slots_[c].reset();
            release_slot_segments(c);
            publish_active_slots();
        }
    }

    // Emits a newly produced token to the HTTP thread and updates scheduler state.
    // Returns true if job should keep decoding, false if job finished (EOS / max_tokens / cancelled).
    bool emit_token(const std::shared_ptr<StreamJob> & job, int32_t tok, bool mark_prefill_ready = false) {
        if (job->cancelled.load(std::memory_order_relaxed) || g_stop_server.load(std::memory_order_relaxed)) {
            finish_job(job, "stop");
            return false;
        }
        if (tok == id_end_sampling_ || tok == id_eot_) {
            finish_job(job, "stop");
            return false;
        }
        ++job->sched_completion_tokens;
        job->live_gen_toks.store(job->sched_completion_tokens, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> qlk(job->q_mu);
            if (mark_prefill_ready) job->prefill_ready = true;
            job->out_toks.push_back(tok);
        }
        job->q_cv.notify_one();

        if (job->sched_completion_tokens >= job->max_tokens) {
            finish_job(job, "length");
            return false;
        }
        job->cur_tok = tok;
        return true;
    }

    void admit_pending_jobs() {
        const int n_slots = eng_.num_slots();
        while (true) {
            std::shared_ptr<StreamJob> next_job;
            int chosen_slot = -1;
            int chosen_span = 1;
            bool chosen_prefix_hit = false;
            bool chosen_ckpt_rollback = false;
            int chosen_prefix_len = 0;
            int chosen_resume_seg = 0;
            int chosen_resume_off = 0;

            // Segments reserved by already-active jobs waiting to expand in-place
            bool expand_reserved[MAX_CONCURRENT_SLOTS] = {};
            int n_active = 0;
            bool any_expanding = false;
            for (int c = 0; c < n_slots; ++c) {
                if (active_slots_[c] != nullptr) {
                    ++n_active;
                    if (active_slots_[c]->waiting_expand) {
                        any_expanding = true;
                        const int next_seg = c + slot_span_[c];
                        if (next_seg < n_slots) expand_reserved[next_seg] = true;
                    }
                }
            }

            {
                std::lock_guard<std::mutex> lk(sched_mu_);
                // Drop any cancelled jobs in pending_
                for (auto it = pending_.begin(); it != pending_.end(); ) {
                    if ((*it)->cancelled.load(std::memory_order_relaxed)) {
                        auto cj = *it;
                        it = pending_.erase(it);
                        finish_job(cj, "stop");
                    } else {
                        ++it;
                    }
                }
                pending_count_.store((int)pending_.size(), std::memory_order_relaxed);
                if (pending_.empty()) break;

                bool queue_reserved[MAX_CONCURRENT_SLOTS] = {};
                for (int s = 0; s < n_slots; ++s) queue_reserved[s] = expand_reserved[s];

                const bool strictly_solo = (n_active == 0 && pending_.size() == 1 && !any_expanding);
                size_t chosen_idx = pending_.size();

                // Non-blocking scan with anti-starvation window reservation:
                // if a multi-segment job at the front cannot fit yet, reserve its best
                // target window so it doesn't starve, while allowing shorter jobs behind
                // it to start immediately on unreserved free segments!
                for (size_t idx = 0; idx < pending_.size(); ++idx) {
                    auto & cand = pending_[idx];
                    const int32_t est_p = cand->has_images ? cand->est_prompt_toks
                                                           : (int32_t)cand->prompt_toks.size();
                    const int min_span = min_span_for_tokens(est_p + 64);
                    // Avoid trapping a near-65K prompt on the last segment (which cannot expand)
                    // when the user requested enough output tokens to cross slot_ctx.
                    const bool avoid_last_slot =
                        (min_span == 1 && n_slots >= 2 && !cfg_.strict_slot_ctx &&
                         (est_p + std::min(cand->max_tokens, 2048) > eng_.slot_ctx()) &&
                         (est_p + 64 < request_max_ctx()));

                    int best_hit_slot = -1;
                    int best_hit_len  = 0;
                    int best_hit_span = min_span;
                    int best_hit_seg  = 0;
                    int best_hit_off  = 0;
                    bool best_hit_ckpt = false;
                    int best_free_slot = -1;
                    int64_t best_free_score = INT64_MIN;

                    for (int c = 0; c <= n_slots - min_span; ++c) {
                        if (avoid_last_slot && c == n_slots - 1) continue;
                        bool ok = true;
                        for (int s = c; s < c + min_span; ++s) {
                            if (seg_owner_[s] != -1 || queue_reserved[s]) {
                                ok = false;
                                break;
                            }
                        }
                        if (!ok) continue;

                        // Check exact prefix cache hit on slot c (text and multimodal).
                        // A cached multimodal prefix can span more segments than est_prompt_toks
                        // (images up to 4096 tokens vs 1024 estimate), so require all cached
                        // segments [c, c + hit_span) to be free and claim them atomically.
                        {
                            int rseg = 0, roff = 0;
                            const int plen = match_slot_prefix(c, *cand, &rseg, &roff);
                            if (plen > 0) {
                                const int hit_span = std::max(min_span, std::max(1, slot_cache_span_[c]));
                                bool hit_span_ok = (c + hit_span <= n_slots);
                                for (int s = c + min_span; hit_span_ok && s < c + hit_span; ++s) {
                                    if (seg_owner_[s] != -1 || queue_reserved[s]) hit_span_ok = false;
                                }
                                // On slot 0 when strictly solo with MTP enabled, only forfeit a tiny
                                // text-only prefix hit (< 2048 tok) to resync the drafter; never throw
                                // away a cached multimodal prefix (re-running ViT + prefill is costly)
                                // or forfeit hits under parallel load (where batched decode skips MTP).
                                const bool skip_for_mtp =
                                    (strictly_solo && c == 0 && drafter_ != nullptr && cand->allow_mtp &&
                                     !drafter_in_sync_ && slot_mm_prologue_[0].empty() && plen < 2048);
                                if (hit_span_ok && !skip_for_mtp && plen > best_hit_len) {
                                    best_hit_len  = plen;
                                    best_hit_slot = c;
                                    best_hit_span = hit_span;
                                    best_hit_seg  = rseg;
                                    best_hit_off  = roff;
                                    best_hit_ckpt = false;
                                }
                            } else if (c == 0) {
                                // Fallback: exact match against slot 0's prompt-end checkpoint
                                // (text or multimodal — recovers from 1-token BPE/JSON drift or
                                // branching turns without cold-prefilling history or re-running ViT).
                                int crseg = 0, croff = 0;
                                const int ckpt_len = match_slot0_checkpoint(*cand, &crseg, &croff);
                                if (ckpt_len > 0) {
                                    const int hit_span = std::max(min_span, std::max(1, slot0_ckpt_span_));
                                    bool hit_span_ok = (hit_span <= n_slots);
                                    for (int s = min_span; hit_span_ok && s < hit_span; ++s) {
                                        if (seg_owner_[s] != -1 || queue_reserved[s]) hit_span_ok = false;
                                    }
                                    const bool skip_for_mtp =
                                        (strictly_solo && drafter_ != nullptr && cand->allow_mtp &&
                                         !slot0_ckpt_drafter_valid_ && slot0_ckpt_mm_prologue_.empty() &&
                                         ckpt_len < 2048);
                                    if (hit_span_ok && !skip_for_mtp && ckpt_len > best_hit_len) {
                                        best_hit_len  = ckpt_len;
                                        best_hit_slot = 0;
                                        best_hit_span = hit_span;
                                        best_hit_seg  = crseg;
                                        best_hit_off  = croff;
                                        best_hit_ckpt = true;
                                    }
                                }
                            }
                        }

                        // Compute placement score for cold / eviction admission
                        int64_t score = 0;
                        if (strictly_solo && c == 0) {
                            score = (1LL << 50); // Solo fast path: always slot 0 (HIP Graph + MTP)
                        } else {
                            // Check if [c, c + min_span) overlaps any warm cached prefix
                            bool overlaps_cache = false;
                            uint64_t max_lru = 0;
                            int max_overlap_span = 0;
                            for (int k = 0; k < n_slots; ++k) {
                                if (!slot_cache_valid_[k]) continue;
                                const int k_span = std::max(1, slot_cache_span_[k]);
                                if (k < c + min_span && c < k + k_span) {
                                    overlaps_cache = true;
                                    if (slot_last_used_[k] > max_lru) max_lru = slot_last_used_[k];
                                    if (k_span > max_overlap_span) max_overlap_span = k_span;
                                }
                            }
                            if (!overlaps_cache) {
                                score += 5000000LL; // Strongly prefer genuinely cold segments
                            } else {
                                // Prefer evicting single-segment caches over multi-segment long caches,
                                // and evict least-recently-used (smallest max_lru) first.
                                score -= (int64_t)max_overlap_span * 500000LL;
                                score -= (int64_t)max_lru;
                            }

                            // Generalized free-run bisection (works for ANY n_slots 1..8, odd or even):
                            // Find the contiguous free & unreserved run [run_start, run_end) around [c, c+min_span).
                            int run_start = c;
                            while (run_start > 0 && seg_owner_[run_start - 1] == -1 && !queue_reserved[run_start - 1]) {
                                --run_start;
                            }
                            int run_end = c + min_span;
                            while (run_end < n_slots && seg_owner_[run_end] == -1 && !queue_reserved[run_end]) {
                                ++run_end;
                            }
                            const bool active_left = (run_start > 0 && seg_owner_[run_start - 1] != -1);
                            const int left_gap  = c - run_start;
                            const int right_gap = run_end - (c + min_span);
                            if (active_left) {
                                // Bisect the free run so BOTH the active slot to our left and this new slot
                                // retain maximal in-place expansion headroom (e.g. 0->2->1->3 for 4 slots,
                                // 0->4->2->6->1->3->5->7 for 8 slots, 0->4->2->6->1->3->5 for 7 slots).
                                const int balanced = std::min(left_gap, right_gap);
                                score += (int64_t)balanced * 200000LL;
                                if (left_gap > 0)  score += 40000LL;
                                if (right_gap > 0) score += 20000LL;
                                score += (int64_t)(left_gap + right_gap) * 1000LL;
                                score += (int64_t)left_gap * 100LL;
                            } else {
                                // No active neighbor on our left (e.g. starts at segment 0 or after a
                                // reservation): anchor at run_start to maximize right expansion headroom
                                // and avoid stranding unusable left fragments.
                                if (left_gap == 0) score += 250000LL;
                                score += (int64_t)right_gap * 20000LL;
                            }

                            // Power-of-2 buddy alignment bonus for multi-segment jobs (span >= 2)
                            if (min_span >= 2) {
                                int align = 1;
                                while (align < min_span) align <<= 1;
                                if (c % align == 0) score += 100000LL;
                            }

                            // Deterministic tie-break favoring lower slot indices (favors s0 & stable graph keys)
                            score += (int64_t)(n_slots - c) * 10LL;
                        }

                        if (score > best_free_score) {
                            best_free_score = score;
                            best_free_slot = c;
                        }
                    }

                    if (best_hit_slot >= 0) {
                        chosen_slot = best_hit_slot;
                        chosen_span = best_hit_span;
                        chosen_prefix_hit = true;
                        chosen_ckpt_rollback = best_hit_ckpt;
                        chosen_prefix_len = best_hit_len;
                        chosen_resume_seg = best_hit_seg;
                        chosen_resume_off = best_hit_off;
                        chosen_idx = idx;
                        break;
                    }
                    if (best_free_slot >= 0) {
                        chosen_slot = best_free_slot;
                        chosen_span = min_span;
                        chosen_prefix_hit = false;
                        chosen_ckpt_rollback = false;
                        chosen_prefix_len = 0;
                        chosen_idx = idx;
                        break;
                    }

                    // Candidate cannot fit right now -> reserve its best target window to prevent starvation
                    if (min_span >= 2 || avoid_last_slot) {
                        int best_w = -1;
                        int best_w_score = INT32_MIN;
                        for (int w = 0; w <= n_slots - min_span; ++w) {
                            if (avoid_last_slot && w == n_slots - 1) continue;
                            bool conflict = false;
                            int busy_cnt = 0;
                            for (int s = w; s < w + min_span; ++s) {
                                if (queue_reserved[s]) { conflict = true; break; }
                                if (seg_owner_[s] != -1) ++busy_cnt;
                            }
                            if (conflict) continue;
                            int wscore = -busy_cnt * 100;
                            if (min_span >= 2) {
                                int align = 1;
                                while (align < min_span) align <<= 1;
                                if (w % align == 0) wscore += 50;
                            }
                            {
                                int rseg = 0, roff = 0, crseg = 0, croff = 0;
                                if (match_slot_prefix(w, *cand, &rseg, &roff) > 0 ||
                                    (w == 0 && match_slot0_checkpoint(*cand, &crseg, &croff) > 0)) {
                                    wscore += 200;
                                }
                            }
                            if (wscore > best_w_score) {
                                best_w_score = wscore;
                                best_w = w;
                            }
                        }
                        if (best_w >= 0) {
                            for (int s = best_w; s < best_w + min_span; ++s) {
                                queue_reserved[s] = true;
                            }
                        }
                    }
                }

                if (chosen_slot < 0) break;

                next_job = pending_[chosen_idx];
                pending_.erase(pending_.begin() + (std::ptrdiff_t)chosen_idx);
                pending_count_.store((int)pending_.size(), std::memory_order_relaxed);
                active_slots_[chosen_slot] = next_job;
                active_slots_pub_[chosen_slot] = next_job;
                slot_span_[chosen_slot] = chosen_span;
                slot_span_pub_[chosen_slot] = chosen_span;
                for (int s = chosen_slot; s < chosen_slot + chosen_span; ++s) {
                    seg_owner_[s] = chosen_slot;
                }
                slot_last_used_[chosen_slot] = ++sched_lru_seq_;
                busy_.store(true, std::memory_order_relaxed);
            }

            // Initialize slot state on GPU
            next_job->slot = chosen_slot;
            next_job->span = chosen_span;
            next_job->waiting_expand = false;
            next_job->state = JOB_PREFILLING;
            next_job->live_slot.store(chosen_slot, std::memory_order_relaxed);
            next_job->live_span.store(chosen_span, std::memory_order_relaxed);
            next_job->live_ctx_limit.store(span_ctx_limit(chosen_slot, chosen_span), std::memory_order_relaxed);
            next_job->live_waiting_expand.store(false, std::memory_order_relaxed);
            next_job->live_state.store(JOB_PREFILLING, std::memory_order_relaxed);
            next_job->t_prefill_start = std::chrono::steady_clock::now();

            // ALWAYS invalidate every other slot's cached prefix overlapping [chosen_slot, chosen_slot + chosen_span)
            invalidate_overlapping_caches(chosen_slot, chosen_span, /*except_slot=*/chosen_slot);

            if (!next_job->has_images) {
                next_job->sched_prompt_tokens = (int)next_job->prompt_toks.size();
                next_job->live_prompt_toks.store(next_job->sched_prompt_tokens, std::memory_order_relaxed);
                const int32_t limit = span_ctx_limit(chosen_slot, chosen_span);
                const int avail = limit - next_job->sched_prompt_tokens - 8;
                if (avail < 1) {
                    finish_job(next_job, "stop", true, 400,
                               "Prompt exceeds context window",
                               context_overflow_error_json(request_max_ctx(), next_job->sched_prompt_tokens));
                    continue;
                }
                // Do NOT crush max_tokens to a temporary span=1 limit when the slot can still expand;
                // only clamp against the physical max reachable context from chosen_slot.
                const int32_t max_reachable = span_ctx_limit(chosen_slot, n_slots - chosen_slot);
                const int max_avail = std::max(1, max_reachable - next_job->sched_prompt_tokens - 8);
                if (next_job->max_tokens > max_avail) next_job->max_tokens = max_avail;

                if (chosen_prefix_hit) {
                    bool hit_ok = true;
                    if (chosen_ckpt_rollback) {
                        const int32_t old_pos = eng_.slot_pos(0);
                        auto t_rb0 = std::chrono::steady_clock::now();
                        hit_ok = eng_.restore_slot0_checkpoint(slot0_ckpt_eng_);
                        if (hit_ok) {
                            if (drafter_ != nullptr) {
                                if (slot0_ckpt_drafter_valid_ &&
                                    drafter_->restore_checkpoint(slot0_ckpt_drafter_)) {
                                    drafter_in_sync_ = true;
                                } else {
                                    drafter_in_sync_ = false;
                                }
                            }
                            slot_mm_prologue_[0].clear();
                            slot_cached_toks_[0] = slot0_ckpt_toks_;
                            slot_cache_valid_[0] = true;
                            update_cache_span(0);
                            const double rb_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_rb0).count();
                            char buf[192];
                            std::snprintf(buf, sizeof(buf),
                                          "Slot s0 prompt-end checkpoint hit: rewound %d -> %d tok in %.1f ms (mtp=%s)",
                                          old_pos, chosen_prefix_len, rb_ms,
                                          (drafter_ != nullptr && drafter_in_sync_) ? "sync" : "off");
                            add_log("INFO", buf);
                        }
                    }
                    if (hit_ok) {
                        if (eng_.slot_span(chosen_slot) < chosen_span) {
                            hit_ok = eng_.expand_slot_span(chosen_slot, chosen_span);
                        } else if (eng_.slot_span(chosen_slot) > chosen_span) {
                            hit_ok = eng_.shrink_slot_span(chosen_slot, chosen_span);
                        }
                    }
                    if (hit_ok) {
                        next_job->sched_cache_hit_tokens = chosen_prefix_len;
                        next_job->prefill_offset = chosen_prefix_len;
                    } else {
                        chosen_prefix_hit = false;
                    }
                }
                if (!chosen_prefix_hit) {
                    eng_.reset_slot(chosen_slot, chosen_span);
                    clear_slot_cache(chosen_slot);
                    if (chosen_slot == 0) {
                        clear_slot0_checkpoint();
                        if (drafter_ != nullptr) {
                            drafter_->reset();
                            drafter_in_sync_ = true;
                        }
                    }
                    next_job->prefill_offset = 0;
                }
            } else {
                // Multimodal turn. With a prefix hit (identical prologue images + text) the
                // slot's KV is replayed and only the remaining segments are prefilled;
                // otherwise reset the slot with its allocated span.
                next_job->live_prompt_toks.store(next_job->est_prompt_toks, std::memory_order_relaxed);
                if (chosen_prefix_hit) {
                    bool hit_ok = true;
                    if (chosen_ckpt_rollback) {
                        const int32_t old_pos = eng_.slot_pos(0);
                        auto t_rb0 = std::chrono::steady_clock::now();
                        hit_ok = eng_.restore_slot0_checkpoint(slot0_ckpt_eng_);
                        if (hit_ok) {
                            if (drafter_ != nullptr) {
                                if (slot0_ckpt_drafter_valid_ &&
                                    drafter_->restore_checkpoint(slot0_ckpt_drafter_)) {
                                    drafter_in_sync_ = true;
                                } else {
                                    drafter_in_sync_ = false;
                                }
                            }
                            slot_mm_prologue_[0] = slot0_ckpt_mm_prologue_;
                            slot_cached_toks_[0] = slot0_ckpt_toks_;
                            slot_cache_valid_[0] = true;
                            update_cache_span(0);
                            const double rb_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_rb0).count();
                            char buf[192];
                            std::snprintf(buf, sizeof(buf),
                                          "Slot s0 prompt-end checkpoint hit (mm): rewound %d -> %d tok in %.1f ms (mtp=%s)",
                                          old_pos, chosen_prefix_len, rb_ms,
                                          (drafter_ != nullptr && drafter_in_sync_) ? "sync" : "off");
                            add_log("INFO", buf);
                        }
                    }
                    if (hit_ok) {
                        if (eng_.slot_span(chosen_slot) < chosen_span) {
                            hit_ok = eng_.expand_slot_span(chosen_slot, chosen_span);
                        } else if (eng_.slot_span(chosen_slot) > chosen_span) {
                            hit_ok = eng_.shrink_slot_span(chosen_slot, chosen_span);
                        }
                    }
                    if (!hit_ok) {
                        chosen_prefix_hit = false;
                    }
                }
                if (chosen_prefix_hit) {
                    apply_mm_prefix_ntokens(chosen_slot, *next_job);
                    next_job->sched_cache_hit_tokens = chosen_prefix_len;
                    next_job->sched_prompt_tokens = chosen_prefix_len;
                    next_job->seg_idx = (size_t)chosen_resume_seg;
                    next_job->seg_offset = 0;
                    next_job->cur_seg_toks.clear();
                    if (chosen_resume_off > 0) {
                        // Resume inside a text segment: pre-load its not-yet-cached remainder.
                        const auto & st = next_job->segs[next_job->seg_idx].toks;
                        next_job->cur_seg_toks.assign(st.begin() + chosen_resume_off, st.end());
                        next_job->sched_prompt_tokens += (int)next_job->cur_seg_toks.size();
                    }
                    next_job->live_prompt_toks.store(
                        std::max(next_job->est_prompt_toks, next_job->sched_prompt_tokens),
                        std::memory_order_relaxed);
                    next_job->cur_vis_loaded = false;
                    next_job->cur_vis_gen = 0;
                } else {
                    next_job->sched_prompt_tokens = 0; // accumulated as segments are encoded
                    eng_.reset_slot(chosen_slot, chosen_span);
                    clear_slot_cache(chosen_slot);
                    if (chosen_slot == 0) {
                        clear_slot0_checkpoint();
                        if (drafter_ != nullptr) {
                            drafter_->reset();
                            drafter_in_sync_ = true;
                        }
                    }
                    next_job->seg_idx = 0;
                    next_job->seg_offset = 0;
                    next_job->cur_seg_toks.clear();
                    next_job->cur_vis_loaded = false;
                    next_job->cur_vis_gen = 0;
                }
            }
        }
    }

    void advance_prefill_chunk(const std::shared_ptr<StreamJob> & job) {
        const int c = job->slot;
        if (job->cancelled.load(std::memory_order_relaxed)) {
            clear_slot_cache(c);
            finish_job(job, "stop");
            return;
        }

        if (!job->has_images) {
            const int total = (int)job->prompt_toks.size();
            const int rem = total - job->prefill_offset;
            const int chunk = std::min(MAX_UBATCH, rem);
            const bool is_last = (job->prefill_offset + chunk == total);
            const int32_t needed_pos = eng_.slot_pos(c) + chunk;
            if (needed_pos > span_ctx_limit(c, job->span)) {
                if (!try_expand_slot(job, needed_pos)) {
                    if (job->waiting_expand) return; // paused until neighbor drains
                    eng_.reset_slot(c, 1);
                    clear_slot_cache(c);
                    if (c == 0) { clear_slot0_checkpoint(); if (drafter_) { drafter_->reset(); drafter_in_sync_ = true; } }
                    finish_job(job, "stop", true, 400, "Prompt exceeds context window",
                               context_overflow_error_json(request_max_ctx(), total));
                    return;
                }
            }

            std::vector<float> logits;
            if (!eng_.prefill_slot_chunk(c, job->prompt_toks.data() + job->prefill_offset, chunk,
                                         is_last, is_last ? &logits : nullptr)) {
                eng_.reset_slot(c, 1);
                clear_slot_cache(c);
                if (c == 0) { clear_slot0_checkpoint(); if (drafter_) { drafter_->reset(); drafter_in_sync_ = true; } }
                finish_job(job, "stop", true, 500, "Engine prefill failed");
                return;
            }
            slot_cached_toks_[c].insert(slot_cached_toks_[c].end(),
                                        job->prompt_toks.begin() + job->prefill_offset,
                                        job->prompt_toks.begin() + job->prefill_offset + chunk);
            job->prefill_offset += chunk;

            if (is_last) {
                slot_cache_valid_[c] = true;
                update_cache_span(c);
                if (c == 0 && total >= SLOT0_CKPT_MIN_TOKENS) {
                    save_slot0_checkpoint();
                }
                job->sched_prompt_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - job->t_prefill_start).count();
                job->t_decode_start = std::chrono::steady_clock::now();
                job->state = JOB_DECODING;
                job->live_state.store(JOB_DECODING, std::memory_order_relaxed);
                int32_t first_tok = sample_logits(logits, job->sp, job->rng);
                (void)emit_token(job, first_tok, /*mark_prefill_ready=*/true);
            }
            return;
        }

        // Multimodal prefill: advance by one <= MAX_UBATCH chunk of text or image embeddings
        while (job->seg_idx < job->segs.size()) {
            const auto & seg = job->segs[job->seg_idx];
            if (!seg.is_image) {
                if (job->seg_offset == 0 && job->cur_seg_toks.empty()) {
                    job->cur_seg_toks = seg.toks;
                    if (job->cur_seg_toks.empty()) {
                        ++job->seg_idx;
                        continue;
                    }
                    job->sched_prompt_tokens += (int)job->cur_seg_toks.size();
                    job->live_prompt_toks.store(job->sched_prompt_tokens, std::memory_order_relaxed);
                }
                const int rem = (int)job->cur_seg_toks.size() - job->seg_offset;
                const int chunk = std::min(MAX_UBATCH, rem);
                const int32_t needed_pos = eng_.slot_pos(c) + chunk;
                if (needed_pos > span_ctx_limit(c, job->span)) {
                    if (!try_expand_slot(job, needed_pos)) {
                        if (job->waiting_expand) return;
                        eng_.reset_slot(c, 1);
                        clear_slot_cache(c);
                        if (c == 0) { clear_slot0_checkpoint(); if (drafter_) { drafter_->reset(); drafter_in_sync_ = true; } }
                        finish_job(job, "stop", true, 400, "Multimodal prompt exceeds context window",
                                   context_overflow_error_json(request_max_ctx(), job->sched_prompt_tokens));
                        return;
                    }
                }
                const bool seg_done = (job->seg_offset + chunk == (int)job->cur_seg_toks.size());
                // Check if any non-empty segment remains after seg_idx
                bool is_last_overall = seg_done;
                if (is_last_overall) {
                    for (size_t k = job->seg_idx + 1; k < job->segs.size(); ++k) {
                        if (job->segs[k].is_image || !job->segs[k].toks.empty()) {
                            is_last_overall = false;
                            break;
                        }
                    }
                }
                std::vector<float> logits;
                if (!eng_.prefill_slot_chunk(c, job->cur_seg_toks.data() + job->seg_offset, chunk,
                                             is_last_overall, is_last_overall ? &logits : nullptr)) {
                    eng_.reset_slot(c, 1);
                    clear_slot_cache(c);
                    if (c == 0) { clear_slot0_checkpoint(); if (drafter_) { drafter_->reset(); drafter_in_sync_ = true; } }
                    finish_job(job, "stop", true, 500, "Text prefill failed");
                    return;
                }
                job->seg_offset += chunk;
                if (seg_done) {
                    job->cur_seg_toks.clear();
                    job->seg_offset = 0;
                    ++job->seg_idx;
                }
                if (is_last_overall) {
                    finish_multimodal_prefill(job, logits);
                }
                return;
            } else {
                if (!job->cur_vis_loaded || job->cur_vis_gen != vis_encode_gen_) {
                    // VisionEncoder::d_proj_out_ is a single shared scratch buffer that may be
                    // overwritten or reallocated on any encode_image_bytes() call. Bump the
                    // generation before encoding so any paused job (e.g. waiting_expand) detects
                    // the overwrite and deterministically re-encodes before resuming its chunks.
                    const bool first_encode = !job->cur_vis_loaded;
                    const uint64_t next_gen = ++vis_encode_gen_;
                    if (!vision_->encode_image_bytes(seg.image_bytes.data(), seg.image_bytes.size(), job->cur_vis) ||
                        (!first_encode && job->cur_vis.n_tokens != job->segs[job->seg_idx].n_tokens) ||
                        job->cur_vis.n_tokens <= (first_encode ? 0 : job->seg_offset)) {
                        job->cur_vis_loaded = false;
                        job->cur_vis_gen = 0;
                        eng_.reset_slot(c, 1);
                        clear_slot_cache(c);
                        if (c == 0) { clear_slot0_checkpoint(); if (drafter_) { drafter_->reset(); drafter_in_sync_ = true; } }
                        finish_job(job, "stop", true, 400, "Failed to decode/encode image");
                        return;
                    }
                    job->cur_vis_loaded = true;
                    job->cur_vis_gen = next_gen;
                    if (first_encode) {
                        job->segs[job->seg_idx].n_tokens = job->cur_vis.n_tokens;
                        job->seg_offset = 0;
                        job->sched_prompt_tokens += job->cur_vis.n_tokens;
                        job->live_prompt_toks.store(job->sched_prompt_tokens, std::memory_order_relaxed);
                    }
                }
                const int rem = job->cur_vis.n_tokens - job->seg_offset;
                const int chunk = std::min(MAX_UBATCH, rem);
                const int32_t needed_pos = eng_.slot_pos(c) + chunk;
                if (needed_pos > span_ctx_limit(c, job->span)) {
                    if (!try_expand_slot(job, needed_pos)) {
                        if (job->waiting_expand) return;
                        job->cur_vis_loaded = false;
                        job->cur_vis_gen = 0;
                        eng_.reset_slot(c, 1);
                        clear_slot_cache(c);
                        if (c == 0) { clear_slot0_checkpoint(); if (drafter_) { drafter_->reset(); drafter_in_sync_ = true; } }
                        finish_job(job, "stop", true, 400, "Multimodal prompt exceeds context window",
                                   context_overflow_error_json(request_max_ctx(), job->sched_prompt_tokens));
                        return;
                    }
                }
                const bool seg_done = (job->seg_offset + chunk == job->cur_vis.n_tokens);
                bool is_last_overall = seg_done;
                if (is_last_overall) {
                    for (size_t k = job->seg_idx + 1; k < job->segs.size(); ++k) {
                        if (job->segs[k].is_image || !job->segs[k].toks.empty()) {
                            is_last_overall = false;
                            break;
                        }
                    }
                }
                std::vector<float> logits;
                const float * ptr = job->cur_vis.d_embeds + (size_t)job->seg_offset * N_EMBD;
                if (!eng_.prefill_embeds_slot_chunk(c, ptr, chunk, is_last_overall,
                                                    is_last_overall ? &logits : nullptr)) {
                    job->cur_vis_loaded = false;
                    job->cur_vis_gen = 0;
                    eng_.reset_slot(c, 1);
                    clear_slot_cache(c);
                    if (c == 0) { clear_slot0_checkpoint(); if (drafter_) { drafter_->reset(); drafter_in_sync_ = true; } }
                    finish_job(job, "stop", true, 500, "Vision embed prefill failed");
                    return;
                }
                job->seg_offset += chunk;
                if (seg_done) {
                    job->cur_vis_loaded = false;
                    job->cur_vis_gen = 0;
                    job->seg_offset = 0;
                    ++job->seg_idx;
                }
                if (is_last_overall) {
                    finish_multimodal_prefill(job, logits);
                }
                return;
            }
        }
        eng_.reset_slot(c, 1);
        clear_slot_cache(c);
        if (c == 0) { clear_slot0_checkpoint(); if (drafter_) { drafter_->reset(); drafter_in_sync_ = true; } }
        finish_job(job, "stop", true, 500, "No logits produced after multimodal prefill");
    }

    // Publish the just-prefilled multimodal prompt as the slot's reusable prefix:
    //   prologue = segments up to and including the last image, tail = text after it.
    // Image segments are identified by content hash (+ their ViT token count), so a later
    // request carrying byte-identical images and identical text replays this KV.
    void commit_multimodal_prefix(const std::shared_ptr<StreamJob> & job) {
        const int c = job->slot;
        clear_slot_cache(c);
        size_t last_img = SIZE_MAX;
        for (size_t i = 0; i < job->segs.size(); ++i) {
            if (job->segs[i].is_image) last_img = i;
        }
        auto & pro = slot_mm_prologue_[c];
        for (size_t i = 0; i <= last_img && last_img != SIZE_MAX; ++i) {
            const auto & sg = job->segs[i];
            if (sg.is_image) {
                SlotCacheSeg cs;
                cs.is_image = true;
                cs.ntok = sg.n_tokens;
                cs.hash1 = sg.hash1;
                cs.hash2 = sg.hash2;
                if (cs.ntok <= 0) { clear_slot_cache(c); return; } // never encoded -> not replayable
                pro.push_back(std::move(cs));
            } else if (!sg.toks.empty()) {
                SlotCacheSeg cs;
                cs.toks = sg.toks;
                pro.push_back(std::move(cs));
            }
        }
        auto & tail = slot_cached_toks_[c];
        if (last_img != SIZE_MAX) {
            for (size_t i = last_img + 1; i < job->segs.size(); ++i) {
                if (!job->segs[i].is_image && !job->segs[i].toks.empty()) {
                    tail.insert(tail.end(), job->segs[i].toks.begin(), job->segs[i].toks.end());
                }
            }
        }
        // The recorded prefix must describe exactly what sits in the slot's KV.
        if (slot_cached_total(c) != (int64_t)eng_.slot_pos(c) || slot_cached_total(c) <= 0) {
            clear_slot_cache(c);
            return;
        }
        slot_cache_valid_[c] = true;
        update_cache_span(c);
        if (c == 0 && slot_cached_total(0) >= SLOT0_CKPT_MIN_TOKENS) {
            save_slot0_checkpoint();
        }
    }

    void finish_multimodal_prefill(const std::shared_ptr<StreamJob> & job, const std::vector<float> & logits) {
        const int c = job->slot;
        commit_multimodal_prefix(job);
        const int32_t max_reachable = span_ctx_limit(c, eng_.num_slots() - c);
        const int max_avail = std::max(1, max_reachable - eng_.slot_pos(c) - 8);
        if (job->max_tokens > max_avail) job->max_tokens = max_avail;
        job->sched_prompt_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - job->t_prefill_start).count();
        job->t_decode_start = std::chrono::steady_clock::now();
        job->state = JOB_DECODING;
        job->live_state.store(JOB_DECODING, std::memory_order_relaxed);
        int32_t first_tok = sample_logits(logits, job->sp, job->rng);
        (void)emit_token(job, first_tok, /*mark_prefill_ready=*/true);
    }

    // Exact lossless DSpark MTP speculative round (solo slot 0 fast path — 0% regression).
    void run_solo_mtp_round(const std::shared_ptr<StreamJob> & job) {
        job->sched_mtp_used = true;
        job->live_mtp.store(true, std::memory_order_relaxed);

        const int32_t needed = eng_.current_pos() + 8;
        if (needed >= span_ctx_limit(0, job->span)) {
            if (!try_expand_slot(job, needed + 1)) {
                finish_job(job, "length");
                return;
            }
        }
        const int32_t anchor = job->cur_tok;
        drafter_->note_tokens(&anchor, 1);

        const int P = eng_.current_pos();
        (void)hipEventRecord(ev0_, eng_.stream());
        DSparkDraft d = drafter_->draft(anchor, P);
        (void)hipEventRecord(ev1_, eng_.stream());
        (void)hipStreamSynchronize(eng_.stream());
        float d_ms = 0.0f;
        (void)hipEventElapsedTime(&d_ms, ev0_, ev1_);
        job->mtp_sum_draft += d_ms;

        int32_t drafts[7] = {0};
        float conf_f[7] = {0.0f};
        (void)hipMemcpy(conf_f, d.conf, sizeof(float) * 7, hipMemcpyDeviceToHost);
        (void)hipMemcpy(drafts, d.tokens, sizeof(int32_t) * 7, hipMemcpyDeviceToHost);

        int lead = 0;
        while (lead < 7 && conf_f[lead] > cfg_.spec_threshold) ++lead;
        const int m = std::max(2, std::min(8, lead + 1));
        int32_t round_tokens[8] = {0};
        round_tokens[0] = anchor;
        for (int r = 1; r < m; ++r) round_tokens[r] = drafts[r - 1];

        (void)hipEventRecord(ev1_, eng_.stream());
        bool vok = eng_.verify_step(round_tokens, m, P);
        (void)hipEventRecord(ev2_, eng_.stream());
        (void)hipStreamSynchronize(eng_.stream());
        if (!vok) {
            clear_slot_cache(0);
            drafter_in_sync_ = false;
            finish_job(job, "length");
            return;
        }
        float v_ms = 0.0f;
        (void)hipEventElapsedTime(&v_ms, ev1_, ev2_);

        int acc = 0;
        int32_t next = 0;
        if (!job->need_logits) {
            const int32_t * target = eng_.host_target_ids();
            while (acc < m - 1 && target[acc] == drafts[acc]) ++acc;
            next = target[acc];
        } else if (use_gpu_full_sampler(job->sp)) {
            // Issue #4: exact speculative sampling with a point-mass proposal
            // q = delta(draft) (the drafter's tokens are deterministic), all on
            // the GPU sampler -- no per-logit host traffic:
            //   accept draft d with probability min(1, p(d) / q(d)) = p(d) = e_d / Z;
            //   on rejection draw from norm(max(0, p - q)) = p(x) / (1 - p(d)), x != d,
            //   which is exactly the GPU draw over keep \ {d} (row r, excl = d);
            //   if every draft is accepted, the bonus token is a plain draw from p
            //   (row m-1, excl = -1).
            // The residual draws use Philox streams independent of the host's
            // acceptance coins, so pre-drawing every row's residual is exact.
            int32_t excl[8];
            int32_t stok[8];
            float   sstat[8 * 4];
            for (int r = 0; r < m; ++r) excl[r] = (r < m - 1) ? drafts[r] : -1;
            (void)hipEventRecord(ev1_, eng_.stream());
            const bool s_ok = eng_.sample_rows_full_to_host(
                m, 1.0f / job->sp.temperature, job->sp.min_p, excl, job->rng(), stok, sstat);
            (void)hipEventRecord(ev2_, eng_.stream());
            (void)hipStreamSynchronize(eng_.stream());
            float tk_ms = 0.0f;
            (void)hipEventElapsedTime(&tk_ms, ev1_, ev2_);
            v_ms += tk_ms;
            bool s_valid = s_ok;
            for (int r = 0; s_valid && r < m; ++r) {
                if (!(sstat[4 * r + 1] > 0.0f) || !std::isfinite(sstat[4 * r + 1])) s_valid = false;
            }
            if (!s_valid) {
                const int32_t * target = eng_.host_target_ids();
                while (acc < m - 1 && target[acc] == drafts[acc]) ++acc;
                next = target[acc];
            } else {
                std::uniform_real_distribution<float> u01(0.0f, 1.0f);
                bool all_accepted = true;
                for (int r = 0; r < m - 1; ++r) {
                    const float z   = sstat[4 * r + 1];
                    const float p_d = (sstat[4 * r + 3] > 0.5f) ? std::min(1.0f, sstat[4 * r + 2] / z) : 0.0f;
                    if (u01(job->rng) < p_d || stok[r] < 0) {
                        // stok < 0: keep \ {d} is empty, i.e. p(d) = 1 (accept).
                        ++acc;
                    } else {
                        next = stok[r];
                        all_accepted = false;
                        break;
                    }
                }
                if (all_accepted) {
                    next = (stok[m - 1] >= 0) ? stok[m - 1] : eng_.host_target_ids()[m - 1];
                }
            }
        } else {
            // Lossless speculative rejection sampling for temperature > 0 (top_k, min_p, top_p).
            std::array<float,   8 * MTP_VERIFY_TOPK> topk_vals{};
            std::array<int32_t, 8 * MTP_VERIFY_TOPK> topk_ids{};
            std::array<float,   8> row_max_l{};
            std::array<float,   8> row_sum_exp{};
            const float inv_t = 1.0f / job->sp.temperature;
            (void)hipEventRecord(ev1_, eng_.stream());
            const bool tok_ok = eng_.verify_topk_to_host(
                m, inv_t, topk_vals.data(), topk_ids.data(), row_max_l.data(), row_sum_exp.data());
            (void)hipEventRecord(ev2_, eng_.stream());
            (void)hipStreamSynchronize(eng_.stream());
            float tk_ms = 0.0f;
            (void)hipEventElapsedTime(&tk_ms, ev1_, ev2_);
            v_ms += tk_ms;

            if (!tok_ok) {
                const int32_t * target = eng_.host_target_ids();
                while (acc < m - 1 && target[acc] == drafts[acc]) ++acc;
                next = target[acc];
            } else {
                bool all_accepted = true;
                for (int r = 0; r < m - 1; ++r) {
                    bool accepted = false;
                    const int32_t tok = sample_speculative_row(
                        eng_, r,
                        topk_vals.data() + (size_t)r * MTP_VERIFY_TOPK,
                        topk_ids.data()  + (size_t)r * MTP_VERIFY_TOPK,
                        row_max_l[(size_t)r], row_sum_exp[(size_t)r],
                        drafts[r], job->sp, job->rng, accepted);
                    if (accepted) {
                        ++acc;
                    } else {
                        next = tok;
                        all_accepted = false;
                        break;
                    }
                }
                if (all_accepted) {
                    bool dummy = false;
                    const int r = m - 1;
                    next = sample_speculative_row(
                        eng_, r,
                        topk_vals.data() + (size_t)r * MTP_VERIFY_TOPK,
                        topk_ids.data()  + (size_t)r * MTP_VERIFY_TOPK,
                        row_max_l[(size_t)r], row_sum_exp[(size_t)r],
                        /*draft_tok=*/-1, job->sp, job->rng, dummy);
                }
            }
        }
        job->mtp_sum_verify += v_ms;

        int emit_acc = acc;
        const int rem = job->max_tokens - job->sched_completion_tokens;
        if (rem >= 0 && emit_acc > rem) emit_acc = rem;
        bool stop_on_draft_eos = false;
        int32_t stop_eos_tok = id_end_sampling_;
        for (int r = 0; r < emit_acc; ++r) {
            if (drafts[r] == id_end_sampling_ || drafts[r] == id_eot_) {
                // Do not commit EOS into KV / prefix cache (matches single-step decode_step
                // so the next chat turn retains a 100% prefix cache hit and drafter sync).
                emit_acc = r;
                stop_on_draft_eos = true;
                stop_eos_tok = drafts[r];
                break;
            }
        }
        const int commit = emit_acc + 1;
        // Always call verify_rollback (even when commit == m) so Quest page metadata
        // is updated for every committed token.
        eng_.verify_rollback(P, commit, m);
        drafter_->commit(eng_.taps(), commit, P);
        if (slot_cache_valid_[0]) {
            for (int r = 0; r < commit; ++r) slot_cached_toks_[0].push_back(round_tokens[r]);
            update_cache_span(0);
        }

        ++job->mtp_rounds;
        job->mtp_accepted += emit_acc;

        for (int r = 0; r < emit_acc; ++r) {
            drafter_->note_tokens(&drafts[r], 1);
            if (!emit_token(job, drafts[r])) {
                if (r + 1 < emit_acc) {
                    clear_slot_cache(0);
                }
                return;
            }
        }
        (void)emit_token(job, stop_on_draft_eos ? stop_eos_tok : next);
    }

    void scheduler_loop() {
        (void)hipEventCreate(&ev0_);
        (void)hipEventCreate(&ev1_);
        (void)hipEventCreate(&ev2_);

        std::vector<float> single_logits;
        int prefill_rr = 0;

        while (!sched_stop_.load()) {
            // 1. Clean up cancelled jobs
            for (int c = 0; c < eng_.num_slots(); ++c) {
                if (active_slots_[c] != nullptr &&
                    active_slots_[c]->cancelled.load(std::memory_order_relaxed)) {
                    finish_job(active_slots_[c], "stop");
                }
            }

            // 1b. Immediately wake up any active slot waiting to expand into a now-free neighbor
            //     (gives in-flight streams priority over new queued jobs).
            for (int c = 0; c < eng_.num_slots(); ++c) {
                auto job = active_slots_[c];
                if (job != nullptr && job->waiting_expand) {
                    const int32_t probe = std::max(job->expand_needed_pos, span_ctx_limit(c, job->span) + 1);
                    (void)try_expand_slot(job, probe);
                }
            }

            // 2. Count active slots
            int n_active = 0;
            for (int c = 0; c < eng_.num_slots(); ++c) {
                if (active_slots_[c] != nullptr) ++n_active;
            }

            // 3. Handle explicit KV reset when idle
            if (reset_requested_.load(std::memory_order_relaxed) && n_active == 0) {
                reset_all_gpu();
                {
                    std::lock_guard<std::mutex> lk(sched_mu_);
                    reset_requested_.store(false);
                    ++reset_gen_;
                }
                reset_cv_.notify_all();
                continue;
            }

            // 4. Admit any queued jobs into free admissible slots (skip if reset requested)
            if (!reset_requested_.load(std::memory_order_relaxed) &&
                pending_count_.load(std::memory_order_relaxed) > 0) {
                admit_pending_jobs();
            }

            // Recount active & classify
            n_active = 0;
            int prefill_slots[MAX_CONCURRENT_SLOTS];
            int n_prefill = 0;
            int decode_slots[MAX_CONCURRENT_SLOTS];
            int n_decode = 0;
            for (int c = 0; c < eng_.num_slots(); ++c) {
                auto & j = active_slots_[c];
                if (j == nullptr) continue;
                ++n_active;
                if (j->waiting_expand) continue;
                if (j->state == JOB_PREFILLING) prefill_slots[n_prefill++] = c;
                else if (j->state == JOB_DECODING) decode_slots[n_decode++] = c;
            }

            // 5. Idle wait
            if (n_active == 0) {
                std::unique_lock<std::mutex> lk(sched_mu_);
                busy_.store(!pending_.empty(), std::memory_order_relaxed);
                sched_cv_.wait(lk, [&]() {
                    return sched_stop_.load() || !pending_.empty() || reset_requested_.load();
                });
                continue;
            }

            // 6. SOLO FAST PATH (active == 1 on slot 0, DECODING, no pending queue):
            //    0% overhead — runs exact lossless DSpark MTP (temp==0 or temp>0) or static HIP Graph.
            if (n_active == 1 && n_decode == 1 && decode_slots[0] == 0 &&
                pending_count_.load(std::memory_order_relaxed) == 0 &&
                !reset_requested_.load(std::memory_order_relaxed)) {
                auto job = active_slots_[0];
                if (drafter_ != nullptr && job->allow_mtp && drafter_in_sync_) {
                    run_solo_mtp_round(job);
                    continue;
                }
                if (eng_.current_pos() + 2 >= span_ctx_limit(0, job->span)) {
                    if (!try_expand_slot(job, eng_.current_pos() + 3)) {
                        finish_job(job, "length");
                        continue;
                    }
                }
                const bool gpu_samp = job->need_logits && use_gpu_full_sampler(job->sp);
                int32_t next = eng_.decode_step(job->cur_tok,
                                                (job->need_logits && !gpu_samp) ? &single_logits : nullptr);
                if (next < 0) {
                    clear_slot_cache(0);
                    finish_job(job, "length");
                    continue;
                }
                if (slot_cache_valid_[0]) {
                    slot_cached_toks_[0].push_back(job->cur_tok);
                    update_cache_span(0);
                }
                if (drafter_ != nullptr) drafter_in_sync_ = false;
                const int32_t sampled = gpu_samp
                    ? sample_decode_row_gpu(eng_, job->sp, job->rng, next)
                    : (job->need_logits ? sample_logits(single_logits, job->sp, job->rng) : next);
                (void)emit_token(job, sampled);
                continue;
            }

            // 7. CHUNKED PREFILL INTERLEAVING (one <= 512-token chunk per tick)
            if (n_prefill > 0) {
                // If an eligible prefilling job currently owns VisionEncoder's shared d_proj_out_
                // buffer, finish draining its remaining image chunks before round-robining to
                // another job (which could overwrite d_proj_out_ and force a redundant re-encode).
                // Jobs paused in waiting_expand are excluded from prefill_slots, avoiding deadlock.
                int pick = -1;
                for (int i = 0; i < n_prefill; ++i) {
                    auto & pj = active_slots_[prefill_slots[i]];
                    if (pj != nullptr && pj->cur_vis_loaded && pj->cur_vis_gen == vis_encode_gen_) {
                        pick = prefill_slots[i];
                        break;
                    }
                }
                if (pick < 0) {
                    pick = prefill_slots[prefill_rr % n_prefill];
                    ++prefill_rr;
                }
                auto job = active_slots_[pick];
                if (job != nullptr && job->state == JOB_PREFILLING && !job->waiting_expand) {
                    advance_prefill_chunk(job);
                }
                // If that was a solo slot-0 request that just finished prefilling,
                // loop straight back to Step 6 so it enters lossless DSpark MTP
                // instead of falling through to Step 8 (which would desync drafter).
                if (n_active == 1 && pick == 0 && active_slots_[0] != nullptr &&
                    active_slots_[0]->state == JOB_DECODING &&
                    pending_count_.load(std::memory_order_relaxed) == 0) {
                    continue;
                }
            }

            // 8. COLLECT DECODING SLOTS & RUN BATCHED DECODE (1x weight stream for B >= 2)
            int32_t b_slots[MAX_CONCURRENT_SLOTS];
            int32_t b_toks[MAX_CONCURRENT_SLOTS];
            int32_t b_argmax[MAX_CONCURRENT_SLOTS];
            std::shared_ptr<StreamJob> b_jobs[MAX_CONCURRENT_SLOTS];
            int B = 0;
            bool any_need_cpu_logits = false;
            for (int c = 0; c < eng_.num_slots(); ++c) {
                auto job = active_slots_[c];
                if (job == nullptr || job->state != JOB_DECODING) continue;
                if (job->cancelled.load(std::memory_order_relaxed)) {
                    finish_job(job, "stop");
                    continue;
                }
                if (eng_.slot_pos(c) + 2 >= span_ctx_limit(c, job->span)) {
                    if (!try_expand_slot(job, eng_.slot_pos(c) + 3)) {
                        if (job->waiting_expand) continue; // paused cleanly waiting for neighbor to drain
                        finish_job(job, "length");
                        continue;
                    }
                }
                b_slots[B] = c;
                b_toks[B]  = job->cur_tok;
                b_jobs[B]  = job;
                if (job->need_logits && !use_gpu_full_sampler(job->sp)) {
                    any_need_cpu_logits = true;
                }
                ++B;
            }

            if (B == 1) {
                const int c = b_slots[0];
                auto & job = b_jobs[0];
                const bool gpu_samp = job->need_logits && use_gpu_full_sampler(job->sp);
                int32_t next = eng_.decode_slot_single(c, b_toks[0],
                                                       (job->need_logits && !gpu_samp) ? &single_logits : nullptr);
                if (next < 0) {
                    clear_slot_cache(c);
                    finish_job(job, "length");
                } else {
                    if (slot_cache_valid_[c]) {
                        slot_cached_toks_[c].push_back(b_toks[0]);
                        update_cache_span(c);
                    }
                    if (c == 0 && drafter_ != nullptr) drafter_in_sync_ = false;
                    const int32_t sampled = gpu_samp
                        ? sample_decode_row_gpu(eng_, job->sp, job->rng, next)
                        : (job->need_logits ? sample_logits(single_logits, job->sp, job->rng) : next);
                    (void)emit_token(job, sampled);
                }
            } else if (B >= 2) {
                if (any_need_cpu_logits) {
                    const size_t need = (size_t)B * VOCAB_UNPADDED;
                    if (batch_logits_.size() < need) batch_logits_.resize(need);
                }
                bool ok = eng_.decode_batch(b_slots, b_toks, B, b_argmax,
                                            any_need_cpu_logits ? batch_logits_.data() : nullptr);
                int32_t gpu_sampled[MAX_CONCURRENT_SLOTS] = {};
                bool    gpu_sampled_ok[MAX_CONCURRENT_SLOTS] = {};
                if (ok) {
                    int      n_gpu = 0;
                    int32_t  g_rows[MAX_CONCURRENT_SLOTS];
                    float    g_inv_t[MAX_CONCURRENT_SLOTS];
                    float    g_min_p[MAX_CONCURRENT_SLOTS];
                    uint64_t g_seeds[MAX_CONCURRENT_SLOTS];
                    for (int r = 0; r < B; ++r) {
                        auto & job = b_jobs[r];
                        if (job->need_logits && use_gpu_full_sampler(job->sp)) {
                            g_rows[n_gpu]  = r;
                            g_inv_t[n_gpu] = 1.0f / job->sp.temperature;
                            g_min_p[n_gpu] = job->sp.min_p;
                            g_seeds[n_gpu] = job->rng();
                            ++n_gpu;
                        }
                    }
                    if (n_gpu > 0) {
                        int32_t g_tok[MAX_CONCURRENT_SLOTS];
                        float   g_stats[MAX_CONCURRENT_SLOTS * 4];
                        if (eng_.sample_batch_rows_full_to_host(n_gpu, g_rows, g_inv_t, g_min_p, g_seeds,
                                                                g_tok, g_stats)) {
                            for (int k = 0; k < n_gpu; ++k) {
                                const int r = g_rows[k];
                                if (g_tok[k] >= 0 && g_tok[k] < VOCAB_UNPADDED && g_stats[4 * k + 1] > 0.0f) {
                                    gpu_sampled[r]    = g_tok[k];
                                    gpu_sampled_ok[r] = true;
                                }
                            }
                        }
                    }
                }
                for (int r = 0; r < B; ++r) {
                    const int c = b_slots[r];
                    auto & job = b_jobs[r];
                    if (c == 0 && drafter_ != nullptr) drafter_in_sync_ = false;
                    if (!ok) {
                        clear_slot_cache(c);
                        finish_job(job, "length");
                        continue;
                    }
                    if (slot_cache_valid_[c]) {
                        slot_cached_toks_[c].push_back(b_toks[r]);
                        update_cache_span(c);
                    }
                    const int32_t sampled = !job->need_logits
                        ? b_argmax[r]
                        : (use_gpu_full_sampler(job->sp)
                            ? (gpu_sampled_ok[r] ? gpu_sampled[r] : b_argmax[r])
                            : sample_logits_ptr(batch_logits_.data() + (size_t)r * VOCAB_UNPADDED,
                                                VOCAB_UNPADDED, job->sp, job->rng));
                    (void)emit_token(job, sampled);
                }
            }
        }

        // Shutdown: fail any remaining active/queued jobs cleanly so HTTP threads unblock
        for (int c = 0; c < eng_.num_slots(); ++c) {
            if (active_slots_[c] != nullptr) finish_job(active_slots_[c], "stop");
        }
        {
            std::lock_guard<std::mutex> lk(sched_mu_);
            while (!pending_.empty()) {
                auto j = pending_.front();
                pending_.pop_front();
                finish_job(j, "stop");
            }
            pending_count_.store(0, std::memory_order_relaxed);
            reset_cv_.notify_all();
        }

        if (ev0_) { (void)hipEventDestroy(ev0_); ev0_ = nullptr; }
        if (ev1_) { (void)hipEventDestroy(ev1_); ev1_ = nullptr; }
        if (ev2_) { (void)hipEventDestroy(ev2_); ev2_ = nullptr; }
    }

    void handle_generate(int fd, const HttpRequest & req, bool is_chat) {
        JsonVal j;
        JsonParser jp{req.body, 0};
        if (!jp.parse_val(j) || !j.is_obj()) {
            send_http_response(fd, 400, "Bad Request", "application/json",
                               "{\"error\":{\"message\":\"Invalid JSON body\"}}");
            return;
        }

        const bool stream = j.get_bool("stream", false);
        const int32_t req_max_ctx = request_max_ctx();
        int max_tokens = (int)j.get_num("max_tokens", j.get_num("max_completion_tokens", 131072));
        if (max_tokens < 1) max_tokens = 1;
        if (max_tokens > req_max_ctx) max_tokens = req_max_ctx;

        SampleParams sp;
        sp.temperature = (float)j.get_num("temperature", cfg_.default_temperature);
        sp.top_p       = (float)j.get_num("top_p", cfg_.default_top_p);
        sp.min_p       = (float)j.get_num("min_p", cfg_.default_min_p);
        sp.top_k       = (int)j.get_num("top_k", cfg_.default_top_k);
        if (const JsonVal * sv = j.get("seed"); sv && sv->is_num()) {
            sp.has_seed = true;
            sp.seed = (uint64_t)sv->num;
        }
        const bool allow_mtp = j.get_bool("mtp", true);

        std::vector<std::string> stop_seqs;
        if (const JsonVal * st = j.get("stop")) {
            if (st->is_str() && !st->str.empty()) stop_seqs.push_back(st->str);
            else if (st->is_arr()) {
                for (const auto & el : st->arr) {
                    if (el.is_str() && !el.str.empty()) stop_seqs.push_back(el.str);
                }
            }
        }

        std::vector<PromptSegment> segs;
        bool has_images = false;
        bool primed_text = false;
        std::string err;
        if (is_chat) {
            if (!build_inkling_chat_segments(j, bpe_, segs, has_images, primed_text, err)) {
                send_http_response(fd, 400, "Bad Request", "application/json",
                                   "{\"error\":{\"message\":" + json_str(err) + "}}");
                return;
            }
            if (has_images && vision_ == nullptr) {
                send_http_response(fd, 400, "Bad Request", "application/json",
                                   "{\"error\":{\"message\":\"Vision (--mmproj) is not loaded on this server\"}}");
                return;
            }
        } else {
            std::string prompt = j.get_str("prompt");
            if (prompt.empty()) {
                send_http_response(fd, 400, "Bad Request", "application/json",
                                   "{\"error\":{\"message\":\"Missing 'prompt' string\"}}");
                return;
            }
            const bool parse_special = j.get_bool("parse_special", true);
            PromptSegment ps;
            ps.is_image = false;
            ps.toks = bpe_.encode(prompt, parse_special);
            segs.push_back(std::move(ps));
        }

        auto job = std::make_shared<StreamJob>();
        job->req_id      = make_id();
        job->is_chat     = is_chat;
        job->stream      = stream;
        job->has_images  = has_images;
        job->primed_text = primed_text;
        job->allow_mtp   = allow_mtp;
        job->need_logits = (sp.temperature > 1e-5f);
        job->max_tokens  = max_tokens;
        job->sp          = sp;
        job->rng.seed(sp.has_seed ? sp.seed : (uint64_t)std::random_device{}());
        job->t_enqueued  = std::chrono::steady_clock::now();

        if (!has_images) {
            size_t total_toks = 0;
            for (const auto & sg : segs) total_toks += sg.toks.size();
            job->prompt_toks.reserve(total_toks);
            for (const auto & sg : segs) {
                job->prompt_toks.insert(job->prompt_toks.end(), sg.toks.begin(), sg.toks.end());
            }
            if (job->prompt_toks.empty()) {
                send_http_response(fd, 400, "Bad Request", "application/json",
                                   "{\"error\":{\"message\":\"Prompt encoded to 0 tokens\"}}");
                return;
            }
            if ((int)job->prompt_toks.size() + 16 >= req_max_ctx) {
                send_http_response(fd, 400, "Bad Request", "application/json",
                                   context_overflow_error_json(req_max_ctx, (int32_t)job->prompt_toks.size()));
                return;
            }
            job->est_prompt_toks = (int)job->prompt_toks.size();
            job->live_prompt_toks.store((int)job->prompt_toks.size(), std::memory_order_relaxed);
        } else {
            // Text segments are already pre-tokenized on the HTTP thread (exact ids; shared by
            // the scheduler's prefix matcher and the multimodal prefill loop).
            int est = 0;
            for (const auto & sg : segs) {
                if (sg.is_image) {
                    est += 1024; // conservative per-image token estimate for initial span sizing
                } else {
                    est += (int)sg.toks.size();
                }
            }
            if (est + 16 >= req_max_ctx) {
                send_http_response(fd, 400, "Bad Request", "application/json",
                                   context_overflow_error_json(req_max_ctx, est));
                return;
            }
            job->est_prompt_toks = est;
            job->live_prompt_toks.store(est, std::memory_order_relaxed);
            job->segs = std::move(segs);
        }

        const std::string req_id = job->req_id;
        const long created = (long)std::time(nullptr);

        // Enqueue to GPU continuous-batching scheduler
        {
            std::lock_guard<std::mutex> lk(sched_mu_);
            pending_.push_back(job);
            pending_count_.store((int)pending_.size(), std::memory_order_relaxed);
            busy_.store(true, std::memory_order_relaxed);
        }
        sched_cv_.notify_one();

        // Wait for prefill completion (polling socket disconnect every 50 ms)
        {
            std::unique_lock<std::mutex> qlk(job->q_mu);
            while (!job->prefill_ready && !job->done) {
                job->q_cv.wait_for(qlk, std::chrono::milliseconds(50), [&]() {
                    return job->prefill_ready || job->done;
                });
                if (g_stop_server.load() || client_disconnected(fd)) {
                    job->cancelled.store(true);
                    sched_cv_.notify_one();
                    return;
                }
            }
            if (job->failed) {
                const int code = job->http_err_code;
                const char * text = (code == 400) ? "Bad Request" : "Internal Server Error";
                std::string body = !job->http_err_raw_json.empty()
                    ? job->http_err_raw_json
                    : ("{\"error\":{\"message\":" + json_str(job->http_err_msg) + "}}");
                send_http_response(fd, code, text, "application/json", body);
                return;
            }
        }

        // Send SSE headers once prefill succeeded
        if (stream && !send_sse_headers(fd)) {
            job->cancelled.store(true);
            return;
        }

        GenResult gr;
        enum ChatPart { PART_IDLE, PART_HEADER, PART_THINKING, PART_CONTENT, PART_TOOL_JSON, PART_TOOL_TEXT };
        ChatPart part = (!is_chat || primed_text) ? PART_CONTENT : PART_IDLE;
        Utf8StreamBuf u8_think, u8_content, u8_tool;
        std::string header_buf, tool_buf;
        bool aborted = false;

        // Incremental SSE tool_calls streaming state for PART_TOOL_JSON
        bool tool_stream_started = false;
        bool tool_args_started   = false;
        bool tool_args_done      = false;
        bool tool_in_str         = false;
        bool tool_esc            = false;
        int  tool_brace_depth    = 0;
        size_t tool_scan_pos     = 0;
        size_t tool_cur_idx      = 0;
        std::string tool_cur_id, tool_cur_name, tool_args_out;

        auto trim_str = [](const std::string & s) -> std::string {
            size_t a = 0, b = s.size();
            while (a < b && (s[a] == ' ' || s[a] == '\n' || s[a] == '\r' || s[a] == '\t')) ++a;
            while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\n' || s[b - 1] == '\r' || s[b - 1] == '\t')) --b;
            return s.substr(a, b - a);
        };

        auto reset_tool_stream = [&]() {
            tool_stream_started = false;
            tool_args_started   = false;
            tool_args_done      = false;
            tool_in_str         = false;
            tool_esc            = false;
            tool_brace_depth    = 0;
            tool_scan_pos       = 0;
            tool_cur_idx        = 0;
            tool_cur_id.clear();
            tool_cur_name.clear();
            tool_args_out.clear();
        };

        auto send_sse_delta = [&](const std::string & field, const std::string & txt) {
            if (txt.empty() || aborted) return;
            std::ostringstream oss;
            if (is_chat) {
                oss << "data: {\"id\":" << json_str(req_id)
                    << ",\"object\":\"chat.completion.chunk\",\"created\":" << created
                    << ",\"model\":" << json_str(cfg_.model_name)
                    << ",\"choices\":[{\"index\":0,\"delta\":{"
                    << json_str(field) << ":" << json_str(txt)
                    << "},\"finish_reason\":null}]}\n\n";
            } else {
                oss << "data: {\"id\":" << json_str(req_id)
                    << ",\"object\":\"text_completion\",\"created\":" << created
                    << ",\"model\":" << json_str(cfg_.model_name)
                    << ",\"choices\":[{\"index\":0,\"text\":" << json_str(txt)
                    << ",\"finish_reason\":null}]}\n\n";
            }
            if (!send_all(fd, oss.str())) aborted = true;
        };

        auto pump_tool_json_stream = [&]() {
            if (part != PART_TOOL_JSON || tool_args_done || aborted) return;
            if (!tool_args_started) {
                size_t ap = tool_buf.find("\"args\"");
                if (ap == std::string::npos) return;
                size_t col = tool_buf.find(':', ap + 6);
                if (col == std::string::npos) return;
                size_t p = col + 1;
                while (p < tool_buf.size() && (tool_buf[p] == ' ' || tool_buf[p] == '\n' || tool_buf[p] == '\r' || tool_buf[p] == '\t')) ++p;
                if (p >= tool_buf.size() || tool_buf[p] != '{') return;

                tool_cur_name = trim_str(header_buf);
                if (tool_cur_name.empty()) {
                    JsonVal prefix;
                    std::string tmp = tool_buf.substr(0, ap) + "\"args\":{}}";
                    JsonParser pp{tmp, 0};
                    if (pp.parse_val(prefix) && prefix.is_obj()) tool_cur_name = prefix.get_str("name");
                }
                if (tool_cur_name.empty()) return;

                tool_args_started   = true;
                tool_stream_started = true;
                tool_scan_pos       = p;
                tool_brace_depth    = 0;
                tool_in_str         = false;
                tool_esc            = false;
                tool_cur_idx        = gr.tool_calls.size();
                tool_cur_id         = "call_" + req_id + "_" + std::to_string(tool_cur_idx);
                tool_args_out.clear();

                if (stream && !aborted) {
                    std::ostringstream oss;
                    oss << "data: {\"id\":" << json_str(req_id)
                        << ",\"object\":\"chat.completion.chunk\",\"created\":" << created
                        << ",\"model\":" << json_str(cfg_.model_name)
                        << ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":" << tool_cur_idx
                        << ",\"id\":" << json_str(tool_cur_id)
                        << ",\"type\":\"function\",\"function\":{\"name\":" << json_str(tool_cur_name)
                        << ",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n";
                    if (!send_all(fd, oss.str())) aborted = true;
                }
            }
            if (tool_args_started && !tool_args_done && !aborted) {
                std::string chunk;
                while (tool_scan_pos < tool_buf.size()) {
                    char ch = tool_buf[tool_scan_pos++];
                    chunk.push_back(ch);
                    tool_args_out.push_back(ch);
                    if (tool_in_str) {
                        if (tool_esc) { tool_esc = false; }
                        else if (ch == '\\') { tool_esc = true; }
                        else if (ch == '"') { tool_in_str = false; }
                    } else {
                        if (ch == '"') { tool_in_str = true; }
                        else if (ch == '{') { ++tool_brace_depth; }
                        else if (ch == '}') {
                            --tool_brace_depth;
                            if (tool_brace_depth == 0) {
                                tool_args_done = true;
                                break;
                            }
                        }
                    }
                }
                if (stream && !chunk.empty() && !aborted) {
                    std::ostringstream oss;
                    oss << "data: {\"id\":" << json_str(req_id)
                        << ",\"object\":\"chat.completion.chunk\",\"created\":" << created
                        << ",\"model\":" << json_str(cfg_.model_name)
                        << ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":" << tool_cur_idx
                        << ",\"function\":{\"arguments\":" << json_str(chunk)
                        << "}}]},\"finish_reason\":null}]}\n\n";
                    if (!send_all(fd, oss.str())) aborted = true;
                }
            }
        };

        auto emit_tool_call = [&]() {
            tool_buf += u8_tool.flush();
            if (part == PART_TOOL_JSON) pump_tool_json_stream();
            if (tool_stream_started) {
                ParsedToolCall tc;
                tc.id        = tool_cur_id;
                tc.name      = tool_cur_name;
                tc.arguments = tool_args_out.empty() ? "{}" : tool_args_out;
                gr.tool_calls.push_back(std::move(tc));
                reset_tool_stream();
                header_buf.clear();
                tool_buf.clear();
                return;
            }
            std::string fn_name = trim_str(header_buf);
            std::string fn_args = "{}";
            std::string trimmed = trim_str(tool_buf);
            if (!trimmed.empty()) {
                JsonVal parsed;
                JsonParser tp{trimmed, 0};
                if (tp.parse_val(parsed) && parsed.is_obj()) {
                    if (std::string pn = parsed.get_str("name"); !pn.empty()) fn_name = pn;
                    if (const JsonVal * a = parsed.get("args")) {
                        fn_args = json_serialize(*a);
                    } else if (const JsonVal * a2 = parsed.get("arguments")) {
                        fn_args = a2->is_str() ? a2->str : json_serialize(*a2);
                    } else {
                        fn_args = json_serialize(parsed);
                    }
                } else {
                    fn_args = trimmed;
                }
            }
            if (!fn_name.empty()) {
                const size_t tc_idx = gr.tool_calls.size();
                ParsedToolCall tc;
                tc.id = "call_" + req_id + "_" + std::to_string(tc_idx);
                tc.name = fn_name;
                tc.arguments = fn_args;
                gr.tool_calls.push_back(tc);
                if (stream && !aborted) {
                    std::ostringstream oss;
                    oss << "data: {\"id\":" << json_str(req_id)
                        << ",\"object\":\"chat.completion.chunk\",\"created\":" << created
                        << ",\"model\":" << json_str(cfg_.model_name)
                        << ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":" << tc_idx
                        << ",\"id\":" << json_str(tc.id)
                        << ",\"type\":\"function\",\"function\":{\"name\":" << json_str(tc.name)
                        << ",\"arguments\":" << json_str(tc.arguments)
                        << "}}]},\"finish_reason\":null}]}\n\n";
                    if (!send_all(fd, oss.str())) aborted = true;
                }
            }
            reset_tool_stream();
            header_buf.clear();
            tool_buf.clear();
        };

        auto process_token = [&](int32_t tok) -> bool {
            if (aborted || g_stop_server.load() || (stream && client_disconnected(fd))) {
                aborted = true;
                gr.finish_reason = "stop";
                return false;
            }
            ++gr.completion_tokens;
            if (is_chat) {
                if (tok == id_model_) {
                    header_buf.clear();
                    part = PART_HEADER;
                    return true;
                }
                if (tok == id_thinking_) {
                    header_buf.clear();
                    part = PART_THINKING;
                    return true;
                }
                if (tok == id_text_) {
                    header_buf.clear();
                    part = PART_CONTENT;
                    return true;
                }
                if (tok == id_invoke_tool_json_ || tok == id_invoke_tool_text_) {
                    reset_tool_stream();
                    tool_buf.clear();
                    (void)u8_tool.flush();
                    part = (tok == id_invoke_tool_json_) ? PART_TOOL_JSON : PART_TOOL_TEXT;
                    return true;
                }
                if (tok == id_end_msg_) {
                    if (part == PART_THINKING) {
                        std::string tail = u8_think.flush();
                        if (!tail.empty()) { gr.reasoning += tail; if (stream) send_sse_delta("reasoning_content", tail); }
                    } else if (part == PART_CONTENT) {
                        std::string tail = u8_content.flush();
                        if (!tail.empty()) { gr.content += tail; if (stream) send_sse_delta("content", tail); }
                    } else if (part == PART_TOOL_JSON || part == PART_TOOL_TEXT) {
                        emit_tool_call();
                    }
                    header_buf.clear();
                    part = PART_IDLE;
                    return true;
                }
                std::string raw = bpe_.decode_token(tok, /*skip_special=*/true);
                if (raw.empty()) return true;
                if (part == PART_HEADER) {
                    header_buf += raw;
                } else if (part == PART_THINKING) {
                    std::string ready = u8_think.push(raw);
                    if (!ready.empty()) {
                        gr.reasoning += ready;
                        if (stream) send_sse_delta("reasoning_content", ready);
                    }
                } else if (part == PART_TOOL_JSON || part == PART_TOOL_TEXT) {
                    std::string ready = u8_tool.push(raw);
                    if (!ready.empty()) {
                        tool_buf += ready;
                        if (part == PART_TOOL_JSON) pump_tool_json_stream();
                    }
                } else {
                    std::string ready = u8_content.push(raw);
                    if (!ready.empty()) {
                        gr.content += ready;
                        if (stream) send_sse_delta("content", ready);
                    }
                }
            } else {
                std::string raw = bpe_.decode_token(tok, /*skip_special=*/false);
                std::string ready = u8_content.push(raw);
                if (!ready.empty()) {
                    gr.content += ready;
                    if (stream) send_sse_delta("text", ready);
                }
            }

            for (const std::string & s : stop_seqs) {
                if (gr.content.size() >= s.size() &&
                    gr.content.compare(gr.content.size() - s.size(), s.size(), s) == 0) {
                    gr.content.erase(gr.content.size() - s.size());
                    gr.finish_reason = "stop";
                    return false;
                }
            }
            return true;
        };

        // Drain token stream from GPU scheduler
        bool stopped_by_http = false;
        while (true) {
            std::deque<int32_t> batch;
            bool job_done = false;
            {
                std::unique_lock<std::mutex> qlk(job->q_mu);
                while (job->out_toks.empty() && !job->done) {
                    job->q_cv.wait_for(qlk, std::chrono::milliseconds(50), [&]() {
                        return !job->out_toks.empty() || job->done;
                    });
                    if (g_stop_server.load() || (stream && client_disconnected(fd))) {
                        aborted = true;
                        stopped_by_http = true;
                        job->cancelled.store(true);
                        break;
                    }
                }
                if (stopped_by_http) break;
                batch.swap(job->out_toks);
                job_done = job->done;
            }
            for (int32_t tok : batch) {
                if (!process_token(tok)) {
                    stopped_by_http = true;
                    job->cancelled.store(true);
                    break;
                }
            }
            if (stopped_by_http || job_done) break;
        }

        // Wait briefly for scheduler to finalize job stats if we stopped early
        if (stopped_by_http) {
            std::unique_lock<std::mutex> qlk(job->q_mu);
            job->q_cv.wait_for(qlk, std::chrono::milliseconds(500), [&]() { return job->done; });
        }

        // Flush remaining UTF-8 bytes / open tool call
        if (part == PART_TOOL_JSON || part == PART_TOOL_TEXT) {
            emit_tool_call();
        }
        if (std::string t = u8_think.flush(); !t.empty()) {
            gr.reasoning += t;
            if (stream) send_sse_delta("reasoning_content", t);
        }
        if (std::string c = u8_content.flush(); !c.empty()) {
            gr.content += c;
            if (stream) send_sse_delta(is_chat ? "content" : "text", c);
        }

        // Copy final telemetry from job
        {
            std::lock_guard<std::mutex> qlk(job->q_mu);
            gr.prompt_tokens     = job->final_prompt_tokens;
            gr.cache_hit_tokens  = job->final_cache_hit_tokens;
            gr.slot              = job->final_slot;
            gr.slot_span         = job->final_slot_span;
            gr.prompt_ms         = job->final_prompt_ms;
            gr.predicted_ms      = job->final_predicted_ms;
            gr.mtp_used          = job->final_mtp_used;
            gr.mtp_tok_per_round = job->final_mtp_tok_per_round;
            gr.mtp_avg_accepted  = job->final_mtp_avg_accepted;
            gr.mtp_draft_ms      = job->final_mtp_draft_ms;
            gr.mtp_verify_ms     = job->final_mtp_verify_ms;
            if (!stopped_by_http) {
                gr.finish_reason = job->finish_reason;
            }
        }
        if (is_chat && !gr.tool_calls.empty() && gr.finish_reason == "stop") {
            gr.finish_reason = "tool_calls";
        }

        const double tps = (gr.predicted_ms > 0.0) ? ((double)gr.completion_tokens / (gr.predicted_ms / 1000.0)) : 0.0;
        char slot_tag[24];
        if (gr.slot_span > 1) {
            std::snprintf(slot_tag, sizeof(slot_tag), "s%d..%d", gr.slot, gr.slot + gr.slot_span - 1);
        } else {
            std::snprintf(slot_tag, sizeof(slot_tag), "s%d", gr.slot);
        }
        char log_buf[384];
        std::snprintf(log_buf, sizeof(log_buf),
                      "%s %s [%s] | prompt=%d (cached=%d, %.0f ms) | gen=%d in %.0f ms (%.1f tok/s%s%s)",
                      is_chat ? "chat" : "cmpl", stream ? "stream" : "sync", slot_tag,
                      gr.prompt_tokens, gr.cache_hit_tokens, gr.prompt_ms,
                      gr.completion_tokens, gr.predicted_ms, tps,
                      gr.mtp_used ? ", MTP" : "",
                      gr.tool_calls.empty() ? "" : ", tools");
        std::printf("[serve] %s\n", log_buf);
        std::fflush(stdout);
        record_activity(gr, stream, has_images);
        add_log("REQ", log_buf);

        // Final response / SSE finish chunk
        if (stream) {
            if (!aborted) {
                std::ostringstream oss;
                if (is_chat) {
                    oss << "data: {\"id\":" << json_str(req_id)
                        << ",\"object\":\"chat.completion.chunk\",\"created\":" << created
                        << ",\"model\":" << json_str(cfg_.model_name)
                        << ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":"
                        << json_str(gr.finish_reason) << "}]"
                        << ",\"usage\":" << usage_json(gr)
                        << ",\"timings\":" << timings_json(gr) << "}\n\n"
                        << "data: [DONE]\n\n";
                } else {
                    oss << "data: {\"id\":" << json_str(req_id)
                        << ",\"object\":\"text_completion\",\"created\":" << created
                        << ",\"model\":" << json_str(cfg_.model_name)
                        << ",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":"
                        << json_str(gr.finish_reason) << "}]"
                        << ",\"usage\":" << usage_json(gr)
                        << ",\"timings\":" << timings_json(gr) << "}\n\n"
                        << "data: [DONE]\n\n";
                }
                (void)send_all(fd, oss.str());
            }
        } else {
            std::ostringstream oss;
            if (is_chat) {
                oss << "{\"id\":" << json_str(req_id)
                    << ",\"object\":\"chat.completion\",\"created\":" << created
                    << ",\"model\":" << json_str(cfg_.model_name)
                    << ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":";
                if (gr.content.empty() && !gr.tool_calls.empty()) {
                    oss << "null";
                } else {
                    oss << json_str(gr.content);
                }
                if (!gr.reasoning.empty()) {
                    oss << ",\"reasoning_content\":" << json_str(gr.reasoning);
                }
                if (!gr.tool_calls.empty()) {
                    oss << ",\"tool_calls\":[";
                    for (size_t i = 0; i < gr.tool_calls.size(); ++i) {
                        if (i > 0) oss << ",";
                        const auto & tc = gr.tool_calls[i];
                        oss << "{\"id\":" << json_str(tc.id)
                            << ",\"type\":\"function\",\"function\":{\"name\":" << json_str(tc.name)
                            << ",\"arguments\":" << json_str(tc.arguments) << "}}";
                    }
                    oss << "]";
                }
                oss << "},\"finish_reason\":" << json_str(gr.finish_reason) << "}]"
                    << ",\"usage\":" << usage_json(gr)
                    << ",\"timings\":" << timings_json(gr) << "}";
            } else {
                oss << "{\"id\":" << json_str(req_id)
                    << ",\"object\":\"text_completion\",\"created\":" << created
                    << ",\"model\":" << json_str(cfg_.model_name)
                    << ",\"choices\":[{\"index\":0,\"text\":" << json_str(gr.content)
                    << ",\"finish_reason\":" << json_str(gr.finish_reason) << "}]"
                    << ",\"usage\":" << usage_json(gr)
                    << ",\"timings\":" << timings_json(gr) << "}";
            }
            send_http_response(fd, 200, "OK", "application/json", oss.str());
        }
    }

    InklingEngine & eng_;
    const BpeTokenizer & bpe_;
    DSparkDrafter * drafter_;
    VisionEncoder * vision_;
    ServerConfig cfg_;

    int32_t id_model_            = -1;
    int32_t id_thinking_         = -1;
    int32_t id_text_             = -1;
    int32_t id_end_msg_          = -1;
    int32_t id_end_sampling_     = -1;
    int32_t id_eot_              = -1;
    int32_t id_invoke_tool_json_ = -1;
    int32_t id_invoke_tool_text_ = -1;

    // Continuous batching scheduler state
    std::mutex sched_mu_;
    std::condition_variable sched_cv_;
    std::deque<std::shared_ptr<StreamJob>> pending_;
    std::atomic<int> pending_count_{0};
    std::shared_ptr<StreamJob> active_slots_[MAX_CONCURRENT_SLOTS];     // GPU thread owned
    std::shared_ptr<StreamJob> active_slots_pub_[MAX_CONCURRENT_SLOTS]; // sched_mu_ guarded snapshot
    int slot_span_[MAX_CONCURRENT_SLOTS] = {};                          // GPU thread owned
    int slot_span_pub_[MAX_CONCURRENT_SLOTS] = {};                      // sched_mu_ guarded snapshot
    int seg_owner_[MAX_CONCURRENT_SLOTS] = {};                          // GPU thread owned (-1 = free, else base slot)
    // Per-slot prefix cache. A cached prefix is [prologue segments][trailing tokens]:
    //   - slot_mm_prologue_  : multimodal prompt up to and including its last image
    //                          (text segments as token ids, images as 128-bit hash + token count)
    //   - slot_cached_toks_  : everything after it (remaining prompt text + committed decode tokens)
    // Both are compared against the incoming request, so a multimodal turn whose image
    // bytes are unchanged replays the KV instead of re-running the ViT + prefill.
    std::vector<SlotCacheSeg> slot_mm_prologue_[MAX_CONCURRENT_SLOTS];
    std::vector<int32_t> slot_cached_toks_[MAX_CONCURRENT_SLOTS];
    bool slot_cache_valid_[MAX_CONCURRENT_SLOTS] = {};
    int slot_cache_span_[MAX_CONCURRENT_SLOTS] = {};
    // Host-RAM prompt-end checkpoint fallback for slot 0 (0 extra VRAM, text + multimodal).
    static constexpr int SLOT0_CKPT_MIN_TOKENS = 512;
    std::vector<SlotCacheSeg> slot0_ckpt_mm_prologue_;
    std::vector<int32_t> slot0_ckpt_toks_;
    InklingEngine::Slot0Checkpoint slot0_ckpt_eng_;
    DSparkDrafter::Checkpoint slot0_ckpt_drafter_;
    bool slot0_ckpt_valid_ = false;
    bool slot0_ckpt_drafter_valid_ = false;
    int  slot0_ckpt_span_ = 0;
    uint64_t slot_last_used_[MAX_CONCURRENT_SLOTS] = {};
    uint64_t sched_lru_seq_ = 0;
    uint64_t vis_encode_gen_ = 0;
    bool drafter_in_sync_ = false;
    std::vector<float> batch_logits_;
    hipEvent_t ev0_ = nullptr, ev1_ = nullptr, ev2_ = nullptr;

    std::atomic<bool> reset_requested_{false};
    uint64_t reset_gen_ = 0;
    std::condition_variable reset_cv_;

    std::thread sched_thread_;
    bool sched_running_ = false;
    std::atomic<bool> sched_stop_{false};

    std::chrono::steady_clock::time_point start_time_;
    std::atomic<bool> busy_{false};
    std::mutex telemetry_mu_;
    std::deque<ActivityRecord> activity_;
    std::deque<LogEntry> logs_;
    uint64_t total_requests_ = 0;
    uint64_t total_prompt_tokens_ = 0;
    uint64_t total_completion_tokens_ = 0;
};

} // namespace

int run_http_server(
    InklingEngine & eng,
    const BpeTokenizer & bpe,
    DSparkDrafter * drafter,
    VisionEncoder * vision,
    const ServerConfig & cfg)
{
    ::signal(SIGPIPE, SIG_IGN);
    ::signal(SIGINT, handle_sig);
    ::signal(SIGTERM, handle_sig);
    g_stop_server.store(false);

    InklingServer srv(eng, bpe, drafter, vision, cfg);
    std::printf("[serve] warming up HIP graphs & multi-slot batch kernels...\n");
    std::fflush(stdout);
    srv.warmup();
    srv.start_scheduler();

    int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) {
        std::perror("[serve] socket");
        srv.stop_scheduler();
        return 1;
    }
    int opt = 1;
    (void)::setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t)cfg.port);
    if (::inet_pton(AF_INET, cfg.host.c_str(), &addr.sin_addr) != 1) {
        addr.sin_addr.s_addr = INADDR_ANY;
    }
    if (::bind(lfd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        std::perror("[serve] bind");
        ::close(lfd);
        srv.stop_scheduler();
        return 1;
    }
    if (::listen(lfd, 64) != 0) {
        std::perror("[serve] listen");
        ::close(lfd);
        srv.stop_scheduler();
        return 1;
    }
    g_listen_fd.store(lfd);

    const double llm_vram_gib = (double)eng.total_vram_bytes() / (1024.0 * 1024.0 * 1024.0);
    const double mtp_vram_gib = drafter ? ((double)drafter->vram_bytes() / (1024.0 * 1024.0 * 1024.0)) : 0.0;
    const double vis_vram_gib = vision  ? ((double)vision->vram_bytes()  / (1024.0 * 1024.0 * 1024.0)) : 0.0;
    const double vram = llm_vram_gib + mtp_vram_gib + vis_vram_gib;
    int32_t req_max = eng.max_ctx();
    if (cfg.strict_slot_ctx && eng.num_slots() > 1) req_max = std::min(req_max, eng.slot_ctx());
    if (cfg.max_slot_ctx > 0) req_max = std::min(req_max, cfg.max_slot_ctx);
    const char * ctx_mode = (eng.num_slots() <= 1) ? "solo"
                          : (req_max <= eng.slot_ctx()) ? "strict"
                          : "elastic";
    std::printf("[nib] v%s listening on http://%s:%d\n",
                NIB_VERSION, cfg.host.c_str(), cfg.port);
    std::printf("[nib] Web UI: http://%s:%d/  |  OpenAI API: http://%s:%d/v1/chat/completions\n",
                cfg.host.c_str(), cfg.port, cfg.host.c_str(), cfg.port);
    std::printf("[nib] Config: ctx=%d slots=%d(%d/slot, req_max=%d, %s) kv=%s sparse_pages=%d temp=%.2f top_p=%.2f top_k=%d min_p=%.2f mtp=%s vision=%s vram=%.2f GiB\n",
                eng.max_ctx(), eng.num_slots(), eng.slot_ctx(), req_max, ctx_mode,
                cfg.kv_type_name.c_str(), cfg.sparse_pages, cfg.default_temperature,
                cfg.default_top_p, cfg.default_top_k, cfg.default_min_p,
                drafter ? "ON" : "OFF", vision ? "ON" : "OFF", vram);
    size_t dev_free_b = 0, dev_total_b = 0;
    if (hipMemGetInfo(&dev_free_b, &dev_total_b) == hipSuccess && dev_total_b > 0) {
        const double free_gib  = (double)dev_free_b  / (1024.0 * 1024.0 * 1024.0);
        const double total_gib = (double)dev_total_b / (1024.0 * 1024.0 * 1024.0);
        const double used_gib  = total_gib - free_gib;
        std::printf("[nib] Device VRAM: %.2f / %.2f GiB used (%.2f GiB free; app=%.2f GiB: llm=%.2f + mtp=%.2f + vision=%.2f)\n",
                    used_gib, total_gib, free_gib, vram, llm_vram_gib, mtp_vram_gib, vis_vram_gib);
        if (dev_free_b < (size_t)512 * 1024 * 1024) {
            std::printf("[nib] WARNING: Only %.0f MiB free GPU VRAM remaining! On Linux (amdgpu TTM) or Windows/WSL2 (WDDM), "
                        "running this close to physical VRAM can silently spill buffers to PCIe GTT (severe decode slowdown) or OOM.\n"
                        "[nib]          Reduce -c or --slots, use --kv-type q4_1/q4_0, or pass --no-vision to regain headroom.\n",
                        (double)dev_free_b / (1024.0 * 1024.0));
        } else if (vision != nullptr && dev_free_b < (size_t)1024 * 1024 * 1024) {
            std::printf("[nib] NOTE: %.0f MiB free GPU VRAM with Vision enabled — each image transiently allocates its ViT workspace "
                        "(~%.0f MiB at 1080p, up to %.0f MiB at max resolution)%s.\n",
                        (double)dev_free_b / (1024.0 * 1024.0),
                        (double)vision->estimate_encode_bytes(1296) / (1024.0 * 1024.0),
                        (double)vision->estimate_max_encode_bytes() / (1024.0 * 1024.0),
                        vision->keep_scratch() ? " and keeps it resident (--vision-keep-scratch)" : ", freed right after the encode");
        }
        if (eng.cpu_embd()) {
            std::printf("[nib] tok_embd=CPU pinned (%.1f MiB of VRAM saved; GPU-mapped host RAM)\n",
                        (double)eng.host_pinned_bytes() / (1024.0 * 1024.0));
        }
    }
    std::fflush(stdout);

    while (!g_stop_server.load()) {
        sockaddr_in caddr {};
        socklen_t clen = sizeof(caddr);
        int cfd = ::accept(lfd, reinterpret_cast<sockaddr *>(&caddr), &clen);
        if (cfd < 0) {
            if (g_stop_server.load()) break;
            continue;
        }
        std::thread([&srv, cfd]() { srv.handle_client(cfd); }).detach();
    }

    int fd = g_listen_fd.exchange(-1);
    if (fd >= 0) ::close(fd);
    srv.stop_scheduler();
    std::printf("\n[serve] shutdown complete.\n");
    return 0;
}

} // namespace inkling
