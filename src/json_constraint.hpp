#pragma once

// ============================================================================
// nib — minimal JSON constrained decoding (response_format)
// ============================================================================
// Emits *canonical* JSON: no whitespace, object keys in schema order, only the
// properties listed in "required" (or all declared properties when the schema
// has no "required"), arrays of any length.  That is a strict subset of what a
// schema allows — exactly what a tool-call / JSON-output generator should
// produce — and it keeps the machine small enough to be obviously correct.
//
// Supported schema keywords: type (object/array/string/integer/number/boolean/
// null), properties, required, items, enum (string values).
// Unsupported ($ref, anyOf/oneOf/allOf, pattern, min/max, additionalProperties
// semantics) degrade to "any JSON value" rather than mis-constraining.
//
// The machine is a byte-level pushdown automaton.  Masking probes each
// vocabulary token by feeding its bytes into a copy of the state; tokens whose
// first byte cannot be accepted are rejected without a copy, which keeps the
// cost per step negligible for canonical JSON (only a few first bytes are ever
// legal, e.g. '"' or '}').

#include <cstdint>
#include <string>
#include <vector>

namespace nibj {

struct Node {
    enum Kind { ANY = 0, OBJ, ARR, STR, NUM, INT, BOOL, NUL, ENUMV } kind = ANY;
    std::vector<std::string> keys;       // OBJ: property names, in emission order
    std::vector<int>         kids;       // OBJ: schema index per key
    int                      items = -1; // ARR: element schema (-1 = any)
    std::vector<std::string> enums;      // ENUMV: allowed string values
};

class Constraint {
public:
    // ---- construction -----------------------------------------------------
    void build_any() {
        nodes_.clear();
        nodes_.push_back(Node{});
        root_ = 0;
        reset();
    }

    // `response_format: {"type":"json_object"}`: a JSON object with arbitrary keys.
    void build_any_object() {
        nodes_.clear();
        Node obj;
        obj.kind = Node::OBJ;      // no declared keys -> any key/value pairs
        nodes_.push_back(std::move(obj));
        root_ = 0;
        reset();
    }

    // Flat object schema: one level of scalar/enum properties.
    void build_object(const std::vector<std::string> & keys,
                      const std::vector<std::vector<std::string>> & enums,
                      const std::vector<int> & kinds,
                      const std::vector<bool> & required) {
        nodes_.clear();
        Node obj;
        obj.kind = Node::OBJ;
        for (size_t i = 0; i < keys.size(); ++i) {
            if (!required.empty() && !required[i]) continue;
            Node k;
            k.kind = (kinds[i] < 0) ? Node::ANY : static_cast<Node::Kind>(kinds[i]);
            if (k.kind == Node::ENUMV) k.enums = enums[i];
            // Nested containers are not modelled here; "any value" is safe.
            if (k.kind == Node::OBJ || k.kind == Node::ARR) k.kind = Node::ANY;
            nodes_.push_back(std::move(k));
            obj.keys.push_back(keys[i]);
            obj.kids.push_back((int)nodes_.size() - 1);
        }
        nodes_.push_back(std::move(obj));
        root_ = (int)nodes_.size() - 1;
        reset();
    }

    // Top-level scalar (string / number / boolean / null) or any value.
    void build_root(int kind, const std::vector<std::string> & enums) {
        nodes_.clear();
        Node n;
        n.kind = (kind < 0) ? Node::ANY : static_cast<Node::Kind>(kind);
        if (n.kind == Node::ENUMV) n.enums = enums;
        if (n.kind == Node::OBJ || n.kind == Node::ARR) n.kind = Node::ANY;
        nodes_.push_back(std::move(n));
        root_ = 0;
        reset();
    }

    // Build from a flat list of (kind, enum values) for an array schema.
    void build_array(int item_kind, const std::vector<std::string> & item_enums) {
        nodes_.clear();
        Node item;
        item.kind = (item_kind < 0) ? Node::ANY : static_cast<Node::Kind>(item_kind);
        if (item.kind == Node::ENUMV) item.enums = item_enums;
        if (item.kind == Node::OBJ || item.kind == Node::ARR) item.kind = Node::ANY;
        nodes_.push_back(std::move(item));
        Node arr;
        arr.kind = Node::ARR;
        arr.items = 0;
        nodes_.push_back(std::move(arr));
        root_ = 1;
        reset();
    }

    void reset() {
        st_.clear();
        need_    = N_VALUE;
        cur_     = root_;
        pending_ = -1;
        in_str_  = false;
        role_    = S_VALUE;
        esc_     = false;
        escu_    = 0;
        in_num_  = false;
        num_digits_ = false;
        num_dot_ = false;
        num_exp_ = false;
        num_zero_ = false;
        num_first_ = false;
        num_exp_sign_ok_ = false;
        lit_     = nullptr;
        lit_pos_ = 0;
        enum_left_.clear();
        enum_lit_.clear();
        done_ = false;
        dead_ = false;
    }

    bool active() const { return !done_ && !dead_; }
    bool done()   const { return done_; }

    // ---- one byte ---------------------------------------------------------
    bool feed(unsigned char c) {
        if (dead_ || done_) return false;

        if (in_str_) return feed_string(c);

        if (in_num_) {
            if (feed_number_byte(c)) return true;
            if (!num_digits_) return false;
            in_num_ = false;                 // number ends here
            if (!after_value()) return false;
            if (done_) return false;         // nothing may follow a complete value
            // fall through: this byte belongs to the post-value state
        }
        if (lit_) return feed_literal(c);

        // Insignificant whitespace is legal between JSON tokens.  Rejecting it
        // forces a canonical layout the model was not trained on, and a model
        // that wants `{\n  "key": …` then spends tokens contorting around the
        // mask (observed: an empty "" key and a dropped property).
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') return true;

        switch (need_) {
        case N_VALUE: return start_value(c);
        case N_KEY:   return start_key(c);
        case N_COLON:
            if (c != ':') return false;
            need_ = N_VALUE;
            cur_  = pending_;
            return true;
        case N_SEP:   return feed_sep(c);
        default:      return false;
        }
    }

    void advance(const std::string & bytes) {
        for (unsigned char c : bytes) {
            if (done_ || dead_) return;
            if (!feed(c) && !done_) { dead_ = true; return; }
        }
    }

    // Renders the allowed-token mask (1 = allowed).
    void fill(std::vector<uint8_t> & mask, const std::vector<std::string> & tok_bytes) const {
        const size_t n = tok_bytes.size();
        if (dead_) { mask.assign(n, 1); return; }   // constraint broke: stop interfering
        if (done_) {
            // The value is complete: nothing but a turn-ending token may follow,
            // otherwise the model appends prose to the JSON.
            mask.assign(n, 0);
            for (size_t id = 0; id < n; ++id) if (tok_bytes[id].empty()) mask[id] = 1;
            return;
        }
        mask.assign(n, 0);

        bool first_ok[256] = {false};
        for (int c = 0; c < 256; ++c) {
            Constraint p = *this;
            first_ok[c] = p.feed((unsigned char)c) || p.done();
        }
        for (size_t id = 0; id < n; ++id) {
            const std::string & s = tok_bytes[id];
            if (s.empty()) continue;   // specials are blocked mid-value: no escaping
            if (!first_ok[(unsigned char)s[0]]) continue;
            Constraint p = *this;
            bool ok = true;
            for (unsigned char ch : s) {
                if (!p.feed(ch)) { if (!p.done()) { ok = false; break; } }
            }
            if (ok) mask[id] = 1;
        }
    }

private:
    enum Need : int { N_VALUE, N_KEY, N_COLON, N_SEP, N_DONE };
    enum Role : int { S_VALUE, S_KEY_NAMED, S_KEY_ANY, S_ENUM };

    struct Frame {
        int  schema = -1;   // -1 = unconstrained container
        int  next   = 0;    // OBJ: index of next property
        bool arr    = false;
    };

    const Node & nd(int i) const { return nodes_[(size_t)i]; }

    bool push_obj(int schema) {
        Frame f;
        f.schema = schema;
        f.arr    = false;
        f.next   = 0;
        st_.push_back(f);
        need_ = N_KEY;
        cur_  = (schema >= 0 && !nd(schema).keys.empty()) ? nd(schema).kids[0] : -1;
        return true;
    }

    bool push_arr(int schema) {
        Frame f;
        f.schema = schema;
        f.arr    = true;
        st_.push_back(f);
        need_ = N_SEP;                       // empty array is legal
        cur_  = (schema >= 0) ? nd(schema).items : -1;
        return true;
    }

    bool start_value(unsigned char c) {
        // The root value must honour the declared type: without this the first
        // byte of the response is unconstrained (cur_ is still -1).
        if (cur_ < 0 && st_.empty() && root_ >= 0 && nd(root_).kind != Node::ANY)
            cur_ = root_;
        const int  kind = (cur_ >= 0) ? nd(cur_).kind : Node::ANY;
        switch (kind) {
        case Node::ANY:
            if (c == '{') return push_obj(-1);
            if (c == '[') return push_arr(-1);
            if (c == '"') { in_str_ = true; role_ = S_VALUE; return true; }
            if (c == '-' || (c >= '0' && c <= '9')) return start_num(c, false);
            if (c == 't') return start_lit("true");
            if (c == 'f') return start_lit("false");
            if (c == 'n') return start_lit("null");
            return false;
        case Node::OBJ: return (c == '{') ? push_obj(cur_) : false;
        case Node::ARR: return (c == '[') ? push_arr(cur_) : false;
        case Node::STR: if (c != '"') return false; in_str_ = true; role_ = S_VALUE; return true;
        case Node::ENUMV:
            if (c != '"') return false;
            in_str_   = true;
            role_     = S_ENUM;
            enum_left_ = nd(cur_).enums;
            enum_lit_.clear();
            return true;
        case Node::NUM: return start_num(c, false);
        case Node::INT: return start_num(c, true);
        case Node::BOOL:
            if (c == 't') return start_lit("true");
            if (c == 'f') return start_lit("false");
            return false;
        case Node::NUL: return start_lit("null");
        }
        return false;
    }

    bool start_key(unsigned char c) {
        if (st_.empty()) return false;
        Frame & f = st_.back();
        if (c == '}') {
            // every object must carry at least one member: an empty object is
            // almost never what the caller asked for
            const bool no_declared = (f.schema < 0) || nd(f.schema).keys.empty();
            if (no_declared && f.next == 0) return false;
            return close_obj();
        }
        if (c != '"') return false;
        in_str_ = true;
        enum_lit_.clear();
        enum_left_.clear();
        if (f.schema < 0 || nd(f.schema).keys.empty()) {
            // unconstrained object (no declared keys): any key is legal
            role_    = S_KEY_ANY;
            pending_ = -1;
        } else {
            const Node & n = nd(f.schema);
            role_       = S_KEY_NAMED;
            enum_left_.push_back(n.keys[(size_t)f.next]);
            pending_    = n.kids[(size_t)f.next];
        }
        return true;
    }

    bool close_obj() {
        if (st_.empty()) return false;
        Frame & f = st_.back();
        if (f.arr) return false;
        if (f.schema >= 0 && (size_t)f.next < nd(f.schema).keys.size()) return false;
        st_.pop_back();
        return after_value();
    }

    bool feed_string(unsigned char c) {
        if (esc_) {
            esc_ = false;
            if (c == 'u') { escu_ = 4; return true; }
            return c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' ||
                   c == 'n' || c == 'r' || c == 't';
        }
        if (escu_ > 0) {
            --escu_;
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
        }
        if (c == '\\') {
            // enum values and canonical keys are literal text: an escape cannot
            // match either, and allowing it would let the string run forever
            if (role_ == S_ENUM || role_ == S_KEY_NAMED) return false;
            esc_ = true;
            return true;
        }
        if (c == '"') {
            in_str_ = false;
            if (role_ == S_ENUM) {
                bool hit = false;
                for (const auto & e : enum_left_) if (e == enum_lit_) { hit = true; break; }
                if (!hit) return false;
            }
            const Role r = role_;
            role_ = S_VALUE;
            if (r == S_KEY_ANY && enum_lit_.empty()) return false;   // no empty keys
            if (r == S_KEY_NAMED) {
                // a key must match the declared name exactly: a prefix is not a
                // key, and accepting one silently drops a required property
                if (enum_left_.empty() || enum_left_[0] != enum_lit_) return false;
                need_ = N_COLON;
                return true;
            }
            if (r == S_KEY_ANY) { need_ = N_COLON; return true; }
            return after_value();
        }
        if (c < 0x20) return false;
        if (role_ == S_ENUM) {
            enum_lit_.push_back((char)c);
            for (const auto & e : enum_left_) {
                if (e.size() >= enum_lit_.size() &&
                    e.compare(0, enum_lit_.size(), enum_lit_) == 0) return true;
            }
            return false;
        }
        if (role_ == S_KEY_NAMED) {
            // canonical key: must match the expected property name
            enum_lit_.push_back((char)c);
            if (enum_left_.empty()) return false;
            const std::string & want = enum_left_[0];
            return want.size() >= enum_lit_.size() &&
                   want.compare(0, enum_lit_.size(), enum_lit_) == 0;
        }
        if (role_ == S_KEY_ANY) { enum_lit_.push_back((char)c); return true; }
        return true;
    }

    bool start_num(unsigned char c, bool integer_only) {
        in_num_          = true;
        num_digits_      = false;
        num_dot_         = false;
        num_exp_         = false;
        num_zero_        = false;
        num_first_       = true;
        num_exp_sign_ok_ = false;
        int_only_        = integer_only;
        return feed_number_byte(c);
    }

    // Canonical JSON numbers only: no leading '+', no leading zeroes, no
    // "1.", no doubled signs.  Anything looser emits documents that parse as
    // invalid, which is the one thing this machine must never do.
    bool feed_number_byte(unsigned char c) {
        const bool first = num_first_;
        num_first_ = false;
        if (c >= '0' && c <= '9') {
            if (num_zero_ && !num_dot_ && !num_exp_) return false;  // 01, -01, 0123
            if (!num_digits_ && c == '0') num_zero_ = true;         // leading zero
            num_digits_      = true;
            num_exp_sign_ok_ = false;
            return true;
        }
        if (c == '-') {
            if (first) return true;                                         // leading sign
            if (num_exp_ && num_exp_sign_ok_) { num_exp_sign_ok_ = false; return true; }
            return false;
        }
        if (c == '+') {
            if (num_exp_ && num_exp_sign_ok_) { num_exp_sign_ok_ = false; return true; }
            return false;
        }
        if (c == '.') {
            if (int_only_ || !num_digits_ || num_dot_ || num_exp_) return false;
            num_dot_ = true; num_digits_ = false; num_zero_ = false; return true;
        }
        if (c == 'e' || c == 'E') {
            if (int_only_ || !num_digits_ || num_exp_) return false;
            num_exp_ = true; num_digits_ = false; num_zero_ = false; num_exp_sign_ok_ = true;
            return true;
        }
        return false;
    }

    bool start_lit(const char * s) {
        lit_ = s;
        lit_pos_ = 1;                        // the first character is already consumed
        if (lit_[1] == '\0') { lit_ = nullptr; return after_value(); }
        return true;
    }

    bool feed_literal(unsigned char c) {
        if (lit_[lit_pos_] != (char)c) return false;
        ++lit_pos_;
        if (lit_[lit_pos_] == '\0') { lit_ = nullptr; return after_value(); }
        return true;
    }

    bool feed_sep(unsigned char c) {
        if (st_.empty()) return false;
        Frame & f = st_.back();
        if (f.arr) {
            if (c == ']') { st_.pop_back(); return after_value(); }
            if (c == ',') { need_ = N_VALUE; return true; }
            if (f.next == 0) return start_value(c);   // first element of an array
            return false;
        }
        if (c == '}') {
            const bool no_declared = (f.schema < 0) || nd(f.schema).keys.empty();
            if (no_declared && f.next == 0) return false;
            return close_obj();
        }
        if (c == ',') {
            need_ = N_KEY;
            if (f.schema < 0 || nd(f.schema).keys.empty()) {
                cur_ = -1;                    // unconstrained object: any number of members
            } else {
                const Node & n = nd(f.schema);
                if ((size_t)f.next >= n.keys.size()) return false;  // no more properties
                cur_ = n.kids[(size_t)f.next];
            }
            return true;
        }
        return false;
    }

    bool after_value() {
        if (st_.empty()) { done_ = true; return true; }
        Frame & f = st_.back();
        ++f.next;
        need_ = N_SEP;                       // a separator or the closing bracket follows
        return true;
    }

    std::vector<Node>  nodes_;
    int                root_ = -1;
    std::vector<Frame> st_;

    Need need_ = N_VALUE;
    int  cur_ = -1;
    int  pending_ = -1;

    bool in_str_ = false;
    Role role_ = S_VALUE;
    bool esc_ = false;
    int  escu_ = 0;
    std::vector<std::string> enum_left_;
    std::string              enum_lit_;

    bool in_num_ = false;
    bool num_digits_ = false;
    bool num_dot_ = false;
    bool num_exp_ = false;
    bool num_zero_ = false;         // the number started with '0'
    bool num_first_ = false;        // the next byte is the number's first byte
    bool num_exp_sign_ok_ = false;  // an exponent sign may still be emitted
    bool int_only_ = false;

    const char * lit_ = nullptr;
    int lit_pos_ = 0;

    bool done_ = false;
    bool dead_ = false;
};

} // namespace nibj
