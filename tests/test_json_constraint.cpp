// Standalone unit test for src/json_constraint.hpp — no HIP, plain g++.
#include "../src/json_constraint.hpp"
#include <cstdio>
#include <string>
#include <vector>

using nibj::Constraint;

static int failures = 0;

static void expect_ok(const char * name, Constraint c, const std::string & s) {
    size_t pos = 0;
    for (char raw : s) {
        unsigned char ch = (unsigned char)raw;
        if (!c.feed(ch)) {
            std::printf("FAIL %-38s rejected at byte %zu (%s)\n", name, pos, s.substr(0, pos).c_str());
            ++failures;
            return;
        }
        ++pos;
    }
    if (!c.done()) {
        std::printf("FAIL %-38s did not complete: %s\n", name, s.c_str());
        ++failures;
        return;
    }
    std::printf("ok   %-38s %s\n", name, s.c_str());
}

static void expect_bad(const char * name, Constraint c, const std::string & s) {
    for (unsigned char ch : s) {
        if (!c.feed(ch) && !c.done()) {
            std::printf("ok   %-38s rejected %s\n", name, s.c_str());
            return;
        }
    }
    if (!c.done()) {                     // incomplete input is not a valid value
        std::printf("ok   %-38s incomplete %s\n", name, s.c_str());
        return;
    }
    std::printf("FAIL %-38s accepted invalid: %s\n", name, s.c_str());
    ++failures;
}

int main() {
    // {city: string}
    auto obj_city = [] {
        Constraint c;
        c.build_object({"city"}, {{}}, {nibj::Node::STR}, {true});
        return c;
    };
    expect_ok("object string", obj_city(), R"({"city":"Lisbon"})");
    expect_ok("object empty string", obj_city(), R"({"city":""})");
    expect_ok("object escaped", obj_city(), R"({"city":"a\"b\\c\n"})");
    expect_ok("object unicode escape", obj_city(), R"({"city":"a\u00e9"})");
    expect_bad("object wrong key", obj_city(), R"({"town":"Lisbon"})");
    expect_bad("object missing value", obj_city(), R"({"city"})");
    expect_bad("object unquoted value", obj_city(), R"({"city":Lisbon)");
    expect_bad("object no close", obj_city(), R"({"city":"Lisbon")");

    // two properties, in schema order
    Constraint two;
    two.build_object({"a", "b"}, {{}, {}}, {nibj::Node::INT, nibj::Node::BOOL}, {true, true});
    expect_ok("two props", two, R"({"a":-42,"b":true})");
    Constraint two2 = two;
    expect_bad("two props reversed", two2, R"({"b":true})");

    // enum
    Constraint en;
    en.build_object({"unit"}, {{"metric", "imperial"}}, {nibj::Node::ENUMV}, {true});
    expect_ok("enum metric", en, R"({"unit":"metric"})");
    expect_ok("enum imperial", en, R"({"unit":"imperial"})");
    Constraint en2 = en, en3 = en;
    expect_bad("enum other", en2, R"({"unit":"kelvin"})");
    expect_bad("enum prefix only", en3, R"({"unit":"metr"})");

    // numbers
    Constraint num;
    num.build_object({"n", "i"}, {{}, {}}, {nibj::Node::NUM, nibj::Node::INT}, {true, true});
    expect_ok("numbers", num, R"({"n":-3.5e10,"i":7})");
    Constraint num2 = num;
    expect_bad("int with fraction", num2, R"({"n":1.5,"i":1.5})");

    // canonical numbers only: the machine must never emit invalid JSON
    Constraint anynum;
    anynum.build_object({"n"}, {{}}, {nibj::Node::ANY}, {true});
    expect_bad("double minus", anynum, R"({"n":--1})");
    expect_bad("plus leading", anynum, R"({"n":+1})");
    expect_bad("double exponent sign", anynum, R"({"n":1e+-5})");
    expect_bad("leading zero", anynum, R"({"n":01})");
    expect_bad("trailing dot", anynum, R"({"n":1.})");
    expect_ok("exponent forms", anynum, R"({"n":1e-5})");
    expect_ok("zero", anynum, R"({"n":0})");
    expect_ok("negative zero fraction", anynum, R"({"n":-0.5})");

    // null / bool
    Constraint nul;
    nul.build_object({"x", "y"}, {{}, {}}, {nibj::Node::NUL, nibj::Node::BOOL}, {true, true});
    expect_ok("null+bool", nul, R"({"x":null,"y":false})");

    // array of strings
    Constraint arr;
    arr.build_array(nibj::Node::STR, {});
    expect_ok("array empty", arr, R"([])");
    expect_ok("array 2", arr, R"(["a","b"])");
    expect_bad("array trailing comma", arr, R"(["a",])");

    // any JSON
    Constraint any;
    any.build_any();
    expect_ok("any object", any, R"({"k":[1,2,{"z":null}]})");
    Constraint any2;
    any2.build_any();
    // a bare number is complete only when something terminates it, so accept
    // "no rejection" rather than "done" here
    bool accepted = true;
    for (char raw : std::string("42")) if (!any2.feed((unsigned char)raw)) accepted = false;
    std::printf("%s any scalar number accepted\n", accepted ? "ok  " : "FAIL");
    if (!accepted) ++failures;

    // required-only: optional props must be skipped
    Constraint req;
    req.build_object({"keep", "drop"}, {{}, {}}, {nibj::Node::STR, nibj::Node::STR}, {true, false});
    expect_ok("required only", req, R"({"keep":"y"})");

    // ---- token masking ---------------------------------------------------
    Constraint m;
    m.build_object({"city"}, {{}}, {nibj::Node::STR}, {true});
    m.advance("{\"city\":\"");
    std::vector<std::string> toks = {
        "",            // 0 special mid-value: blocked, or the model escapes the grammar
        "Lisbon",      // 1 inside the string: allowed
        "\"",          // 2 closes the string: allowed
        "}",           // 3 closes the object: allowed after close
        "abc",         // 4 plain text: not allowed inside a JSON string? it is
        "\n",          // 5 raw control char: not allowed inside a string
        "{",           // 6 structural noise inside string: allowed as content
        "Lisbon\"}",   // 7 completes the whole value: allowed
        "x y z",       // 8 spaces inside string are content: allowed
        "\x01",        // 9 control char: rejected
    };
    std::vector<uint8_t> mask;
    m.fill(mask, toks);
    auto allowed = [&](size_t i) { return mask[i] != 0; };
    std::printf("mask: special=%d 'Lisbon\"}'=%d control=%d newline=%d\n",
                allowed(0), allowed(7), allowed(9), allowed(5));
    if (allowed(0))  { std::printf("FAIL special token must be blocked mid-value\n"); ++failures; }
    if (!allowed(7)) { std::printf("FAIL completing token must be allowed\n"); ++failures; }
    if (allowed(9))  { std::printf("FAIL raw control char must be masked\n"); ++failures; }
    if (allowed(5))  { std::printf("FAIL raw newline inside string must be masked\n"); ++failures; }

    // once the value is complete only turn-ending tokens may follow, otherwise
    // the model appends prose after the JSON
    Constraint fin;
    fin.build_object({"city"}, {{}}, {nibj::Node::STR}, {true});
    fin.advance(R"({"city":"Lisbon"})");
    std::vector<std::string> ftoks = {"", " sure"};
    std::vector<uint8_t> fmask;
    fin.fill(fmask, ftoks);
    if (!fmask[0]) { std::printf("FAIL special must be allowed after completion\n"); ++failures; }
    if (fmask[1])  { std::printf("FAIL prose must be masked after completion\n"); ++failures; }

    // an unconstrained object may not close empty
    Constraint anyobj;
    anyobj.build_any_object();
    if (anyobj.feed('}')) { std::printf("FAIL empty unconstrained object was accepted\n"); ++failures; }
    expect_ok("any object with one member", anyobj, R"({"a":1})");

    // whitespace between tokens is legal and must not be rejected: models emit
    // pretty-printed JSON, and forcing a canonical layout mangles their output
    expect_ok("pretty printed object", anyobj, "{\n  \"a\": 1,\n  \"b\": [1, 2]\n}");
    Constraint ws;
    ws.build_object({"city"}, {{}}, {nibj::Node::STR}, {true});
    expect_ok("pretty printed typed object", ws, "{\n  \"city\": \"Lisbon\"\n}");
    // but an empty key is not a key
    Constraint emptykey;
    emptykey.build_any_object();
    expect_bad("empty key", emptykey, R"({"":"Tokyo"})");

    // a declared key must be complete: a prefix would silently drop the property
    Constraint prefix;
    prefix.build_object({"condition"}, {{}}, {nibj::Node::STR}, {true});
    expect_bad("truncated key", prefix, R"({"co":"Sunny"})");
    expect_ok("full key", prefix, R"({"condition":"Sunny"})");

    // after a completed value the machine deactivates
    Constraint d;
    d.build_object({"city"}, {{}}, {nibj::Node::STR}, {true});
    d.advance(R"({"city":"Lisbon"})");
    if (!d.done()) { std::printf("FAIL should be done after a full object\n"); ++failures; }

    std::printf(failures ? "\n%d FAILURES\n" : "\nall tests passed\n", failures);
    return failures ? 1 : 0;
}
