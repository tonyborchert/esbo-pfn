// Command-line parsing, a minimal JSON reader and writer, file output and the chunk scheduler.
#pragma once

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace esbo {

#ifndef ESBO_GENERATOR_VERSION
#define ESBO_GENERATOR_VERSION "1.0.0"
#endif
inline const char* generator_version() { return ESBO_GENERATOR_VERSION; }

// ------------------------------------------------------------------ arguments
struct Args {
    std::map<std::string, std::string> kv;

    bool has(const std::string& k) const { return kv.count(k) != 0; }
    std::string get(const std::string& k, const std::string& def = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? def : it->second;
    }
    long long geti(const std::string& k, long long def) const {
        return has(k) ? std::stoll(get(k)) : def;
    }
    std::uint64_t getu(const std::string& k, std::uint64_t def) const {
        return has(k) ? std::stoull(get(k)) : def;
    }
    double getd(const std::string& k, double def) const { return has(k) ? std::stod(get(k)) : def; }
    std::string require(const std::string& k) const {
        if (!has(k)) throw std::runtime_error("missing required option --" + k);
        return get(k);
    }
};

// `--key value` and `--flag` (value "1"). Dashes inside keys become underscores.
inline Args parse_args(int argc, char** argv, int from) {
    Args a;
    for (int i = from; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument " + s);
        s = s.substr(2);
        std::string val = "1";
        const std::size_t eq = s.find('=');
        if (eq != std::string::npos) {
            val = s.substr(eq + 1);
            s = s.substr(0, eq);
        } else if (i + 1 < argc) {
            const std::string nxt = argv[i + 1];
            const bool is_number = nxt.size() > 1 && nxt[0] == '-' &&
                                   (std::isdigit(static_cast<unsigned char>(nxt[1])) || nxt[1] == '.');
            if (nxt.rfind("--", 0) != 0 || is_number) {
                val = nxt;
                ++i;
            }
        }
        for (char& c : s)
            if (c == '-') c = '_';
        a.kv[s] = val;
    }
    return a;
}

inline std::vector<double> parse_doubles(const std::string& s) {
    std::vector<double> out;
    std::string cur;
    for (char c : s + ",") {
        if (c == ',' || c == ' ' || c == '[' || c == ']') {
            if (!cur.empty()) out.push_back(std::stod(cur));
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    return out;
}

// "0-9,12,20-21" -> sorted unique indices.
inline std::vector<long long> parse_index_list(const std::string& s) {
    std::set<long long> out;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) continue;
        const std::size_t dash = part.find('-');
        if (dash == std::string::npos) {
            out.insert(std::stoll(part));
        } else {
            const long long a = std::stoll(part.substr(0, dash));
            const long long b = std::stoll(part.substr(dash + 1));
            for (long long i = a; i <= b; ++i) out.insert(i);
        }
    }
    return std::vector<long long>(out.begin(), out.end());
}

// ------------------------------------------------------------------ JSON writing
inline std::string jnum(double v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

inline std::string jstr(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') o.push_back('\\');
        o.push_back(c);
    }
    o.push_back('"');
    return o;
}

template <class T>
std::string jarr(const T* v, int n) {
    std::string o = "[";
    for (int i = 0; i < n; ++i) {
        if (i) o += ", ";
        o += jnum(static_cast<double>(v[i]));
    }
    return o + "]";
}

inline std::string jstrs(const std::vector<std::string>& v) {
    std::string o = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) o += ", ";
        o += jstr(v[i]);
    }
    return o + "]";
}

// ------------------------------------------------------------------ JSON reading
// Enough JSON for instance files: objects, arrays, numbers (incl. Infinity/NaN), strings, bools.
struct JValue {
    enum Kind { Null, Num, Str, Bool, Arr, Obj } kind = Null;
    double num = 0.0;
    std::string str;
    std::vector<JValue> arr;
    std::vector<std::pair<std::string, JValue>> obj;

    const JValue* find(const std::string& k) const {
        for (const auto& kv : obj)
            if (kv.first == k) return &kv.second;
        return nullptr;
    }
};

class JParser {
public:
    explicit JParser(const std::string& s) : s_(s) {}

    JValue parse() {
        JValue v = value();
        ws();
        if (p_ != s_.size()) fail("trailing characters");
        return v;
    }

private:
    const std::string& s_;
    std::size_t p_ = 0;

    [[noreturn]] void fail(const std::string& m) {
        throw std::runtime_error("JSON parse error at " + std::to_string(p_) + ": " + m);
    }
    void ws() {
        while (p_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[p_]))) ++p_;
    }
    bool lit(const char* t) {
        const std::size_t n = std::strlen(t);
        if (s_.compare(p_, n, t) == 0) {
            p_ += n;
            return true;
        }
        return false;
    }
    JValue value() {
        ws();
        if (p_ >= s_.size()) fail("unexpected end");
        JValue v;
        const char c = s_[p_];
        if (c == '{') {
            v.kind = JValue::Obj;
            ++p_;
            ws();
            if (s_[p_] == '}') {
                ++p_;
                return v;
            }
            for (;;) {
                ws();
                JValue key = value();
                if (key.kind != JValue::Str) fail("object key must be a string");
                ws();
                if (s_[p_] != ':') fail("expected ':'");
                ++p_;
                v.obj.emplace_back(key.str, value());
                ws();
                if (s_[p_] == ',') {
                    ++p_;
                    continue;
                }
                if (s_[p_] == '}') {
                    ++p_;
                    return v;
                }
                fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            v.kind = JValue::Arr;
            ++p_;
            ws();
            if (s_[p_] == ']') {
                ++p_;
                return v;
            }
            for (;;) {
                v.arr.push_back(value());
                ws();
                if (s_[p_] == ',') {
                    ++p_;
                    continue;
                }
                if (s_[p_] == ']') {
                    ++p_;
                    return v;
                }
                fail("expected ',' or ']'");
            }
        }
        if (c == '"') {
            v.kind = JValue::Str;
            ++p_;
            while (p_ < s_.size() && s_[p_] != '"') {
                if (s_[p_] == '\\') ++p_;
                v.str.push_back(s_[p_++]);
            }
            ++p_;
            return v;
        }
        if (lit("true")) {
            v.kind = JValue::Bool;
            v.num = 1;
            return v;
        }
        if (lit("false")) {
            v.kind = JValue::Bool;
            return v;
        }
        if (lit("null")) return v;
        v.kind = JValue::Num;
        if (lit("Infinity")) {
            v.num = std::numeric_limits<double>::infinity();
            return v;
        }
        if (lit("-Infinity")) {
            v.num = -std::numeric_limits<double>::infinity();
            return v;
        }
        if (lit("NaN")) {
            v.num = std::numeric_limits<double>::quiet_NaN();
            return v;
        }
        const char* start = s_.c_str() + p_;
        char* end = nullptr;
        v.num = std::strtod(start, &end);
        if (end == start) fail("bad value");
        p_ += static_cast<std::size_t>(end - start);
        return v;
    }
};

inline std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ------------------------------------------------------------------ files
inline void mkdirs(const std::string& path) {
    std::string acc;
    std::size_t i = 0;
    if (!path.empty() && path[0] == '/') {
        acc = "/";
        i = 1;
    }
    while (i <= path.size()) {
        std::size_t s = path.find('/', i);
        if (s == std::string::npos) s = path.size();
        acc += path.substr(i, s - i);
        if (!acc.empty() && acc != "/") ::mkdir(acc.c_str(), 0775);
        acc += "/";
        i = s + 1;
    }
}

inline bool file_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

// Write `text` to `path` through a temporary file and an atomic rename.
inline void write_atomic(const std::string& path, const std::string& text) {
    const std::string tmp = path + ".tmp" + std::to_string(::getpid());
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("cannot write " + tmp);
        f << text;
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0)
        throw std::runtime_error("cannot rename " + tmp + " to " + path);
}

// Positional writes into a (possibly sparse) binary file shared by several workers and processes.
class BinFile {
public:
    explicit BinFile(const std::string& path) : path_(path) {
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT, 0664);
        if (fd_ < 0) throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
    }
    ~BinFile() {
        if (fd_ >= 0) ::close(fd_);
    }
    BinFile(const BinFile&) = delete;
    BinFile& operator=(const BinFile&) = delete;

    void pwrite_all(const void* data, std::size_t bytes, std::uint64_t offset) {
        const char* p = static_cast<const char*>(data);
        while (bytes > 0) {
            const ssize_t n = ::pwrite(fd_, p, bytes, static_cast<off_t>(offset));
            if (n < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error("write to " + path_ + " failed: " + std::strerror(errno));
            }
            p += n;
            bytes -= static_cast<std::size_t>(n);
            offset += static_cast<std::uint64_t>(n);
        }
    }
    void sync() { ::fsync(fd_); }

private:
    std::string path_;
    int fd_ = -1;
};

// The append-only completion log `chunks.log`: one line "<chunk> <units>" per finished chunk.
class ChunkLog {
public:
    explicit ChunkLog(std::string path) : path_(std::move(path)) {}

    std::set<long long> done() const {
        std::set<long long> out;
        std::ifstream f(path_);
        long long c = 0, n = 0;
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            if (ss >> c >> n) out.insert(c);
        }
        return out;
    }

    void mark(long long chunk, long long units) {
        std::lock_guard<std::mutex> g(mu_);
        const int fd = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0664);
        if (fd < 0) throw std::runtime_error("cannot open " + path_);
        const std::string line = std::to_string(chunk) + " " + std::to_string(units) + "\n";
        if (::write(fd, line.data(), line.size()) != static_cast<ssize_t>(line.size())) {
            ::close(fd);
            throw std::runtime_error("cannot append to " + path_);
        }
        ::fsync(fd);
        ::close(fd);
    }

private:
    std::string path_;
    std::mutex mu_;
};

// Run fn(chunk) for every chunk on `workers` threads. The first exception stops the run.
inline void run_chunks(const std::vector<long long>& chunks, int workers,
                       const std::function<void(long long)>& fn) {
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::string error;
    std::mutex emu;
    auto body = [&]() {
        for (;;) {
            if (failed.load()) return;
            const std::size_t i = next.fetch_add(1);
            if (i >= chunks.size()) return;
            try {
                fn(chunks[i]);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> g(emu);
                if (!failed.exchange(true)) error = e.what();
                return;
            }
        }
    };
    workers = std::max(1, std::min<int>(workers, static_cast<int>(chunks.size())));
    if (workers == 1) {
        body();
    } else {
        std::vector<std::thread> ts;
        for (int w = 0; w < workers; ++w) ts.emplace_back(body);
        for (auto& t : ts) t.join();
    }
    if (failed.load()) throw std::runtime_error(error);
}

}  // namespace esbo
