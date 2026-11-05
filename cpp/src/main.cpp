// Command-line front end: docscan ref.jpg view2.jpg [view3.jpg ...] -o out [--focal-px F]
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "docscan.hpp"

#ifdef __GLIBC__
#include <malloc.h>
#endif

// FocalLengthIn35mmFilm (EXIF tag 0xA405) from a JPEG, 0 if absent.
static double exif_focal_35mm(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<unsigned char> d((std::istreambuf_iterator<char>(f)), {});
    size_t p = 2;
    if (d.size() < 4 || d[0] != 0xFF || d[1] != 0xD8) return 0;
    while (p + 4 <= d.size() && d[p] == 0xFF) {
        int marker = d[p + 1];
        size_t len = (d[p + 2] << 8) | d[p + 3];
        if (marker == 0xE1 && p + 10 <= d.size() && std::memcmp(&d[p + 4], "Exif\0\0", 6) == 0) {
            size_t t = p + 10, end = std::min(d.size(), p + 2 + len);
            if (t + 8 > end) return 0;
            bool le = d[t] == 'I';
            auto u16 = [&](size_t o) -> unsigned {
                if (t + o + 2 > end) return 0;
                return le ? d[t + o] | d[t + o + 1] << 8 : d[t + o] << 8 | d[t + o + 1];
            };
            auto u32 = [&](size_t o) -> unsigned {
                return le ? u16(o) | u16(o + 2) << 16 : u16(o) << 16 | u16(o + 2);
            };
            auto find = [&](size_t ifd, unsigned tag, unsigned& val) {
                unsigned n = u16(ifd);
                for (unsigned i = 0; i < n; i++) {
                    size_t e = ifd + 2 + 12 * i;
                    if (u16(e) == tag) {
                        val = u16(e + 2) == 3 ? u16(e + 8) : u32(e + 8);  // SHORT or LONG
                        return true;
                    }
                }
                return false;
            };
            size_t ifd0 = u32(4);
            unsigned v = 0, sub = 0;
            if (find(ifd0, 0x8769, sub) && find(sub, 0xA405, v) && v) return v;
            if (find(ifd0, 0xA405, v)) return v;
            return 0;
        }
        if (marker == 0xDA) break;  // start of scan: no EXIF before it
        p += 2 + len;
    }
    return 0;
}

static void usage() {
    std::fprintf(stderr,
                 "usage: docscan ref.jpg view2.jpg [view3.jpg view4.jpg] [-o OUT_DIR] [--focal-px F]\n"
                 "               [--page WxH] [--dpi N]\n"
                 "  The first photo is the reference that gets rendered; 4 views recommended.\n"
                 "  With 2 views the focal length is kept fixed: pass --focal-px (Pixel 8 16:9: 3030).\n");
}

int main(int argc, char** argv) {
#ifdef __GLIBC__
    // glibc keeps a heap per thread and holds on to freed memory: ~0.5 GB more peak memory
    mallopt(M_ARENA_MAX, 2);
#endif
    std::vector<std::string> paths;
    std::string out = "out";
    docscan::Options opt;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-o" || a == "--out") out = next();
        else if (a == "--focal-px") opt.focal_px = std::stod(next());
        else if (a == "--dpi") opt.dpi = std::stoi(next());
        else if (a == "--page") {
            std::string v = next();
            if (std::sscanf(v.c_str(), "%lfx%lf", &opt.page_w, &opt.page_h) != 2) {
                usage();
                return 2;
            }
        } else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else if (!a.empty() && a[0] == '-') {
            usage();
            return 2;
        } else paths.push_back(a);
    }
    if (paths.size() < 2) {
        usage();
        return 2;
    }
    auto t0 = std::chrono::steady_clock::now();
    std::vector<cv::Mat> imgs;
    for (auto& p : paths) {
        imgs.push_back(cv::imread(p, cv::IMREAD_COLOR));
        if (imgs.back().empty()) {
            std::fprintf(stderr, "cannot read %s\n", p.c_str());
            return 1;
        }
    }
    if (opt.focal_px <= 0) {
        double f35 = exif_focal_35mm(paths[0]);
        if (f35 > 0) {
            opt.focal_px = f35 * std::hypot(imgs[0].cols, imgs[0].rows) / 43.27;
            std::fprintf(stderr, "focal from EXIF (35mm-equiv %.0f mm)\n", f35);
        }
    }
    try {
        cv::Mat flat = docscan::process(imgs, opt);
        std::filesystem::create_directories(out);
        cv::imwrite(out + "/flat.png", flat, {cv::IMWRITE_PNG_COMPRESSION, 9});
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "done -> %s/flat.png (%.1f s)\n", out.c_str(), sec);
    return 0;
}
