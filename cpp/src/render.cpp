#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include <algorithm>
#include <functional>

#include "stages.hpp"
#ifdef DOCSCAN_GPU
#include "gpu_sift.hpp"
#endif

#include <chrono>

namespace docscan {

static inline uchar trunc8(double v) { return (uchar)std::clamp(v, 0.0, 255.0); }  // numpy astype(uint8)

// bilinear lookup in a CV_32F map with clamped coordinates; NaN if any neighbour is NaN
static inline float lookup(const cv::Mat& m, float x, float y) {
    x = std::clamp(x, 0.f, (float)m.cols - 1);
    y = std::clamp(y, 0.f, (float)m.rows - 1);
    int x0 = std::min((int)x, m.cols - 2), y0 = std::min((int)y, m.rows - 2);
    float a = x - x0, b = y - y0;
    const float* r0 = m.ptr<float>(y0);
    const float* r1 = m.ptr<float>(y0 + 1);
    return (1 - a) * (1 - b) * r0[x0] + a * (1 - b) * r0[x0 + 1] + (1 - a) * b * r1[x0] + a * b * r1[x0 + 1];
}

// Flat page -> reference photo: for every output pixel its (u, v) from the inverse flat map,
// height and normal from the surface, then the pixel in the reference view (R = I, t = 0).
Render render_flat(const cv::Mat& img, const Cam& cam, const Frame& F, const Surface& S, const cv::Mat& ur,
                   const cv::Mat& vr, double res, double W, double H, int dpi) {
    double ppm = dpi / 25.4;
    int ow = (int)std::lround(W * ppm), oh = (int)std::lround(H * ppm);
    std::vector<double> zu, zv;
    grid_gradient(S.g, S.Z, zu, zv);
    cv::Mat mx(oh, ow, CV_32F), my(oh, ow, CV_32F);
    Render R;
    R.valid.create(oh, ow, CV_8U);
    R.page.create(oh, ow, CV_8U);
    R.nu.create(oh, ow, CV_32F);
    R.nv.create(oh, ow, CV_32F);
    parallel_for(cv::Range(0, oh), [&](const cv::Range& rg) {
        for (int y = rg.start; y < rg.end; y++) {
            float sy = (float)((y + 0.5) / ppm / res);
            for (int x = 0; x < ow; x++) {
                float sx = (float)((x + 0.5) / ppm / res);
                float U = lookup(ur, sx, sy), V = lookup(vr, sx, sy);
                bool pg = std::isfinite(U) && std::isfinite(V);
                if (!pg) U = V = -1e4f;
                double z = S.g.sample(S.Z, U, V);
                Vec3 X = F.from_page(Vec3(U, V, z));
                mx.at<float>(y, x) = (float)(cam.f * X.x() / X.z() + cam.cx);
                my.at<float>(y, x) = (float)(cam.f * X.y() / X.z() + cam.cy);
                R.nu.at<float>(y, x) = (float)-S.g.sample(zu, U, V);
                R.nv.at<float>(y, x) = (float)-S.g.sample(zv, U, V);
                R.page.at<uchar>(y, x) = pg;
                float px = mx.at<float>(y, x), py = my.at<float>(y, x);
                R.valid.at<uchar>(y, x) =
                    X.z() > 0 && px >= 2 && py >= 2 && px < img.cols - 3 && py < img.rows - 3;
            }
        }
    });
    cv::Mat out;
    cv::remap(img, out, mx, my, cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    out.convertTo(R.img, CV_32FC3);
    return R;
}

cv::Mat gray_u8(const cv::Mat& img32, const cv::Mat& page) {
    cv::Mat g(img32.size(), CV_8U);
    for (int y = 0; y < img32.rows; y++)
        for (int x = 0; x < img32.cols; x++) {
            const cv::Vec3f& p = img32.at<cv::Vec3f>(y, x);
            g.at<uchar>(y, x) = page.at<uchar>(y, x) ? trunc8((p[0] + p[1] + p[2]) / 3.0f) : 255;
        }
    return g;
}

// Divide out illumination: log-luminance of the paper modelled as a low-order function of
// position and surface normal (fold shading), fitted to the upper envelope (ink is darker
// and down-weighted), then a medium-scale residual correction.
// Wide Gaussian blur on a downscaled grid (the fields blurred here are smooth anyway).
static void wide_blur(const cv::Mat& src, cv::Mat& dst, double sig) {
    int f = (int)(sig / 4);
    if (f < 2) {
        cv::GaussianBlur(src, dst, cv::Size(0, 0), sig);
        return;
    }
    cv::Mat sm;
    cv::resize(src, sm, cv::Size(), 1.0 / f, 1.0 / f, cv::INTER_AREA);
    // area averaging already blurs by about f/sqrt(12) px
    cv::GaussianBlur(sm, sm, cv::Size(0, 0), std::sqrt(std::max(sig * sig / (f * f) - 1.0 / 12, 0.25)));
    cv::resize(sm, dst, src.size(), 0, 0, cv::INTER_LINEAR);
}

cv::Mat remove_shading(const cv::Mat& img, const cv::Mat& valid, const cv::Mat& nu, const cv::Mat& nv) {
    const int sub = 4, oh = img.rows, ow = img.cols, nf = 15;
    cv::Mat L(oh, ow, CV_32F);
    parallel_for(cv::Range(0, oh), [&](const cv::Range& rg) {
        for (int y = rg.start; y < rg.end; y++)
            for (int x = 0; x < ow; x++) {
                const cv::Vec3f& p = img.at<cv::Vec3f>(y, x);
                L.at<float>(y, x) = std::log(std::max((p[0] + p[1] + p[2]) / 3.0f, 1.0f));
            }
    });
    auto feats = [](double s, double t, double a, double b, double* f) {
        double v[nf] = {1, s, t, s * s, s * t, t * t, s * s * s, s * s * t, s * t * t, t * t * t, a, b, a * a, b * b, a * b};
        std::copy(v, v + nf, f);
    };
    std::vector<double> Phi, ys;
    for (int y = 0; y < oh; y += sub)
        for (int x = 0; x < ow; x += sub) {
            if (!valid.at<uchar>(y, x)) continue;
            double f[nf];
            feats((double)x / ow * 2 - 1, (double)y / oh * 2 - 1, nu.at<float>(y, x), nv.at<float>(y, x), f);
            Phi.insert(Phi.end(), f, f + nf);
            ys.push_back(L.at<float>(y, x));
        }
    const size_t n = ys.size();
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, nf, Eigen::RowMajor>> P(Phi.data(), n, nf);
    Eigen::Map<Eigen::VectorXd> yv(ys.data(), n);
    Eigen::VectorXd w = Eigen::VectorXd::Ones(n);
    Eigen::Matrix<double, nf, 1> beta = Eigen::Matrix<double, nf, 1>::Zero();
    // normal equations summed over fixed row blocks in parallel (same result for any thread count)
    const int nb = 64, bl = (int)((n + nb - 1) / nb);
    std::vector<Eigen::Matrix<double, nf, nf>> Ab(nb);
    std::vector<Eigen::Matrix<double, nf, 1>> rb(nb);
    for (int it = 0; it < 12; it++) {
        parallel_for(cv::Range(0, nb), [&](const cv::Range& rg) {
            for (int b = rg.start; b < rg.end; b++) {
                int a = std::min((int)n, b * bl), len = std::min((int)n, a + bl) - a;
                auto Pb = P.middleRows(a, len);
                auto wb = w.segment(a, len);
                Ab[b] = Pb.transpose() * wb.asDiagonal() * Pb;
                rb[b] = Pb.transpose() * wb.cwiseProduct(yv.segment(a, len));
            }
        });
        Eigen::Matrix<double, nf, nf> A = Eigen::Matrix<double, nf, nf>::Zero();
        Eigen::Matrix<double, nf, 1> rhs = Eigen::Matrix<double, nf, 1>::Zero();
        for (int b = 0; b < nb; b++) A += Ab[b], rhs += rb[b];
        beta = A.completeOrthogonalDecomposition().solve(rhs);
        double sn = it < 3 ? 0.25 : 0.08;  // start tolerant, then hug the paper level
        parallel_for(cv::Range(0, (int)n), [&](const cv::Range& rg) {
            for (int i = rg.start; i < rg.end; i++) {
                double r = yv[i] - P.row(i).dot(beta);
                w[i] = r < 0 ? 1 / (1 + std::pow(r / sn, 2)) : 1 / (1 + std::pow(r / 0.05, 2));
            }
        });
    }
    cv::Mat S(oh, ow, CV_32F), resm(oh, ow, CV_32F), paper(oh, ow, CV_32F);
    parallel_for(cv::Range(0, oh), [&](const cv::Range& rg) {
        for (int y = rg.start; y < rg.end; y++)
            for (int x = 0; x < ow; x++) {
                double f[nf];
                feats((double)x / ow * 2 - 1, (double)y / oh * 2 - 1, nu.at<float>(y, x), nv.at<float>(y, x), f);
                double s = 0;
                for (int k = 0; k < nf; k++) s += f[k] * beta[k];
                S.at<float>(y, x) = (float)s;
                float r = L.at<float>(y, x) - (float)s;
                bool pp = valid.at<uchar>(y, x) && r > -0.10f && r < 0.10f;
                paper.at<float>(y, x) = pp;
                resm.at<float>(y, x) = pp ? r : 0.f;
            }
    });
    // Fold shadows: darker regions that reach the page edge and run across a good part of the page
    // are shading, not print (printed grey areas rarely touch the edge). The 3D normals explain
    // most of them; with one photo nothing does. Text is removed from the residual by a closing.
    {
        const int f = 4;
        const double ppm = std::max(oh, ow) / 297.0 / f;  // pixels per mm at 1/f size
        cv::Mat R(oh, ow, CV_32F, cv::Scalar(0)), Rs, vs, Rc;
        parallel_for(cv::Range(0, oh), [&](const cv::Range& rg) {
            for (int y = rg.start; y < rg.end; y++)
                for (int x = 0; x < ow; x++)
                    if (valid.at<uchar>(y, x)) R.at<float>(y, x) = L.at<float>(y, x) - S.at<float>(y, x);
        });
        cv::resize(R, Rs, cv::Size(), 1.0 / f, 1.0 / f, cv::INTER_AREA);
        cv::resize(valid, vs, Rs.size(), 0, 0, cv::INTER_NEAREST);
        int k = std::max(3, (int)std::lround(1.5 * ppm)) | 1;
        cv::morphologyEx(Rs, Rc, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k)));
        cv::Mat cand = (Rc < -0.08f) & (Rc > -0.7f) & (vs > 0), edge, lab, stats, cent;
        cv::dilate(vs == 0, edge, cv::Mat::ones(3, 3, CV_8U));
        int n = cv::connectedComponentsWithStats(cand, lab, stats, cent, 8, CV_32S);
        std::vector<char> keep(n, 0), touch(n, 0);
        for (int y = 0; y < lab.rows; y++)
            for (int x = 0; x < lab.cols; x++) {
                int l = lab.at<int>(y, x);
                if (l && (edge.at<uchar>(y, x) || x == 0 || y == 0 || x == lab.cols - 1 || y == lab.rows - 1))
                    touch[l] = 1;
            }
        int kept = 0;
        for (int l = 1; l < n; l++) {
            int w = stats.at<int>(l, cv::CC_STAT_WIDTH), h = stats.at<int>(l, cv::CC_STAT_HEIGHT);
            keep[l] = touch[l] && stats.at<int>(l, cv::CC_STAT_AREA) > 0.002 * lab.total() &&
                      (w > 0.3 * lab.cols || h > 0.3 * lab.rows);
            kept += keep[l];
        }
        if (kept) {
            cv::Mat sh(lab.size(), CV_8U), shf, Rcf;
            for (int i = 0; i < (int)lab.total(); i++) sh.data[i] = keep[((int*)lab.data)[i]] ? 255 : 0;
            cv::dilate(sh, sh, cv::Mat::ones(3, 3, CV_8U));
            cv::resize(sh, shf, valid.size(), 0, 0, cv::INTER_NEAREST);
            cv::resize(Rc, Rcf, valid.size(), 0, 0, cv::INTER_LINEAR);
            parallel_for(cv::Range(0, oh), [&](const cv::Range& rg) {
                for (int y = rg.start; y < rg.end; y++)
                    for (int x = 0; x < ow; x++) {
                        if (!shf.at<uchar>(y, x) || !valid.at<uchar>(y, x)) continue;
                        float r = R.at<float>(y, x);
                        if (r > Rcf.at<float>(y, x) - 0.10f && r < 0.10f) {
                            paper.at<float>(y, x) = 1.f;
                            resm.at<float>(y, x) = r;
                        }
                    }
            });
        }
        log(fmt("  shading: %d fold shadow region%s", kept, kept == 1 ? "" : "s"));
    }
    // residual correction on paper pixels (normalized convolution), wide fallback where
    // paper evidence is sparse
    double sig = std::max(oh, ow) / 200.0;
    cv::Mat num, den, num2, den2;
    parallel_for(cv::Range(0, 4), [&](const cv::Range& rg) {  // OpenCV runs each blur on one core
        for (int k = rg.start; k < rg.end; k++) {
            if (k == 0) wide_blur(resm, num, sig);
            if (k == 1) wide_blur(paper, den, sig);
            if (k == 2) wide_blur(resm, num2, 6 * sig);
            if (k == 3) wide_blur(paper, den2, 6 * sig);
        }
    });
    cv::Mat alb(oh, ow, CV_32FC3);
    parallel_for(cv::Range(0, oh), [&](const cv::Range& rg) {
        for (int y = rg.start; y < rg.end; y++)
            for (int x = 0; x < ow; x++) {
                float d = den.at<float>(y, x), d2 = den2.at<float>(y, x);
                float corr = d > 0.05f ? num.at<float>(y, x) / std::max(d, 1e-6f) : 0.f;
                float corr2 = d2 > 1e-3f ? num2.at<float>(y, x) / std::max(d2, 1e-6f) : 0.f;
                float blend = std::clamp((d - 0.05f) / 0.25f, 0.f, 1.f);
                float s = S.at<float>(y, x) + blend * corr + (1 - blend) * corr2;
                alb.at<cv::Vec3f>(y, x) = img.at<cv::Vec3f>(y, x) / std::exp(s);
            }
    });
    float gains[3];
    parallel_for(cv::Range(0, 3), [&](const cv::Range& rg) {
        for (int c = rg.start; c < rg.end; c++) {
            std::vector<double> ch;
            for (int y = 0; y < oh; y++)
                for (int x = 0; x < ow; x++)
                    if (paper.at<float>(y, x) > 0) ch.push_back(alb.at<cv::Vec3f>(y, x)[c]);
            gains[c] = ch.empty() ? 1.f : (float)median(ch);
        }
    });
    parallel_for(cv::Range(0, oh), [&](const cv::Range& rg) {
        for (int y = rg.start; y < rg.end; y++)
            for (int x = 0; x < ow; x++) {
                cv::Vec3f& a = alb.at<cv::Vec3f>(y, x);
                for (int c = 0; c < 3; c++) a[c] /= gains[c];
            }
    });
    return alb;
}

// Paper to `white` (clipping at 1.0 would flatten the bright side of every edge);
// near-neutral pixels made neutral, real colours kept.
cv::Mat to_uint8(const cv::Mat& alb, double white) {
    cv::Mat out(alb.size(), CV_8UC3);
    parallel_for(cv::Range(0, alb.rows), [&](const cv::Range& rg) {
        for (int y = rg.start; y < rg.end; y++)
            for (int x = 0; x < alb.cols; x++) {
                const cv::Vec3f& a = alb.at<cv::Vec3f>(y, x);
                float g = (a[0] + a[1] + a[2]) / 3;
                float sat = std::max({std::abs(a[0] - g), std::abs(a[1] - g), std::abs(a[2] - g)});
                float keep = std::clamp((sat - 0.03f) / 0.05f, 0.f, 1.f);
                cv::Vec3b& o = out.at<cv::Vec3b>(y, x);
                for (int c = 0; c < 3; c++) o[c] = trunc8((g + (a[c] - g) * keep) * 255 * white);
            }
    });
    return out;
}

// A neighbourhood filter of radius r run on horizontal strips in parallel (OpenCV runs some
// filters on one core). Each strip is filtered with r extra rows, so the result is identical.
static void by_strips(const cv::Mat& src, cv::Mat& dst, int r,
                      const std::function<void(const cv::Mat&, cv::Mat&)>& filter) {
    const int h = src.rows, n = std::max(1, std::min(32, h / std::max(4 * r, 32)));
    cv::Mat out(src.size(), src.type());
    parallel_for(cv::Range(0, n), [&](const cv::Range& rg) {
        for (int k = rg.start; k < rg.end; k++) {
            int a = k * h / n, b = (k + 1) * h / n, a0 = std::max(0, a - r), b0 = std::min(h, b + r);
            cv::Mat part = src.rowRange(a0, b0).clone(), f;
            filter(part, f);
            f.rowRange(a - a0, b - a0).copyTo(out.rowRange(a, b));
        }
    });
    dst = out;
}

// paper level and spread (median / MAD of bright, locally flat pixels)
static bool paper_level(const cv::Mat& x, double& P, double& s) {
    cv::Mat x8(x.size(), CV_8U), bg;
    const int N = (int)x.total(), nb = 64;
    parallel_for(cv::Range(0, N), [&](const cv::Range& rg) {
        for (int i = rg.start; i < rg.end; i++) x8.data[i] = trunc8(((float*)x.data)[i]);
    });
    by_strips(x8, bg, 10, [](const cv::Mat& a, cv::Mat& b) { cv::medianBlur(a, b, 21); });
    std::vector<std::vector<double>> part(nb);  // per block, concatenated in order
    parallel_for(cv::Range(0, nb), [&](const cv::Range& rg) {
        for (int k = rg.start; k < rg.end; k++)
            for (int i = (int)((long)k * N / nb); i < (int)((long)(k + 1) * N / nb); i++)
                if (bg.data[i] > 180 && std::abs((int)x8.data[i] - (int)bg.data[i]) < 25)
                    part[k].push_back(((float*)x.data)[i]);
    });
    std::vector<double> v;
    for (auto& p : part) v.insert(v.end(), p.begin(), p.end());
    if (v.size() < 1000) return false;
    P = median(v);
    for (auto& q : v) q = std::abs(q - P);
    s = 1.4826 * median(v);
    return true;
}

// cv::fastNlMeansDenoising(g8, gd, 4, 7, 21) (a search window of 11 is 2.5x faster but loses
// umlaut dots). With a GPU, its top part runs there while the CPU does the rest; the split
// follows the speeds measured last time. Each part gets 13 extra rows (search + patch radius),
// so the seam is exact.
static void denoise(const cv::Mat& g8, cv::Mat& gd, bool gpu) {
#ifdef DOCSCAN_GPU
    static double share = 0.5;  // of the rows for the GPU
    GpuSift* G = gpu ? GpuSift::get() : nullptr;
    const int H = g8.rows, r = 13;
    if (G && H > 4 * r) {
        int split = std::clamp((int)std::lround(H * share), 2 * r, H - 2 * r);
        cv::Mat top = g8.rowRange(0, split + r).clone(), bot = g8.rowRange(split - r, H).clone(), dt, db;
        bool ok = false;
        double tg = 0, tc = 0;
        {
            Thread th([&] {
                auto t = std::chrono::steady_clock::now();
                ok = G->denoise(top, dt, 4.0f);
                tg = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
            });
            auto t = std::chrono::steady_clock::now();
            cv::fastNlMeansDenoising(bot, db, 4.0f, 7, 21);
            tc = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
        }
        if (ok) {
            gd.create(g8.size(), CV_8U);
            dt.rowRange(0, split).copyTo(gd.rowRange(0, split));
            db.rowRange(r, db.rows).copyTo(gd.rowRange(split, H));
            double rg = split / std::max(tg, 1e-3), rc = (H - split) / std::max(tc, 1e-3);
            log(fmt("    denoising: GPU %d rows %.2f s, CPU %d rows %.2f s", split, tg, H - split, tc));
            share = std::clamp(rg / (rg + rc), 0.05, 0.95);
            return;
        }
    }
#else
    (void)gpu;
#endif
    cv::fastNlMeansDenoising(g8, gd, 4.0f, 7, 21);
}

// Scanner-like paper: denoise, even out small paper-level variations, push the paper to
// white with a ramp. Pages without real colour come back single-channel.
cv::Mat clean_paper(const cv::Mat& img, bool gpu, bool doc_tone) {
    const double near = 0.10;
    int oh = img.rows, ow = img.cols;
    cv::Mat gray(oh, ow, CV_32F), colmask(oh, ow, CV_8U), g8(oh, ow, CV_8U);
    parallel_for(cv::Range(0, oh), [&](const cv::Range& rg) {
        for (int y = rg.start; y < rg.end; y++)
            for (int x = 0; x < ow; x++) {
                const cv::Vec3b& p = img.at<cv::Vec3b>(y, x);
                double g = (p[0] + p[1] + p[2]) / 3.0;
                gray.at<float>(y, x) = (float)g;
                double ch = std::max({std::abs(p[0] - g), std::abs(p[1] - g), std::abs(p[2] - g)});
                colmask.at<uchar>(y, x) = ch > 15;
                g8.at<uchar>(y, x) = trunc8(g);
            }
    });
    cv::medianBlur(colmask, colmask, 5);
    bool colour = cv::mean(colmask)[0] > 1e-3;
    cv::Mat gd, g;
    auto t = std::chrono::steady_clock::now();
    denoise(g8, gd, gpu);
    log(fmt("    denoising %.2f s", std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count()));
    gd.convertTo(g, CV_32F);
    double P0, s0;
    bool found = paper_level(g, P0, s0);
    // local paper level: ink removed by a closing, then smoothed
    cv::Mat bg;
    cv::morphologyEx(g, bg, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
    by_strips(bg, bg, 25, [](const cv::Mat& a, cv::Mat& b) { cv::GaussianBlur(a, b, cv::Size(0, 0), 6.0); });
    if (!found) {
        if (colour) return img;
        return g8;
    }
    cv::Mat e(oh, ow, CV_32F);
    parallel_for(cv::Range(0, oh * ow), [&](const cv::Range& rg) {
        for (int i = rg.start; i < rg.end; i++) {
            double ratio = ((float*)bg.data)[i] / P0;
            double w = std::clamp((ratio - (1 - near)) / (near / 2), 0.0, 1.0);
            ((float*)e.data)[i] = (float)(((float*)g.data)[i] * (1 + w * (1 / std::max(ratio, 1e-3) - 1)));
        }
    });
    double P = P0, s = s0;
    paper_level(e, P, s);
    s = std::max(s, 2.0);
    double lo = P - 4 * s - 4, hi = P - s;
    cv::Mat Y(oh, ow, CV_32F);
    parallel_for(cv::Range(0, oh), [&](const cv::Range& rg) {
        for (int y = rg.start; y < rg.end; y++)
            for (int x = 0; x < ow; x++) {
                double v = e.at<float>(y, x);
                double yv = v <= lo ? v : lo + (v - lo) * (255 - lo) / std::max(hi - lo, 1e-3);
                Y.at<float>(y, x) = (float)std::clamp(yv, 0.0, 255.0);
            }
    });
    if (doc_tone) {
        // Scanner-app look: black ink on white paper. The black point follows the darkest ink
        // within ~3 mm, so thin or small print that the camera blurs to grey is darkened, not
        // faded (a global curve lost 10 % of the words in a form's fine print).
        const double ppm = std::min(ow, oh) / 210.0;
        cv::Mat blk;
        int k = (int)(3 * ppm) | 1;
        cv::erode(Y, blk, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k)));
        by_strips(blk, blk, (int)(5 * ppm), [&](const cv::Mat& a, cv::Mat& b) { cv::GaussianBlur(a, b, cv::Size(0, 0), 1.5 * ppm); });
        parallel_for(cv::Range(0, oh), [&](const cv::Range& rg) {
            for (int y = rg.start; y < rg.end; y++)
                for (int x = 0; x < ow; x++) {
                    double b0 = std::clamp((double)blk.at<float>(y, x), 40.0, 150.0);
                    double tc = std::clamp((Y.at<float>(y, x) - b0) / (245.0 - b0), 0.0, 1.0);
                    Y.at<float>(y, x) = (float)(255.0 * tc * tc * (3 - 2 * tc));
                }
        });
    }
    cv::Mat out(oh, ow, colour ? CV_8UC3 : CV_8U);
    parallel_for(cv::Range(0, oh), [&](const cv::Range& rg) {
        for (int y = rg.start; y < rg.end; y++)
            for (int x = 0; x < ow; x++) {
                double yv = Y.at<float>(y, x);
                if (!colour) out.at<uchar>(y, x) = (uchar)yv;
                else {
                    double k = yv / std::max((double)gray.at<float>(y, x), 1.0);
                    const cv::Vec3b& p = img.at<cv::Vec3b>(y, x);
                    for (int c = 0; c < 3; c++) out.at<cv::Vec3b>(y, x)[c] = trunc8(p[c] * k);
                }
            }
    });
    return out;
}

// Text lines of an upright grey page at about 6 px/mm: ink per row of each line, summed as
// ascender minus descender ink (positive for upright Latin text); n: lines found.
static void line_profile(const cv::Mat& g, double ppm, int& n, double& score) {
    const int px = std::max(1, (int)std::lround(ppm));
    cv::Mat ink, blob, lab, stats, cent;
    cv::adaptiveThreshold(g, ink, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 15 * px | 1, 12);
    cv::morphologyEx(ink, blob, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4 * px, 1)));
    int nc = cv::connectedComponentsWithStats(blob, lab, stats, cent, 8, CV_32S);
    n = 0;
    double up = 0, down = 0;
    for (int k = 1; k < nc; k++) {
        int x0 = stats.at<int>(k, cv::CC_STAT_LEFT), y0 = stats.at<int>(k, cv::CC_STAT_TOP);
        int w = stats.at<int>(k, cv::CC_STAT_WIDTH), h = stats.at<int>(k, cv::CC_STAT_HEIGHT);
        if (w < 25 * ppm || h > 7 * ppm || h < 1.5 * ppm || w < 6 * h) continue;
        std::vector<int> prof(h, 0);
        for (int y = 0; y < h; y++) {
            const int* l = lab.ptr<int>(y0 + y);
            const uchar* q = ink.ptr<uchar>(y0 + y);
            for (int x = x0; x < x0 + w; x++) prof[y] += l[x] == k && q[x];
        }
        int mx = *std::max_element(prof.begin(), prof.end());
        int top = 0, bot = h - 1;
        while (top < h && prof[top] < mx / 2) top++;
        while (bot > 0 && prof[bot] < mx / 2) bot--;
        if (bot - top < 2) continue;  // a ruled line, not text
        double a = 0, b = 0, c = 0;
        for (int y = 0; y < h; y++) (y < top ? a : y > bot ? b : c) += prof[y];
        if (c <= 0) continue;
        up += a / c, down += b / c;
        n++;
    }
    score = n ? (up - down) / n : 0;
}

int text_rotation(const cv::Mat& img, double ppm, std::string* info) {
    cv::Mat g = img;
    if (g.channels() == 3) cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
    double f = std::min(1.0, 6.0 / ppm);
    if (f < 1) cv::resize(g, g, cv::Size(), f, f, cv::INTER_AREA);
    ppm *= f;
    int n[4];
    double sc[4];
    cv::Mat r[4];
    r[0] = g;
    cv::rotate(g, r[1], cv::ROTATE_90_CLOCKWISE);
    cv::rotate(g, r[2], cv::ROTATE_180);
    cv::rotate(g, r[3], cv::ROTATE_90_COUNTERCLOCKWISE);
    parallel_for(cv::Range(0, 4), [&](const cv::Range& rg) {
        for (int k = rg.start; k < rg.end; k++) line_profile(r[k], ppm, n[k], sc[k]);
    });
    // lines found upright vs upside down are the same lines: compare 0/180 with 90/270
    int hz = n[0] + n[2], vt = n[1] + n[3];
    int best = hz >= vt ? (sc[0] >= sc[2] ? 0 : 2) : (sc[1] >= sc[3] ? 1 : 3);
    if (info)
        *info = fmt("text lines %d horizontal, %d vertical; ascender score %+.2f / %+.2f", hz / 2, vt / 2,
                    hz >= vt ? sc[0] : sc[1], hz >= vt ? sc[2] : sc[3]);
    if (std::max(hz, vt) < 2 * 3) return 0;
    return best * 90;
}

}  // namespace docscan
