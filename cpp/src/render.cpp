#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include <algorithm>
#include <functional>

#include "stages.hpp"

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

// Scanner-like paper: denoise, even out small paper-level variations, push the paper to
// white with a ramp. Pages without real colour come back single-channel.
cv::Mat clean_paper(const cv::Mat& img) {
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
    cv::fastNlMeansDenoising(g8, gd, 4.0f, 7, 11);  // search 11 instead of 21: 2.5x faster, mean change 0.1 levels
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
    cv::Mat out(oh, ow, colour ? CV_8UC3 : CV_8U);
    parallel_for(cv::Range(0, oh), [&](const cv::Range& rg) {
        for (int y = rg.start; y < rg.end; y++)
            for (int x = 0; x < ow; x++) {
                double v = e.at<float>(y, x);
                double yv = v <= lo ? v : lo + (v - lo) * (255 - lo) / std::max(hi - lo, 1e-3);
                yv = std::clamp(yv, 0.0, 255.0);
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

}  // namespace docscan
