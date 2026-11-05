#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <thread>
#include <chrono>

#include "stages.hpp"
#ifdef DOCSCAN_GPU
#include "gpu_sift.hpp"
#endif

namespace docscan {

// ------------------------------------------------------------ page

// Largest bright region, approximated by a quadrilateral (tl, tr, br, bl).
bool find_page_quad(const cv::Mat& img, std::array<Vec2, 4>& quad, cv::Mat& mask, int work_side) {
    cv::Mat g, sm, th;
    if (img.channels() == 3) cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
    else g = img;
    double small_s = (double)work_side / std::max(g.rows, g.cols);
    cv::resize(g, sm, cv::Size(), small_s, small_s, cv::INTER_AREA);
    auto ks = [&](int k) { return 2 * (int)std::lround((k - 1) / 2.0 * work_side / 1000.0) + 1; };  // odd, scaled
    // close text strokes so the page becomes one blob
    cv::morphologyEx(sm, sm, cv::MORPH_CLOSE, cv::Mat::ones(ks(25), ks(25), CV_8U));
    cv::GaussianBlur(sm, th, cv::Size(ks(9), ks(9)), 0);
    cv::threshold(th, th, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::morphologyEx(th, th, cv::MORPH_OPEN, cv::Mat::ones(ks(15), ks(15), CV_8U));
    std::vector<std::vector<cv::Point>> cnts;
    cv::findContours(th, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (cnts.empty()) return false;
    size_t best = 0;
    for (size_t i = 1; i < cnts.size(); i++)
        if (cv::contourArea(cnts[i]) > cv::contourArea(cnts[best])) best = i;
    const auto& c = cnts[best];
    if (cv::contourArea(c) < 0.1 * sm.total()) return false;
    cv::Mat msmall = cv::Mat::zeros(sm.size(), CV_8U);
    cv::drawContours(msmall, cnts, (int)best, 255, -1);
    cv::resize(msmall, mask, g.size(), 0, 0, cv::INTER_NEAREST);

    std::vector<cv::Point> hull;
    cv::convexHull(c, hull);
    double peri = cv::arcLength(hull, true);
    std::vector<cv::Point2d> q;
    for (int k = 0; k < 15; k++) {
        double eps = 0.01 + k * (0.07 / 14);
        std::vector<cv::Point> ap;
        cv::approxPolyDP(hull, ap, eps * peri, true);
        if (ap.size() == 4) {
            for (auto& p : ap) q.emplace_back(p.x, p.y);
            break;
        }
    }
    if (q.empty()) {
        cv::Point2f bp[4];
        cv::minAreaRect(c).points(bp);
        for (auto& p : bp) q.emplace_back(p.x, p.y);
    }
    std::array<Vec2, 4> Q;
    for (int k = 0; k < 4; k++) Q[k] = Vec2(q[k].x, q[k].y) / small_s;
    auto arg = [&](auto key, bool mx) {
        int b = 0;
        for (int k = 1; k < 4; k++)
            if (mx ? key(Q[k]) > key(Q[b]) : key(Q[k]) < key(Q[b])) b = k;
        return b;
    };
    auto sum = [](const Vec2& p) { return p.x() + p.y(); };
    auto diff = [](const Vec2& p) { return p.y() - p.x(); };
    quad = {Q[arg(sum, false)], Q[arg(diff, false)], Q[arg(sum, true)], Q[arg(diff, true)]};
    return true;
}

// np.gradient of a 1D profile: central differences inside, one-sided at the ends
static void gradient_abs(const std::vector<double>& p, std::vector<double>& g) {
    size_t n = p.size();
    g.resize(n);
    g[0] = std::abs(p[1] - p[0]);
    g[n - 1] = std::abs(p[n - 1] - p[n - 2]);
    for (size_t i = 1; i + 1 < n; i++) g[i] = std::abs(0.5 * (p[i + 1] - p[i - 1]));
}

// Subpixel corners: fit lines to the strongest edge along short normal profiles near each
// corner, then intersect. Only the part of each edge near the corner is used.
std::array<Vec2, 4> refine_quad(const cv::Mat& gray, const std::array<Vec2, 4>& quad, int half, int n) {
    cv::Mat g;
    gray.convertTo(g, CV_32F);
    cv::GaussianBlur(g, g, cv::Size(0, 0), 1.2);
    std::array<Vec2, 4> out = quad;
    auto edge_pts = [&](const Vec2& a, const Vec2& b, double t0, double t1) {
        Vec2 d = (b - a).normalized(), nrm(-d.y(), d.x());
        std::vector<cv::Point2f> pts;
        std::vector<double> prof(2 * half + 1), gr;
        for (int s = 0; s < n; s++) {
            double tt = t0 + (t1 - t0) * s / (n - 1);
            Vec2 p = a + tt * (b - a);
            for (int o = -half; o <= half; o++) {
                Vec2 xy = p + o * nrm;
                prof[o + half] = sample_clamped(g, xy.x(), xy.y());
            }
            gradient_abs(prof, gr);
            int k = (int)(std::max_element(gr.begin(), gr.end()) - gr.begin());
            if (!(0 < k && k < (int)gr.size() - 1) || gr[k] < 5) continue;
            double den = gr[k - 1] - 2 * gr[k] + gr[k + 1];
            double sub = std::abs(den) > 1e-9 ? 0.5 * (gr[k - 1] - gr[k + 1]) / den : 0.0;
            Vec2 e = p + (k - half + sub) * nrm;
            pts.emplace_back((float)e.x(), (float)e.y());
        }
        return pts;
    };
    for (int i = 0; i < 4; i++) {
        const Vec2 &prv = quad[(i + 3) % 4], &cur = quad[i], &nxt = quad[(i + 1) % 4];
        auto e1 = edge_pts(prv, cur, 0.72, 0.97), e2 = edge_pts(cur, nxt, 0.03, 0.28);
        if (e1.size() < 8 || e2.size() < 8) continue;
        cv::Vec4f l1, l2;
        cv::fitLine(e1, l1, cv::DIST_HUBER, 0, 0.01, 0.01);
        cv::fitLine(e2, l2, cv::DIST_HUBER, 0, 0.01, 0.01);
        Vec2 p(l1[2], l1[3]), r(l1[0], l1[1]), q(l2[2], l2[3]), s(l2[0], l2[1]);
        Eigen::Matrix2d A;
        A << r, -s;
        if (std::abs(A.determinant()) < 1e-6) continue;
        double lam = A.partialPivLu().solve(q - p)[0];
        Vec2 c = p + lam * r;
        if ((c - cur).norm() < 3 * half) out[i] = c;
    }
    return out;
}

// Closed, ordered, subpixel page boundary (strongest edge along the normal of the smoothed
// mask contour).
void page_outline(const cv::Mat& gray, const cv::Mat& mask, Pts2& out, std::vector<char>& ok, double step,
                  int half) {
    std::vector<std::vector<cv::Point>> cnts;
    cv::findContours(mask.clone(), cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    size_t best = 0;
    for (size_t i = 1; i < cnts.size(); i++)
        if (cv::contourArea(cnts[i]) > cv::contourArea(cnts[best])) best = i;
    std::vector<Vec2> c;
    for (auto& p : cnts[best]) c.emplace_back(p.x, p.y);
    c.push_back(c[0]);
    std::vector<double> cum(c.size(), 0.0);
    for (size_t i = 1; i < c.size(); i++) cum[i] = cum[i - 1] + (c[i] - c[i - 1]).norm();
    std::vector<double> tt = arange(0, cum.back(), step);
    size_t n = tt.size();
    Pts2 p(n);
    size_t seg = 0;
    for (size_t i = 0; i < n; i++) {
        while (seg + 2 < cum.size() && cum[seg + 1] <= tt[i]) seg++;
        double L = cum[seg + 1] - cum[seg];
        double a = L > 0 ? (tt[i] - cum[seg]) / L : 0;
        p[i] = c[seg] + a * (c[seg + 1] - c[seg]);
    }
    // gaussian_filter1d(sigma=4, mode="wrap")
    const int R = 16;
    std::vector<double> w(2 * R + 1);
    double ws = 0;
    for (int k = -R; k <= R; k++) ws += w[k + R] = std::exp(-0.5 * k * k / 16.0);
    Pts2 ps(n, Vec2::Zero());
    for (size_t i = 0; i < n; i++)
        for (int k = -R; k <= R; k++) ps[i] += w[k + R] / ws * p[((long)i + k + (long)n * 4) % n];
    cv::Mat g;
    gray.convertTo(g, CV_32F);
    cv::GaussianBlur(g, g, cv::Size(0, 0), 1.2);
    out.resize(n);
    ok.resize(n);
    std::vector<double> prof(2 * half + 1), gr;
    for (size_t i = 0; i < n; i++) {
        Vec2 tan = ps[(i + 1) % n] - ps[(i + n - 1) % n];
        tan /= tan.norm() + 1e-12;
        Vec2 nrm(-tan.y(), tan.x());
        for (int o = -half; o <= half; o++) {
            Vec2 xy = ps[i] + o * nrm;
            prof[o + half] = sample_clamped(g, xy.x(), xy.y());
        }
        gradient_abs(prof, gr);
        int k = (int)(std::max_element(gr.begin(), gr.end()) - gr.begin());
        int L = (int)gr.size();
        ok[i] = k > 0 && k < L - 1 && gr[k] > 4;
        int kk = std::clamp(k, 1, L - 2);
        double l = gr[kk - 1], m = gr[kk], r = gr[kk + 1], den = l - 2 * m + r;
        double sub = std::abs(den) > 1e-9 ? 0.5 * (l - r) / den : 0;
        out[i] = ps[i] + (kk - half + sub) * nrm;
    }
}

// ------------------------------------------------------------ features

// SIFT on overlapping tiles (OpenCV upsamples 2x internally: ~2 GB on a full photo).
// Each keypoint is kept by the tile whose core contains it; strongest n_features kept.
void sift_detect(const cv::Mat& g, const cv::Mat& mask, int n_features, Pts2& pts, cv::Mat& desc, bool gpu) {
    const int ny = 4, nx = 3, overlap = 200;
    int h = g.rows, w = g.cols;
    // tiles run in parallel (SIFT itself is mostly sequential); merged in tile order
    struct Tile {
        std::vector<Vec2> P;
        std::vector<float> resp;
        std::vector<cv::Mat> ds;
    };
    std::vector<Tile> tiles(ny * nx);
    bool done = false;
#ifdef DOCSCAN_GPU
    // GPU: same tiles one after another (each fills the GPU); keypoints filtered by the tile mask and
    // core region before descriptors are computed
    if (GpuSift* G = gpu ? GpuSift::get() : nullptr) {
        done = true;
        auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < ny * nx && done; t++) {
            int i = t / nx, j = t % nx;
            int y0 = (int)(i * ((double)h / ny)), y1 = (int)((i + 1) * ((double)h / ny));
            int x0 = (int)(j * ((double)w / nx)), x1 = (int)((j + 1) * ((double)w / nx));
            int Y0 = std::max(y0 - overlap, 0), Y1 = std::min(y1 + overlap, h);
            int X0 = std::max(x0 - overlap, 0), X1 = std::min(x1 + overlap, w);
            cv::Rect roi(X0, Y0, X1 - X0, Y1 - Y0);
            cv::Mat m;
            if (!mask.empty()) {
                m = mask(roi);
                if (cv::countNonZero(m) == 0) continue;
            }
            auto keep = [&](const cv::Point2f& p) {
                if (!m.empty() && !m.at<uchar>((int)(p.y + 0.5f), (int)(p.x + 0.5f))) return false;
                double x = (double)p.x + X0, y = (double)p.y + Y0;
                return x0 <= x && x < x1 && y0 <= y && y < y1;
            };
            std::vector<cv::KeyPoint> kp;
            cv::Mat d;
            if (!G->detect(g(roi).clone(), 0.02f, keep, kp, d)) {
                done = false;  // GPU error: redo everything on the CPU
                break;
            }
            Tile& T = tiles[t];
            for (size_t k = 0; k < kp.size(); k++) {
                T.P.emplace_back((float)((double)kp[k].pt.x + X0), (float)((double)kp[k].pt.y + Y0));
                T.resp.push_back(kp[k].response);
                T.ds.push_back(d.row((int)k));
            }
        }
        if (!done) tiles.assign(ny * nx, Tile());
        auto st = G->take_stats();
        double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (done && st.pyramid > 0)
            log(fmt("    gpu: pyramid %.2f s, extrema + orientation %.2f s, descriptors %.2f s, rest %.2f s", st.pyramid,
                    st.extrema, st.descriptors, wall - st.pyramid - st.extrema - st.descriptors));
    }
#endif
    if (!done) parallel_for(cv::Range(0, ny * nx), [&](const cv::Range& rg) {
        auto sift = cv::SIFT::create(0, 3, 0.02);
        for (int t = rg.start; t < rg.end; t++) {
            int i = t / nx, j = t % nx;
            int y0 = (int)(i * ((double)h / ny)), y1 = (int)((i + 1) * ((double)h / ny));
            int x0 = (int)(j * ((double)w / nx)), x1 = (int)((j + 1) * ((double)w / nx));
            int Y0 = std::max(y0 - overlap, 0), Y1 = std::min(y1 + overlap, h);
            int X0 = std::max(x0 - overlap, 0), X1 = std::min(x1 + overlap, w);
            cv::Rect roi(X0, Y0, X1 - X0, Y1 - Y0);
            cv::Mat m;
            if (!mask.empty()) {
                m = mask(roi);
                if (cv::countNonZero(m) == 0) continue;
            }
            std::vector<cv::KeyPoint> kp;
            cv::Mat d;
            sift->detectAndCompute(g(roi).clone(), m.empty() ? cv::noArray() : cv::InputArray(m.clone()), kp, d);
            Tile& T = tiles[t];
            for (size_t k = 0; k < kp.size(); k++) {
                double x = (double)kp[k].pt.x + X0, y = (double)kp[k].pt.y + Y0;
                if (x0 <= x && x < x1 && y0 <= y && y < y1) {
                    T.P.emplace_back((float)x, (float)y);  // float32 like cv::KeyPoint
                    T.resp.push_back(kp[k].response);
                    T.ds.push_back(d.row((int)k));
                }
            }
        }
    });
    std::vector<Vec2> P;
    std::vector<float> resp;
    std::vector<cv::Mat> ds;
    for (auto& T : tiles) {
        P.insert(P.end(), T.P.begin(), T.P.end());
        resp.insert(resp.end(), T.resp.begin(), T.resp.end());
        ds.insert(ds.end(), T.ds.begin(), T.ds.end());
    }
    std::vector<size_t> order(P.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return resp[a] > resp[b]; });
    if (n_features > 0 && order.size() > (size_t)n_features) order.resize(n_features);
    pts.clear();
    desc.create((int)order.size(), 128, CV_32F);
    for (size_t k = 0; k < order.size(); k++) {
        pts.push_back(P[order[k]]);
        ds[order[k]].copyTo(desc.row((int)k));
    }
}

cv::Ptr<cv::flann::Index> flann_index(const cv::Mat& d) {
    if (d.rows < 2) return nullptr;
    return cv::makePtr<cv::flann::Index>(d, cv::flann::KDTreeIndexParams(4));
}

// 2 nearest neighbours in the indexed train set of every query row (FLANN, as cv::FlannBasedMatcher
// with 4 trees / 32 checks), searched in parallel. good[i]: ratio test 0.85 passed.
static void knn_ratio(const cv::Mat& query, cv::flann::Index* index, std::vector<int>& best,
                      std::vector<char>& good) {
    best.assign(query.rows, -1);
    good.assign(query.rows, 0);
    if (!index || query.rows == 0) return;
    cv::Mat ind(query.rows, 2, CV_32S), dist(query.rows, 2, CV_32F);
    parallel_for(cv::Range(0, query.rows), [&](const cv::Range& r) {
        cv::Mat ii = ind.rowRange(r.start, r.end), dd = dist.rowRange(r.start, r.end);
        index->knnSearch(query.rowRange(r.start, r.end), ii, dd, 2, cv::flann::SearchParams(32));
    });
    for (int i = 0; i < query.rows; i++) {
        const int* id = ind.ptr<int>(i);
        const float* d = dist.ptr<float>(i);  // squared L2; the matcher compares L2
        if (id[0] < 0 || id[1] < 0) continue;
        best[i] = id[0];
        good[i] = std::sqrt(d[0]) < 0.85 * std::sqrt(d[1]);
    }
}

// Mutual nearest neighbours with ratio test 0.85. index0: FLANN index of d0 (built once per
// reference). The index of d is built in a second thread while d is searched in index0.
void match_ref(const cv::Mat& d0, cv::flann::Index* index0, const Pts2& pts, const cv::Mat& d, int seed,
               std::vector<int>& i0, Pts2& pj) {
    std::vector<int> b01, b10;
    std::vector<char> g01, g10;
    Thread t([&] {
        cv::setRNGSeed(seed);  // FLANN tree construction is random
        auto index = flann_index(d);
        knn_ratio(d0, index.get(), b01, g01);
    });
    knn_ratio(d, index0, b10, g10);
    t.join();
    i0.clear();
    pj.clear();
    for (int i = 0; i < d0.rows; i++)
        if (g01[i] && g10[b01[i]] && b10[b01[i]] == i) {
            i0.push_back(i);
            pj.push_back(pts[b01[i]]);
        }
}

static std::vector<cv::Point2d> cvpts(const Pts2& p) {
    std::vector<cv::Point2d> o;
    for (auto& q : p) o.emplace_back(q.x(), q.y());
    return o;
}

// Essential matrix (MAGSAC) with the most inliers in front of both cameras: a single run
// occasionally returns a wrong model; a few seeded runs make this rare and reproducible.
cv::Mat essential(const Pts2& q0, const Pts2& qj, const Cam& cam, double threshold, std::vector<char>& inl) {
    cv::Mat K = (cv::Mat_<double>(3, 3) << cam.f, 0, cam.cx, 0, cam.f, cam.cy, 0, 0, 1);
    auto a = cvpts(q0), b = cvpts(qj);
    cv::Mat best;
    int best_good = -1;
    inl.assign(q0.size(), 0);
    if (q0.size() < 5) return best;
    for (int k = 0; k < 4; k++) {
        cv::setRNGSeed(k);
        cv::Mat m;
        cv::Mat E = cv::findEssentialMat(a, b, K, cv::USAC_MAGSAC, 0.99999, threshold, 1000, m);
        if (E.empty()) continue;
        E = E.rowRange(0, 3).clone();  // degenerate input can return several stacked solutions
        std::vector<cv::Point2d> ai, bi;
        std::vector<char> in(q0.size());
        for (size_t i = 0; i < q0.size(); i++)
            if ((in[i] = m.at<uchar>((int)i) != 0)) {
                ai.push_back(a[i]);
                bi.push_back(b[i]);
            }
        int good = 0;
        if (ai.size() >= 5) {
            cv::Mat R, t;
            good = cv::recoverPose(E, ai, bi, K, R, t);
        }
        if (good > best_good) {
            best_good = good;
            best = E;
            inl = in;
        }
        if (good >= 0.95 * ai.size()) break;
    }
    return best;
}

void recover_pose(const cv::Mat& E, const Pts2& q0, const Pts2& qj, const Cam& cam, Pose& P) {
    cv::Mat K = (cv::Mat_<double>(3, 3) << cam.f, 0, cam.cx, 0, cam.f, cam.cy, 0, 0, 1), R, t;
    cv::recoverPose(E, cvpts(q0), cvpts(qj), K, R, t);
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) P.R(a, b) = R.at<double>(a, b);
        P.t[a] = t.at<double>(a);
    }
}

// Share of page cells (n x n over the page bbox) containing >= 3 points.
double page_coverage(const Pts2& pts, const cv::Mat& mask, int n) {
    cv::Rect bb = cv::boundingRect(mask);
    int x0 = bb.x, x1 = bb.x + bb.width, y0 = bb.y, y1 = bb.y + bb.height;
    std::vector<int> hist(n * n, 0);
    for (auto& p : pts) {
        if (p.y() < y0 || p.y() > y1 || p.x() < x0 || p.x() > x1) continue;
        int i = std::min((int)((p.y() - y0) / (y1 - y0) * n), n - 1);
        int j = std::min((int)((p.x() - x0) / (x1 - x0) * n), n - 1);
        hist[i * n + j]++;
    }
    int cells = 0, hit = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            cv::Rect r(x0 + (x1 - x0) * j / n, y0 + (y1 - y0) * i / n, 0, 0);
            r.width = x0 + (x1 - x0) * (j + 1) / n - r.x;
            r.height = y0 + (y1 - y0) * (i + 1) / n - r.y;
            if (r.area() <= 0) continue;
            if (cv::countNonZero(mask(r)) > 0.5 * r.area()) {
                cells++;
                hit += hist[i * n + j] >= 3;
            }
        }
    return (double)hit / std::max(cells, 1);
}

// Median angle between corresponding rays after removing the rotation (~0: no baseline).
double parallax_deg(const Pts2& q0, const Pts2& qj, const Mat3& R, const Cam& cam) {
    std::vector<double> a;
    for (size_t i = 0; i < q0.size(); i++) {
        Vec2 x0 = cam.norm(q0[i]), xj = cam.norm(qj[i]);
        Vec3 d0 = Vec3(x0.x(), x0.y(), 1).normalized();
        Vec3 dj = (R.transpose() * Vec3(xj.x(), xj.y(), 1)).normalized();
        a.push_back(std::acos(std::clamp(d0.dot(dj), -1.0, 1.0)));
    }
    return median(a) * 180 / M_PI;
}

// ------------------------------------------------------------ subpixel refinement

// Affine map img1->img2 around each query point, fitted to the k nearest matches.
void local_affines(const Pts2& p1, const Pts2& p2, const Pts2& q, std::vector<Eigen::Matrix2d>& A, Pts2& pred,
                   int k) {
    KdTree2 tree(p1);
    k = std::min<int>(k, (int)p1.size());
    A.resize(q.size());
    pred.resize(q.size());
    parallel_for(cv::Range(0, (int)q.size()), [&](const cv::Range& rg) {
        std::vector<int> idx;
        std::vector<double> dist;
        for (int i = rg.start; i < rg.end; i++) {
            tree.knn(q[i], k, idx, dist);
            Vec2 m1 = Vec2::Zero(), m2 = Vec2::Zero();
            for (int j : idx) {
                m1 += p1[j];
                m2 += p2[j];
            }
            m1 /= idx.size();
            m2 /= idx.size();
            Eigen::MatrixXd D1(idx.size(), 2), D2(idx.size(), 2);  // dynamic: thin SVD needs it
            for (size_t r = 0; r < idx.size(); r++) {
                D1.row(r) = (p1[idx[r]] - m1).transpose();
                D2.row(r) = (p2[idx[r]] - m2).transpose();
            }
            // pseudo-inverse solution (also covers collinear neighbours)
            Eigen::Matrix2d sol = D1.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(D2);
            A[i] = sol.transpose();
            pred[i] = m2 + A[i] * (q[i] - m1);
        }
    });
}

static inline float sample_zero(const cv::Mat& g, double x, double y) {
    if (!(x >= 0 && y >= 0 && x <= g.cols - 1 && y <= g.rows - 1)) return 0.f;
    int x0 = std::min((int)x, g.cols - 2), y0 = std::min((int)y, g.rows - 2);
    float a = (float)(x - x0), b = (float)(y - y0);
    const uchar* r0 = g.ptr<uchar>(y0);
    const uchar* r1 = g.ptr<uchar>(y0 + 1);
    return (1 - a) * (1 - b) * r0[x0] + a * (1 - b) * r0[x0 + 1] + (1 - a) * b * r1[x0] + a * b * r1[x0 + 1];
}

// NCC template matching of img1 patches (pre-warped by the local affine A) in a +-search
// window around p2, with parabolic subpixel peak (cv::matchTemplate TM_CCOEFF_NORMED semantics).
void refine_subpixel(const cv::Mat& g1, const cv::Mat& g2, const Pts2& p1, const Pts2& p2,
                     const std::vector<Eigen::Matrix2d>& A, Pts2& out, std::vector<char>& ok, int win, int search) {
    const int r = win / 2, S = win + 2 * search, K = 2 * search + 1;
    const int h1 = g1.rows, w1 = g1.cols, h2 = g2.rows, w2 = g2.cols;
    out = p2;
    ok.assign(p1.size(), 0);
    parallel_for(cv::Range(0, (int)p1.size()), [&](const cv::Range& rg) {
        std::vector<float> tpl(win * win), reg(S * S), res(K * K);
        std::vector<double> ci((S + 1) * (S + 1)), ci2((S + 1) * (S + 1));
        for (int i = rg.start; i < rg.end; i++) {
            double x1 = p1[i].x(), y1 = p1[i].y(), x2 = p2[i].x(), y2 = p2[i].y();
            if (!(r + 1 <= x1 && x1 < w1 - r - 1 && r + 1 <= y1 && y1 < h1 - r - 1 && r + search + 1 <= x2 &&
                  x2 < w2 - r - search - 1 && r + search + 1 <= y2 && y2 < h2 - r - search - 1))
                continue;
            Eigen::Matrix2d Ai = A[i].inverse();
            double mean = 0;
            for (int yy = 0; yy < win; yy++)
                for (int xx = 0; xx < win; xx++) {
                    double dx = xx - r, dy = yy - r;
                    float v = sample_zero(g1, x1 + Ai(0, 0) * dx + Ai(0, 1) * dy, y1 + Ai(1, 0) * dx + Ai(1, 1) * dy);
                    tpl[yy * win + xx] = v;
                    mean += v;
                }
            mean /= win * win;
            double tn2 = 0;
            for (auto& v : tpl) {
                v = (float)(v - mean);
                tn2 += (double)v * v;
            }
            bool flat = std::sqrt(tn2 / (win * win)) < 4;
            double c0 = (S - 1) / 2.0;
            for (int yy = 0; yy < S; yy++)
                for (int xx = 0; xx < S; xx++) reg[yy * S + xx] = sample_zero(g2, x2 + xx - c0, y2 + yy - c0);
            for (int yy = 0; yy <= S; yy++) ci[yy * (S + 1)] = ci2[yy * (S + 1)] = 0;
            for (int xx = 0; xx <= S; xx++) ci[xx] = ci2[xx] = 0;
            for (int yy = 1; yy <= S; yy++) {
                double rs = 0, rs2 = 0;
                for (int xx = 1; xx <= S; xx++) {
                    double v = reg[(yy - 1) * S + xx - 1];
                    rs += v;
                    rs2 += v * v;
                    ci[yy * (S + 1) + xx] = ci[(yy - 1) * (S + 1) + xx] + rs;
                    ci2[yy * (S + 1) + xx] = ci2[(yy - 1) * (S + 1) + xx] + rs2;
                }
            }
            auto box = [&](const std::vector<double>& c, int oy, int ox) {
                return c[(oy + win) * (S + 1) + ox + win] - c[oy * (S + 1) + ox + win] - c[(oy + win) * (S + 1) + ox] +
                       c[oy * (S + 1) + ox];
            };
            int best = 0;
            for (int oy = 0; oy < K; oy++)
                for (int ox = 0; ox < K; ox++) {
                    float num = 0;
                    for (int yy = 0; yy < win; yy++) {
                        const float* rr = &reg[(oy + yy) * S + ox];
                        const float* tt = &tpl[yy * win];
                        for (int xx = 0; xx < win; xx++) num += rr[xx] * tt[xx];
                    }
                    double ws = box(ci, oy, ox), wq = box(ci2, oy, ox);
                    double wvar = std::max(wq - ws * ws / (win * win), 0.0);
                    res[oy * K + ox] = (float)(num / std::sqrt(std::max(wvar * tn2, 1e-12)));
                    if (res[oy * K + ox] > res[best]) best = oy * K + ox;
                }
            int cy = best / K, cx = best % K;
            float mx = res[best];
            ok[i] = !flat && mx >= 0.85f && cx > 0 && cx < K - 1 && cy > 0 && cy < K - 1;
            int cxc = std::clamp(cx, 1, K - 2), cyc = std::clamp(cy, 1, K - 2);
            auto para = [](double l, double c, double rr) {
                double d = l - 2 * c + rr;
                return std::abs(d) < 1e-9 ? 0.0 : 0.5 * (l - rr) / d;
            };
            double sx = para(res[cyc * K + cxc - 1], res[cyc * K + cxc], res[cyc * K + cxc + 1]);
            double sy = para(res[(cyc - 1) * K + cxc], res[cyc * K + cxc], res[(cyc + 1) * K + cxc]);
            out[i] = Vec2(x2 - search + cx + sx, y2 - search + cy + sy);
        }
    });
}

// Guided matching from reference corners into every other view: position predicted by
// projecting nearby reconstructed points, refined by NCC.
void densify(const std::vector<cv::Mat>& gs, const cv::Mat& mask, const Pts2& p_ref,
             const Pts3& X, const Cam& cam, const std::vector<Pose>& poses, int max_pts, int min_dist, int search,
             Obs& dobs) {
    const int V = (int)poses.size();
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(gs[0], corners, max_pts, 0.005, min_dist, mask, 5, false, 0.04);
    KdTree2 tree(p_ref);
    Pts2 q;
    for (auto& c : corners) {
        Vec2 p(c.x, c.y);
        if (tree.nearest(p).second < 200) q.push_back(p);
    }
    Obs obs(q.size(), V);
    for (size_t i = 0; i < q.size(); i++) obs.at(i, 0) = q[i];
    for (int j = 1; j < V; j++) {
        Pts2 pj(X.size());
        for (size_t i = 0; i < X.size(); i++) pj[i] = project(X[i], poses[j], cam).first;
        std::vector<Eigen::Matrix2d> A;
        Pts2 pred, r;
        std::vector<char> ok;
        local_affines(p_ref, pj, q, A, pred, 12);
        refine_subpixel(gs[0], gs[j], q, pred, A, r, ok, 21, search);
        for (size_t i = 0; i < q.size(); i++)
            if (ok[i] && (r[i] - pred[i]).norm() < search - 1) obs.at(i, j) = r[i];
    }
    std::vector<size_t> keep;
    for (size_t i = 0; i < q.size(); i++) {
        bool any = false;
        for (int j = 1; j < V; j++) any |= obs.has(i, j);
        if (any) keep.push_back(i);
    }
    dobs = obs.rows(keep);
}

}  // namespace docscan
