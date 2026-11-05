#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <map>

#include "stages.hpp"

namespace docscan {

using SpMat = Eigen::SparseMatrix<double>;
using Trip = Eigen::Triplet<double>;

// ------------------------------------------------------------ page frame

void fit_plane(const Pts3& X, Vec3& c, Vec3& n) {
    std::vector<double> xs[3];
    for (auto& p : X)
        for (int a = 0; a < 3; a++) xs[a].push_back(p[a]);
    c = Vec3(median(xs[0]), median(xs[1]), median(xs[2]));
    auto normal_of = [&](const std::vector<char>& keep, const Vec3& ctr) {
        Mat3 C = Mat3::Zero();
        for (size_t i = 0; i < X.size(); i++)
            if (keep[i]) C += (X[i] - ctr) * (X[i] - ctr).transpose();
        Eigen::SelfAdjointEigenSolver<Mat3> es(C);
        return Vec3(es.eigenvectors().col(0));  // smallest eigenvalue
    };
    std::vector<char> keep(X.size(), 1);
    for (int it = 0; it < 3; it++) {
        n = normal_of(std::vector<char>(X.size(), 1), c);
        std::vector<double> d(X.size());
        for (size_t i = 0; i < X.size(); i++) d[i] = std::abs((X[i] - c).dot(n));
        double thr = 3 * median(d) + 1e-12;
        Vec3 s = Vec3::Zero();
        int cnt = 0;
        for (size_t i = 0; i < X.size(); i++)
            if ((keep[i] = d[i] < thr)) {
                s += X[i];
                cnt++;
            }
        c = s / cnt;
    }
    n = normal_of(keep, c);
    if (n.dot(c) > 0) n = -n;  // normal points towards the reference camera (origin)
}

static Vec3 ray_of(const Vec2& px, const Cam& cam) {
    Vec2 x = cam.norm(px);
    return Vec3(x.x(), x.y(), 1);
}

// Origin at a page corner, eu along the short edge, ev along the long edge, en normal.
Frame page_frame(const Cam& cam, const std::array<Vec2, 4>& quad, const Vec3& c, Vec3 n, double W, double H) {
    Vec3 C[4];
    for (int k = 0; k < 4; k++) {
        Vec3 r = ray_of(quad[k], cam);
        C[k] = r * (c.dot(n) / r.dot(n));
    }
    double top = (C[1] - C[0]).norm(), right = (C[2] - C[1]).norm();
    double bottom = (C[2] - C[3]).norm(), left = (C[3] - C[0]).norm();
    double horiz = (top + bottom) / 2, vert = (left + right) / 2;
    Frame F;
    Vec3 ue, ve;
    double short_, long_;
    if (horiz <= vert) {
        F.o = C[0], ue = C[1] - C[0], ve = C[3] - C[0];
        short_ = horiz, long_ = vert;
        int m[4] = {0, 1, 2, 3};
        std::copy(m, m + 4, F.edge_map);
    } else {
        F.o = C[3], ue = C[0] - C[3], ve = C[2] - C[3];
        short_ = vert, long_ = horiz;
        int m[4] = {1, 2, 3, 0};  // image edge k -> page side
        std::copy(m, m + 4, F.edge_map);
    }
    F.eu = (ue - ue.dot(n) * n).normalized();
    F.ev = n.cross(F.eu);
    if (F.ev.dot(ve) < 0) {
        n = -n;
        F.ev = n.cross(F.eu);
    }
    F.en = n;
    F.scale = 0.5 * (W / short_ + H / long_);
    return F;
}

// Height-field extent: the nominal page plus whatever the data covers.
std::array<double, 4> grid_bounds(const Pts2& uv, double W, double H, double margin) {
    std::vector<double> u, v;
    for (auto& p : uv) {
        u.push_back(p.x());
        v.push_back(p.y());
    }
    double u0 = std::min(-margin, percentile(u, 0.5) - margin), u1 = std::max(W + margin, percentile(u, 99.5) + margin);
    double v0 = std::min(-margin, percentile(v, 0.5) - margin), v1 = std::max(H + margin, percentile(v, 99.5) + margin);
    double mu = median(u), mv = median(v);
    return {std::max(u0, mu - 1.2 * W), std::min(u1, mu + 1.2 * W), std::max(v0, mv - 1.2 * H),
            std::min(v1, mv + 1.2 * H)};
}

// ------------------------------------------------------------ height field

// Robust smooth z(u,v): bilinear data term + second-difference regularizer, IRLS
// (Huber on data, Cauchy on curvature so creases stay sharp-ish).
Surface fit_heightfield(const Pts3& uvz, const std::array<double, 4>& b, double step, double lam, double ruling_w,
                        std::vector<double>* resid) {
    Surface S;
    S.g.u0 = b[0], S.g.v0 = b[2], S.g.step = step;
    S.g.nu = (int)arange(b[0], b[1] + step, step).size();
    S.g.nv = (int)arange(b[2], b[3] + step, step).size();
    const int nu = S.g.nu, nv = S.g.nv, N = nu * nv;

    std::vector<Trip> ta;
    std::vector<double> z;
    for (auto& p : uvz) {
        double fi = (p.x() - S.g.u0) / step, fj = (p.y() - S.g.v0) / step;
        if (!(fi >= 0 && fi < nu - 1 && fj >= 0 && fj < nv - 1)) continue;
        int i0 = (int)fi, j0 = (int)fj, r = (int)z.size();
        double a = fi - i0, bb = fj - j0;
        ta.emplace_back(r, j0 * nu + i0, (1 - a) * (1 - bb));
        ta.emplace_back(r, j0 * nu + i0 + 1, a * (1 - bb));
        ta.emplace_back(r, (j0 + 1) * nu + i0, (1 - a) * bb);
        ta.emplace_back(r, (j0 + 1) * nu + i0 + 1, a * bb);
        z.push_back(p.z());
    }
    const int nz = (int)z.size();
    SpMat A(nz, N);
    A.setFromTriplets(ta.begin(), ta.end());
    std::vector<Trip> td;
    std::vector<double> dir_w;
    int k = 0;
    for (int j = 0; j < nv; j++)
        for (int i = 0; i + 2 < nu; i++, k++) {
            td.emplace_back(k, j * nu + i, 1.0);
            td.emplace_back(k, j * nu + i + 1, -2.0);
            td.emplace_back(k, j * nu + i + 2, 1.0);
            dir_w.push_back(ruling_w);  // extra stiffness along u when u is the fold direction
        }
    for (int j = 0; j + 2 < nv; j++)
        for (int i = 0; i < nu; i++, k++) {
            td.emplace_back(k, j * nu + i, 1.0);
            td.emplace_back(k, (j + 1) * nu + i, -2.0);
            td.emplace_back(k, (j + 2) * nu + i, 1.0);
            dir_w.push_back(1.0);
        }
    SpMat D(k, N);
    D.setFromTriplets(td.begin(), td.end());
    SpMat At = A.transpose(), Dt = D.transpose();
    Eigen::Map<Eigen::VectorXd> zv(z.data(), nz);

    double area = (b[1] - b[0]) * (b[3] - b[2]);
    double dens = nz / area * step * step;
    Eigen::VectorXd wd = Eigen::VectorXd::Ones(nz), wr(k), zz;
    for (int r = 0; r < k; r++) wr[r] = dir_w[r] * lam * dens;
    SpMat I(N, N);
    I.setIdentity();
    Eigen::SimplicialLDLT<SpMat> solver;
    for (int it = 0; it < 6; it++) {
        SpMat M = At * wd.asDiagonal() * A + Dt * wr.asDiagonal() * D + 1e-9 * I;
        if (it == 0) solver.analyzePattern(M);
        solver.factorize(M);
        zz = solver.solve(At * wd.cwiseProduct(zv));
        Eigen::VectorXd rd = A * zz - zv, rr = D * zz;
        std::vector<double> ard(nz), arr(k);
        for (int r = 0; r < nz; r++) ard[r] = std::abs(rd[r]);
        for (int r = 0; r < k; r++) arr[r] = std::abs(rr[r]);
        double sd = 1.4826 * median(ard) + 1e-6, sr = 1.4826 * median(arr) + 1e-6;
        for (int r = 0; r < nz; r++) wd[r] = std::min(1.0, 2 * sd / (ard[r] + 1e-12));
        for (int r = 0; r < k; r++) wr[r] = dir_w[r] * lam * dens / (1 + std::pow(rr[r] / (3 * sr), 2));
    }
    S.Z.assign(zz.data(), zz.data() + N);
    if (resid) {
        Eigen::VectorXd rd = A * zz - zv;
        resid->assign(rd.data(), rd.data() + nz);
    }
    return S;
}

// Intersect reference-camera rays with the surface: scan for all crossings and refine the
// one nearest the plane estimate by bisection.
void ray_surface(const Pts2& px, const Cam& cam, const Frame& F, const Surface& S, const Vec3& c, const Vec3& n,
                 Pts3& X, std::vector<char>& ok) {
    const int n_scan = 41, iters = 40;
    X.resize(px.size());
    ok.resize(px.size());
    for (size_t i = 0; i < px.size(); i++) {
        Vec3 ray = ray_of(px[i], cam);
        double lam0 = c.dot(n) / ray.dot(n);
        auto f = [&](double lam) {
            Vec3 u = F.to_page(ray * lam);
            return u.z() - S.g.sample(S.Z, u.x(), u.y());
        };
        double vals[n_scan], fr[n_scan];
        for (int k = 0; k < n_scan; k++) {
            fr[k] = 0.6 + 0.8 * k / (n_scan - 1);
            vals[k] = f(lam0 * fr[k]);
        }
        int pick = -1;
        double bestc = 1e300;
        for (int k = 0; k + 1 < n_scan; k++)
            if (vals[k] * vals[k + 1] <= 0) {
                double cc = std::abs(0.5 * (fr[k] + fr[k + 1]) - 1.0);
                if (cc < bestc) bestc = cc, pick = k;
            }
        ok[i] = pick >= 0;
        if (!ok[i]) {
            X[i] = ray * lam0;
            continue;
        }
        double lo = lam0 * fr[pick], hi = lam0 * fr[pick + 1], flo = f(lo);
        for (int it = 0; it < iters; it++) {
            double mid = 0.5 * (lo + hi), fm = f(mid);
            if (flo * fm <= 0) hi = mid;
            else lo = mid, flo = fm;
        }
        X[i] = ray * (0.5 * (lo + hi));
    }
}

// np.gradient along u (columns) and v (rows), edge_order 1
void grid_gradient(const Grid& g, const std::vector<double>& Z, std::vector<double>& zu, std::vector<double>& zv) {
    int nu = g.nu, nv = g.nv;
    zu.resize(Z.size());
    zv.resize(Z.size());
    for (int j = 0; j < nv; j++)
        for (int i = 0; i < nu; i++) {
            auto at = [&](int jj, int ii) { return Z[(size_t)jj * nu + ii]; };
            zu[(size_t)j * nu + i] = i == 0        ? (at(j, 1) - at(j, 0)) / g.step
                                     : i == nu - 1 ? (at(j, nu - 1) - at(j, nu - 2)) / g.step
                                                   : (at(j, i + 1) - at(j, i - 1)) / (2 * g.step);
            zv[(size_t)j * nu + i] = j == 0        ? (at(1, i) - at(0, i)) / g.step
                                     : j == nv - 1 ? (at(nv - 1, i) - at(nv - 2, i)) / g.step
                                                   : (at(j + 1, i) - at(j - 1, i)) / (2 * g.step);
        }
}

// Dominant bending direction from the curvature: in-plane angle of the fold lines, anisotropy.
std::pair<double, double> fold_direction(const Surface& S) {
    std::vector<double> zu, zv, zuu, zuv, zvu, zvv;
    grid_gradient(S.g, S.Z, zu, zv);
    grid_gradient(S.g, zu, zuu, zuv);
    grid_gradient(S.g, zv, zvu, zvv);
    Eigen::Matrix2d T = Eigen::Matrix2d::Zero();
    for (size_t i = 0; i < S.Z.size(); i++) {
        Eigen::Matrix2d Hm;
        Hm << zuu[i], zuv[i], zuv[i], zvv[i];
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(Hm);
        int k = std::abs(es.eigenvalues()[1]) > std::abs(es.eigenvalues()[0]) ? 1 : 0;
        double w = std::abs(es.eigenvalues()[k]) - std::abs(es.eigenvalues()[1 - k]);
        T += w * es.eigenvectors().col(k) * es.eigenvectors().col(k).transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(T);
    Eigen::Vector2d across = es.eigenvectors().col(1);
    double aniso = (es.eigenvalues()[1] - es.eigenvalues()[0]) / (es.eigenvalues()[1] + es.eigenvalues()[0] + 1e-12);
    double ang = std::atan2(across.x(), -across.y());  // ruling = (-across.y, across.x)
    ang = std::fmod(ang + M_PI / 2, M_PI);
    if (ang < 0) ang += M_PI;
    return {ang - M_PI / 2, aniso};
}

// ------------------------------------------------------------ unrolling

// Unroll the height-field mesh into the plane preserving 3D edge lengths (Gauss-Newton,
// direct sparse solve). Returns flat (s, t) per grid vertex in mm.
Pts2 flatten_grid(const Surface& S, double& strain_med, double& strain_max) {
    const int nu = S.g.nu, nv = S.g.nv, N = nu * nv;
    std::vector<std::pair<int, int>> E;
    for (int j = 0; j < nv; j++)
        for (int i = 0; i + 1 < nu; i++) E.push_back({j * nu + i, j * nu + i + 1});
    for (int j = 0; j + 1 < nv; j++)
        for (int i = 0; i < nu; i++) E.push_back({j * nu + i, (j + 1) * nu + i});
    for (int j = 0; j + 1 < nv; j++)
        for (int i = 0; i + 1 < nu; i++) E.push_back({j * nu + i, (j + 1) * nu + i + 1});
    for (int j = 0; j + 1 < nv; j++)
        for (int i = 1; i < nu; i++) E.push_back({j * nu + i, (j + 1) * nu + i - 1});
    const size_t m = E.size();
    std::vector<Vec3> P(N);
    for (int j = 0; j < nv; j++)
        for (int i = 0; i < nu; i++) P[j * nu + i] = Vec3(S.g.gu(i), S.g.gv(j), S.Z[(size_t)j * nu + i]);
    std::vector<double> L(m);
    for (size_t e = 0; e < m; e++) L[e] = (P[E[e].first] - P[E[e].second]).norm();
    Eigen::VectorXd x0(2 * N);
    for (int i = 0; i < N; i++) x0.segment<2>(2 * i) = P[i].head<2>();
    const double anchor = 1e-3;
    auto cost_of = [&](const Eigen::VectorXd& x, std::vector<double>* rd) {
        double c = 0;
        if (rd) rd->resize(m);
        for (size_t e = 0; e < m; e++) {
            double r = (x.segment<2>(2 * E[e].first) - x.segment<2>(2 * E[e].second)).norm() - L[e];
            if (rd) (*rd)[e] = r;
            c += r * r;
        }
        c += anchor * anchor * (x - x0).squaredNorm();
        return 0.5 * c;
    };
    Eigen::VectorXd x = x0;
    std::vector<double> rd;
    double c = cost_of(x, &rd);
    Eigen::SimplicialLDLT<SpMat> solver;
    std::vector<Trip> trip;
    for (int it = 0; it < 30; it++) {
        trip.clear();
        Eigen::VectorXd grad = anchor * anchor * (x - x0);
        for (size_t e = 0; e < m; e++) {
            int a = E[e].first, b = E[e].second;
            Vec2 d = x.segment<2>(2 * a) - x.segment<2>(2 * b);
            Vec2 g = d / std::max(d.norm(), 1e-12);
            int idx[4] = {2 * a, 2 * a + 1, 2 * b, 2 * b + 1};
            double val[4] = {g.x(), g.y(), -g.x(), -g.y()};
            for (int p = 0; p < 4; p++) {
                grad[idx[p]] += val[p] * rd[e];
                for (int q = 0; q < 4; q++) trip.emplace_back(idx[p], idx[q], val[p] * val[q]);
            }
        }
        for (int i = 0; i < 2 * N; i++) trip.emplace_back(i, i, anchor * anchor);
        SpMat H(2 * N, 2 * N);
        H.setFromTriplets(trip.begin(), trip.end());
        if (it == 0) solver.analyzePattern(H);
        solver.factorize(H);
        Eigen::VectorXd step = solver.solve(-grad);
        double a = 1.0, cn;
        Eigen::VectorXd xn;
        std::vector<double> rdn;
        while (true) {
            xn = x + a * step;
            cn = cost_of(xn, &rdn);
            if (cn <= c || a < 1e-4) break;
            a *= 0.5;
        }
        bool done = c - cn <= 1e-10 * c;
        x = xn, c = cn, rd = rdn;
        if (done) break;
    }
    std::vector<double> strain(m);
    for (size_t e = 0; e < m; e++) strain[e] = std::abs(rd[e]) / L[e];
    strain_med = median(strain);
    strain_max = *std::max_element(strain.begin(), strain.end());
    Pts2 st(N);
    for (int i = 0; i < N; i++) st[i] = x.segment<2>(2 * i);
    return st;
}

// Bilinear sample of the per-vertex flat map at (u, v), clamped to the grid.
Vec2 sample_st(const Pts2& st, const Grid& g, double u, double v) {
    double fi = std::clamp((u - g.u0) / g.step, 0.0, (double)g.nu - 1);
    double fj = std::clamp((v - g.v0) / g.step, 0.0, (double)g.nv - 1);
    int i0 = (int)fi, j0 = (int)fj;
    int i1 = std::min(i0 + 1, g.nu - 1), j1 = std::min(j0 + 1, g.nv - 1);
    double a = fi - i0, b = fj - j0;
    return (1 - a) * (1 - b) * st[j0 * g.nu + i0] + a * (1 - b) * st[j0 * g.nu + i1] +
           (1 - a) * b * st[j1 * g.nu + i0] + a * b * st[j1 * g.nu + i1];
}

// Invert the grid map (u, v) -> (s, t) on a regular raster (res_mm): the mesh triangles are
// rasterised in (s, t) with barycentric interpolation of (u, v); NaN outside the mesh.
void flat_to_uv(const Pts2& st, const Grid& g, double W, double H, double res, cv::Mat& ur, cv::Mat& vr) {
    int ns = (int)arange(0, W + res, res).size(), nt = (int)arange(0, H + res, res).size();
    ur.create(nt, ns, CV_32F);
    vr.create(nt, ns, CV_32F);
    ur.setTo(NaN);
    vr.setTo(NaN);
    auto tri = [&](int a, int b, int c, double ua, double va, double ub, double vb, double uc, double vc) {
        Vec2 A = st[a] / res, B = st[b] / res, C = st[c] / res;
        double det = (B.x() - A.x()) * (C.y() - A.y()) - (C.x() - A.x()) * (B.y() - A.y());
        if (std::abs(det) < 1e-12) return;
        int x0 = std::max(0, (int)std::ceil(std::min({A.x(), B.x(), C.x()})));
        int x1 = std::min(ns - 1, (int)std::floor(std::max({A.x(), B.x(), C.x()})));
        int y0 = std::max(0, (int)std::ceil(std::min({A.y(), B.y(), C.y()})));
        int y1 = std::min(nt - 1, (int)std::floor(std::max({A.y(), B.y(), C.y()})));
        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++) {
                double l1 = ((x - A.x()) * (C.y() - A.y()) - (C.x() - A.x()) * (y - A.y())) / det;
                double l2 = ((B.x() - A.x()) * (y - A.y()) - (x - A.x()) * (B.y() - A.y())) / det;
                double l0 = 1 - l1 - l2;
                if (l0 < -1e-9 || l1 < -1e-9 || l2 < -1e-9) continue;
                ur.at<float>(y, x) = (float)(l0 * ua + l1 * ub + l2 * uc);
                vr.at<float>(y, x) = (float)(l0 * va + l1 * vb + l2 * vc);
            }
    };
    for (int j = 0; j + 1 < g.nv; j++)
        for (int i = 0; i + 1 < g.nu; i++) {
            int p00 = j * g.nu + i, p10 = p00 + 1, p01 = p00 + g.nu, p11 = p01 + 1;
            double u0 = g.gu(i), u1 = g.gu(i + 1), v0 = g.gv(j), v1 = g.gv(j + 1);
            tri(p00, p10, p11, u0, v0, u1, v0, u1, v1);
            tri(p00, p11, p01, u0, v0, u1, v1, u0, v1);
        }
}

// ------------------------------------------------------------ outline

// Edge index per outline point (0 top, 1 right, 2 bottom, 3 left in image quad order);
// -1 near corners. Uses the contour order between the corners.
std::vector<int> assign_edges(const Pts2& pts, const std::array<Vec2, 4>& quad, double corner_excl) {
    long n = (long)pts.size();
    long ci[4];
    for (int k = 0; k < 4; k++) {
        long b = 0;
        for (long i = 1; i < n; i++)
            if ((pts[i] - quad[k]).norm() < (pts[b] - quad[k]).norm()) b = i;
        ci[k] = b;
    }
    auto fwd = [&](long a, long b) { return ((b - a) % n + n) % n; };
    bool forward = fwd(ci[0], ci[1]) < fwd(ci[0], ci[2]) && fwd(ci[0], ci[1]) < fwd(ci[0], ci[3]);
    std::vector<int> eid(n, -1);
    for (int k = 0; k < 4; k++) {
        long a = ci[k], b = ci[(k + 1) % 4];
        long L = forward ? fwd(a, b) : fwd(b, a);
        for (long m = 0; m <= L; m++)
            if (corner_excl * L < m && m < (1 - corner_excl) * L) eid[forward ? (a + m) % n : ((a - m) % n + n) % n] = k;
    }
    return eid;
}

// Match reference-outline points to the other views' outlines along epipolar lines;
// X_pred picks the right crossing.
void triangulate_outline(const Pts2& out0, const std::vector<Pts2>& outs, const Cam& cam,
                         const std::vector<Pose>& poses, const Pts3& X_pred, Obs& obs) {
    const double min_sin = 0.35, max_px = 40.0;
    const int V = (int)poses.size();
    obs = Obs(out0.size(), V);
    for (size_t i = 0; i < out0.size(); i++) obs.at(i, 0) = out0[i];
    for (int j = 1; j < V; j++) {
        if (outs[j].empty()) continue;
        const Pts2& pj = outs[j];
        size_t m = pj.size();
        std::vector<Vec3> xj(m);
        for (size_t s = 0; s < m; s++) xj[s] = ray_of(pj[s], cam);
        const Vec3& t = poses[j].t;
        Mat3 tx;
        tx << 0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0;
        Mat3 Ej = tx * poses[j].R;
        for (size_t i = 0; i < out0.size(); i++) {
            Vec3 l = Ej * ray_of(out0[i], cam);
            Vec2 pred = project(X_pred[i], poses[j], cam).first;
            double best = 1e300;
            Vec2 bestp = Vec2::Zero();
            size_t bests = 0;
            for (size_t s = 0; s < m; s++) {
                const Vec3 &a = xj[s], &b = xj[(s + 1) % m];
                double va = a.dot(l), vb = b.dot(l);
                if (!(va * vb < 0)) continue;
                double tp = va / (va - vb);
                Vec2 pn = a.head<2>() + tp * (b.head<2>() - a.head<2>());
                Vec2 ppx(cam.f * pn.x() + cam.cx, cam.f * pn.y() + cam.cy);
                double d = (ppx - pred).norm();
                if (d < best) best = d, bestp = ppx, bests = s;
            }
            if (best > max_px) continue;
            Vec2 seg = xj[(bests + 1) % m].head<2>() - xj[bests].head<2>();
            seg /= seg.norm() + 1e-12;
            Vec2 ldir = Vec2(l[1], -l[0]) / (std::hypot(l[0], l[1]) + 1e-12);
            if (std::abs(ldir.x() * seg.y() - ldir.y() * seg.x()) < min_sin) continue;
            obs.at(i, j) = bestp;
        }
    }
}

// ------------------------------------------------------------ rectangle and corrections

Vec2 rect_apply(const Eigen::Vector4d& x, const Vec2& p) {
    double c = std::cos(x[1]), s = std::sin(x[1]);
    return Vec2(x[0] * (c * p.x() - s * p.y()) + x[2], x[0] * (s * p.x() + c * p.y()) + x[3]);
}

// Similarity (k, theta, tx, ty) putting flattened outline points on the W x H rectangle;
// side: 0 top, 1 right, 2 bottom, 3 left.
Eigen::Vector4d fit_rectangle(const Pts2& st, const std::vector<int>& side, double W, double H,
                              const Eigen::Vector4d& x0, std::vector<double>& resid) {
    const double target[4] = {0, W, H, 0};
    const int axis[4] = {1, 0, 1, 0};
    auto fun = [&](const Eigen::VectorXd& x, Eigen::VectorXd& r, Eigen::MatrixXd* J) {
        r.resize(st.size());
        if (J) J->resize(st.size(), 4);
        double c = std::cos(x[1]), s = std::sin(x[1]);
        for (size_t i = 0; i < st.size(); i++) {
            int ax = axis[side[i]];
            Vec2 q = rect_apply(x, st[i]);
            r[i] = q[ax] - target[side[i]];
            if (J) {
                Vec2 Rp(c * st[i].x() - s * st[i].y(), s * st[i].x() + c * st[i].y());
                Vec2 dth(x[0] * (-s * st[i].x() - c * st[i].y()), x[0] * (c * st[i].x() - s * st[i].y()));
                (*J)(i, 0) = Rp[ax];
                (*J)(i, 1) = dth[ax];
                (*J)(i, 2) = ax == 0;
                (*J)(i, 3) = ax == 1;
            }
        }
    };
    Eigen::VectorXd x = least_squares(fun, x0, Loss::SoftL1, 0.3, 100);
    Eigen::VectorXd r;
    fun(x, r, nullptr);
    resid.assign(r.data(), r.data() + r.size());
    return x;
}

Eigen::Vector4d corners_init(const Pts2& st_c, const Pts2& dst) {
    Vec2 ma = Vec2::Zero(), mb = Vec2::Zero();
    for (size_t k = 0; k < st_c.size(); k++) ma += st_c[k], mb += dst[k];
    ma /= st_c.size();
    mb /= dst.size();
    Eigen::Matrix2d C = Eigen::Matrix2d::Zero();
    for (size_t k = 0; k < st_c.size(); k++) C += (dst[k] - mb) * (st_c[k] - ma).transpose();
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(C, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d U = svd.matrixU(), Vt = svd.matrixV().transpose();
    Eigen::Matrix2d D = Eigen::Matrix2d::Identity();
    D(1, 1) = (U * Vt).determinant() < 0 ? -1 : 1;
    Eigen::Matrix2d R = U * D * Vt;
    Vec2 t = mb - R * ma;
    return Eigen::Vector4d(1.0, std::atan2(R(1, 0), R(0, 0)), t.x(), t.y());
}

// Value at each q of a straight line fitted to the ok samples (a, r) within +-half of q
// (at least the k_min nearest ones).
std::vector<double> local_line(const std::vector<double>& a_, const std::vector<double>& r_,
                               const std::vector<char>& ok, const std::vector<double>& q, double half, int k_min) {
    std::vector<double> a, r;
    for (size_t i = 0; i < a_.size(); i++)
        if (ok[i]) a.push_back(a_[i]), r.push_back(r_[i]);
    std::vector<double> out(q.size(), 0.0);
    if (a.empty()) return out;
    std::vector<double> d(a.size());
    for (size_t qi = 0; qi < q.size(); qi++) {
        double x = q[qi];
        for (size_t i = 0; i < a.size(); i++) d[i] = std::abs(a[i] - x);
        double thr = half;
        if (std::count_if(d.begin(), d.end(), [&](double v) { return v <= half; }) < k_min) {
            std::vector<double> s = d;
            std::sort(s.begin(), s.end());
            thr = s[std::min<size_t>(k_min, s.size()) - 1];
        }
        double mn = 1e300, mx = -1e300, sx = 0, sy = 0, sxx = 0, sxy = 0;
        int n = 0;
        for (size_t i = 0; i < a.size(); i++)
            if (d[i] <= thr) {
                double xx = a[i] - x;
                mn = std::min(mn, a[i]), mx = std::max(mx, a[i]);
                sx += xx, sy += r[i], sxx += xx * xx, sxy += xx * r[i], n++;
            }
        if (mx - mn < 1.0) out[qi] = sy / n;
        else {
            double den = n * sxx - sx * sx;
            double slope = (n * sxy - sx * sy) / den;
            out[qi] = (sy - slope * sx) / n;
        }
    }
    return out;
}

// least-squares polynomial (highest degree first, like numpy.polyfit)
std::vector<double> polyfit(const std::vector<double>& x, const std::vector<double>& y, int deg) {
    Eigen::MatrixXd A(x.size(), deg + 1);
    Eigen::VectorXd b(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        for (int k = 0; k <= deg; k++) A(i, k) = std::pow(x[i], deg - k);
        b[i] = y[i];
    }
    Eigen::VectorXd c = A.colPivHouseholderQr().solve(b);
    return std::vector<double>(c.data(), c.data() + c.size());
}

double polyval(const std::vector<double>& c, double x) {
    double v = 0;
    for (double k : c) v = v * x + k;
    return v;
}

// Near-axis line segments in a flat render, in flat mm.
void detect_lines(const cv::Mat& g8, double W, double H, std::vector<Eigen::Vector4d>& segs,
                  std::vector<char>& horiz) {
    const double min_mm = 8.0, max_dev = 3.0, border = 4.0;
    std::vector<cv::Vec4f> L;
    cv::createLineSegmentDetector(cv::LSD_REFINE_STD)->detect(g8, L);
    segs.clear();
    horiz.clear();
    double sc = W / g8.cols;
    auto inside = [&](double x, double y) { return x > border && x < W - border && y > border && y < H - border; };
    for (auto& l : L) {
        Eigen::Vector4d s(l[0] * sc, l[1] * sc, l[2] * sc, l[3] * sc);
        double dx = s[2] - s[0], dy = s[3] - s[1];
        double len = std::hypot(dx, dy);
        double ang = std::fmod(std::atan2(dy, dx) * 180 / M_PI + 360.0, 180.0);
        double dev = std::fmod(ang + 45, 90.0) - 45;
        bool hz = std::abs(std::fmod(ang + 90, 180.0) - 90) < 45;
        if (len >= min_mm && std::abs(dev) < max_dev && inside(s[0], s[1]) && inside(s[2], s[3])) {
            segs.push_back(s);
            horiz.push_back(hz);
        }
    }
}

Vec2 Disp::operator()(const Vec2& q) const {
    double fi = std::clamp((q.x() - s0) / step, 0.0, ns - 1.001);
    double fj = std::clamp((q.y() - t0) / step, 0.0, nt - 1.001);
    int i0 = (int)fi, j0 = (int)fj;
    double a = fi - i0, b = fj - j0;
    Vec2 out;
    for (int ax = 0; ax < 2; ax++) {
        const std::vector<double>& f = field[ax];
        out[ax] = (1 - a) * (1 - b) * f[j0 * ns + i0] + a * (1 - b) * f[j0 * ns + i0 + 1] +
                  (1 - a) * b * f[(j0 + 1) * ns + i0] + a * b * f[(j0 + 1) * ns + i0 + 1];
    }
    return out;
}

// One smooth 2D displacement field on the flat page, per axis, from
//   edges:   point + D = target (paper edge on the rectangle)
//   lines:   all samples of a segment share one cross-axis coordinate (left free)
//   anchors: points that should stay put
// plus second-difference smoothness.
Disp fit_correction(double W, double H, const std::vector<EdgeTerm>& edges, const std::vector<Eigen::Vector4d>& segs,
                    const std::vector<char>& horiz, const Pts2& anchors, double w_anchor) {
    const double step = 4.0, lam = 30.0, w_edge = 1.0, w_line = 1.0;
    Disp D;
    D.step = step;
    D.s0 = -2 * step;
    D.t0 = -2 * step;
    D.ns = (int)arange(-2 * step, W + 3 * step, step).size();
    D.nt = (int)arange(-2 * step, H + 3 * step, step).size();
    const int ns = D.ns, nt = D.nt, N = ns * nt;
    auto bil = [&](const Vec2& P, int cols[4], double vals[4]) {
        double fi = std::clamp((P.x() - D.s0) / step, 0.0, ns - 1.001);
        double fj = std::clamp((P.y() - D.t0) / step, 0.0, nt - 1.001);
        int i0 = (int)fi, j0 = (int)fj;
        double a = fi - i0, b = fj - j0;
        cols[0] = j0 * ns + i0, cols[1] = j0 * ns + i0 + 1, cols[2] = (j0 + 1) * ns + i0, cols[3] = (j0 + 1) * ns + i0 + 1;
        vals[0] = (1 - a) * (1 - b), vals[1] = a * (1 - b), vals[2] = (1 - a) * b, vals[3] = a * b;
    };
    // smoothness D^T D (both directions of second differences)
    std::vector<Trip> tdd;
    auto add_dd = [&](int p0, int p1, int p2) {
        int c[3] = {p0, p1, p2};
        double v[3] = {1, -2, 1};
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++) tdd.emplace_back(c[a], c[b], lam * v[a] * v[b]);
    };
    for (int j = 0; j < nt; j++)
        for (int i = 0; i + 2 < ns; i++) add_dd(j * ns + i, j * ns + i + 1, j * ns + i + 2);
    for (int j = 0; j + 2 < nt; j++)
        for (int i = 0; i < ns; i++) add_dd(j * ns + i, (j + 1) * ns + i, (j + 2) * ns + i);

    for (int axis = 0; axis < 2; axis++) {
        std::vector<Trip> trip = tdd;
        std::vector<std::pair<int, double>> rhs_acc;
        int n_c = 0, rows = 0;
        std::vector<double> rhs_vec;
        auto add_row = [&](const int* c, const double* v, int nn, double w, double rhs) {
            for (int a = 0; a < nn; a++) {
                for (int b = 0; b < nn; b++) trip.emplace_back(c[a], c[b], w * w * v[a] * v[b]);
                rhs_acc.push_back({c[a], w * w * v[a] * rhs});
            }
            rows++;
        };
        for (auto& e : edges) {
            if (e.axis != axis) continue;
            for (size_t i = 0; i < e.P.size(); i++) {
                int c[4];
                double v[4];
                bil(e.P[i], c, v);
                add_row(c, v, 4, w_edge, e.target[i] - e.P[i][axis]);
            }
        }
        // horizontal segments constrain t (axis 1), vertical ones s (axis 0)
        for (size_t k = 0; k < segs.size(); k++) {
            if ((bool)horiz[k] != (axis == 1)) continue;
            Vec2 a = segs[k].head<2>(), b = segs[k].tail<2>();
            int n = std::max((int)((b - a).norm() / 2.0), 2);
            double w = w_line / std::sqrt(n / 10.0 + 1);
            for (int s = 0; s < n; s++) {
                Vec2 P = a + (double)s / (n - 1) * (b - a);
                int c[5];
                double v[5];
                bil(P, c, v);
                c[4] = N + n_c;  // P_axis + D(P) - c_k = 0, c_k unknown
                v[4] = -1;
                add_row(c, v, 5, w, -P[axis]);
            }
            n_c++;
        }
        for (auto& p : anchors) {
            int c[4];
            double v[4];
            bil(p, c, v);
            add_row(c, v, 4, w_anchor, 0.0);
        }
        if (rows == 0) {
            D.field[axis].assign(N, 0.0);
            continue;
        }
        const int nu = N + n_c;
        for (int i = 0; i < nu; i++) trip.emplace_back(i, i, 1e-6);
        SpMat M(nu, nu);
        M.setFromTriplets(trip.begin(), trip.end());
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(nu);
        for (auto& [i, v] : rhs_acc) rhs[i] += v;
        Eigen::SimplicialLDLT<SpMat> solver(M);
        Eigen::VectorXd x = solver.solve(rhs);
        D.field[axis].assign(x.data(), x.data() + N);
    }
    return D;
}

}  // namespace docscan
