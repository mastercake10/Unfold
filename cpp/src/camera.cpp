#include <opencv2/calib3d.hpp>

#include "common.hpp"

namespace docscan {

std::pair<Vec2, double> project(const Vec3& X, const Pose& P, const Cam& c) {
    Vec3 Xc = P.R * X + P.t;
    return {Vec2(c.f * Xc.x() / Xc.z() + c.cx, c.f * Xc.y() / Xc.z() + c.cy), Xc.z()};
}

// Linear least-squares N-view triangulation: each observation gives two rows a.X = -c of
// the DLT system; the 3x3 normal equations are solved per point.
Pts3 triangulate(const Obs& obs, const std::vector<Pose>& poses, const Cam& cam) {
    size_t N = obs.n();
    Pts3 X(N);
    for (size_t i = 0; i < N; i++) {
        Mat3 M = Mat3::Zero();
        Vec3 rhs = Vec3::Zero();
        for (int v = 0; v < obs.V; v++) {
            if (!obs.has(i, v)) continue;
            Vec2 xn = cam.norm(obs.at(i, v));
            const Mat3& R = poses[v].R;
            const Vec3& t = poses[v].t;
            for (int k = 0; k < 2; k++) {
                Vec3 a = xn[k] * R.row(2).transpose() - R.row(k).transpose();
                double c = xn[k] * t[2] - t[k];
                M += a * a.transpose();
                rhs -= a * c;
            }
        }
        // tiny ridge keeps points seen in one view only from being singular
        M.diagonal().array() += 1e-12 * M.trace() + 1e-300;
        X[i] = M.ldlt().solve(rhs);
    }
    return X;
}

void reproj_errors(const Pts3& X, const Obs& obs, const std::vector<Pose>& poses, const Cam& cam,
                   std::vector<double>& err, std::vector<char>& depth_ok) {
    size_t N = obs.n();
    int V = obs.V;
    err.assign(N * V, NaN);
    depth_ok.assign(N, 1);
    for (size_t i = 0; i < N; i++)
        for (int v = 0; v < V; v++) {
            auto [p, z] = project(X[i], poses[v], cam);
            if (!(z > 0)) depth_ok[i] = 0;
            if (obs.has(i, v)) err[i * V + v] = (p - obs.at(i, v)).norm();
        }
}

// Drop single observations with reprojection error > thr, retriangulate, and keep points
// still seen in the reference and at least one other view.
void clean_obs(Pts3& X, Obs& obs, const std::vector<Pose>& poses, const Cam& cam, double thr,
               std::vector<double>* err_out) {
    int V = obs.V;
    std::vector<double> err;
    std::vector<char> dok;
    for (int pass = 0; pass < 2; pass++) {
        reproj_errors(X, obs, poses, cam, err, dok);
        std::vector<size_t> keep;
        for (size_t i = 0; i < obs.n(); i++) {
            int others = 0;
            for (int v = 0; v < V; v++)
                if (err[i * V + v] > thr) obs.at(i, v) = Vec2(NaN, NaN);
            for (int v = 1; v < V; v++) others += obs.has(i, v);
            if (dok[i] && obs.has(i, 0) && others) keep.push_back(i);
        }
        obs = obs.rows(keep);
        X = triangulate(obs, poses, cam);
    }
    reproj_errors(X, obs, poses, cam, err, dok);
    std::vector<size_t> keep;
    for (size_t i = 0; i < obs.n(); i++) {
        bool bad = false;
        for (int v = 0; v < V; v++) bad |= err[i * V + v] > thr;
        if (dok[i] && !bad) keep.push_back(i);
    }
    obs = obs.rows(keep);
    X = select(X, keep);
    if (err_out) {
        err_out->clear();
        for (size_t i : keep)
            for (int v = 0; v < V; v++) err_out->push_back(err[i * V + v]);
    }
}

static Mat3 rodrigues(const Vec3& r, Eigen::Matrix<double, 3, 9>* jac = nullptr) {
    cv::Mat rv = (cv::Mat_<double>(3, 1) << r[0], r[1], r[2]), R, J;
    cv::Rodrigues(rv, R, J);
    Mat3 o;
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) o(a, b) = R.at<double>(a, b);
    if (jac)
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 9; b++) (*jac)(a, b) = J.at<double>(a, b);
    return o;
}

// Reduced bundle adjustment: focal (optional) and poses of views 1.. are the only
// parameters; points are re-triangulated inside every evaluation. Gauge: view 0 fixed,
// |t_1| fixed. Exact Jacobian: X solves M X = b (M = sum a a^T, b = -sum a c); differentiating
// gives M dX = -sum [da e + a (da^T X + dc)] with e = a^T X + c.
void bundle_adjust(Cam& cam, std::vector<Pose>& poses, const Obs& obs, bool refine_f) {
    const int V = obs.V;
    const size_t N = obs.n();
    const int nf = refine_f ? 1 : 0, P = nf + 6 * (V - 1);
    const double f0 = cam.f, t1n = poses[1].t.norm();
    Eigen::VectorXd x0(P);
    if (refine_f) x0[0] = 0;
    for (int j = 1; j < V; j++) {
        cv::Mat R(3, 3, CV_64F), rv;
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++) R.at<double>(a, b) = poses[j].R(a, b);
        cv::Rodrigues(R, rv);
        for (int a = 0; a < 3; a++) {
            x0[nf + 6 * (j - 1) + a] = rv.at<double>(a);
            x0[nf + 6 * (j - 1) + 3 + a] = poses[j].t[a];
        }
    }
    struct Unpacked {
        double f;
        std::vector<Mat3> R;
        std::vector<Vec3> t;
        std::vector<Eigen::Matrix<double, 3, 9>> dR;
    };
    auto unpack = [&](const Eigen::VectorXd& x) {
        Unpacked u;
        u.f = refine_f ? f0 * (1 + x[0]) : f0;
        u.R.assign(V, Mat3::Identity());
        u.t.assign(V, Vec3::Zero());
        u.dR.assign(V, Eigen::Matrix<double, 3, 9>::Zero());
        for (int j = 1; j < V; j++) {
            int o = nf + 6 * (j - 1);
            u.R[j] = rodrigues(x.segment<3>(o), &u.dR[j]);
            u.t[j] = x.segment<3>(o + 3);
        }
        return u;
    };

    auto fun = [&](const Eigen::VectorXd& x, Eigen::VectorXd& r, Eigen::MatrixXd* J) {
        Unpacked u = unpack(x);
        const double f = u.f;
        r.setZero(2 * N * V + 1);
        if (J) J->setZero(2 * N * V + 1, P);
        std::vector<Vec2> xn(V);
        std::vector<char> has(V);
        Vec3 A[8][2];
        double C[8][2], E[8][2];
        for (size_t i = 0; i < N; i++) {
            Mat3 M = Mat3::Zero();
            Vec3 b = Vec3::Zero();
            for (int v = 0; v < V; v++) {
                has[v] = obs.has(i, v);
                if (!has[v]) continue;
                xn[v] = Vec2((obs.at(i, v).x() - cam.cx) / f, (obs.at(i, v).y() - cam.cy) / f);
                for (int k = 0; k < 2; k++) {
                    A[v][k] = xn[v][k] * u.R[v].row(2).transpose() - u.R[v].row(k).transpose();
                    C[v][k] = xn[v][k] * u.t[v][2] - u.t[v][k];
                    M += A[v][k] * A[v][k].transpose();
                    b -= A[v][k] * C[v][k];
                }
            }
            M.diagonal().array() += 1e-12 * M.trace() + 1e-300;
            Mat3 Minv = M.inverse();
            Vec3 X = Minv * b;
            Vec3 Xc[8];
            Vec2 xh[8];
            double z[8];
            bool live[8];
            for (int v = 0; v < V; v++) {
                if (has[v])
                    for (int k = 0; k < 2; k++) E[v][k] = A[v][k].dot(X) + C[v][k];
                Xc[v] = u.R[v] * X + u.t[v];
                z[v] = Xc[v].z();
                xh[v] = Xc[v].head<2>() / z[v];
                live[v] = has[v] && z[v] > 0;
                if (live[v]) {
                    size_t row = (i * V + v) * 2;
                    r[row] = f * xh[v].x() + cam.cx - obs.at(i, v).x();
                    r[row + 1] = f * xh[v].y() + cam.cy - obs.at(i, v).y();
                }
            }
            if (!J) continue;
            // derivative columns; one parameter at a time
            auto column = [&](int p, const Vec3& s, int view, const Mat3* dR, const Vec3* dt, double df) {
                Vec3 dX = -(Minv * s);
                for (int v = 0; v < V; v++) {
                    if (!live[v]) continue;
                    Vec3 dXc = u.R[v] * dX;
                    if (v == view) {
                        if (dR) dXc += *dR * X;
                        if (dt) dXc += *dt;
                    }
                    Vec2 du = f * (dXc.head<2>() - xh[v] * dXc.z()) / z[v] + df * xh[v];
                    size_t row = (i * V + v) * 2;
                    (*J)(row, p) = du.x();
                    (*J)(row + 1, p) = du.y();
                }
            };
            if (refine_f) {
                double sf = f0 / f;  // d xn / d phi = -xn * sf
                Vec3 s = Vec3::Zero();
                for (int v = 0; v < V; v++) {
                    if (!has[v]) continue;
                    for (int k = 0; k < 2; k++) {
                        Vec3 da = -sf * xn[v][k] * u.R[v].row(2).transpose();
                        double dc = -sf * xn[v][k] * u.t[v][2];
                        s += da * E[v][k] + A[v][k] * (da.dot(X) + dc);
                    }
                }
                column(0, s, -1, nullptr, nullptr, f0);
            }
            for (int j = 1; j < V; j++) {
                int o = nf + 6 * (j - 1);
                for (int a = 0; a < 3; a++) {
                    Mat3 dR;
                    for (int q = 0; q < 9; q++) dR(q / 3, q % 3) = u.dR[j](a, q);
                    Vec3 s = Vec3::Zero();
                    if (has[j])
                        for (int k = 0; k < 2; k++) {
                            Vec3 da = xn[j][k] * dR.row(2).transpose() - dR.row(k).transpose();
                            s += da * E[j][k] + A[j][k] * da.dot(X);
                        }
                    column(o + a, s, j, &dR, nullptr, 0);
                }
                for (int a = 0; a < 3; a++) {
                    Vec3 dt = Vec3::Unit(a);
                    Vec3 s = Vec3::Zero();
                    if (has[j])
                        for (int k = 0; k < 2; k++) s += A[j][k] * (xn[j][k] * dt[2] - dt[k]);
                    column(o + 3 + a, s, j, nullptr, &dt, 0);
                }
            }
        }
        r[2 * N * V] = 1e3 * (u.t[1].norm() - t1n) / t1n;
        if (J)
            for (int a = 0; a < 3; a++) (*J)(2 * N * V, nf + 3 + a) = 1e3 * u.t[1][a] / u.t[1].norm() / t1n;
    };
    Eigen::VectorXd x = least_squares(fun, x0, Loss::Huber, 1.0, 100);
    Unpacked u = unpack(x);
    cam.f = u.f;
    for (int j = 1; j < V; j++) {
        poses[j].R = u.R[j];
        poses[j].t = u.t[j];
    }
}

}  // namespace docscan
