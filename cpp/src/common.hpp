// Internal types and helpers shared by the pipeline stages.
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <opencv2/core.hpp>

#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace docscan {

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Pts2 = std::vector<Vec2>;
using Pts3 = std::vector<Vec3>;

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

void log(const std::string& s);
std::string fmt(const char* f, ...);

// ------------------------------------------------------------ camera
// One pinhole camera (no distortion) shared by all views. View 0 is the reference:
// R = I, t = 0; world frame = reference camera frame.
struct Cam {
    double f, cx, cy;
    Vec2 norm(const Vec2& px) const { return {(px.x() - cx) / f, (px.y() - cy) / f}; }
    Mat3 K() const {
        Mat3 k;
        k << f, 0, cx, 0, f, cy, 0, 0, 1;
        return k;
    }
};

struct Pose {
    Mat3 R = Mat3::Identity();
    Vec3 t = Vec3::Zero();
};

// Observations: N points x V views, pixel positions, NaN where not observed.
struct Obs {
    int V = 0;
    std::vector<Vec2> p;
    Obs() = default;
    Obs(size_t n, int v) : V(v), p(n * v, Vec2(NaN, NaN)) {}
    size_t n() const { return V ? p.size() / V : 0; }
    Vec2& at(size_t i, int v) { return p[i * V + v]; }
    const Vec2& at(size_t i, int v) const { return p[i * V + v]; }
    bool has(size_t i, int v) const { return std::isfinite(p[i * V + v].x()); }
    Obs rows(const std::vector<size_t>& idx) const;
    void append(const Obs& o);
};

std::pair<Vec2, double> project(const Vec3& X, const Pose& P, const Cam& c);
Pts3 triangulate(const Obs& obs, const std::vector<Pose>& poses, const Cam& cam);
// reprojection error per (point, view), NaN where not observed; depth_ok: in front of all cameras
void reproj_errors(const Pts3& X, const Obs& obs, const std::vector<Pose>& poses, const Cam& cam,
                   std::vector<double>& err, std::vector<char>& depth_ok);
void clean_obs(Pts3& X, Obs& obs, const std::vector<Pose>& poses, const Cam& cam, double thr,
               std::vector<double>* err_out = nullptr);
void bundle_adjust(Cam& cam, std::vector<Pose>& poses, const Obs& obs, bool refine_f);

// ------------------------------------------------------------ robust least squares
// Small dense Levenberg-Marquardt with robust loss (IRLS weights). fun fills residuals
// and, if J != nullptr, the Jacobian.
enum class Loss { Linear, Huber, SoftL1 };
using ResidFn = std::function<void(const Eigen::VectorXd& x, Eigen::VectorXd& r, Eigen::MatrixXd* J)>;
Eigen::VectorXd least_squares(const ResidFn& fun, Eigen::VectorXd x, Loss loss, double f_scale,
                              int max_iter = 100);

// ------------------------------------------------------------ small utilities
double median(std::vector<double> v);
double percentile(std::vector<double> v, double q);  // linear interpolation, like numpy
template <class T> std::vector<T> select(const std::vector<T>& v, const std::vector<size_t>& idx) {
    std::vector<T> o;
    o.reserve(idx.size());
    for (size_t i : idx) o.push_back(v[i]);
    return o;
}
std::vector<size_t> where(const std::vector<char>& m);
// numpy.arange(a, b, step)
std::vector<double> arange(double a, double b, double step);
// bilinear sample of a CV_32F image with clamped coordinates (scipy map_coordinates, mode="nearest")
float sample_clamped(const cv::Mat& g, double x, double y);

// Exact k-nearest-neighbour search in 2D.
class KdTree2 {
public:
    explicit KdTree2(const Pts2& pts);
    // indices (and distances) of the k nearest points, nearest first
    void knn(const Vec2& q, int k, std::vector<int>& idx, std::vector<double>& dist) const;
    std::pair<int, double> nearest(const Vec2& q) const;

private:
    struct Node { int lo, hi, axis; double split; int left, right; };
    Pts2 pts_;
    std::vector<int> order_;
    std::vector<Node> nodes_;
    int build(int lo, int hi, int depth);
};

// ------------------------------------------------------------ page geometry
struct Frame {
    Vec3 o, eu, ev, en;
    double scale;
    int edge_map[4];  // image quad edge k (0 top, 1 right, 2 bottom, 3 left) -> page side (same coding)
    Vec3 to_page(const Vec3& X) const {
        Vec3 d = X - o;
        return Vec3(d.dot(eu), d.dot(ev), d.dot(en)) * scale;
    }
    Vec3 from_page(const Vec3& U) const {
        Vec3 u = U / scale;
        return o + u.x() * eu + u.y() * ev + u.z() * en;
    }
};

// Regular grid over (u, v) in mm; values row-major (v rows, u columns).
struct Grid {
    double u0 = 0, v0 = 0, step = 1;
    int nu = 0, nv = 0;
    double gu(int i) const { return u0 + i * step; }
    double gv(int j) const { return v0 + j * step; }
    double sample(const std::vector<double>& Z, double u, double v) const;
};

// A thread with a 16 MB stack (Android gives new threads about 1 MB); joined on destruction.
// Exceptions thrown by fn are rethrown by join().
class Thread {
public:
    explicit Thread(std::function<void()> fn);
    ~Thread();
    void join();
    Thread(const Thread&) = delete;
    Thread& operator=(const Thread&) = delete;

private:
    struct Impl;
    Impl* p;
};

// fn over [r.start, r.end) in chunks on all cores (cv::getNumThreads()). Used instead of
// cv::parallel_for_, which runs serially while any other thread is inside one (on the phone:
// the page detection of the camera preview). Nested calls run serially.
void parallel_for(const cv::Range& r, const std::function<void(const cv::Range&)>& fn);

}  // namespace docscan
