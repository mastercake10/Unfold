#include "common.hpp"
#include "docscan.hpp"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <memory>
#include <mutex>
#include <queue>
#include <exception>
#include <thread>

#include <pthread.h>

namespace docscan {

static std::mutex log_mu;
static std::function<void(const std::string&)> log_sink;

void set_log_sink(std::function<void(const std::string&)> sink) {
    std::lock_guard<std::mutex> lk(log_mu);
    log_sink = std::move(sink);
}

void log(const std::string& s) {
    std::lock_guard<std::mutex> lk(log_mu);
    if (log_sink) log_sink(s);
    else std::fprintf(stderr, "%s\n", s.c_str());
}

std::string fmt(const char* f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

Obs Obs::rows(const std::vector<size_t>& idx) const {
    Obs o(idx.size(), V);
    for (size_t k = 0; k < idx.size(); k++)
        for (int v = 0; v < V; v++) o.at(k, v) = at(idx[k], v);
    return o;
}

void Obs::append(const Obs& o) { p.insert(p.end(), o.p.begin(), o.p.end()); }

double median(std::vector<double> v) {
    if (v.empty()) return NaN;
    size_t n = v.size(), m = n / 2;
    std::nth_element(v.begin(), v.begin() + m, v.end());
    double hi = v[m];
    if (n % 2) return hi;
    double lo = *std::max_element(v.begin(), v.begin() + m);
    return 0.5 * (lo + hi);
}

double percentile(std::vector<double> v, double q) {
    if (v.empty()) return NaN;
    std::sort(v.begin(), v.end());
    double pos = q / 100.0 * (v.size() - 1);
    size_t i = (size_t)std::floor(pos);
    if (i + 1 >= v.size()) return v.back();
    double a = pos - i;
    return v[i] + a * (v[i + 1] - v[i]);
}

std::vector<size_t> where(const std::vector<char>& m) {
    std::vector<size_t> o;
    for (size_t i = 0; i < m.size(); i++)
        if (m[i]) o.push_back(i);
    return o;
}

std::vector<double> arange(double a, double b, double step) {
    long n = (long)std::ceil((b - a) / step);
    std::vector<double> o(std::max(n, 0L));
    for (long i = 0; i < n; i++) o[i] = a + i * step;
    return o;
}

float sample_clamped(const cv::Mat& g, double x, double y) {
    x = std::clamp(x, 0.0, (double)g.cols - 1);
    y = std::clamp(y, 0.0, (double)g.rows - 1);
    int x0 = (int)x, y0 = (int)y;
    int x1 = std::min(x0 + 1, g.cols - 1), y1 = std::min(y0 + 1, g.rows - 1);
    float a = (float)(x - x0), b = (float)(y - y0);
    const float* r0 = g.ptr<float>(y0);
    const float* r1 = g.ptr<float>(y1);
    return (1 - a) * (1 - b) * r0[x0] + a * (1 - b) * r0[x1] + (1 - a) * b * r1[x0] + a * b * r1[x1];
}

double Grid::sample(const std::vector<double>& Z, double u, double v) const {
    double fi = std::clamp((u - u0) / step, 0.0, nu - 1.001);
    double fj = std::clamp((v - v0) / step, 0.0, nv - 1.001);
    int i0 = (int)fi, j0 = (int)fj;
    double a = fi - i0, b = fj - j0;
    const double* r0 = &Z[(size_t)j0 * nu];
    const double* r1 = r0 + nu;
    return (1 - a) * (1 - b) * r0[i0] + a * (1 - b) * r0[i0 + 1] + (1 - a) * b * r1[i0] + a * b * r1[i0 + 1];
}

// ------------------------------------------------------------ kd-tree

KdTree2::KdTree2(const Pts2& pts) : pts_(pts), order_(pts.size()) {
    for (size_t i = 0; i < pts.size(); i++) order_[i] = (int)i;
    if (!pts.empty()) build(0, (int)pts.size(), 0);
}

int KdTree2::build(int lo, int hi, int depth) {
    int id = (int)nodes_.size();
    nodes_.push_back({lo, hi, -1, 0, -1, -1});
    if (hi - lo <= 8) return id;
    double mn[2] = {1e300, 1e300}, mx[2] = {-1e300, -1e300};
    for (int i = lo; i < hi; i++)
        for (int a = 0; a < 2; a++) {
            mn[a] = std::min(mn[a], pts_[order_[i]][a]);
            mx[a] = std::max(mx[a], pts_[order_[i]][a]);
        }
    int axis = (mx[0] - mn[0] >= mx[1] - mn[1]) ? 0 : 1;
    int mid = (lo + hi) / 2;
    std::nth_element(order_.begin() + lo, order_.begin() + mid, order_.begin() + hi,
                     [&](int a, int b) { return pts_[a][axis] < pts_[b][axis]; });
    double split = pts_[order_[mid]][axis];
    int l = build(lo, mid, depth + 1);
    int r = build(mid, hi, depth + 1);
    nodes_[id].axis = axis;
    nodes_[id].split = split;
    nodes_[id].left = l;
    nodes_[id].right = r;
    return id;
}

void KdTree2::knn(const Vec2& q, int k, std::vector<int>& idx, std::vector<double>& dist) const {
    idx.clear();
    dist.clear();
    if (pts_.empty() || k <= 0) return;
    k = std::min<int>(k, (int)pts_.size());
    std::priority_queue<std::pair<double, int>> heap;  // max-heap on squared distance
    std::function<void(int)> visit = [&](int id) {
        const Node& nd = nodes_[id];
        if (nd.axis < 0) {
            for (int i = nd.lo; i < nd.hi; i++) {
                int p = order_[i];
                double d2 = (pts_[p] - q).squaredNorm();
                if ((int)heap.size() < k) heap.push({d2, p});
                else if (d2 < heap.top().first) {
                    heap.pop();
                    heap.push({d2, p});
                }
            }
            return;
        }
        double diff = q[nd.axis] - nd.split;
        int near = diff < 0 ? nd.left : nd.right, far = diff < 0 ? nd.right : nd.left;
        visit(near);
        if ((int)heap.size() < k || diff * diff < heap.top().first) visit(far);
    };
    visit(0);
    idx.resize(heap.size());
    dist.resize(heap.size());
    for (int i = (int)heap.size() - 1; i >= 0; i--) {
        idx[i] = heap.top().second;
        dist[i] = std::sqrt(heap.top().first);
        heap.pop();
    }
}

std::pair<int, double> KdTree2::nearest(const Vec2& q) const {
    std::vector<int> i;
    std::vector<double> d;
    knn(q, 1, i, d);
    return i.empty() ? std::make_pair(-1, 1e300) : std::make_pair(i[0], d[0]);
}

// ------------------------------------------------------------ least squares

Eigen::VectorXd least_squares(const ResidFn& fun, Eigen::VectorXd x, Loss loss, double fs, int max_iter) {
    auto robust = [&](const Eigen::VectorXd& r, Eigen::VectorXd* w) {
        double cost = 0;
        if (w) w->resize(r.size());
        for (Eigen::Index i = 0; i < r.size(); i++) {
            double z = (r[i] / fs) * (r[i] / fs), rho, d1;
            switch (loss) {
                case Loss::Linear: rho = z; d1 = 1; break;
                case Loss::Huber:
                    if (z <= 1) { rho = z; d1 = 1; }
                    else { double sz = std::sqrt(z); rho = 2 * sz - 1; d1 = 1 / sz; }
                    break;
                default: { double s = std::sqrt(1 + z); rho = 2 * (s - 1); d1 = 1 / s; }
            }
            cost += 0.5 * fs * fs * rho;
            if (w) (*w)[i] = d1;
        }
        return cost;
    };
    Eigen::VectorXd r, rn, w;
    Eigen::MatrixXd J;
    double lambda = 1e-3;
    for (int it = 0; it < max_iter; it++) {
        fun(x, r, &J);
        double cost = robust(r, &w);
        Eigen::MatrixXd H = J.transpose() * w.asDiagonal() * J;
        Eigen::VectorXd g = J.transpose() * w.cwiseProduct(r);
        Eigen::VectorXd dH = H.diagonal().cwiseMax(1e-12);
        bool accepted = false, done = false;
        while (lambda < 1e12) {
            Eigen::MatrixXd A = H;
            A.diagonal() += lambda * dH;
            Eigen::VectorXd dx = A.ldlt().solve(-g);
            Eigen::VectorXd xn = x + dx;
            fun(xn, rn, nullptr);
            double cn = robust(rn, nullptr);
            if (cn <= cost) {
                accepted = true;
                done = (cost - cn) <= 1e-10 * cost || dx.norm() <= 1e-10 * (x.norm() + 1e-10);
                x = xn;
                lambda = std::max(lambda / 3, 1e-12);
                break;
            }
            lambda *= 4;
        }
        if (!accepted || done) break;
    }
    return x;
}

struct Thread::Impl {
    std::function<void()> fn;
    std::exception_ptr err;
    pthread_t t;
    bool running = false;
    static void* run(void* arg) {
        auto* p = (Impl*)arg;
        try {
            p->fn();
        } catch (...) {
            p->err = std::current_exception();
        }
        return nullptr;
    }
};

Thread::Thread(std::function<void()> fn) : p(new Impl) {
    p->fn = std::move(fn);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 16u << 20);
    p->running = pthread_create(&p->t, &attr, Impl::run, p) == 0;
    pthread_attr_destroy(&attr);
    if (!p->running) Impl::run(p);  // no thread available: run here
}

void Thread::join() {
    if (p->running) pthread_join(p->t, nullptr);
    p->running = false;
    if (p->err) {
        auto e = p->err;
        p->err = nullptr;
        std::rethrow_exception(e);
    }
}

Thread::~Thread() {
    if (p->running) pthread_join(p->t, nullptr);
    delete p;
}

static thread_local bool in_parallel = false;

void parallel_for(const cv::Range& r, const std::function<void(const cv::Range&)>& fn) {
    const int n = r.size();
    if (n <= 0) return;
    const int threads = std::min(std::max(cv::getNumThreads(), 1), n);
    if (threads == 1 || in_parallel) {
        fn(r);
        return;
    }
    const int chunk = std::max(1, n / (threads * 8));
    std::atomic<int> next{0};
    auto work = [&] {
        bool outer = in_parallel;
        in_parallel = true;
        for (int a; (a = next.fetch_add(chunk)) < n;) fn(cv::Range(r.start + a, r.start + std::min(n, a + chunk)));
        in_parallel = outer;
    };
    std::vector<std::unique_ptr<Thread>> th;
    for (int i = 1; i < threads; i++) th.push_back(std::make_unique<Thread>(work));
    std::exception_ptr err;
    try {
        work();
    } catch (...) {
        err = std::current_exception();
        next = n;  // stop handing out chunks
    }
    for (auto& t : th) try {
            t->join();
        } catch (...) {
            if (!err) err = std::current_exception();
            next = n;
        }
    if (err) std::rethrow_exception(err);
}

}  // namespace docscan
