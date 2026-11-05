#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>
#include <random>
#include <stdexcept>
#include <thread>

#include "docscan.hpp"
#include "stages.hpp"

namespace docscan {

namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error(msg); }

Pts3 to_page(const Frame& F, const Pts3& X) {
    Pts3 o(X.size());
    for (size_t i = 0; i < X.size(); i++) o[i] = F.to_page(X[i]);
    return o;
}

Pts3 concat(const Pts3& a, const Pts3& b) {
    Pts3 o = a;
    o.insert(o.end(), b.begin(), b.end());
    return o;
}

Pts2 uv_of(const Pts3& p) {
    Pts2 o(p.size());
    for (size_t i = 0; i < p.size(); i++) o[i] = p[i].head<2>();
    return o;
}

Pts2 column(const Obs& obs, int v) {
    Pts2 o(obs.n());
    for (size_t i = 0; i < obs.n(); i++) o[i] = obs.at(i, v);
    return o;
}

// observations of views a and b only (as a 2-view Obs)
Obs pick_views(const Obs& obs, const std::vector<size_t>& rows, int a, int b) {
    Obs o(rows.size(), 2);
    for (size_t k = 0; k < rows.size(); k++) {
        o.at(k, 0) = obs.at(rows[k], a);
        o.at(k, 1) = obs.at(rows[k], b);
    }
    return o;
}

// some views could not be used; process() retries without them
struct BadViews {
    std::vector<int> views;
    std::string first_error;
};

struct ViewMatch {
    std::vector<int> i0;
    Pts2 pr;
    std::vector<char> ok;
    Pose pose;
    std::string msg, error;
};

}  // namespace

// ---------------------------------------------------------------- per-photo analysis

struct ViewData {
    cv::Mat gray;  // downscaled to max_side
    cv::Mat mask;  // page mask (may be empty for views > 0)
    ViewMatch m;   // views > 0: matches with the reference
    size_t n_kp = 0;
};

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point& t) {  // seconds since t, and restart t
    auto now = Clock::now();
    double d = std::chrono::duration<double>(now - t).count();
    t = now;
    return d;
}

struct Session::Impl {
    Options opt;
    int w = 0, h = 0;
    double s = 1, f_full = 0;
    Cam cam{};
    cv::Mat ref_full;  // reference photo at full size (rendered)
    std::array<Vec2, 4> quad;
    cv::Mat mask_er;
    Pts2 P0;
    cv::Mat d0;
    cv::Ptr<cv::flann::Index> index0;  // of d0, shared by all views
    std::vector<ViewData> views;

    void prepare(const cv::Mat& img, cv::Mat& small, cv::Mat& gray) const {
        if (img.empty() || img.type() != CV_8UC3) fail("photos must be 8-bit colour images");
        if (img.cols != w || img.rows != h) fail("all photos must have the same size");
        small = img;
        if (s < 1) cv::resize(img, small, cv::Size(), s, s, cv::INTER_AREA);
        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
    }

    void add_ref(const cv::Mat& img) {
        auto t = Clock::now();
        cv::Mat small;
        ViewData v;
        prepare(img, small, v.gray);
        std::array<Vec2, 4> q;
        if (!find_page_quad(small, q, v.mask)) fail("could not find the page in the first photo");
        quad = refine_quad(v.gray, q);
        cv::erode(v.mask, mask_er, cv::Mat::ones(15, 15, CV_8U));
        ref_full = img;
        views.push_back(std::move(v));
        log(fmt("  photo 1: page %.2f s", secs(t)));
    }

    // features of the reference, needed once a second photo comes (one photo needs none)
    void ref_features() {
        if (!d0.empty() || !P0.empty()) return;
        auto t = Clock::now();
        sift_detect(views[0].gray, mask_er, opt.features, P0, d0, opt.gpu);
        double t_sift = secs(t);
        cv::setRNGSeed(0);  // FLANN tree construction is random
        index0 = flann_index(d0);
        views[0].n_kp = P0.size();
        log(fmt("  photo 1: %zu keypoints, features %.2f s, index %.2f s", P0.size(), t_sift, secs(t)));
    }

    // features of view j and its matches with the reference; r.error set if unusable
    ViewData analyse(const cv::Mat& img, int j) const {
        auto t = Clock::now();
        cv::Mat small;
        ViewData v;
        prepare(img, small, v.gray);
        std::array<Vec2, 4> q;
        cv::Mat mj;
        if (find_page_quad(small, q, v.mask)) cv::dilate(v.mask, mj, cv::Mat::ones(41, 41, CV_8U));
        else v.mask.release();
        small.release();
        double t_page = secs(t);
        Pts2 kps;
        cv::Mat descs;
        sift_detect(v.gray, mj, opt.features, kps, descs, opt.gpu);
        double t_sift = secs(t);
        v.n_kp = kps.size();
        ViewMatch& r = v.m;
        cv::setRNGSeed(j);  // FLANN/MAGSAC randomness per view, independent of threading
        std::vector<int> i0;
        Pts2 pj;
        match_ref(d0, index0.get(), kps, descs, j, i0, pj);
        std::vector<Vec2> a(i0.size());
        for (size_t k = 0; k < i0.size(); k++) a[k] = P0[i0[k]];
        // identical or unmoved photo: the essential matrix is degenerate and its pose meaningless
        std::vector<double> disp(i0.size());
        for (size_t k = 0; k < i0.size(); k++) disp[k] = (pj[k] - a[k]).norm();
        double dmed = median(disp), diag = std::hypot(v.gray.cols, v.gray.rows);
        if (!disp.empty() && dmed < 0.01 * diag) {
            r.error = fmt("photo %d: the phone did not move since photo 1 (points moved %.0f px). Move the "
                          "phone 10-20 cm between shots.", j + 1, dmed);
            return v;
        }
        double t_match = secs(t);
        std::vector<char> inl;
        cv::Mat E = essential(a, pj, cam, 1.0, inl);
        double t_pose = secs(t);
        Pts2 q0, qj;
        for (size_t k = 0; k < i0.size(); k++)
            if (inl[k]) {
                r.i0.push_back(i0[k]);
                q0.push_back(a[k]);
                qj.push_back(pj[k]);
            }
        double cov = page_coverage(q0, mask_er);
        if (E.empty() || q0.size() < 8) {
            r.error = fmt("photo %d: does not match photo 1 (is the same page in view?)", j + 1);
            return v;
        }
        recover_pose(E, q0, qj, cam, r.pose);
        double par = parallax_deg(q0, qj, r.pose.R, cam);
        if (par < 1.5) {
            r.error = fmt("photo %d: almost no viewpoint change since photo 1 (parallax %.2f deg). Move the "
                          "phone 10-20 cm between shots instead of tilting it.", j + 1, par);
            return v;
        }
        std::vector<Eigen::Matrix2d> A;
        Pts2 pred;
        local_affines(q0, qj, q0, A, pred, 12);
        refine_subpixel(views[0].gray, v.gray, q0, qj, A, r.pr, r.ok, 15, 4);
        int nok = 0;
        for (size_t k = 0; k < q0.size(); k++) nok += r.ok[k] = r.ok[k] && (r.pr[k] - qj[k]).norm() < 1.5;
        r.msg = fmt("  photo %d: %zu keypoints, %zu inliers, %d refined, page coverage %.2f%s\n"
                    "    page %.2f s, features %.2f s, matching %.2f s, pose %.2f s, refinement %.2f s",
                    j + 1, kps.size(), q0.size(), nok, cov,
                    cov < 0.4 ? " (low: steep folds need tilt simulation, not ported)" : "", t_page, t_sift,
                    t_match, t_pose, secs(t));
        return v;
    }

    cv::Mat solve(const std::vector<int>& vidx) const;  // indices into views
    // state handed from the 3D part to edge/line correction and rendering
    struct Flat {
        Surface S;
        Frame F;
        Cam cam;
        Pts2 st, sto, out0, ctarget;
        std::vector<char> hit, use;
        std::vector<int> side;
        std::vector<size_t> cidx;
        Pts3 uvz;  // measured surface points (page frame)
        std::vector<double> rres;
    };
    cv::Mat solve_single() const;  // one photo: page plane from its corners
    cv::Mat correct_render(Flat& fl) const;
};

Session::Session(int width, int height, const Options& opt) : p(std::make_unique<Impl>()) {
    p->opt = opt;
    p->w = width;
    p->h = height;
    p->s = std::min(1.0, (double)opt.max_side / std::max(width, height));
    p->f_full = opt.focal_px > 0 ? opt.focal_px : 26.0 * std::hypot(width, height) / 43.27;
    const double s = p->s;
    p->cam = Cam{p->f_full * s, ((width - 1) / 2.0 + 0.5) * s - 0.5, ((height - 1) / 2.0 + 0.5) * s - 0.5};
}

Session::~Session() = default;

int Session::size() const { return (int)p->views.size(); }

std::string Session::add(const cv::Mat& img) {
    try {
        if (p->views.empty()) {
            p->add_ref(img);
            return "";
        }
        if (p->views.size() >= 8) return "at most 8 photos";
        p->ref_features();
        ViewData v = p->analyse(img, (int)p->views.size());
        if (!v.m.error.empty()) {
            log("  " + v.m.error);
            return v.m.error;
        }
        log(v.m.msg);
        p->views.push_back(std::move(v));
        return "";
    } catch (const std::exception& e) {
        return e.what();
    }
}

void Session::remove_last() {
    if (p->views.empty()) return;
    p->views.pop_back();
    if (p->views.empty()) {
        p->ref_full.release();
        p->P0.clear();
        p->index0.release();
        p->d0.release();
    }
}

cv::Mat Session::finish() {
    std::vector<int> use;
    for (int j = 0; j < size(); j++) use.push_back(j);
    while (true) {
        if (use.empty()) fail("no usable photo");
        try {
            return p->solve(use);
        } catch (const BadViews& b) {
            if ((int)use.size() - (int)b.views.size() < 1) fail(b.first_error);
            std::vector<int> keep;
            std::string dropped;
            for (size_t k = 0; k < use.size(); k++) {
                if (std::find(b.views.begin(), b.views.end(), (int)k) == b.views.end()) keep.push_back(use[k]);
                else dropped += fmt(" %d", use[k] + 1);
            }
            log("leaving out photo" + dropped + ", retrying with " + std::to_string(keep.size()) + " photos");
            use.swap(keep);
        }
    }
}

cv::Mat process(const std::vector<cv::Mat>& images, const Options& opt) {
    if (images.empty()) fail("no images");
    if (images.size() > 8) fail("at most 8 images");
    Session S(images[0].cols, images[0].rows, opt);
    log(fmt("%zu views, focal start %.1f px", images.size(), S.p->f_full));
    log("matching ...");
    std::string err = S.add(images[0]);
    if (!err.empty()) fail(err);
    if (images.size() > 1) S.p->ref_features();
    // the other views in parallel, kept in input order
    const int V = (int)images.size();
    std::vector<ViewData> vd(V);
    {
        std::vector<std::thread> th;
        for (int j = 1; j < V; j++)
            th.emplace_back([&, j] {
                try {
                    vd[j] = S.p->analyse(images[j], j);
                } catch (const std::exception& e) {
                    vd[j].m.error = e.what();
                }
            });
        for (auto& t : th) t.join();
    }
    std::string first_error;
    for (int j = 1; j < V; j++) {
        if (!vd[j].m.error.empty()) {
            log("  " + vd[j].m.error + " -> leaving it out");
            if (first_error.empty()) first_error = vd[j].m.error;
            continue;
        }
        log(vd[j].m.msg);
        S.p->views.push_back(std::move(vd[j]));
    }
    if (S.size() < 1) fail(first_error);
    return S.finish();
}

// ---------------------------------------------------------------- 3D, unrolling, rendering

cv::Mat Session::Impl::solve(const std::vector<int>& vidx) const {
    if (vidx.size() == 1) return solve_single();
    const int V = (int)vidx.size();
    const double W = opt.page_w, H = opt.page_h;
    const double grid = opt.grid > 0 ? opt.grid : (V >= 3 ? 2.5 : 4.0);
    const double smooth = opt.smooth > 0 ? opt.smooth : (V >= 3 ? 10.0 : 30.0);
    const bool fixed_focal = opt.fixed_focal || V == 2;
    Cam cam = this->cam;
    std::vector<cv::Mat> gs(V), masks(V);
    std::vector<const ViewMatch*> vmp(V, nullptr);
    for (int k = 0; k < V; k++) {
        gs[k] = views[vidx[k]].gray;
        masks[k] = views[vidx[k]].mask;
        if (k > 0) vmp[k] = &views[vidx[k]].m;
    }
    auto vm = [&](int j) -> const ViewMatch& { return *vmp[j]; };
    log(fmt("%d photos: 3D ...", V));

    std::vector<Pose> poses(V);
    Obs obs(P0.size(), V);
    for (size_t i = 0; i < P0.size(); i++) obs.at(i, 0) = P0[i];
    for (int j = 1; j < V; j++) {
        for (size_t k = 0; k < vm(j).i0.size(); k++)
            if (vm(j).ok[k]) obs.at(vm(j).i0[k], j) = vm(j).pr[k];
        poses[j] = vm(j).pose;
    }
    {
        std::vector<size_t> keep;
        for (size_t i = 0; i < obs.n(); i++)
            for (int j = 1; j < V; j++)
                if (obs.has(i, j)) {
                    keep.push_back(i);
                    break;
                }
        obs = obs.rows(keep);
    }
    // put all views on the scale of view 1: depth ratios at nearby reference pixels
    {
        std::vector<size_t> base;
        for (size_t i = 0; i < obs.n(); i++)
            if (obs.has(i, 1)) base.push_back(i);
        Obs b = pick_views(obs, base, 0, 1);
        Pts3 X1 = triangulate(b, {poses[0], poses[1]}, cam);
        KdTree2 tree1(column(b, 0));
        for (int j = 2; j < V; j++) {
            std::vector<size_t> mj;
            for (size_t i = 0; i < obs.n(); i++)
                if (obs.has(i, j)) mj.push_back(i);
            Obs o = pick_views(obs, mj, 0, j);
            Pts3 Xj = triangulate(o, {poses[0], poses[j]}, cam);
            std::vector<double> ratio;
            for (size_t k = 0; k < mj.size(); k++) {
                auto [nn, d] = tree1.nearest(o.at(k, 0));
                if (d < 15) ratio.push_back(X1[nn].z() / Xj[k].z());
            }
            if (ratio.size() < 20) {
                log(fmt("  photo %d overlaps too little with photo 2", vidx[j] + 1));
                throw BadViews{{j}, fmt("photo %d overlaps too little with photo 2", vidx[j] + 1)};
            }
            poses[j].t *= median(ratio);
        }
    }
    Pts3 X = triangulate(obs, poses, cam);
    clean_obs(X, obs, poses, cam, 3.0);

    // ---------------------------------------------------------------- bundle adjustment
    log("bundle adjustment ...");
    std::vector<double> err;
    for (int it = 0; it < 2; it++) {
        std::vector<size_t> sel(X.size());
        for (size_t i = 0; i < sel.size(); i++) sel[i] = i;
        std::mt19937 rng(it);
        size_t k = std::min<size_t>(opt.ba_points, sel.size());
        for (size_t i = 0; i < k; i++) std::swap(sel[i], sel[i + rng() % (sel.size() - i)]);
        sel.resize(k);
        bundle_adjust(cam, poses, obs.rows(sel), !fixed_focal);
        X = triangulate(obs, poses, cam);
        clean_obs(X, obs, poses, cam, it ? 1.0 : 2.0, &err);
    }
    if (X.size() < 50)
        fail(fmt("only %zu reliable 3D points; the photos do not overlap or differ too little in viewpoint", X.size()));
    {
        double se = 0;
        int ne = 0;
        for (double e : err)
            if (std::isfinite(e)) se += e * e, ne++;
        log(fmt("  points %zu, reprojection rms %.3f px, focal %.1f px", X.size(), std::sqrt(se / ne), cam.f / s));
    }

    Vec3 c, n;
    fit_plane(X, c, n);
    Frame F = page_frame(cam, quad, c, n, W, H);
    Pts3 uvz = to_page(F, X);
    auto bnd = grid_bounds(uv_of(uvz), W, H);

    // page outline in all photos (image processing only), alongside the densification
    Pts2 out0;
    std::vector<Pts2> outs(V);
    Thread outline_thread([&] {
        Pts2 o;
        std::vector<char> ok;
        page_outline(gs[0], masks[0], o, ok);
        for (size_t i = 0; i < o.size(); i++)
            if (ok[i]) out0.push_back(o[i]);
        for (int j = 1; j < V; j++)
            if (!masks[j].empty()) page_outline(gs[j], masks[j], outs[j], ok);
    });

    // ---------------------------------------------------------------- densification
    log("densifying ...");
    {
        Surface S0 = fit_heightfield(uvz, bnd, 2 * grid, smooth);
        int mind = std::max(4, (int)std::nearbyint(std::min(gs[0].rows, gs[0].cols) / 400.0));
        Obs dobs;
        densify(gs, mask_er, column(obs, 0), X, cam, poses, opt.dense, mind, 6, dobs);
        size_t cand = dobs.n();
        Pts3 Xd = triangulate(dobs, poses, cam);
        std::vector<double> derr;
        clean_obs(Xd, dobs, poses, cam, 1.0, &derr);
        std::vector<size_t> ok;
        for (size_t i = 0; i < Xd.size(); i++) {
            Vec3 u = F.to_page(Xd[i]);
            if (std::abs(u.z() - S0.g.sample(S0.Z, u.x(), u.y())) < 1.0) ok.push_back(i);
        }
        log(fmt("  candidates %zu, kept %zu", cand, ok.size()));
        Pts3 Xa = concat(X, select(Xd, ok));
        obs.append(dobs.rows(ok));
        for (size_t i : ok) err.insert(err.end(), derr.begin() + i * V, derr.begin() + (i + 1) * V);
        // one point per 2 px cell of the reference view
        std::map<std::pair<double, double>, size_t> first;
        for (size_t i = 0; i < obs.n(); i++) {
            auto key = std::make_pair(std::nearbyint(obs.at(i, 0).x() / 2.0), std::nearbyint(obs.at(i, 0).y() / 2.0));
            first.emplace(key, i);
        }
        std::vector<size_t> dup;
        for (auto& [k, i] : first) dup.push_back(i);
        X = select(Xa, dup);
        obs = obs.rows(dup);
        uvz = to_page(F, X);
        log(fmt("  total points %zu", X.size()));
    }

    // ---------------------------------------------------------------- page outline
    outline_thread.join();
    std::vector<int> side;
    for (int e : assign_edges(out0, quad)) side.push_back(e >= 0 ? F.edge_map[e] : -1);
    Pts3 Xb;
    std::vector<int> bside;
    std::vector<size_t> bidx;
    {
        log("outline ...");
        Surface S0 = fit_heightfield(uvz, bnd, 2 * grid, smooth);
        Pts3 Xp;
        std::vector<char> hitp;
        ray_surface(out0, cam, F, S0, c, n, Xp, hitp);
        Obs bobs;
        triangulate_outline(out0, outs, cam, poses, Xp, bobs);
        for (size_t i = 0; i < bobs.n(); i++)
            for (int j = 1; j < V; j++)
                if (bobs.has(i, j)) {
                    bidx.push_back(i);
                    break;
                }
        bobs = bobs.rows(bidx);
        for (size_t i : bidx) bside.push_back(side[i]);
        Xb = triangulate(bobs, poses, cam);
        std::vector<double> berr;
        std::vector<char> bdok;
        auto rowmax = [&](size_t i) {
            double m = -1;
            for (int v = 0; v < V; v++)
                if (std::isfinite(berr[i * V + v])) m = std::max(m, berr[i * V + v]);
            return m;
        };
        // drop the worst inconsistent observation per point until the rest agree
        for (int pass = 0; pass + 1 < V && !Xb.empty(); pass++) {
            reproj_errors(Xb, bobs, poses, cam, berr, bdok);
            bool any = false;
            for (size_t i = 0; i < bobs.n(); i++) {
                if (rowmax(i) < 1.5) continue;
                int worst = 1;
                for (int v = 2; v < V; v++) {
                    double ew = std::isfinite(berr[i * V + worst]) ? berr[i * V + worst] : -1;
                    double ev = std::isfinite(berr[i * V + v]) ? berr[i * V + v] : -1;
                    if (ev > ew) worst = v;
                }
                bobs.at(i, worst) = Vec2(NaN, NaN);
                any = true;
            }
            if (!any) break;
            std::vector<size_t> still;
            for (size_t i = 0; i < bobs.n(); i++)
                for (int j = 1; j < V; j++)
                    if (bobs.has(i, j)) {
                        still.push_back(i);
                        break;
                    }
            bobs = bobs.rows(still);
            bside = select(bside, still);
            bidx = select(bidx, still);
            Xb = triangulate(bobs, poses, cam);
        }
        if (!Xb.empty()) {
            reproj_errors(Xb, bobs, poses, cam, berr, bdok);
            std::vector<size_t> okb;
            for (size_t i = 0; i < Xb.size(); i++) {
                Vec3 u = F.to_page(Xb[i]);
                double dz = u.z() - S0.g.sample(S0.Z, u.x(), u.y());
                if (bdok[i] && rowmax(i) < 1.5 && std::abs(dz) < 8.0 && bside[i] >= 0) okb.push_back(i);
            }
            Xb = select(Xb, okb);
            bside = select(bside, okb);
            bidx = select(bidx, okb);
        }
        int cnt[4] = {0, 0, 0, 0};
        for (int sd : bside) cnt[sd]++;
        log(fmt("  outline points triangulated: top %d, right %d, bottom %d, left %d (of %zu)", cnt[0], cnt[1],
                cnt[2], cnt[3], out0.size()));
    }
    // from here on only the reference photo is needed
    gs.clear();
    masks.clear();

    auto all_data = [&]() { return Xb.empty() ? uvz : concat(uvz, to_page(F, Xb)); };
    if (!Xb.empty()) bnd = grid_bounds(uv_of(all_data()), W, H);

    // Folds are parallel creases: if the surface bends in one direction only, turn the frame
    // so u runs along the folds and keep the paper straight along them.
    double ruling_w = 1.0;
    if (opt.ruling > 1) {
        Surface S0 = fit_heightfield(all_data(), bnd, 2 * grid, smooth);
        auto [ang, aniso] = fold_direction(S0);
        log(fmt("  fold direction %.1f deg, anisotropy %.2f", ang * 180 / M_PI, aniso));
        if (aniso > 0.6) {
            double ca = std::cos(ang), sa = std::sin(ang);
            Vec3 eu = ca * F.eu + sa * F.ev, ev = -sa * F.eu + ca * F.ev;
            F.eu = eu, F.ev = ev;
            uvz = to_page(F, X);
            bnd = grid_bounds(uv_of(all_data()), W, H);
            ruling_w = opt.ruling;
        }
    }
    // physical page corners: image quad corner k joins image edges k-1 and k
    std::vector<size_t> cidx(4);
    Pts2 ctarget(4);
    for (int k = 0; k < 4; k++) {
        size_t b = 0;
        for (size_t i = 1; i < out0.size(); i++)
            if ((out0[i] - quad[k]).norm() < (out0[b] - quad[k]).norm()) b = i;
        cidx[k] = b;
        int sa = F.edge_map[(k + 3) % 4], sb = F.edge_map[k];
        bool top = sa == 0 || sb == 0, left = sa == 3 || sb == 3;
        ctarget[k] = Vec2(left ? 0.0 : W, top ? 0.0 : H);
    }

    // ---------------------------------------------------------------- height field + unrolling
    Surface S;
    Pts2 st, sto;
    std::vector<char> hit;
    std::vector<char> use;
    Eigen::Vector4d rect;
    std::vector<double> rres;
    for (int it = 0; it < 2; it++) {
        log(fmt("height field + flattening (pass %d), grid u %.0f..%.0f v %.0f..%.0f mm ...", it + 1, bnd[0], bnd[1],
                bnd[2], bnd[3]));
        auto tp = Clock::now();
        S = fit_heightfield(all_data(), bnd, it == 0 ? 2 * grid : grid, smooth, ruling_w);  // pass 1 only sets the scale
        double t_h = secs(tp);
        double smed, smax;
        st = flatten_grid(S, smed, smax);
        double t_f = secs(tp);
        Pts3 Xo;
        ray_surface(out0, cam, F, S, c, n, Xo, hit);
        log(fmt("    grid %dx%d: height field %.2f s, flattening %.2f s, rays %.2f s", S.g.nu, S.g.nv, t_h, t_f,
                secs(tp)));
        for (size_t k = 0; k < bidx.size(); k++) {
            Xo[bidx[k]] = Xb[k];  // where the outline was triangulated, its 3D is measured
            hit[bidx[k]] = 1;
        }
        sto.resize(out0.size());
        for (size_t i = 0; i < out0.size(); i++) {
            Vec3 u = F.to_page(Xo[i]);
            sto[i] = sample_st(st, S.g, u.x(), u.y());
        }
        use.assign(out0.size(), 0);
        Pts2 su;
        std::vector<int> sd;
        for (size_t i = 0; i < out0.size(); i++)
            if ((use[i] = hit[i] && side[i] >= 0)) su.push_back(sto[i]), sd.push_back(side[i]);
        Eigen::Vector4d x0 = it == 0 ? corners_init(select(sto, cidx), ctarget) : rect;
        rect = fit_rectangle(su, sd, W, H, x0, rres);
        double rms = 0;
        for (double r : rres) rms += r * r;
        log(fmt("  outline fit: scale %.4f, rms %.3f mm (%zu outline points)", rect[0], std::sqrt(rms / rres.size()),
                su.size()));
        if (it == 0) {
            F.scale *= rect[0];
            uvz = to_page(F, X);
            rect[0] = 1.0;
        }
    }
    for (auto& p : st) p = rect_apply(rect, p);
    for (auto& p : sto) p = rect_apply(rect, p);

    Flat fl{std::move(S), F, cam, std::move(st), std::move(sto), std::move(out0), ctarget, std::move(hit),
            std::move(use), std::move(side), cidx, std::move(uvz), std::move(rres)};
    return correct_render(fl);
}

cv::Mat Session::Impl::correct_render(Flat& fl) const {
    const double W = opt.page_w, H = opt.page_h;
    const Surface& S = fl.S;
    const Frame& F = fl.F;
    const Cam& cam = fl.cam;
    Pts2 &st = fl.st, &sto = fl.sto;
    const Pts2 &out0 = fl.out0, &ctarget = fl.ctarget;
    const std::vector<char>& hit = fl.hit;
    const std::vector<char>& use = fl.use;
    const std::vector<int>& side = fl.side;
    const std::vector<size_t>& cidx = fl.cidx;
    const Pts3& uvz = fl.uvz;
    const std::vector<double>& rres = fl.rres;
    const char* tl_env = std::getenv("DOCSCAN_TEXTLINES");  // testing: 0 turns text lines off
    const bool text_lines = !tl_env || std::string(tl_env) != "0";

    // ---------------------------------------------------------------- edge and line correction
    const Cam camf{cam.f / s, (cam.cx + 0.5) / s - 0.5, (cam.cy + 0.5) / s - 0.5};
    std::vector<EdgeTerm> edges;
    {
        // Outline points must agree with their neighbours along the edge: a smooth offset of a
        // whole edge is real and gets corrected, isolated jumps are detection errors. Points at
        // the ends of an edge that the whole-edge fit rejects are restored if they agree with
        // their local neighbours (extrapolated page ends can turn fast).
        struct Fit { int ax; std::vector<double> a, r; };
        std::map<int, Fit> edge_fit;
        const int sds[4] = {3, 1, 0, 2}, axs[4] = {0, 0, 1, 1};
        const double tgs[4] = {0.0, W, 0.0, H};
        const char* names[4] = {"top", "right", "bottom", "left"};
        for (int e = 0; e < 4; e++) {
            int sd = sds[e], ax = axs[e];
            double tg = tgs[e];
            std::vector<size_t> m;
            for (size_t i = 0; i < out0.size(); i++)
                if (use[i] && side[i] == sd) m.push_back(i);
            if (m.size() < 10) continue;
            size_t nm = m.size();
            std::vector<double> along(nm), r(nm);
            for (size_t k = 0; k < nm; k++) {
                along[k] = sto[m[k]][1 - ax];
                r[k] = tg - sto[m[k]][ax];
            }
            auto refit = [&](std::vector<char>& ok, bool local) {
                for (int pass = 0; pass < 3; pass++) {
                    std::vector<double> pred;
                    if (local) pred = local_line(along, r, ok, along);
                    else {
                        std::vector<double> xa, ra;
                        for (size_t k = 0; k < nm; k++)
                            if (ok[k]) xa.push_back(along[k]), ra.push_back(r[k]);
                        auto cf = polyfit(xa, ra, 2);
                        for (double a : along) pred.push_back(polyval(cf, a));
                    }
                    std::vector<double> dev(nm), adev;
                    for (size_t k = 0; k < nm; k++) {
                        dev[k] = r[k] - pred[k];
                        if (ok[k]) adev.push_back(std::abs(dev[k]));
                    }
                    double mad = 1.4826 * median(adev) + 1e-6;
                    for (size_t k = 0; k < nm; k++)
                        ok[k] = std::abs(r[k]) < 25.0 && std::abs(dev[k]) < std::max(1.0, 4 * mad);
                }
            };
            std::vector<char> ok(nm), ok_l(nm);
            for (size_t k = 0; k < nm; k++) ok[k] = ok_l[k] = std::abs(r[k]) < 25.0;
            refit(ok, false);
            refit(ok_l, true);
            std::vector<size_t> o(nm);
            for (size_t k = 0; k < nm; k++) o[k] = k;
            std::sort(o.begin(), o.end(), [&](size_t a, size_t b) { return along[a] < along[b]; });
            long first = -1, last = -1;
            for (size_t k = 0; k < nm; k++)
                if (ok[o[k]]) {
                    if (first < 0) first = (long)k;
                    last = (long)k;
                }
            if (first >= 0)
                for (size_t k = 0; k < nm; k++)
                    if (((long)k < first || (long)k > last) && ok_l[o[k]]) ok[o[k]] = 1;
            Fit fit{ax, {}, {}};
            EdgeTerm et{{}, ax, {}};
            std::vector<double> rv = r;
            for (size_t k = 0; k < nm; k++)
                if (ok[k]) {
                    fit.a.push_back(along[k]);
                    fit.r.push_back(r[k]);
                    et.P.push_back(sto[m[k]]);
                    et.target.push_back(tg);
                }
            log(fmt("    edge %s: offset median %+.2f mm, range %+.2f..%+.2f, kept %zu/%zu", names[sd], median(rv),
                    percentile(rv, 5), percentile(rv, 95), et.P.size(), nm));
            edge_fit[sd] = fit;
            edges.push_back(et);
        }
        // corners: keep only if consistent with the trend of the adjacent edge near them
        for (int ax = 0; ax < 2; ax++) {
            EdgeTerm et{{}, ax, {}};
            for (int k = 0; k < 4; k++) {
                bool keep = hit[cidx[k]];
                const Vec2& Pc = sto[cidx[k]];
                for (auto& [sd, fit] : edge_fit) {
                    double edge_tg = (sd == 3 || sd == 0) ? 0.0 : (sd == 1 ? W : H);
                    if (fit.ax == ax && std::abs(ctarget[k][ax] - edge_tg) < 1e-6) {
                        double pred = local_line(fit.a, fit.r, std::vector<char>(fit.a.size(), 1), {Pc[1 - ax]}, 30.0)[0];
                        if (std::abs((ctarget[k][ax] - Pc[ax]) - pred) > 3.0) keep = false;
                    }
                }
                if (keep) {
                    et.P.push_back(Pc);
                    et.target.push_back(ctarget[k][ax]);
                }
            }
            edges.push_back(et);
        }
    }
    Pts2 anchors(uvz.size());
    for (size_t i = 0; i < uvz.size(); i++) anchors[i] = sample_st(st, S.g, uvz[i].x(), uvz[i].y());
    if (anchors.size() > 4000) {
        std::mt19937 rng(0);
        std::shuffle(anchors.begin(), anchors.end(), rng);
        anchors.resize(4000);
    }
    // 1) edges: measured content holds firmly, so the correction stays in the margins
    {
        Disp de = fit_correction(W, H, edges, {}, {}, anchors, 1.0);
        for (auto& p : st) p += de(p);
        for (auto& p : sto) p += de(p);
        std::vector<double> dd;
        for (auto& p : anchors) {
            p += de(p);
            dd.push_back(de(p).norm());
        }
        if (!dd.empty()) log(fmt("  edge correction: shift at measured content median %.3f mm", median(dd)));
    }
    // 2) lines: printed lines may override the measurement (weak hold); the corrected
    //    outline stays where it is
    {
        std::vector<EdgeTerm> hold;
        for (int ax = 0; ax < 2; ax++) {
            EdgeTerm et{{}, ax, {}};
            for (size_t i = 0; i < out0.size(); i++)
                if (use[i]) et.P.push_back(sto[i]), et.target.push_back(sto[i][ax]);
            hold.push_back(et);
        }
        const Pts2 st1 = st, sto1 = sto;
        Disp dl;
        bool have = false;
        for (int it = 0; it < 2; it++) {
            cv::Mat ur, vr;
            flat_to_uv(st, S.g, W, H, 0.5, ur, vr);
            Render R = render_flat(ref_full, camf, F, S, ur, vr, 0.5, W, H, 150);
            std::vector<Eigen::Vector4d> segs;
            std::vector<char> hz;
            cv::Mat g8 = gray_u8(R.img, R.page);
            detect_lines(g8, W, H, segs, hz);
            std::vector<Pts2> text, text_v;
            if (text_lines) {
                text = detect_text_lines(g8, W, H);
                text_v = detect_text_lines(g8, W, H, true);
            }
            if (have) {
                for (auto& sg : segs) {
                    Vec2 a = sg.head<2>(), b = sg.tail<2>();
                    sg << a - dl(a), b - dl(b);
                }
                for (auto* tv : {&text, &text_v})
                    for (auto& tl : *tv)
                        for (auto& P : tl) P -= dl(P);
            }
            dl = fit_correction(W, H, hold, segs, hz, anchors, opt.anchor_w, text, text_v);
            if (text_lines) log(fmt("    text lines: %zu horizontal, %zu vertical", text.size(), text_v.size()));
            have = true;
            std::vector<double> dd;
            for (size_t i = 0; i < st.size(); i++) {
                Vec2 d = dl(st1[i]);
                st[i] = st1[i] + d;
                dd.push_back(d.norm());
            }
            for (size_t i = 0; i < sto.size(); i++) sto[i] = sto1[i] + dl(sto1[i]);
            log(fmt("  line correction %d: %zu line segments, shift median %.3f mm, max %.2f mm", it + 1, segs.size(),
                    median(dd), *std::max_element(dd.begin(), dd.end())));
        }
    }
    {
        const double tg[4] = {0.0, W, H, 0.0};
        const int ax[4] = {1, 0, 1, 0};
        double a = 0, b = 0;
        int nn = 0;
        for (double r : rres) a += r * r;
        for (size_t i = 0; i < out0.size(); i++)
            if (use[i]) {
                double r = sto[i][ax[side[i]]] - tg[side[i]];
                b += r * r;
                nn++;
            }
        log(fmt("  outline on rectangle: rms %.3f -> %.3f mm", std::sqrt(a / rres.size()), std::sqrt(b / nn)));
    }

    // ---------------------------------------------------------------- render
    log("rendering ...");
    auto t = Clock::now();
    cv::Mat ur, vr;
    flat_to_uv(st, S.g, W, H, 0.5, ur, vr);
    Render R = render_flat(ref_full, camf, F, S, ur, vr, 0.5, W, H, opt.dpi);
    double t_render = secs(t);
    cv::Mat valid = R.valid & R.page;
    cv::Mat alb = remove_shading(R.img, valid, R.nu, R.nv);
    double t_shading = secs(t);
    cv::Mat out = clean_paper(to_uint8(alb), opt.gpu);
    double t_paper = secs(t);
    if (opt.rotate) {
        std::string info;
        int rot = text_rotation(out, opt.dpi / 25.4, &info);
        log(fmt("  orientation: %s -> rotate %d deg (%.2f s)", info.c_str(), rot, secs(t)));
        if (rot) cv::rotate(out, out, rot == 90 ? cv::ROTATE_90_CLOCKWISE : rot == 180 ? cv::ROTATE_180
                                                                                   : cv::ROTATE_90_COUNTERCLOCKWISE);
    }
    log(fmt("  warp %.2f s, shading %.2f s, paper %.2f s", t_render, t_shading, t_paper));
    return out;
}

// One photo: no 3D measurement. The page is taken as the plane through its four corners (known
// focal), the curved outline and printed lines then straighten what the folds bend.
cv::Mat Session::Impl::solve_single() const {
    const double W = opt.page_w, H = opt.page_h;
    const double step = opt.grid > 0 ? opt.grid : 4.0;
    log("1 photo: page plane from its corners ...");
    Flat fl;
    fl.cam = cam;
    double horiz = ((quad[1] - quad[0]).norm() + (quad[2] - quad[3]).norm()) / 2;
    double vert = ((quad[3] - quad[0]).norm() + (quad[2] - quad[1]).norm()) / 2;
    double pw = horiz <= vert ? W : H, ph = horiz <= vert ? H : W;
    std::vector<cv::Point2f> src{{0, 0}, {(float)pw, 0}, {(float)pw, (float)ph}, {0, (float)ph}}, dst;
    for (auto& q : quad) dst.emplace_back((float)q.x(), (float)q.y());
    cv::Mat Hc = cv::getPerspectiveTransform(src, dst);
    Eigen::Matrix3d Hm, K;
    for (int r = 0; r < 3; r++)
        for (int k = 0; k < 3; k++) Hm(r, k) = Hc.at<double>(r, k);
    K << cam.f, 0, cam.cx, 0, cam.f, cam.cy, 0, 0, 1;
    Eigen::Matrix3d M = K.inverse() * Hm;
    Vec3 h1 = M.col(0), h2 = M.col(1);
    Vec3 n = h1.cross(h2).normalized();
    Vec3 c = M.col(2) * (2 / (h1.norm() + h2.norm()));
    if (c.z() < 0) c = -c;
    log(fmt("  page tilt %.1f deg", std::acos(std::min(1.0, std::abs(n.z()))) * 180 / M_PI));
    fl.F = page_frame(cam, quad, c, n, W, H);
    const Frame& F = fl.F;

    {
        Pts2 o;
        std::vector<char> ok;
        page_outline(views[0].gray, views[0].mask, o, ok);
        for (size_t i = 0; i < o.size(); i++)
            if (ok[i]) fl.out0.push_back(o[i]);
    }
    const Pts2& out0 = fl.out0;
    if (out0.size() < 50) fail("could not trace the page outline");
    for (int e : assign_edges(out0, quad)) fl.side.push_back(e >= 0 ? F.edge_map[e] : -1);

    // flat surface over the page and its outline
    Pts2 uv;
    for (auto& p : out0) {
        Vec3 r = Vec3((p.x() - cam.cx) / cam.f, (p.y() - cam.cy) / cam.f, 1.0);
        Vec3 u = F.to_page(r * (c.dot(n) / r.dot(n)));
        uv.push_back(u.head<2>());
    }
    auto bnd = grid_bounds(uv, W, H);
    Surface& S = fl.S;
    S.g.u0 = bnd[0], S.g.v0 = bnd[2], S.g.step = step;
    S.g.nu = (int)std::ceil((bnd[1] + step - bnd[0]) / step);
    S.g.nv = (int)std::ceil((bnd[3] + step - bnd[2]) / step);
    S.Z.assign((size_t)S.g.nu * S.g.nv, 0.0);
    fl.st.resize(S.Z.size());
    for (int j = 0; j < S.g.nv; j++)
        for (int i = 0; i < S.g.nu; i++) fl.st[(size_t)j * S.g.nu + i] = Vec2(S.g.gu(i), S.g.gv(j));

    Pts3 Xo;
    ray_surface(out0, cam, F, S, c, n, Xo, fl.hit);
    fl.sto.resize(out0.size());
    for (size_t i = 0; i < out0.size(); i++) {
        Vec3 u = F.to_page(Xo[i]);
        fl.sto[i] = sample_st(fl.st, S.g, u.x(), u.y());
    }
    fl.cidx.resize(4);
    fl.ctarget.resize(4);
    for (int k = 0; k < 4; k++) {
        size_t b = 0;
        for (size_t i = 1; i < out0.size(); i++)
            if ((out0[i] - quad[k]).norm() < (out0[b] - quad[k]).norm()) b = i;
        fl.cidx[k] = b;
        int sa = F.edge_map[(k + 3) % 4], sb = F.edge_map[k];
        bool top = sa == 0 || sb == 0, left = sa == 3 || sb == 3;
        fl.ctarget[k] = Vec2(left ? 0.0 : W, top ? 0.0 : H);
    }
    fl.use.assign(out0.size(), 0);
    Pts2 su;
    std::vector<int> sd;
    for (size_t i = 0; i < out0.size(); i++)
        if ((fl.use[i] = fl.hit[i] && fl.side[i] >= 0)) su.push_back(fl.sto[i]), sd.push_back(fl.side[i]);
    Eigen::Vector4d rect = fit_rectangle(su, sd, W, H, corners_init(select(fl.sto, fl.cidx), fl.ctarget), fl.rres);
    double rms = 0;
    for (double r : fl.rres) rms += r * r;
    log(fmt("  outline fit: scale %.4f, rms %.3f mm (%zu outline points)", rect[0], std::sqrt(rms / fl.rres.size()),
            su.size()));
    for (auto& p : fl.st) p = rect_apply(rect, p);
    for (auto& p : fl.sto) p = rect_apply(rect, p);
    return correct_render(fl);
}

}  // namespace docscan
