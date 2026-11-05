// Pipeline stages (see docscan3d.py for the reference implementation of each).
#pragma once

#include <array>

#include <opencv2/flann.hpp>

#include "common.hpp"

namespace docscan {

// ------------------------------------------------------------ page and matching (matching.cpp)
// img: BGR or grey; work_side: long side of the internal working image (1000 in the pipeline)
bool find_page_quad(const cv::Mat& img, std::array<Vec2, 4>& quad, cv::Mat& mask, int work_side = 1000);
std::array<Vec2, 4> refine_quad(const cv::Mat& gray, const std::array<Vec2, 4>& quad, int half = 25, int n = 40);
void page_outline(const cv::Mat& gray, const cv::Mat& mask, Pts2& out, std::vector<char>& ok, double step = 4.0,
                  int half = 18);
// gpu: use GpuSift if built in and available (else OpenCV on the CPU)
void sift_detect(const cv::Mat& g, const cv::Mat& mask, int n_features, Pts2& pts, cv::Mat& desc, bool gpu = false);
// FLANN index (4 random kd-trees) of descriptors; keeps a reference to d
cv::Ptr<cv::flann::Index> flann_index(const cv::Mat& d);
void match_ref(const cv::Mat& d0, cv::flann::Index* index0, const Pts2& pts, const cv::Mat& d, int seed,
               std::vector<int>& i0, Pts2& pj);
cv::Mat essential(const Pts2& q0, const Pts2& qj, const Cam& cam, double threshold, std::vector<char>& inl);
void recover_pose(const cv::Mat& E, const Pts2& q0, const Pts2& qj, const Cam& cam, Pose& P);
double page_coverage(const Pts2& pts, const cv::Mat& mask, int n = 8);
double parallax_deg(const Pts2& q0, const Pts2& qj, const Mat3& R, const Cam& cam);
void local_affines(const Pts2& p1, const Pts2& p2, const Pts2& q, std::vector<Eigen::Matrix2d>& A, Pts2& pred,
                   int k = 12);
// g1, g2: CV_8U grey images
void refine_subpixel(const cv::Mat& g1, const cv::Mat& g2, const Pts2& p1, const Pts2& p2,
                     const std::vector<Eigen::Matrix2d>& A, Pts2& out, std::vector<char>& ok, int win = 15,
                     int search = 4);
void densify(const std::vector<cv::Mat>& gs, const cv::Mat& mask, const Pts2& p_ref,
             const Pts3& X, const Cam& cam, const std::vector<Pose>& poses, int max_pts, int min_dist, int search,
             Obs& dobs);

// ------------------------------------------------------------ surface (surface.cpp)
struct Surface {
    Grid g;
    std::vector<double> Z;
};

void fit_plane(const Pts3& X, Vec3& c, Vec3& n);
Frame page_frame(const Cam& cam, const std::array<Vec2, 4>& quad, const Vec3& c, Vec3 n, double W, double H);
std::array<double, 4> grid_bounds(const Pts2& uv, double W, double H, double margin = 15.0);
Surface fit_heightfield(const Pts3& uvz, const std::array<double, 4>& bounds, double step, double lam,
                        double ruling_w = 1.0, std::vector<double>* resid = nullptr);
void ray_surface(const Pts2& px, const Cam& cam, const Frame& F, const Surface& S, const Vec3& c, const Vec3& n,
                 Pts3& X, std::vector<char>& ok);
void grid_gradient(const Grid& g, const std::vector<double>& Z, std::vector<double>& zu, std::vector<double>& zv);
std::pair<double, double> fold_direction(const Surface& S);
Pts2 flatten_grid(const Surface& S, double& strain_med, double& strain_max);
Vec2 sample_st(const Pts2& st, const Grid& g, double u, double v);
void flat_to_uv(const Pts2& st, const Grid& g, double W, double H, double res, cv::Mat& ur, cv::Mat& vr);
std::vector<int> assign_edges(const Pts2& pts, const std::array<Vec2, 4>& quad, double corner_excl = 0.06);
void triangulate_outline(const Pts2& out0, const std::vector<Pts2>& outs, const Cam& cam,
                         const std::vector<Pose>& poses, const Pts3& X_pred, Obs& obs);

Vec2 rect_apply(const Eigen::Vector4d& x, const Vec2& p);
Eigen::Vector4d fit_rectangle(const Pts2& st, const std::vector<int>& side, double W, double H,
                              const Eigen::Vector4d& x0, std::vector<double>& resid);
Eigen::Vector4d corners_init(const Pts2& st_c, const Pts2& dst);
std::vector<double> local_line(const std::vector<double>& a, const std::vector<double>& r,
                               const std::vector<char>& ok, const std::vector<double>& q, double half = 10.0,
                               int k_min = 8);
std::vector<double> polyfit(const std::vector<double>& x, const std::vector<double>& y, int deg);
double polyval(const std::vector<double>& c, double x);
void detect_lines(const cv::Mat& g8, double W, double H, std::vector<Eigen::Vector4d>& segs,
                  std::vector<char>& horiz);

struct EdgeTerm {
    Pts2 P;
    int axis;
    std::vector<double> target;
};
// smooth displacement field on the flat page (mm)
struct Disp {
    double s0 = 0, t0 = 0, step = 4;
    int ns = 0, nt = 0;
    std::vector<double> field[2];
    Vec2 operator()(const Vec2& q) const;
};
Disp fit_correction(double W, double H, const std::vector<EdgeTerm>& edges, const std::vector<Eigen::Vector4d>& segs,
                    const std::vector<char>& horiz, const Pts2& anchors, double w_anchor,
                    const std::vector<Pts2>& curves = {}, const std::vector<Pts2>& curves_v = {});
// Text lines (and other long horizontal ink structures) as polylines in mm: centre of the ink
// per 3 mm column, for straightening across folds where a single view has no 3D.
// vertical: lines of text that runs top to bottom (landscape page)
std::vector<Pts2> detect_text_lines(const cv::Mat& g8, double W, double H, bool vertical = false);

// ------------------------------------------------------------ rendering (render.cpp)
struct Render {
    cv::Mat img;    // CV_32FC3 reference photo resampled onto the flat page
    cv::Mat valid;  // CV_8U, inside the photo
    cv::Mat page;   // CV_8U, covered by the unrolled surface
    cv::Mat nu, nv; // CV_32F, surface normal (page frame)
};
Render render_flat(const cv::Mat& img, const Cam& cam, const Frame& F, const Surface& S, const cv::Mat& ur,
                   const cv::Mat& vr, double res, double W, double H, int dpi);
cv::Mat gray_u8(const cv::Mat& img32, const cv::Mat& page);
cv::Mat remove_shading(const cv::Mat& img, const cv::Mat& valid, const cv::Mat& nu, const cv::Mat& nv);
cv::Mat to_uint8(const cv::Mat& alb, double white = 0.95);
cv::Mat clean_paper(const cv::Mat& img, bool gpu = false);
// Clockwise rotation (0, 90, 180, 270) that makes the text upright, judged by the direction of
// the text lines and by ascenders (above the lowercase band) outweighing descenders.
// ppm: pixels per mm of img. Returns 0 if there is too little text to tell.
int text_rotation(const cv::Mat& img, double ppm, std::string* info = nullptr);

}  // namespace docscan
