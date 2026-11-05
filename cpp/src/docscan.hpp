// docscan: photos of a folded/curved page -> measured 3D surface -> flat, scanner-like page.
// Minimal C++ port of docscan3d.py (no ASIFT fallback, no view fusion, no lens distortion).
#pragma once

#include <opencv2/core.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace docscan {

struct Options {
    double page_w = 210, page_h = 297;  // mm (A4)
    double focal_px = 0;                // focal length in pixels at full resolution (0: 26 mm equiv.)
    bool fixed_focal = false;           // always on with 2 views (focal not observable)
    int dpi = 200;                      // output resolution
    int max_side = 4100;                // photos are downscaled to this for the 3D part
    int features = 60000;               // SIFT keypoints per view
    int ba_points = 5000;               // points used in bundle adjustment
    int dense = 40000;                  // max candidates for densification
    double grid = 0, smooth = 0;        // height field spacing (mm) / smoothness; 0 = by view count
    double ruling = 20;                 // stiffness along the folds when the page bends one way
    double anchor_w = 0.1;              // how strongly measured content resists line straightening
    bool gpu = true;                    // SIFT on the GPU (Vulkan) if built in and available
    bool rotate = true;                 // turn the output so the text is upright
    bool doc_tone = false;              // ink to black, light grey to white (cleaner; fades light grey text)
};

// images: reference view first (rendered), then 1-7 more views, BGR, same camera and size.
// Returns the flattened page (CV_8UC1, or CV_8UC3 if the page has colour). Throws
// std::runtime_error with a user-facing message if the input cannot be processed.
cv::Mat process(const std::vector<cv::Mat>& images, const Options& opt = Options());

// Incremental use (the app): add photos as they are taken, the heavy per-photo work (page outline,
// features, matching with photo 1) runs inside add(). The first photo is the reference.
class Session {
public:
    Session(int width, int height, const Options& opt = Options());
    ~Session();
    // Returns "" if the photo is usable, else a message for the user (the photo is then not kept).
    std::string add(const cv::Mat& image);
    void remove_last();
    int size() const;
    // 3D surface, unrolling, rendering. Photos that turn out unusable here are left out.
    // Throws std::runtime_error.
    cv::Mat finish();

    struct Impl;
    std::unique_ptr<Impl> p;
};

// Progress messages go to stderr unless a sink is set (may be called from worker threads).
void set_log_sink(std::function<void(const std::string&)> sink);

}  // namespace docscan
