// SIFT on the GPU (Vulkan compute): the same algorithm and parameters as
// cv::SIFT::create(0, 3, contrast, 10, 1.6), including the upscaled first octave.
#pragma once

#include <opencv2/core.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace docscan {

class GpuSift {
public:
    // Shared instance, or nullptr if there is no usable Vulkan device.
    // DOCSCAN_GPU=0 disables the GPU; DOCSCAN_GPU=cpu picks a software device (testing).
    static GpuSift* get();

    // g: CV_8U image. keep(pt) selects keypoints (image coordinates) before descriptors are
    // computed. Returns false on a GPU error. Calls are serialised.
    bool detect(const cv::Mat& g, float contrast, const std::function<bool(const cv::Point2f&)>& keep,
                std::vector<cv::KeyPoint>& kps, cv::Mat& desc);

    const std::string& name() const;

    // GPU time (s) of the passes since the last call (0 if the device has no timestamps)
    struct Stats {
        double pyramid = 0, extrema = 0, descriptors = 0;
    };
    Stats take_stats();
    ~GpuSift();

    struct Impl;

private:
    GpuSift();
    std::unique_ptr<Impl> p;
};

}  // namespace docscan
