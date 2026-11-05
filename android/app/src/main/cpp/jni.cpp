// JNI bridge: de.spaceio.docscan.Native
#include <android/log.h>
#include <jni.h>

#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <sched.h>
#include <time.h>

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "docscan.hpp"
#include "gpu_sift.hpp"
#include "stages.hpp"

namespace {
std::mutex mu;
std::string last_msg, full_log;
std::chrono::steady_clock::time_point t_start;

std::string jstr(JNIEnv* env, jstring s) {
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string r(c);
    env->ReleaseStringUTFChars(s, c);
    return r;
}
}  // namespace

namespace {
std::unique_ptr<docscan::Session> session;  // only touched from the app's single worker thread

double cpu_seconds() {  // CPU time of all threads of the app
    timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

double wall_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// "wall 5.1 s, cpu 20.3 s (4.0 cores busy)": how much of the phone a step really used
std::string usage(double w0, double c0) {
    double w = wall_seconds() - w0, c = cpu_seconds() - c0;
    return docscan::fmt("wall %.1f s, cpu %.1f s (%.1f cores busy)", w, c, c / std::max(w, 1e-6));
}

void start_log() {
    {
        std::lock_guard<std::mutex> lk(mu);
        last_msg.clear();
        full_log.clear();
        t_start = std::chrono::steady_clock::now();
    }
    docscan::set_log_sink([](const std::string& s) {
        __android_log_print(ANDROID_LOG_INFO, "docscan", "%s", s.c_str());
        std::lock_guard<std::mutex> lk(mu);
        double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        char ts[32];
        std::snprintf(ts, sizeof ts, "[%6.1f s] ", t);
        full_log += ts + s + "\n";
        last_msg = s;
    });
}
}  // namespace

// Analyses one photo (page outline, features, matching with photo 1). The first photo starts a
// new session with the given focal length. Returns null if the photo is usable, else a message.
extern "C" JNIEXPORT jstring JNICALL Java_de_spaceio_docscan_Native_add(JNIEnv* env, jclass, jstring jpath,
                                                                        jdouble focal) {
    auto path = jstr(env, jpath);
    if (!session || session->size() == 0) start_log();
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);  // applies EXIF orientation
    if (img.empty()) return env->NewStringUTF(("cannot read " + path).c_str());
    if (!session || session->size() == 0) {
        docscan::Options opt;
        opt.focal_px = focal;
        // 3D part on downscaled photos: 3200 px on the app's 4:3 shots gives the page about the same
        // pixels/mm as the full-size 16:9 photos the pipeline was tuned on; rendering uses full size
        opt.max_side = 3200;
        session = std::make_unique<docscan::Session>(img.cols, img.rows, opt);
        docscan::log(docscan::fmt("photos %dx%d, focal %.1f px", img.cols, img.rows, focal));
        cpu_set_t cs;
        int aff = sched_getaffinity(0, sizeof cs, &cs) == 0 ? CPU_COUNT(&cs) : -1;
        docscan::log(docscan::fmt("threads: opencv %d, cpus %d, affinity %d", cv::getNumThreads(),
                                  cv::getNumberOfCPUs(), aff));
        auto* G = docscan::GpuSift::get();
        docscan::log(G ? "GPU: " + G->name() : "GPU: not available");
    }
    docscan::log(docscan::fmt("analysing photo %d ...", session->size() + 1));
    double w0 = wall_seconds(), c0 = cpu_seconds();
    std::string err = session->add(img);
    docscan::log("    " + usage(w0, c0));
    return err.empty() ? nullptr : env->NewStringUTF(err.c_str());
}

// Sets up the GPU (shader compilation takes ~1.5 s on a Pixel 8) before the first photo.
extern "C" JNIEXPORT void JNICALL Java_de_spaceio_docscan_Native_warmup(JNIEnv*, jclass) { docscan::GpuSift::get(); }

extern "C" JNIEXPORT void JNICALL Java_de_spaceio_docscan_Native_removeLast(JNIEnv*, jclass) {
    if (session) session->remove_last();
}

extern "C" JNIEXPORT void JNICALL Java_de_spaceio_docscan_Native_reset(JNIEnv*, jclass) { session.reset(); }

// 3D surface, unrolling and rendering from the added photos; writes a PNG.
// Returns null on success, else the error message.
extern "C" JNIEXPORT jstring JNICALL Java_de_spaceio_docscan_Native_finish(JNIEnv* env, jclass, jstring jout) {
    if (!session || session->size() < 1) return env->NewStringUTF("no photo");
    try {
        double w0 = wall_seconds(), c0 = cpu_seconds();
        cv::Mat flat = session->finish();
        docscan::log("  3D + rendering: " + usage(w0, c0));
        auto out = jstr(env, jout);
        if (!cv::imwrite(out, flat, {cv::IMWRITE_PNG_COMPRESSION, 3}))
            return env->NewStringUTF(("cannot write " + out).c_str());
        docscan::log("done");
        return nullptr;
    } catch (const std::exception& e) {
        docscan::log(std::string("error: ") + e.what());
        return env->NewStringUTF(e.what());
    }
}

extern "C" JNIEXPORT jstring JNICALL Java_de_spaceio_docscan_Native_progress(JNIEnv* env, jclass) {
    std::lock_guard<std::mutex> lk(mu);
    return env->NewStringUTF(last_msg.c_str());
}

extern "C" JNIEXPORT jstring JNICALL Java_de_spaceio_docscan_Native_log(JNIEnv* env, jclass) {
    std::lock_guard<std::mutex> lk(mu);
    return env->NewStringUTF(full_log.c_str());
}

// Live page detection on a camera frame (Y plane). Writes the 4 corners as x,y pairs normalised
// to [0,1] in the upright (rotated) frame.
extern "C" JNIEXPORT jboolean JNICALL Java_de_spaceio_docscan_Native_detectPage(JNIEnv* env, jclass, jobject buf,
                                                                               jint w, jint h, jint stride,
                                                                               jint rotation, jfloatArray jout) {
    auto* data = (uint8_t*)env->GetDirectBufferAddress(buf);
    if (!data) return false;
    cv::Mat y(h, w, CV_8U, data, stride), up;
    if (rotation == 90) cv::rotate(y, up, cv::ROTATE_90_CLOCKWISE);
    else if (rotation == 180) cv::rotate(y, up, cv::ROTATE_180);
    else if (rotation == 270) cv::rotate(y, up, cv::ROTATE_90_COUNTERCLOCKWISE);
    else up = y;
    std::array<docscan::Vec2, 4> quad;
    cv::Mat mask;
    if (!docscan::find_page_quad(up, quad, mask, 400)) return false;  // small: runs while photos are analysed
    float out[8];
    for (int k = 0; k < 4; k++) {
        out[2 * k] = (float)(quad[k].x() / up.cols);
        out[2 * k + 1] = (float)(quad[k].y() / up.rows);
    }
    env->SetFloatArrayRegion(jout, 0, 8, out);
    return true;
}
