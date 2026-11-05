#include "gpu_sift.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>

#include "common.hpp"

namespace docscan {

namespace {

// SPIR-V, compiled by CMake from shaders/*.comp
const uint32_t kUpscale[] = {
#include "upscale.spv.inc"
};
const uint32_t kBlur[] = {
#include "blur.spv.inc"
};
const uint32_t kDown[] = {
#include "downsample.spv.inc"
};
const uint32_t kExtrema[] = {
#include "extrema.spv.inc"
};
const uint32_t kPrep[] = {
#include "prep_indirect.spv.inc"
};
const uint32_t kOrient[] = {
#include "orientation.spv.inc"
};
const uint32_t kDescr[] = {
#include "descriptor.spv.inc"
};
const uint32_t kNlm[] = {
#include "nlm.spv.inc"
};

constexpr int kLayers = 3;
constexpr double kSigma = 1.6;
constexpr int kMaxOct = 16;
constexpr int kBindings = 10;
// IO buffer layout in 32-bit words, as in shaders/common.glsl
constexpr int kExtCap = 65535;  // per octave; also the max workgroup count of one dispatch
constexpr int kKpCap = 131072;
constexpr int kKernStride = 64;
constexpr size_t kCnt = 0, kKern = 64, kExt = 448, kKp = kExt + (size_t)kExtCap * 8, kSrc = kKp + (size_t)kKpCap * 8;

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw std::runtime_error(fmt("Vulkan error %d in %s", (int)r, what));
}

struct Buf {
    VkBuffer b = VK_NULL_HANDLE;
    VkDeviceMemory m = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* map = nullptr;
    bool coherent = true;
};

struct PcUp { int sw, sh, dw, dh, dst; };
struct PcBlur { int w, h, k, r; };
struct PcDown { int sw, dw, dh; };
struct PcExt { int o, w, h; float contrast, edge; };
struct PcOct { int a, w, h; };
struct PcNlm { int w, h; float k; int y0; };

// OpenCV's getGaussianKernel(n, sigma, CV_32F) with n = cvRound(sigma*8+1)|1 (float images)
std::vector<float> gauss_kernel(double sigma) {
    int n = cvRound(sigma * 8 + 1) | 1;
    std::vector<double> k(n);
    double s = 0, scale2 = -0.5 / (sigma * sigma);
    for (int i = 0; i < n; i++) {
        double x = i - (n - 1) * 0.5;
        k[i] = std::exp(scale2 * x * x);
        s += k[i];
    }
    std::vector<float> f(n);
    for (int i = 0; i < n; i++) f[i] = (float)(k[i] / s);
    return f;
}

}  // namespace

struct GpuSift::Impl {
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    VkPhysicalDeviceMemoryProperties memp{};
    VkDeviceSize max_range = 0, align = 256;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSet sets[kMaxOct] = {};  // one per octave
    VkPipeline p_bh[6] = {}, p_bv[6] = {};  // blur level l -> scratch, scratch -> level l
    VkPipeline p_up = VK_NULL_HANDLE, p_down = VK_NULL_HANDLE, p_ext = VK_NULL_HANDLE,
               p_prep = VK_NULL_HANDLE, p_orient = VK_NULL_HANDLE, p_descr = VK_NULL_HANDLE,
               p_nlm = VK_NULL_HANDLE;
    Buf pyr, io, dsc;  // pyramid levels + scratch (device), small buffers, descriptors
    std::string name;
    std::mutex mu;
    VkQueryPool qpool = VK_NULL_HANDLE;
    double ts_period = 0;  // ns per timestamp tick, 0: no timestamps
    GpuSift::Stats stats;
    int nq = 0;  // timestamps written in the current command buffer

    void init() {
        const char* env = std::getenv("DOCSCAN_GPU");
        bool want_cpu = env && std::string(env) == "cpu";
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "docscan";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        check(vkCreateInstance(&ici, nullptr, &inst), "vkCreateInstance");
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(inst, &n, nullptr);
        std::vector<VkPhysicalDevice> devs(n);
        vkEnumeratePhysicalDevices(inst, &n, devs.data());
        int best = -1;
        for (auto d : devs) {
            VkPhysicalDeviceProperties pr;
            vkGetPhysicalDeviceProperties(d, &pr);
            int score = pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU     ? 3
                        : pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2
                        : pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    ? 1
                                                                                  : -1;
            if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) score = want_cpu ? 4 : -1;
            uint32_t nq = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
            std::vector<VkQueueFamilyProperties> qs(nq);
            vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, qs.data());
            int fam = -1;
            for (uint32_t i = 0; i < nq; i++)
                if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    fam = (int)i;
                    break;
                }
            if (fam < 0 || score <= best) continue;
            best = score;
            pd = d;
            qfam = (uint32_t)fam;
            name = pr.deviceName;
            max_range = pr.limits.maxStorageBufferRange;
            ts_period = qs[fam].timestampValidBits && pr.limits.timestampComputeAndGraphics ? pr.limits.timestampPeriod : 0;
            align = std::max<VkDeviceSize>(256, pr.limits.minStorageBufferOffsetAlignment);
        }
        if (!pd) throw std::runtime_error("no Vulkan GPU");
        vkGetPhysicalDeviceMemoryProperties(pd, &memp);

        float prio = 1;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = qfam;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        check(vkCreateDevice(pd, &dci, nullptr, &dev), "vkCreateDevice");
        vkGetDeviceQueue(dev, qfam, 0, &queue);

        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = qfam;
        check(vkCreateCommandPool(dev, &pci, nullptr, &pool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(dev, &cai, &cb), "vkAllocateCommandBuffers");
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(dev, &fci, nullptr, &fence), "vkCreateFence");

        VkDescriptorSetLayoutBinding b[kBindings];
        for (int i = 0; i < kBindings; i++)
            b[i] = {(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = kBindings;
        lci.pBindings = b;
        check(vkCreateDescriptorSetLayout(dev, &lci, nullptr, &dsl), "vkCreateDescriptorSetLayout");
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsl;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        check(vkCreatePipelineLayout(dev, &plci, nullptr, &layout), "vkCreatePipelineLayout");
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kBindings * kMaxOct};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = kMaxOct;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        check(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool), "vkCreateDescriptorPool");
        std::vector<VkDescriptorSetLayout> dsls(kMaxOct, dsl);
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = dpool;
        dsai.descriptorSetCount = kMaxOct;
        dsai.pSetLayouts = dsls.data();
        check(vkAllocateDescriptorSets(dev, &dsai, sets), "vkAllocateDescriptorSets");

        p_up = pipeline(kUpscale, sizeof kUpscale);
        for (int l = 0; l < 6; l++) {
            p_bh[l] = pipeline(kBlur, sizeof kBlur, {l, 6, 0});
            p_bv[l] = pipeline(kBlur, sizeof kBlur, {6, l, 1});
        }
        p_down = pipeline(kDown, sizeof kDown);
        p_ext = pipeline(kExtrema, sizeof kExtrema);
        p_prep = pipeline(kPrep, sizeof kPrep);
        p_orient = pipeline(kOrient, sizeof kOrient);
        p_descr = pipeline(kDescr, sizeof kDescr);
        p_nlm = pipeline(kNlm, sizeof kNlm);

        if (ts_period > 0) {
            VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qpci.queryCount = 16;
            check(vkCreateQueryPool(dev, &qpci, nullptr, &qpool), "vkCreateQueryPool");
        }
        alloc(io, (kSrc << 2) + (2 << 20), true);
        alloc(dsc, 4096 * 128 * 4, true);
        alloc(pyr, 64 << 20, false);
    }

    VkPipeline pipeline(const uint32_t* code, size_t bytes, std::vector<int> spec = {}) {
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = bytes;
        smci.pCode = code;
        VkShaderModule sm;
        check(vkCreateShaderModule(dev, &smci, nullptr, &sm), "vkCreateShaderModule");
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = sm;
        cpci.stage.pName = "main";
        std::vector<VkSpecializationMapEntry> me(spec.size());
        for (size_t i = 0; i < spec.size(); i++) me[i] = {(uint32_t)i, (uint32_t)(4 * i), 4};
        VkSpecializationInfo si{(uint32_t)me.size(), me.data(), spec.size() * 4, spec.data()};
        if (!spec.empty()) cpci.stage.pSpecializationInfo = &si;
        cpci.layout = layout;
        VkPipeline p;
        VkResult r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &p);
        vkDestroyShaderModule(dev, sm, nullptr);
        check(r, "vkCreateComputePipelines");
        return p;
    }

    uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags nice) {
        for (int pass = 0; pass < 2; pass++)
            for (uint32_t i = 0; i < memp.memoryTypeCount; i++) {
                VkMemoryPropertyFlags f = memp.memoryTypes[i].propertyFlags;
                if ((bits & (1u << i)) && (f & want) == want && (pass == 1 || (f & nice) == nice)) return i;
            }
        throw std::runtime_error("no suitable Vulkan memory type");
    }

    void free(Buf& b) {
        if (b.map) vkUnmapMemory(dev, b.m);
        if (b.b) vkDestroyBuffer(dev, b.b, nullptr);
        if (b.m) vkFreeMemory(dev, b.m, nullptr);
        b = Buf();
    }

    void alloc(Buf& b, VkDeviceSize size, bool host) {
        free(b);
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        check(vkCreateBuffer(dev, &bci, nullptr, &b.b), "vkCreateBuffer");
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(dev, b.b, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        // host buffers: cached memory if possible (reading uncached memory is slow on phones)
        mai.memoryTypeIndex =
            host ? mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
                 : mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        b.coherent = memp.memoryTypes[mai.memoryTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        check(vkAllocateMemory(dev, &mai, nullptr, &b.m), "vkAllocateMemory");
        check(vkBindBufferMemory(dev, b.b, b.m, 0), "vkBindBufferMemory");
        if (host) check(vkMapMemory(dev, b.m, 0, size, 0, &b.map), "vkMapMemory");
        b.size = size;
    }

    void flush(Buf& b) {
        if (b.coherent || !b.map) return;
        VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, b.m, 0, VK_WHOLE_SIZE};
        vkFlushMappedMemoryRanges(dev, 1, &r);
    }
    void invalidate(Buf& b) {
        if (b.coherent || !b.map) return;
        VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, b.m, 0, VK_WHOLE_SIZE};
        vkInvalidateMappedMemoryRanges(dev, 1, &r);
    }

    void ensure(Buf& b, VkDeviceSize size, bool host) {
        if (b.size < size) alloc(b, std::max(size, b.size * 3 / 2), host);
    }

    // octave o's set: its 6 levels, the blur scratch, the previous octave's level 3, IO, descriptors
    struct Range {
        VkDeviceSize off, size;
    };
    void bind_octave(int o, const Range* lv, Range tmp, Range prev) {
        VkDescriptorBufferInfo bi[kBindings];
        for (int l = 0; l < kLayers + 3; l++) bi[l] = {pyr.b, lv[l].off, lv[l].size};
        bi[6] = {pyr.b, tmp.off, tmp.size};
        bi[7] = {pyr.b, prev.off, prev.size};
        bi[8] = {io.b, 0, io.size};
        bi[9] = {dsc.b, 0, dsc.size};
        VkWriteDescriptorSet w[kBindings];
        for (int i = 0; i < kBindings; i++) {
            if (bi[i].range > max_range) throw std::runtime_error("image level exceeds the GPU buffer limit");
            w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[i].dstSet = sets[o];
            w[i].dstBinding = (uint32_t)i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(dev, kBindings, w, 0, nullptr);
    }

    void begin() {
        check(vkResetCommandBuffer(cb, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
        nq = 0;
        if (qpool) vkCmdResetQueryPool(cb, qpool, 0, 16);
    }

    void stamp() {
        if (qpool) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, nq++);
    }

    // seconds between consecutive timestamps of the last submit
    std::vector<double> stamps() {
        std::vector<double> d;
        if (!qpool || nq < 2) return d;
        uint64_t t[16];
        if (vkGetQueryPoolResults(dev, qpool, 0, nq, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return d;
        for (int i = 1; i < nq; i++) d.push_back((t[i] - t[i - 1]) * ts_period * 1e-9);
        return d;
    }

    void use_octave(int o) { vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &sets[o], 0, nullptr); }

    template <class PC>
    void dispatch(VkPipeline p, const PC& pc, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
        vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
        vkCmdDispatch(cb, gx, gy, gz);
    }

    void barrier() {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                           VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
                           VK_ACCESS_TRANSFER_WRITE_BIT;
        const VkPipelineStageFlags st = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
                                        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
        vkCmdPipelineBarrier(cb, st, st, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    void submit() {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
        check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
        flush(io);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        check(vkQueueSubmit(queue, 1, &si, fence), "vkQueueSubmit");
        check(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        check(vkResetFences(dev, 1, &fence), "vkResetFences");
        invalidate(io);
        invalidate(dsc);
    }

    // binding s -> binding d through the scratch binding 6; k: kernel index
    void blur(int w, int h, int s, int d, int k, int r) {
        dispatch(p_bh[s], PcBlur{w, h, k * kKernStride, r}, (uint32_t)((w + 255) / 256), (uint32_t)h);
        barrier();
        dispatch(p_bv[d], PcBlur{w, h, k * kKernStride, r}, (uint32_t)((w + 255) / 256), (uint32_t)h);
        barrier();
    }

    void nlm(const cv::Mat& g, float h, cv::Mat& out) {
        const int w = g.cols, H = g.rows;
        ensure(io, (kSrc << 2) + (size_t)w * H + 4, true);
        ensure(dsc, (size_t)w * H * 4, true);
        // set 0 needs all bindings; the levels are unused here
        Range dummy{0, std::min<VkDeviceSize>(pyr.size, 1 << 16)};
        Range lv[kLayers + 3];
        for (auto& r : lv) r = dummy;
        bind_octave(0, lv, dummy, dummy);
        std::memcpy((uint8_t*)io.map + (kSrc << 2), g.data, (size_t)w * H);
        begin();
        use_octave(0);
        // bands of rows: short GPU jobs (phone drivers may stop long ones)
        const float k = (float)(1.4426950408889634 / (h * h * 49.0));
        for (int y0 = 0; y0 < H; y0 += 128)
            dispatch(p_nlm, PcNlm{w, H, k, y0}, (uint32_t)((w + 15) / 16), (uint32_t)((std::min(128, H - y0) + 15) / 16));
        submit();
        out.create(H, w, CV_8U);
        const float* f = (const float*)dsc.map;
        for (int i = 0; i < w * H; i++) out.data[i] = cv::saturate_cast<uchar>(f[i]);
    }

    void run(const cv::Mat& g, float contrast, const std::function<bool(const cv::Point2f&)>& keep,
             std::vector<cv::KeyPoint>& kps, cv::Mat& dsc_out) {
        const int w = g.cols, h = g.rows, W = 2 * w, H = 2 * h;
        const int nOct = cvRound(std::log((double)std::min(W, H)) / std::log(2.) - 2) + 1;
        if (nOct < 1 || nOct > kMaxOct) throw std::runtime_error("image too small for SIFT");
        std::vector<int> ow(nOct), oh(nOct);
        std::vector<std::array<Range, kLayers + 3>> lv(nOct);
        VkDeviceSize total = 0;
        auto take = [&](VkDeviceSize bytes) {
            Range r{total, bytes};
            total += (bytes + align - 1) / align * align;
            return r;
        };
        for (int o = 0; o < nOct; o++) {
            ow[o] = o ? ow[o - 1] / 2 : W;
            oh[o] = o ? oh[o - 1] / 2 : H;
            for (int l = 0; l < kLayers + 3; l++) lv[o][l] = take((VkDeviceSize)ow[o] * oh[o] * 4);
        }
        const Range tmp = take((VkDeviceSize)W * H * 4);
        ensure(pyr, total, false);
        const size_t io_bytes = (kSrc << 2) + (size_t)w * h + 4;
        ensure(io, io_bytes, true);
        for (int o = 0; o < nOct; o++) bind_octave(o, lv[o].data(), tmp, o ? lv[o - 1][kLayers] : lv[o][0]);

        uint32_t* IO = (uint32_t*)io.map;
        // blur sigmas: initial (the upscaled image has sigma 1.0) and the 5 per-octave steps
        double sig[kLayers + 3];
        sig[0] = std::sqrt(std::max(kSigma * kSigma - 0.5 * 0.5 * 4, 0.01));
        double k = std::pow(2., 1. / kLayers);
        for (int i = 1; i < kLayers + 3; i++) {
            double prev = std::pow(k, (double)(i - 1)) * kSigma, tot = prev * k;
            sig[i] = std::sqrt(tot * tot - prev * prev);
        }
        int rad[kLayers + 3];
        float* K = (float*)(IO + kKern);
        for (int i = 0; i < kLayers + 3; i++) {
            auto kv = gauss_kernel(sig[i]);
            rad[i] = (int)kv.size() / 2;
            std::copy(kv.begin(), kv.end(), K + i * kKernStride);
        }
        uint8_t* S8 = (uint8_t*)(IO + kSrc);
        for (int y = 0; y < h; y++) std::memcpy(S8 + (size_t)y * w, g.ptr(y), w);
        std::memset(IO + kCnt, 0, 64);

        begin();
        stamp();
        use_octave(0);
        dispatch(p_up, PcUp{w, h, W, H, 0}, (uint32_t)((W + 15) / 16), (uint32_t)((H + 15) / 16));
        barrier();
        blur(W, H, 0, 0, 0, rad[0]);
        for (int o = 0; o < nOct; o++) {
            use_octave(o);
            if (o > 0) {
                dispatch(p_down, PcDown{ow[o - 1], ow[o], oh[o]}, (uint32_t)((ow[o] + 15) / 16),
                         (uint32_t)((oh[o] + 15) / 16));
                barrier();
            }
            for (int l = 1; l < kLayers + 3; l++) blur(ow[o], oh[o], l - 1, l, l, rad[l]);
        }
        stamp();
        for (int o = 0; o < nOct; o++) {
            if (ow[o] <= 2 * 5 + 2 || oh[o] <= 2 * 5 + 2) continue;
            use_octave(o);
            vkCmdFillBuffer(cb, io.b, kCnt * 4, 4, 0);  // extrema counter of this octave
            barrier();
            dispatch(p_ext, PcExt{o, ow[o], oh[o], contrast, 10.f}, (uint32_t)((ow[o] + 15) / 16),
                     (uint32_t)((oh[o] + 15) / 16));
            barrier();
            dispatch(p_prep, PcOct{0, 0, 0}, 1);
            barrier();
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_orient);
            PcOct pc{o, ow[o], oh[o]};
            vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
            vkCmdDispatchIndirect(cb, io.b, (kCnt + 4) * 4);
            barrier();
        }
        stamp();
        submit();
        if (auto d = stamps(); d.size() == 2) {
            stats.pyramid += d[0];
            stats.extrema += d[1];
        }

        const int n = (int)std::min<uint32_t>(IO[kCnt + 1], kKpCap);
        const float* R = (const float*)(IO + kKp);

        // keypoints in image coordinates (the first octave is the upscaled image)
        struct Cand {
            cv::KeyPoint k;
            int rec;
        };
        std::vector<Cand> cand(n);
        for (int i = 0; i < n; i++) {
            const float* r = R + i * 8;
            float s = (float)(1 << (int)r[2]) * 0.5f;
            cand[i].k = cv::KeyPoint(r[0] * s, r[1] * s, r[4] * s, r[6], r[5], (int)r[2] - 1);
            cand[i].rec = i;
        }
        // KeyPointsFilter::removeDuplicatedSorted
        std::sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) {
            const cv::KeyPoint &p = a.k, &q = b.k;
            if (p.pt.x != q.pt.x) return p.pt.x < q.pt.x;
            if (p.pt.y != q.pt.y) return p.pt.y < q.pt.y;
            if (p.size != q.size) return p.size > q.size;
            if (p.angle != q.angle) return p.angle < q.angle;
            if (p.response != q.response) return p.response > q.response;
            return a.rec < b.rec;
        });
        std::vector<float> sel;
        kps.clear();
        for (int i = 0; i < n; i++) {
            const cv::KeyPoint& c = cand[i].k;
            if (i > 0) {
                const cv::KeyPoint& p = cand[i - 1].k;
                if (p.pt == c.pt && p.size == c.size && p.angle == c.angle) continue;
            }
            if (!keep(c.pt)) continue;
            kps.push_back(c);
            sel.insert(sel.end(), R + cand[i].rec * 8, R + cand[i].rec * 8 + 8);
        }
        const int m = (int)kps.size();
        dsc_out.create(m, 128, CV_32F);
        if (m == 0) return;
        // records grouped by octave (each octave has its own bindings), output in kps order
        std::vector<int> order(m);
        for (int i = 0; i < m; i++) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sel[a * 8 + 2] < sel[b * 8 + 2]; });
        float* KP = (float*)(IO + kKp);
        for (int i = 0; i < m; i++) std::memcpy(KP + i * 8, &sel[order[i] * 8], 32);
        if (dsc.size < (VkDeviceSize)m * 128 * 4) {
            ensure(dsc, (VkDeviceSize)m * 128 * 4, true);
            for (int o = 0; o < nOct; o++) bind_octave(o, lv[o].data(), tmp, o ? lv[o - 1][kLayers] : lv[o][0]);
        }
        begin();
        stamp();
        for (int i = 0; i < m;) {
            int o = (int)KP[i * 8 + 2], j = i;
            while (j < m && (int)KP[j * 8 + 2] == o) j++;
            use_octave(o);
            for (int b = i; b < j; b += kExtCap)
                dispatch(p_descr, PcOct{b, ow[o], oh[o]}, (uint32_t)std::min(kExtCap, j - b));
            i = j;
        }
        stamp();
        submit();
        if (auto d = stamps(); d.size() == 1) stats.descriptors += d[0];
        const float* D = (const float*)dsc.map;
        for (int i = 0; i < m; i++) std::memcpy(dsc_out.ptr<float>(order[i]), D + (size_t)i * 128, 128 * 4);
    }

    ~Impl() {
        if (dev) {
            vkDeviceWaitIdle(dev);
            for (Buf* b : {&pyr, &io, &dsc}) free(*b);
            for (VkPipeline p : p_bh) vkDestroyPipeline(dev, p, nullptr);
            for (VkPipeline p : p_bv) vkDestroyPipeline(dev, p, nullptr);
            for (VkPipeline p : {p_up, p_down, p_ext, p_prep, p_orient, p_descr, p_nlm})
                if (p) vkDestroyPipeline(dev, p, nullptr);
            if (qpool) vkDestroyQueryPool(dev, qpool, nullptr);
            if (dpool) vkDestroyDescriptorPool(dev, dpool, nullptr);
            if (layout) vkDestroyPipelineLayout(dev, layout, nullptr);
            if (dsl) vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
            if (fence) vkDestroyFence(dev, fence, nullptr);
            if (pool) vkDestroyCommandPool(dev, pool, nullptr);
            vkDestroyDevice(dev, nullptr);
        }
        if (inst) vkDestroyInstance(inst, nullptr);
    }
};

GpuSift::GpuSift() : p(std::make_unique<Impl>()) {}
GpuSift::~GpuSift() = default;

const std::string& GpuSift::name() const { return p->name; }

GpuSift::Stats GpuSift::take_stats() {
    std::lock_guard<std::mutex> lk(p->mu);
    Stats s = p->stats;
    p->stats = Stats();
    return s;
}

GpuSift* GpuSift::get() {
    static std::once_flag once;
    static std::unique_ptr<GpuSift> inst;
    std::call_once(once, [] {
        const char* env = std::getenv("DOCSCAN_GPU");
        if (env && std::string(env) == "0") return;
        std::unique_ptr<GpuSift> g(new GpuSift());
        try {
            g->p->init();
            log("GPU: " + g->p->name);
            inst = std::move(g);
        } catch (const std::exception& e) {
            log(std::string("GPU not used: ") + e.what());
        }
    });
    return inst.get();
}

bool GpuSift::denoise(const cv::Mat& g, cv::Mat& out, float h) {
    CV_Assert(g.type() == CV_8U && g.isContinuous());
    std::lock_guard<std::mutex> lk(p->mu);
    try {
        p->nlm(g, h, out);
        return true;
    } catch (const std::exception& e) {
        log(std::string("GPU denoising failed: ") + e.what());
        return false;
    }
}

bool GpuSift::detect(const cv::Mat& g, float contrast, const std::function<bool(const cv::Point2f&)>& keep,
                     std::vector<cv::KeyPoint>& kps, cv::Mat& desc) {
    CV_Assert(g.type() == CV_8U);
    std::lock_guard<std::mutex> lk(p->mu);
    try {
        p->run(g, contrast, keep, kps, desc);
        return true;
    } catch (const std::exception& e) {
        log(std::string("GPU SIFT failed: ") + e.what());
        return false;
    }
}

}  // namespace docscan
