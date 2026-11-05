// Bindings shared by all SIFT shaders. Each pass works on one octave: its 6 Gaussian levels are
// bound separately (some GPUs limit one binding to 128 MB), plus blur scratch and the previous
// octave's level 3 (input of the downsampling). Small buffers are packed into IO.
layout(std430, binding = 0) buffer B0 { float v[]; } L0;
layout(std430, binding = 1) buffer B1 { float v[]; } L1;
layout(std430, binding = 2) buffer B2 { float v[]; } L2;
layout(std430, binding = 3) buffer B3 { float v[]; } L3;
layout(std430, binding = 4) buffer B4 { float v[]; } L4;
layout(std430, binding = 5) buffer B5 { float v[]; } L5;
layout(std430, binding = 6) buffer B6 { float v[]; } TMP;
layout(std430, binding = 7) buffer B7 { float v[]; } PREV;
layout(std430, binding = 8) buffer B8 { uint v[]; } IO;
layout(std430, binding = 9) buffer B9 { float v[]; } DSC;

// IO layout (32-bit words): counters | blur kernels | extrema | keypoints | 8-bit input image
const int CNT = 0;      // [0] extrema of this octave [1] keypoints [4..6] indirect dispatch
const int KERN = 64;    // 6 kernels, 64 taps each
const int EXT = 448;    // 8 words per extremum
const int EXT_CAP = 65535;
const int KP = EXT + EXT_CAP * 8;  // 8 words per keypoint
const int KP_CAP = 131072;
const int SRC = KP + KP_CAP * 8;

const int LAYERS = 3;   // nOctaveLayers
const float SIGMA = 1.6;
const int BORDER = 5;   // SIFT_IMG_BORDER

float rd(int b, int i) {
    switch (b) {
        case 0: return L0.v[i];
        case 1: return L1.v[i];
        case 2: return L2.v[i];
        case 3: return L3.v[i];
        case 4: return L4.v[i];
        case 5: return L5.v[i];
        case 6: return TMP.v[i];
        default: return PREV.v[i];
    }
}

void wr(int b, int i, float x) {
    switch (b) {
        case 0: L0.v[i] = x; break;
        case 1: L1.v[i] = x; break;
        case 2: L2.v[i] = x; break;
        case 3: L3.v[i] = x; break;
        case 4: L4.v[i] = x; break;
        case 5: L5.v[i] = x; break;
        default: TMP.v[i] = x; break;
    }
}

float iof(int i) { return uintBitsToFloat(IO.v[i]); }
void iow(int i, float x) { IO.v[i] = floatBitsToUint(x); }
