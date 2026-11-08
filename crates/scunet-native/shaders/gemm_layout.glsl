// What the two gemm kernels share: their buffers, push constants, GELU and
// the mapping of rows onto buffers (see gemm.comp).
layout(std430, binding = 0) readonly restrict buffer Src { float16_t src[]; };
layout(std430, binding = 0) readonly restrict buffer Src2 { f16vec2 src2[]; };
layout(std430, binding = 0) readonly restrict buffer Src4 { f16vec4 src4[]; };
layout(std430, binding = 1) restrict buffer Dst { f16vec2 dst[]; };
layout(std430, binding = 1) restrict buffer Dst16 { uvec4 dst16[]; };
layout(std430, binding = 2) readonly restrict buffer Params { f16vec2 params[]; };
layout(std430, binding = 2) readonly restrict buffer Scalars { float16_t ps[]; };
layout(push_constant) uniform PC {
    int k; int n; int src_stride; int src_off; int dst_stride; int dst_off;
    int w_off; int b_off; int gelu; int accumulate; int in_mode; int out_mode;
    int width; int height; int shift; int base; int norm; int g_off; int beta_off; int rows;
} pc;

#include "gelu.glsl"

// Where row r starts in modes 0 and 2; where pixel (2y, 2x) does in mode 1.
int row_start(int mode, int r, int stride, int off) {
    if (mode == 1) {
        int y = r / pc.width;
        int x = r - y * pc.width;
        return (2 * y * 2 * pc.width + 2 * x) * stride;
    }
    if (mode == 2) {
        int g = pc.base + r;
        int w = g >> 6;
        int i = g & 63;
        int columns = pc.width >> 3;
        int y = ((w / columns) * 8 + (i >> 3) + pc.shift) % pc.height;
        int x = ((w % columns) * 8 + (i & 7) + pc.shift) % pc.width;
        return (y * pc.width + x) * stride + off;
    }
    return r * stride + off;
}

