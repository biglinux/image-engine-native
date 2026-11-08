// What the two gemm kernels share: their buffers, push constants, the
// activations and the mapping of rows onto buffers.
//
// out[r][col] (=, +=) act(... + sum_k in[r][k] * W[k][col] + bias[col]) in
// half floats with sums over blocks of 16 k in floats, the activation (none,
// ReLU, GELU or sigmoid) taken after the sum, optionally on the
// layer-normalised input rows. W is stored [k][n padded to 64] in half
// floats, the bias and the normalisation's scale and shift in floats; k is a
// multiple of 16. Rows map onto the buffers in one of these ways:
// - mode 0: row r is `r * stride + off`;
// - mode 1 (patch merging, input only): row r is token (y, x) of a grid
//   `width` wide, and k runs over the tokens (2y, 2x), (2y + 1, 2x),
//   (2y, 2x + 1), (2y + 1, 2x + 1) of the grid twice as wide;
// - mode 2: row `base + r` is token i of 7 x 7 window w of the grid `width`
//   wide padded to `padded`, rolled by `shift`; a token in the padding reads
//   zeros and is not written;
// - mode 3 (input only): row `base + r` is pixel (y, x) of a
//   `width x height` map, and k runs over the taps of a `kernel x kernel`
//   convolution centred on it, each tap's channels (a multiple of 16)
//   together, at `src_off` within its pixel; taps past the edge read zeros.
//
// With `splits` above 1, workgroup z sums only its share of k, a multiple
// of 16, and writes floats `z * rows * dst_stride` past the usual place, for
// reduce.comp to add up.
layout(std430, binding = 0) readonly restrict buffer Src { f16vec4 src4[]; };
layout(std430, binding = 1) restrict buffer Dst { f16vec2 dst[]; };
layout(std430, binding = 1) restrict buffer DstScalar { float16_t dst1[]; };
layout(std430, binding = 1) restrict buffer DstFloat { float dstf[]; };
layout(std430, binding = 1) restrict buffer Dst16 { uvec4 dst16[]; };
layout(std430, binding = 2) readonly restrict buffer Params { f16vec2 params[]; };
layout(std430, binding = 2) readonly restrict buffer Floats { float pf[]; };
layout(push_constant) uniform PC {
    int rows; int k; int n; int src_stride; int src_off; int dst_stride; int dst_off;
    int w_off; int b_off; int act; int accumulate; int in_mode; int out_mode;
    int width; int padded; int shift; int base; int norm; int g_off; int beta_off;
    int kernel; int height; int out_float; int splits;
} pc;

float gelu(float x) {
    float z = clamp(x * 0.70710678, -4.0, 4.0);
    float z2 = z * z;
    float p = -2.7261423e-10;
    p = p * z2 + 2.7706814e-8;
    p = p * z2 - 2.101024e-6;
    p = p * z2 - 5.6925064e-5;
    p = p * z2 - 7.3499063e-4;
    p = p * z2 - 2.9546e-3;
    p = p * z2 - 1.6096033e-2;
    float q = -1.4566072e-5;
    q = q * z2 - 2.1337406e-4;
    q = q * z2 - 1.682827e-3;
    q = q * z2 - 7.373329e-3;
    q = q * z2 - 1.426474e-2;
    return 0.5 * x * (1.0 + z * p / q);
}

vec2 activate(vec2 v) {
    if (pc.act == 1)
        return max(v, vec2(0.0));
    if (pc.act == 2)
        return vec2(gelu(v.x), gelu(v.y));
    if (pc.act == 3)
        return 1.0 / (1.0 + exp(-v));
    return v;
}

// Where row r starts, or -1 for a padding token; in mode 1, where token
// (2y, 2x) does; in mode 3, its pixel as y << 16 | x.
int row_start(int mode, int r, int stride, int off) {
    if (mode == 1) {
        int y = r / pc.width;
        int x = r - y * pc.width;
        return (2 * y * 2 * pc.width + 2 * x) * stride;
    }
    if (mode == 2) {
        int g = pc.base + r;
        int w = g / 49;
        int i = g - w * 49;
        int columns = pc.padded / 7;
        int y = ((w / columns) * 7 + i / 7 + pc.shift) % pc.padded;
        int x = ((w % columns) * 7 + i % 7 + pc.shift) % pc.padded;
        if (y >= pc.width || x >= pc.width)
            return -1;
        return (y * pc.width + x) * stride + off;
    }
    if (mode == 3) {
        int p = pc.base + r;
        int y = p / pc.width;
        return y << 16 | (p - y * pc.width);
    }
    return r * stride + off;
}

// Where the input chunk k0..k0 + 16 lies relative to each row's start:
// `off` past it, and in mode 3 past the pixel moved by (dy, dx). The same
// for every row, so computed once per chunk.
struct Chunk {
    int dy;
    int dx;
    int off;
};

Chunk chunk(int k0) {
    Chunk ch = Chunk(0, 0, k0);
    if (pc.in_mode == 1) {
        int quarter = pc.k >> 2;
        int q = k0 / quarter;
        ch.off = ((q & 1) * 2 * pc.width + (q >> 1)) * pc.src_stride + k0 - q * quarter;
    } else if (pc.in_mode == 3) {
        int cin = pc.k / (pc.kernel * pc.kernel);
        int tap = k0 / cin;
        int ty = tap / pc.kernel;
        ch.dy = ty - (pc.kernel >> 1);
        ch.dx = tap - ty * pc.kernel - (pc.kernel >> 1);
        ch.off = pc.src_off + k0 - tap * cin;
    }
    return ch;
}

// Values 4 q .. 4 q + 3 of the chunk `ch` of the row `start` (from
// row_start), zero for padding and taps past the edge. Every row start and
// chunk offset is a multiple of 4.
f16vec4 load4(int start, Chunk ch, int q) {
    if (start < 0)
        return f16vec4(0.0);
    if (pc.in_mode == 3) {
        int y = (start >> 16) + ch.dy;
        int x = (start & 0xffff) + ch.dx;
        if (y < 0 || y >= pc.height || x < 0 || x >= pc.width)
            return f16vec4(0.0);
        return src4[((y * pc.width + x) * pc.src_stride + ch.off >> 2) + q];
    }
    return src4[(start + ch.off >> 2) + q];
}

// Store columns col..col + 1 (col alone past n) of a row starting at `at`:
// in floats when `out_float`, else in half floats, in pairs where n is even
// and one by one where it is odd.
void store(int at, int col, vec2 v) {
    if (pc.b_off >= 0)
        v += vec2(pf[pc.b_off + col], col + 1 < pc.n ? pf[pc.b_off + col + 1] : 0.0);
    if (pc.out_float != 0) {
        for (int h = 0; h < 2 && col + h < pc.n; h++)
            dstf[at + col + h] = activate(vec2(v[h] + (pc.accumulate != 0 ? dstf[at + col + h] : 0.0))).x;
        return;
    }
    if ((pc.n & 1) == 0) {
        int i = (at + col) >> 1;
        if (pc.accumulate != 0)
            v += vec2(dst[i]);
        dst[i] = f16vec2(activate(v));
        return;
    }
    for (int h = 0; h < 2 && col + h < pc.n; h++) {
        float x = v[h];
        if (pc.accumulate != 0)
            x += float(dst1[at + col + h]);
        dst1[at + col + h] = float16_t(activate(vec2(x)).x);
    }
}

// Store columns col..col + 7 of a row starting at `at` in one 16-byte write,
// where the output is half floats and those columns exist and are aligned;
// false otherwise, for store() to take them a pair at a time. Neighbouring
// invocations' rows are apart, so narrower writes each cost a transaction.
bool store8(int at, int col, vec2 v[4]) {
    if (pc.out_float != 0 || col + 8 > pc.n || ((at + col) & 7) != 0)
        return false;
    int i = (at + col) >> 3;
    uvec4 old = pc.accumulate != 0 ? dst16[i] : uvec4(0);
    uvec4 w;
    [[unroll]] for (int h = 0; h < 4; h++) {
        vec2 x = v[h];
        if (pc.b_off >= 0)
            x += vec2(pf[pc.b_off + col + 2 * h], pf[pc.b_off + col + 2 * h + 1]);
        if (pc.accumulate != 0)
            x += vec2(unpackFloat2x16(old[h]));
        w[h] = packFloat2x16(f16vec2(activate(x)));
    }
    dst16[i] = w;
    return true;
}

// The share of k that workgroup z sums: k0 from `first` to `last`.
void k_range(out int first, out int last) {
    int share = ((pc.k + pc.splits - 1) / pc.splits + 15) & ~15;
    first = int(gl_WorkGroupID.z) * share;
    last = min(pc.k, first + share);
}
