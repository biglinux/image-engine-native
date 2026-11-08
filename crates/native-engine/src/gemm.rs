//! Matrix products: every linear layer and convolution of the networks.
//!
//! The weights of a layer are a `k x n` [`Matrix`], packed once at load into
//! panels of 16 columns, `[n / 16][k][16]`, so the 16 weights one step of the
//! micro-kernel needs are one contiguous run, 64 bytes as floats or 32 as half
//! floats. Activations are NHWC: one row per pixel or token, channels
//! contiguous.
//!
//! The left operand is never copied. An [`Input`] tells, for each of the six
//! rows a micro-kernel works on and each kernel tap, where that row's channels
//! start: in a plain row-major matrix there is one tap, and in a convolution
//! each tap leads to a neighbouring pixel, to its mirror image past the edge,
//! or to zeros. This is the indirect
//! convolution of XNNPACK; it costs nothing per element, where im2col copies
//! every input value `size * size` times.
//!
//! [`multiply`] covers a range of rows on the calling thread. The reduction
//! runs in blocks of `KC` values: a 16-column panel block of `KC x 16`
//! weights (16 KB in half floats) meets every six-row group of the range, and
//! the rows of the range stay in L2 across panels. The order of the sums
//! depends only on `k`, never on the range or on the thread.

use std::ops::Range;

use crate::half::{f32_to_half, half_to_f32};

mod kernel;

use kernel::{KernelCall, kernels};

/// Columns of a weight panel and of a micro-tile.
pub const NR: usize = 16;
/// Rows of a micro-tile.
pub const MR: usize = 6;
/// Length of one block of the reduction. Each block loads and stores every
/// output tile once, which costs more than a float panel block outgrowing L1.
const KC: usize = 512;

static ZEROS: [f32; KC] = [0.0; KC];

/// Weight values as a weights file stores them: 32-bit floats, or IEEE half floats
/// that the AVX2 kernel widens as it loads them.
pub enum Weights {
    F32(Vec<f32>),
    F16(Vec<u16>),
}

impl Weights {
    pub(crate) fn len(&self) -> usize {
        match self {
            Weights::F32(v) => v.len(),
            Weights::F16(v) => v.len(),
        }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            Weights::F32(v) => v.clone(),
            Weights::F16(v) => v.iter().map(|&h| half_to_f32(h)).collect(),
        }
    }

    /// The values as half floats: stored halves as they are, floats rounded
    /// to the nearest.
    pub fn into_f16(self) -> Vec<u16> {
        match self {
            Weights::F32(v) => v.iter().map(|&x| f32_to_half(x)).collect(),
            Weights::F16(v) => v,
        }
    }

    /// Rows `rows` of a row-major matrix `n` wide.
    pub fn rows(&self, rows: Range<usize>, n: usize) -> Weights {
        match self {
            Weights::F32(v) => Weights::F32(v[rows.start * n..rows.end * n].to_vec()),
            Weights::F16(v) => Weights::F16(v[rows.start * n..rows.end * n].to_vec()),
        }
    }
}

/// One step of a panel: 16 weights, aligned so that a vector load of them
/// never straddles two cache lines (Sandy Bridge pays for every split load).
#[derive(Clone, Copy, Default)]
#[repr(C, align(64))]
struct Row([f32; NR]);

/// [`Row`] as half floats.
#[derive(Clone, Copy, Default)]
#[repr(C, align(32))]
struct HalfRow([u16; NR]);

enum Panels {
    F32(Vec<Row>),
    F16(Vec<HalfRow>),
}

/// Packed weights of one layer, with an optional bias.
pub struct Matrix {
    pub k: usize,
    pub n: usize,
    panels: Panels,
    /// Padded to whole panels with zeros; all zeros when the layer has none.
    bias: Vec<f32>,
}

impl Matrix {
    /// Pack row-major `k x n` weights `w`. Half floats stay half where the
    /// AVX2 kernel runs, and are widened for the others.
    pub fn new(w: &Weights, k: usize, n: usize, bias: Option<&[f32]>) -> Self {
        assert_eq!(w.len(), k * n, "weights are not {k} x {n}");
        let count = n.div_ceil(NR);
        fn pack<T: Copy + Default, R: Copy + Default>(
            w: &[T],
            k: usize,
            n: usize,
            count: usize,
            values: impl Fn(&mut R) -> &mut [T; NR],
        ) -> Vec<R> {
            let mut panels = vec![R::default(); count * k];
            for p in 0..count {
                let cols = (n - p * NR).min(NR);
                for row in 0..k {
                    values(&mut panels[p * k + row])[..cols]
                        .copy_from_slice(&w[row * n + p * NR..][..cols]);
                }
            }
            panels
        }
        let panels = match w {
            Weights::F16(h) if crate::simd_tier() >= crate::TIER_AVX2 => {
                Panels::F16(pack(h, k, n, count, |r: &mut HalfRow| &mut r.0))
            }
            Weights::F16(_) => Panels::F32(pack(&w.to_f32(), k, n, count, |r: &mut Row| &mut r.0)),
            Weights::F32(f) => Panels::F32(pack(f, k, n, count, |r: &mut Row| &mut r.0)),
        };
        let mut padded = vec![0.0; count * NR];
        if let Some(b) = bias {
            assert_eq!(b.len(), n, "bias is not {n} long");
            padded[..n].copy_from_slice(b);
        }
        Self {
            k,
            n,
            panels,
            bias: padded,
        }
    }

    /// A `k x n` matrix of float panels and no bias, all zeros, for
    /// [`Matrix::panel_mut`] to fill.
    pub(crate) fn zeros(k: usize, n: usize) -> Self {
        let count = n.div_ceil(NR);
        Self {
            k,
            n,
            panels: Panels::F32(vec![Row::default(); count * k]),
            bias: vec![0.0; count * NR],
        }
    }

    /// Make this matrix `k x n` in floats, its panels' values unspecified.
    pub(crate) fn resize(&mut self, k: usize, n: usize) {
        let count = n.div_ceil(NR);
        match &mut self.panels {
            Panels::F32(v) => v.resize(count * k, Row::default()),
            Panels::F16(_) => self.panels = Panels::F32(vec![Row::default(); count * k]),
        }
        self.bias.resize(count * NR, 0.0);
        (self.k, self.n) = (k, n);
    }

    /// Make this matrix `k x n` in half floats, its panels' values
    /// unspecified.
    pub(crate) fn resize_half(&mut self, k: usize, n: usize) {
        let count = n.div_ceil(NR);
        let zero = HalfRow([0; NR]);
        match &mut self.panels {
            Panels::F16(v) => v.resize(count * k, zero),
            Panels::F32(_) => self.panels = Panels::F16(vec![zero; count * k]),
        }
        self.bias.resize(count * NR, 0.0);
        (self.k, self.n) = (k, n);
    }

    /// Whether the panels hold half floats.
    pub(crate) fn is_half(&self) -> bool {
        matches!(self.panels, Panels::F16(_))
    }

    /// A half-float matrix's panels as one slice, as [`Matrix::floats_mut`].
    pub(crate) fn halves_mut(&mut self) -> &mut [u16] {
        const { assert!(size_of::<HalfRow>() == NR * size_of::<u16>()) };
        match &mut self.panels {
            // SAFETY: HalfRow is a repr(C) [u16; NR] without padding
            // (asserted above), so the rows are contiguous halves.
            Panels::F16(v) => unsafe {
                std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u16>(), v.len() * NR)
            },
            Panels::F32(_) => panic!("float panels are not halves"),
        }
    }

    /// A float matrix's panels as one slice: row `row` of panel `p` is
    /// floats `(p * k + row) * NR..` of it.
    pub(crate) fn floats_mut(&mut self) -> &mut [f32] {
        const { assert!(size_of::<Row>() == NR * size_of::<f32>()) };
        match &mut self.panels {
            // SAFETY: Row is a repr(C) [f32; NR] without padding (asserted
            // above), so the rows are contiguous floats.
            Panels::F32(v) => unsafe {
                std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<f32>(), v.len() * NR)
            },
            Panels::F16(_) => panic!("half-float panels are read-only"),
        }
    }

    /// The bias, padded to whole panels with zeros.
    pub(crate) fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// The 16 weights of row `row` of panel `p`, as floats.
    pub(crate) fn panel_floats(&self, p: usize, row: usize) -> [f32; NR] {
        let at = p * self.k + row;
        match &self.panels {
            Panels::F32(v) => v[at].0,
            // SAFETY: half-float panels exist only at the AVX2 tier, whose
            // CPUs have F16C.
            #[cfg(target_arch = "x86_64")]
            Panels::F16(v) => unsafe { widen(&v[at]) },
            #[cfg(not(target_arch = "x86_64"))]
            Panels::F16(v) => v[at].0.map(half_to_f32),
        }
    }

    /// Address of weight row `row` of panel `p`.
    fn panel_row(&self, p: usize, row: usize) -> *const u8 {
        let at = p * self.k + row;
        match &self.panels {
            Panels::F32(v) => v[at..].as_ptr().cast(),
            Panels::F16(v) => v[at..].as_ptr().cast(),
        }
    }
}

/// The 16 half floats of `row`, widened.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
fn widen(row: &HalfRow) -> [f32; NR] {
    use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm256_cvtph_ps, _mm256_storeu_ps};
    let mut out = [0.0; NR];
    let src = row.0.as_ptr().cast::<__m128i>();
    // SAFETY: the row holds two runs of 8 half floats, and `out` 16 floats.
    unsafe {
        _mm256_storeu_ps(out.as_mut_ptr(), _mm256_cvtph_ps(_mm_loadu_si128(src)));
        _mm256_storeu_ps(
            out[8..].as_mut_ptr(),
            _mm256_cvtph_ps(_mm_loadu_si128(src.add(1))),
        );
    }
    out
}

/// How a convolution reads past the edge of its image.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Pad {
    Zero,
    /// The image mirrored about its edge pixel, as PyTorch's "reflect".
    Reflect,
}

/// Where the rows of a product read their input.
#[derive(Clone, Copy)]
pub enum Taps {
    /// A centred `size x size` convolution, `size` odd, every `step` pixels:
    /// output pixel `(y, x)` reads input pixels
    /// `(step y + ky - size / 2, step x + kx - size / 2)`.
    Conv { size: usize, step: usize, pad: Pad },
    /// One output phase of a stride-2 transposed convolution: its pixel
    /// `(y, x)` reads input pixels `(y + dy, x + dx)` for the first `count`
    /// offsets, zeros past the edge.
    Offsets {
        offsets: [(usize, usize); 4],
        count: usize,
    },
}

/// The left operand of a product.
#[derive(Clone, Copy)]
pub enum Input<'a> {
    /// Row `r` is `data[r * stride..][..k]`.
    Rows { data: &'a [f32], stride: usize },
    /// Taps of a `height x width` NHWC image whose pixel `p` is
    /// `data[p * stride..][..channels]`. Row `r` is output pixel `r`, and the
    /// reduction runs over taps in order, channels within each.
    Image {
        data: &'a [f32],
        height: usize,
        width: usize,
        channels: usize,
        stride: usize,
        taps: Taps,
    },
}

/// `i` mirrored into `0..n` about the edge pixels, for `i` at most `n - 1`
/// past either edge.
pub(crate) fn reflect(i: isize, n: usize) -> usize {
    let last = n as isize - 1;
    i.abs().min(2 * last - i) as usize
}

impl<'a> Input<'a> {
    /// The stride-1, zero-padded, centred `size x size` convolution of a
    /// `height x width` image of `channels` channels, its pixels `stride`
    /// floats apart.
    pub fn conv(
        data: &'a [f32],
        height: usize,
        width: usize,
        channels: usize,
        stride: usize,
        size: usize,
    ) -> Self {
        Input::Image {
            data,
            height,
            width,
            channels,
            stride,
            taps: Taps::Conv {
                size,
                step: 1,
                pad: Pad::Zero,
            },
        }
    }

    /// Rows and columns of an image input's output grid.
    ///
    /// # Panics
    /// For [`Input::Rows`], which has none.
    pub fn out_size(&self) -> (usize, usize) {
        match *self {
            Input::Rows { .. } => panic!("plain rows have no output grid"),
            Input::Image {
                height,
                width,
                taps: Taps::Conv { step, .. },
                ..
            } => ((height - 1) / step + 1, (width - 1) / step + 1),
            Input::Image {
                height,
                width,
                taps: Taps::Offsets { .. },
                ..
            } => (height, width),
        }
    }

    /// Taps per row, and channels per tap (unbounded for plain rows).
    fn taps(&self) -> (usize, usize) {
        match *self {
            Input::Rows { .. } => (1, usize::MAX),
            Input::Image {
                channels,
                taps: Taps::Conv { size, .. },
                ..
            } => (size * size, channels),
            Input::Image {
                channels,
                taps: Taps::Offsets { count, .. },
                ..
            } => (count, channels),
        }
    }

    /// Where row `row` is: its output pixel (y, x) for an image, `(row, 0)`
    /// for plain rows.
    fn position(&self, row: usize) -> (usize, usize) {
        match *self {
            Input::Rows { .. } => (row, 0),
            Input::Image { .. } => {
                let (_, out_width) = self.out_size();
                (row / out_width, row % out_width)
            }
        }
    }

    /// Where tap `tap` reads from relative to its row's position, before any
    /// padding: (ky, kx) of a convolution, or the tap's offset.
    fn tap(&self, tap: usize) -> (usize, usize) {
        match *self {
            Input::Rows { .. } => (0, 0),
            Input::Image {
                taps: Taps::Conv { size, .. },
                ..
            } => (tap / size, tap % size),
            Input::Image {
                taps: Taps::Offsets { offsets, .. },
                ..
            } => offsets[tap],
        }
    }

    /// Pointer to channel `c0` of the tap at `tap` (from [`Input::tap`]) of
    /// the row at `position` (from [`Input::position`]), or to zeros.
    fn at(&self, (oy, ox): (usize, usize), (ty, tx): (usize, usize), c0: usize) -> *const f32 {
        let (data, height, width, stride, taps) = match *self {
            Input::Rows { data, stride } => return data[oy * stride + c0..].as_ptr(),
            Input::Image {
                data,
                height,
                width,
                stride,
                taps,
                ..
            } => (data, height, width, stride, taps),
        };
        let pixel = match taps {
            Taps::Conv { size, step, pad } => {
                let half = (size / 2) as isize;
                let y = (oy * step + ty) as isize - half;
                let x = (ox * step + tx) as isize - half;
                let inside = (0..height as isize).contains(&y) && (0..width as isize).contains(&x);
                match (inside, pad) {
                    (true, _) => Some((y as usize, x as usize)),
                    (false, Pad::Zero) => None,
                    (false, Pad::Reflect) => Some((reflect(y, height), reflect(x, width))),
                }
            }
            Taps::Offsets { .. } => {
                let (y, x) = (oy + ty, ox + tx);
                (y < height && x < width).then_some((y, x))
            }
        };
        match pixel {
            Some((y, x)) => data[(y * width + x) * stride + c0..].as_ptr(),
            None => ZEROS.as_ptr(),
        }
    }

    fn check(&self, rows: &Range<usize>, k: usize) {
        let (data, height, width, channels, stride, taps) = match *self {
            Input::Rows { data, stride } => {
                assert!(
                    rows.end == 0 || (rows.end - 1) * stride + k <= data.len(),
                    "rows out of bounds"
                );
                return;
            }
            Input::Image {
                data,
                height,
                width,
                channels,
                stride,
                taps,
            } => (data, height, width, channels, stride, taps),
        };
        let pixels = height * width;
        let (out_height, out_width) = self.out_size();
        assert_eq!(
            self.taps().0 * channels,
            k,
            "input does not match the weights"
        );
        assert!(
            stride >= channels
                && rows.end <= out_height * out_width
                && pixels > 0
                && data.len() >= (pixels - 1) * stride + channels
        );
        match taps {
            Taps::Conv { size, step, pad } => assert!(
                size % 2 == 1
                    && step > 0
                    && (pad == Pad::Zero || (size / 2 < height && size / 2 < width))
            ),
            Taps::Offsets { count, .. } => assert!(count <= 4),
        }
    }
}

/// What happens to a result before it is stored.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Act {
    None,
    Relu,
    Gelu,
}

/// Pointers of the six rows of each group and tap of one reduction block.
#[derive(Default)]
pub struct GemmScratch {
    ptrs: Vec<*const f32>,
    /// `Input::position` of each row of the call, and `Input::tap` of each
    /// tap of the block.
    positions: Vec<(usize, usize)>,
    taps: Vec<(usize, usize)>,
}

// SAFETY: the pointers are only meaningful during the `multiply` call that
// writes them; a scratch carries no live borrow between calls.
unsafe impl Send for GemmScratch {}

/// `out[r] = act(input[r] * w + bias)` for the rows `rows`, or with
/// `accumulate`, `out[r] = act(out[r] + input[r] * w + bias)`. Row `r` of the
/// result is `out[(r - rows.start) * stride..][..w.n]`.
#[expect(
    clippy::too_many_arguments,
    reason = "a product's input, weights and output layout"
)]
pub fn multiply(
    input: Input,
    rows: Range<usize>,
    w: &Matrix,
    out: &mut [f32],
    stride: usize,
    act: Act,
    accumulate: bool,
    scratch: &mut GemmScratch,
) {
    let m = rows.len();
    if m == 0 {
        return;
    }
    input.check(&rows, w.k);
    assert!(
        stride >= w.n && out.len() >= (m - 1) * stride + w.n,
        "output too small"
    );
    let (taps, channels) = input.taps();
    let channels = channels.min(w.k);
    // A block is several whole taps of a narrow layer or part of one tap.
    let (taps_per_block, chunk) = if channels >= KC {
        (1, KC)
    } else {
        (KC / channels, channels)
    };
    let groups = m.div_ceil(MR);
    let panels = w.n.div_ceil(NR);
    let kernels = kernels(matches!(w.panels, Panels::F16(_)));
    let mut tile = [0.0f32; 2 * MR * NR];
    // Each row's position, once per call.
    scratch.positions.clear();
    scratch
        .positions
        .extend((0..m).map(|i| input.position(rows.start + i)));
    let mut t0 = 0;
    let mut first = true;
    while t0 < taps {
        let t1 = (t0 + taps_per_block).min(taps);
        let mut c0 = 0;
        while c0 < channels {
            let c1 = (c0 + chunk).min(channels);
            let last = t1 == taps && c1 == channels;
            let block_taps = t1 - t0;
            if c0 == 0 {
                scratch.ptrs.clear();
                scratch.taps.clear();
                scratch.taps.extend((t0..t1).map(|t| input.tap(t)));
                for g in 0..groups {
                    for &tap in &scratch.taps {
                        for i in 0..MR {
                            let position = scratch.positions[(g * MR + i).min(m - 1)];
                            scratch.ptrs.push(input.at(position, tap, c0));
                        }
                    }
                }
            } else {
                // The same taps' next chunk of channels: each pointer moves on
                // by the chunk, and padding stays on the zeros.
                for p in &mut scratch.ptrs {
                    if *p != ZEROS.as_ptr() {
                        // SAFETY: `p` points at channel `c0 - chunk` of a pixel
                        // with `channels` readable floats (`Input::check`), and
                        // `c0 < channels`.
                        *p = unsafe { p.add(chunk) };
                    }
                }
            }
            let relu = last && act == Act::Relu;
            for p in 0..panels {
                let b = w.panel_row(p, t0 * channels + c0);
                let bias = if first {
                    w.bias[p * NR..].as_ptr()
                } else {
                    std::ptr::null()
                };
                let cols = (w.n - p * NR).min(NR);
                let kernel = if cols <= NR / 2 {
                    kernels.narrow
                } else {
                    kernels.full
                };
                let load = accumulate || !first;
                // A full panel writes its full row groups in place, in one call.
                let full = if cols == NR { m / MR } else { 0 };
                if full > 0 {
                    let call = KernelCall {
                        taps: block_taps,
                        channels: c1 - c0,
                        a: scratch.ptrs.as_ptr(),
                        b,
                        bias,
                        load,
                        relu,
                        groups: full,
                    };
                    // SAFETY: rows 0..full*6 and columns p*16..p*16+16 are
                    // inside `out` (asserted above with cols == 16); the
                    // pointers of group g start at g*block_taps*6 and point to
                    // at least `channels` readable floats per tap
                    // (`Input::check`), and `b` holds block_taps*channels rows
                    // of 16 floats of this panel.
                    unsafe { kernel(&call, out[p * NR..].as_mut_ptr(), stride) };
                }
                // The rest through a tile of whole 6 x 16 groups: two at a
                // time for a narrow panel, whose kernel then keeps twelve
                // sums in flight.
                let mut g = full;
                while g < groups {
                    let pair = if cols <= NR / 2 && g + 1 < groups {
                        2
                    } else {
                        1
                    };
                    let ptrs = &scratch.ptrs[g * block_taps * MR..][..pair * block_taps * MR];
                    let valid = (m - g * MR).min(pair * MR);
                    let base = g * MR * stride + p * NR;
                    let call = KernelCall {
                        taps: block_taps,
                        channels: c1 - c0,
                        a: ptrs.as_ptr(),
                        b,
                        bias,
                        load,
                        relu,
                        groups: pair,
                    };
                    if load {
                        for i in 0..valid {
                            tile[i * NR..][..cols]
                                .copy_from_slice(&out[base + i * stride..][..cols]);
                        }
                    }
                    // SAFETY: as above, for `pair` groups, with the tile as
                    // their 6 x 16 outputs.
                    unsafe { kernel(&call, tile.as_mut_ptr(), NR) };
                    for i in 0..valid {
                        out[base + i * stride..][..cols].copy_from_slice(&tile[i * NR..][..cols]);
                    }
                    g += pair;
                }
            }
            first = false;
            c0 = c1;
        }
        t0 = t1;
    }
    if act == Act::Gelu {
        for r in 0..m {
            crate::math::gelu(&mut out[r * stride..][..w.n]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::kernel::*;
    use super::*;

    fn values(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.5)
            .collect()
    }

    /// Textbook product in f64, the oracle for every path.
    fn reference(
        a: impl Fn(usize, usize) -> f32,
        m: usize,
        k: usize,
        n: usize,
        w: &[f32],
        bias: &[f32],
    ) -> Vec<f64> {
        let mut out = vec![0.0; m * n];
        for r in 0..m {
            for j in 0..n {
                let mut s = f64::from(bias[j]);
                for x in 0..k {
                    s += f64::from(a(r, x)) * f64::from(w[x * n + j]);
                }
                out[r * n + j] = s;
            }
        }
        out
    }

    fn close(got: &[f32], want: &[f64], stride: usize, n: usize) {
        for (r, row) in want.chunks(n).enumerate() {
            for (j, &v) in row.iter().enumerate() {
                let g = f64::from(got[r * stride + j]);
                assert!(
                    (g - v).abs() <= 1e-4 * (1.0 + v.abs()),
                    "row {r} col {j}: {g} vs {v}"
                );
            }
        }
    }

    /// Each kernel this CPU can run, against the portable one, on one tile
    /// with loading, bias and ReLU, over several taps.
    #[test]
    fn kernels_agree_on_a_tile() {
        let (taps, channels) = (3, 37);
        let a = values(MR * taps * channels, 0.2);
        let b = values(taps * channels * NR, 0.6);
        let bias = values(NR, 1.3);
        let ptrs: Vec<*const f32> = (0..taps)
            .flat_map(|t| (0..MR).map(move |i| (t * MR + i) * channels))
            .map(|at| a[at..].as_ptr())
            .collect();
        let call = KernelCall {
            taps,
            channels,
            a: ptrs.as_ptr(),
            b: b.as_ptr().cast(),
            bias: bias.as_ptr(),
            load: true,
            relu: true,
            groups: 1,
        };
        let start = values(MR * NR, 2.4);
        let run = |kernel: Kernel| {
            let mut c = start.clone();
            // SAFETY: the tile, pointers and weights above have the sizes
            // `KernelCall` describes.
            unsafe { kernel(&call, c.as_mut_ptr(), NR) };
            c
        };
        let want = run(kernel_portable);
        let mut kernels: Vec<Kernel> = Vec::new();
        #[cfg(target_arch = "x86_64")]
        {
            kernels.push(kernel_sse::<2>);
            if crate::simd_tier() >= crate::TIER_AVX {
                kernels.push(kernel_avx);
            }
            if crate::simd_tier() >= crate::TIER_AVX2 {
                kernels.push(kernel_avx2);
            }
        }
        for kernel in kernels {
            for (g, w) in run(kernel).iter().zip(&want) {
                assert!((g - w).abs() <= 1e-5 * (1.0 + w.abs()), "{g} vs {w}");
            }
        }
        // The narrow kernels write the first 8 columns of each row alone.
        let mut narrow: Vec<Kernel> = Vec::new();
        #[cfg(target_arch = "x86_64")]
        {
            narrow.push(kernel_sse::<1>);
            if crate::simd_tier() >= crate::TIER_AVX {
                narrow.push(kernel_avx_narrow);
            }
            if crate::simd_tier() >= crate::TIER_AVX2 {
                narrow.push(kernel_avx2_narrow);
            }
        }
        for kernel in narrow {
            let got = run(kernel);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                let expected = if i % NR < NR / 2 { *w } else { start[i] };
                assert!(
                    (g - expected).abs() <= 1e-5 * (1.0 + expected.abs()),
                    "{i}: {g} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn half_panels_match_reference() {
        if crate::simd_tier() < crate::TIER_AVX2 {
            return;
        }
        let (m, k, n) = (11, 600, 40);
        let a = values(m * k, 0.1);
        let half: Vec<u16> = (0..k * n).map(|i| 0x3000 + (i % 977) as u16).collect();
        let w: Vec<f32> = half.iter().map(|&h| half_to_f32(h)).collect();
        let bias = values(n, 0.5);
        let matrix = Matrix::new(&Weights::F16(half), k, n, Some(&bias));
        let mut out = vec![0.0f32; m * n];
        multiply(
            Input::Rows {
                data: &a,
                stride: k,
            },
            0..m,
            &matrix,
            &mut out,
            n,
            Act::None,
            false,
            &mut GemmScratch::default(),
        );
        close(
            &out,
            &reference(|r, x| a[r * k + x], m, k, n, &w, &bias),
            n,
            n,
        );
    }

    #[test]
    fn dense_rows_match_reference() {
        // Sizes cover partial row groups, partial panels and several k blocks.
        for &(m, k, n) in &[(1, 3, 1), (7, 40, 17), (13, 1100, 48), (6, 512, 16)] {
            let a = values(m * k, 0.3);
            let w = values(k * n, 1.1);
            let bias = values(n, 2.0);
            let matrix = Matrix::new(&Weights::F32(w.clone()), k, n, Some(&bias));
            let stride = n + 3;
            let mut out = vec![f32::NAN; (m - 1) * stride + n];
            let mut scratch = GemmScratch::default();
            multiply(
                Input::Rows {
                    data: &a,
                    stride: k,
                },
                0..m,
                &matrix,
                &mut out,
                stride,
                Act::None,
                false,
                &mut scratch,
            );
            close(
                &out,
                &reference(|r, x| a[r * k + x], m, k, n, &w, &bias),
                stride,
                n,
            );
        }
    }

    #[test]
    fn row_ranges_accumulate_onto_output() {
        let (m, k, n) = (20, 70, 33);
        let a = values(m * k, 0.7);
        let w = values(k * n, -0.4);
        let matrix = Matrix::new(&Weights::F32(w.clone()), k, n, None);
        let want = reference(|r, x| a[r * k + x], m, k, n, &w, &vec![0.0; n]);
        let mut out = vec![1.0f32; m * n];
        let mut scratch = GemmScratch::default();
        for range in [0..5, 5..17, 17..20] {
            let part = &mut out[range.start * n..range.end * n];
            multiply(
                Input::Rows {
                    data: &a,
                    stride: k,
                },
                range,
                &matrix,
                part,
                n,
                Act::None,
                true,
                &mut scratch,
            );
        }
        let shifted: Vec<f64> = want.iter().map(|v| v + 1.0).collect();
        close(&out, &shifted, n, n);
    }

    #[test]
    fn same_convolution_matches_reference() {
        // The fourth value is unused channels between pixels.
        for &(h, wd, c, gap, size, n) in &[
            (5, 7, 3, 0, 3, 20),
            (4, 9, 64, 0, 7, 16),
            (3, 3, 600, 0, 3, 5),
            (6, 5, 2, 0, 1, 3),
            (5, 6, 32, 32, 3, 32),
        ] {
            let stride = c + gap;
            let img = values(h * wd * stride, 0.9);
            let k = size * size * c;
            let w = values(k * n, 0.2);
            let bias = values(n, -1.0);
            let matrix = Matrix::new(&Weights::F32(w.clone()), k, n, Some(&bias));
            let half = (size / 2) as isize;
            let pixel = |r: usize, x: usize| {
                let (tap, ch) = (x / c, x % c);
                let y = (r / wd) as isize + (tap / size) as isize - half;
                let xx = (r % wd) as isize + (tap % size) as isize - half;
                if y < 0 || xx < 0 || y >= h as isize || xx >= wd as isize {
                    0.0
                } else {
                    img[(y as usize * wd + xx as usize) * stride + ch]
                }
            };
            let mut out = vec![0.0f32; h * wd * n];
            let mut scratch = GemmScratch::default();
            let input = Input::conv(&img, h, wd, c, stride, size);
            multiply(
                input,
                0..h * wd,
                &matrix,
                &mut out,
                n,
                Act::Relu,
                false,
                &mut scratch,
            );
            let want: Vec<f64> = reference(pixel, h * wd, k, n, &w, &bias)
                .iter()
                .map(|v| v.max(0.0))
                .collect();
            close(&out, &want, n, n);
        }
    }
}
