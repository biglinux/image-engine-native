//! The 2-D real discrete Fourier transform of LaMa's Fourier units: PyTorch's
//! `rfftn` and `irfftn` over the two spatial axes with "ortho" scaling, on
//! NHWC maps, by fast Fourier transforms over whole pixels: every operation
//! runs on all channels of a pixel at once, in vector registers.
//!
//! A spectrum has `width / 2 + 1` columns, and its pixels hold the real
//! parts of their channels, then the imaginary parts. The inverse reads only
//! the real part of the first column and, for an even width, of the last, as
//! `irfftn` does.
//!
//! Rows are real, so two of them go through one complex transform, one as
//! the real part and one as the imaginary part, and are told apart by the
//! symmetry of a real signal's spectrum. Columns of the spectrum are complex
//! and take a transform each.

use crate::pool::{Pool, Shared};

/// A complex transform of `n` points, `out[k]` the sum over `j` of
/// `x[j] * exp(sign 2 pi i j k / n)`, each point a pixel of complex
/// channels. Mixed-radix decimation in time: radix 4 while it divides `n`,
/// then 2, then each odd prime factor by its own direct sum.
struct Fft {
    n: usize,
    radices: Vec<usize>,
    /// `exp(2 pi i j / n)` for `j` in `0..n`, computed in double precision.
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Fft {
    fn new(n: usize) -> Self {
        let mut radices = Vec::new();
        let mut left = n;
        for p in [4, 2] {
            while left.is_multiple_of(p) {
                radices.push(p);
                left /= p;
            }
        }
        let mut p = 3;
        while left > 1 {
            while left.is_multiple_of(p) {
                radices.push(p);
                left /= p;
            }
            p += 2;
        }
        let angle = |j: usize| std::f64::consts::TAU * j as f64 / n as f64;
        Self {
            n,
            radices,
            cos: (0..n).map(|j| angle(j).cos() as f32).collect(),
            sin: (0..n).map(|j| angle(j).sin() as f32).collect(),
        }
    }

    /// Floats of `temp` [`Fft::run`] needs for pixels of `pixel` floats.
    fn temp_len(&self, pixel: usize) -> usize {
        2 * self.radices.iter().max().unwrap_or(&1) * pixel
    }

    /// The transform of the `n` pixels of `x` into `out`.
    fn run(&self, x: &[f32], sign: f32, pixel: usize, out: &mut [f32], temp: &mut [f32]) {
        self.step(
            x,
            1,
            &self.radices,
            sign,
            pixel,
            &mut out[..self.n * pixel],
            temp,
        );
    }

    /// The transform of points `x[j * stride]`, `j` in `0..out.len() / pixel`,
    /// with the radices left.
    #[expect(clippy::too_many_arguments, reason = "the recursion's state")]
    fn step(
        &self,
        x: &[f32],
        stride: usize,
        radices: &[usize],
        sign: f32,
        pixel: usize,
        out: &mut [f32],
        temp: &mut [f32],
    ) {
        let n = out.len() / pixel;
        let Some((&p, rest)) = radices.split_first() else {
            out.copy_from_slice(&x[..pixel]);
            return;
        };
        let m = n / p;
        for (q, part) in out.chunks_exact_mut(m * pixel).enumerate() {
            self.step(
                &x[q * stride * pixel..],
                stride * p,
                rest,
                sign,
                pixel,
                part,
                temp,
            );
        }
        // Each output k + t m of this step combines the p sub-transforms at
        // k, the q-th turned by exp(sign 2 pi i q k / n).
        let (inputs, outputs) = temp[..2 * p * pixel].split_at_mut(p * pixel);
        let turn = self.n / n;
        for k in 0..m {
            for (q, input) in inputs.chunks_exact_mut(pixel).enumerate() {
                let j = q * k * turn % self.n;
                rotate(
                    &out[(q * m + k) * pixel..][..pixel],
                    self.cos[j],
                    sign * self.sin[j],
                    input,
                );
            }
            match p {
                2 => radix2(inputs, outputs),
                4 => radix4(inputs, sign, outputs),
                _ => {
                    // exp(sign 2 pi i q t / p) is entry (q t mod p) n / p of
                    // the table.
                    for (t, output) in outputs.chunks_exact_mut(pixel).enumerate() {
                        output.fill(0.0);
                        for (q, input) in inputs.chunks_exact(pixel).enumerate() {
                            let j = (q * t % p) * (self.n / p);
                            turn_add(input, self.cos[j], sign * self.sin[j], output);
                        }
                    }
                }
            }
            for (t, output) in outputs.chunks_exact(pixel).enumerate() {
                out[(k + t * m) * pixel..][..pixel].copy_from_slice(output);
            }
        }
    }
}

dispatch! {
    /// `out = x * (c + i s)` for every channel of a pixel.
    rotate(x: &[f32], c: f32, s: f32, out: &mut [f32]) {
        let half = x.len() / 2;
        let (re, im) = x.split_at(half);
        let (o_re, o_im) = out.split_at_mut(half);
        for (((a, b), r), i) in o_re.iter_mut().zip(o_im.iter_mut()).zip(re).zip(im) {
            *a = r * c - i * s;
            *b = r * s + i * c;
        }
    }
}

dispatch! {
    /// `out += x * (c + i s)` for every channel of a pixel.
    turn_add(x: &[f32], c: f32, s: f32, out: &mut [f32]) {
        let half = x.len() / 2;
        let (re, im) = x.split_at(half);
        let (o_re, o_im) = out.split_at_mut(half);
        for (((a, b), r), i) in o_re.iter_mut().zip(o_im.iter_mut()).zip(re).zip(im) {
            *a += r * c - i * s;
            *b += r * s + i * c;
        }
    }
}

dispatch! {
    /// The 2-point transform of the two pixels of `x`.
    radix2(x: &[f32], out: &mut [f32]) {
        let (a, b) = x.split_at(x.len() / 2);
        let (o0, o1) = out.split_at_mut(a.len());
        for (((o0, o1), a), b) in o0.iter_mut().zip(o1.iter_mut()).zip(a).zip(b) {
            *o0 = a + b;
            *o1 = a - b;
        }
    }
}

dispatch! {
    /// The 4-point transform of the four pixels of `x`, whose turn of a
    /// quarter is `sign i`.
    radix4(x: &[f32], sign: f32, out: &mut [f32]) {
        let pixel = x.len() / 4;
        let half = pixel / 2;
        let at = |q: usize| x[q * pixel..][..pixel].split_at(half);
        let ((a0r, a0i), (a1r, a1i), (a2r, a2i), (a3r, a3i)) = (at(0), at(1), at(2), at(3));
        let (y0, rest) = out.split_at_mut(pixel);
        let (y1, rest) = rest.split_at_mut(pixel);
        let (y2, y3) = rest.split_at_mut(pixel);
        let ((y0r, y0i), (y1r, y1i)) = (y0.split_at_mut(half), y1.split_at_mut(half));
        let ((y2r, y2i), (y3r, y3i)) = (y2.split_at_mut(half), y3.split_at_mut(half));
        for c in 0..half {
            let (b0r, b0i) = (a0r[c] + a2r[c], a0i[c] + a2i[c]);
            let (b1r, b1i) = (a0r[c] - a2r[c], a0i[c] - a2i[c]);
            let (b2r, b2i) = (a1r[c] + a3r[c], a1i[c] + a3i[c]);
            let (b3r, b3i) = (a1r[c] - a3r[c], a1i[c] - a3i[c]);
            y0r[c] = b0r + b2r;
            y0i[c] = b0i + b2i;
            y2r[c] = b0r - b2r;
            y2i[c] = b0i - b2i;
            y1r[c] = b1r - sign * b3i;
            y1i[c] = b1i + sign * b3r;
            y3r[c] = b1r + sign * b3i;
            y3i[c] = b1i - sign * b3r;
        }
    }
}

dispatch! {
    /// Spectrum pixels `k` of two real rows `a` and `b` from pixels `k` and
    /// `n - k` of the transform of `a + i b`: `a` gets `(z + conj z') / 2`
    /// and `b` gets `(z - conj z') / 2i`.
    split(z: &[f32], z_mirror: &[f32], a: &mut [f32], b: &mut [f32]) {
        let half = z.len() / 2;
        let ((zr, zi), (mr, mi)) = (z.split_at(half), z_mirror.split_at(half));
        let ((ar, ai), (br, bi)) = (a.split_at_mut(half), b.split_at_mut(half));
        for c in 0..half {
            ar[c] = 0.5 * (zr[c] + mr[c]);
            ai[c] = 0.5 * (zi[c] - mi[c]);
            br[c] = 0.5 * (zi[c] + mi[c]);
            bi[c] = 0.5 * (mr[c] - zr[c]);
        }
    }
}

dispatch! {
    /// Pixel `k` of the transform of `a + i b`, from the spectrum pixels of
    /// two real rows at `k`, conjugated when `mirror` (`k` past the half the
    /// spectrum stores), with imaginary parts read as zero when `real`.
    merge(a: &[f32], b: &[f32], mirror: bool, real: bool, z: &mut [f32]) {
        let half = z.len() / 2;
        let ((ar, ai), (br, bi)) = (a.split_at(half), b.split_at(half));
        let (zr, zi) = z.split_at_mut(half);
        let sign = if mirror { -1.0 } else { 1.0 };
        let keep = if real { 0.0 } else { sign };
        for c in 0..half {
            zr[c] = ar[c] - keep * bi[c];
            zi[c] = keep * ai[c] + br[c];
        }
    }
}

dispatch! {
    /// `x *= scale`.
    scaled(x: &mut [f32], scale: f32) {
        for v in x.iter_mut() {
            *v *= scale;
        }
    }
}

dispatch! {
    /// `x += scale y`.
    add_scaled(x: &mut [f32], y: &[f32], scale: f32) {
        for (x, y) in x.iter_mut().zip(y) {
            *x += scale * y;
        }
    }
}

/// The transforms for maps of one size.
pub struct Fourier {
    height: usize,
    width: usize,
    rows: Fft,
    columns: Fft,
}

impl Fourier {
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            height,
            width,
            rows: Fft::new(width),
            columns: Fft::new(height),
        }
    }

    /// Rows and columns of the maps.
    pub fn size(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    /// Columns of a spectrum.
    pub fn half(&self) -> usize {
        self.width / 2 + 1
    }

    /// Spectrum of `x`, a `height x width` map of `channels` channels, into
    /// `out`, using `work` of the same size.
    pub fn forward(
        &self,
        pool: &mut Pool,
        x: &[f32],
        channels: usize,
        work: &mut [f32],
        out: &mut [f32],
    ) {
        let (h, w, half) = (self.height, self.width, self.half());
        let pixel = 2 * channels;
        let row = pixel * half;
        let rows = Shared::new(work);
        pool.split_by(h.div_ceil(2), 1, 2, |pairs, scratch| {
            let temp_len = self.rows.temp_len(pixel);
            let (buf, _) = scratch.floats(2 * w * pixel + temp_len);
            let (packed, rest) = buf.split_at_mut(w * pixel);
            let (z, temp) = rest.split_at_mut(w * pixel);
            for pair in pairs {
                let y = 2 * pair;
                let second = y + 1 < h;
                for (xx, p) in packed.chunks_exact_mut(pixel).enumerate() {
                    let (re, im) = p.split_at_mut(channels);
                    re.copy_from_slice(&x[(y * w + xx) * channels..][..channels]);
                    if second {
                        im.copy_from_slice(&x[((y + 1) * w + xx) * channels..][..channels]);
                    } else {
                        im.fill(0.0);
                    }
                }
                self.rows.run(packed, -1.0, pixel, z, temp);
                // SAFETY: tasks own disjoint pairs of rows.
                let a = unsafe { rows.part(y * row, row) };
                if !second {
                    // A lone row went in as the real part alone.
                    a.copy_from_slice(&z[..row]);
                    continue;
                }
                // SAFETY: as above.
                let b = unsafe { rows.part((y + 1) * row, row) };
                for k in 0..half {
                    let mirror = (w - k) % w;
                    split(
                        &z[k * pixel..][..pixel],
                        &z[mirror * pixel..][..pixel],
                        &mut a[k * pixel..][..pixel],
                        &mut b[k * pixel..][..pixel],
                    );
                }
            }
        });
        let work = &*work;
        self.columns(pool, work, channels, -1.0, out);
    }

    /// The map whose spectrum is `spectrum`, added to `out` (`height x width`
    /// of `channels`), using `work` of the spectrum's size.
    pub fn inverse(
        &self,
        pool: &mut Pool,
        spectrum: &[f32],
        channels: usize,
        work: &mut [f32],
        out: &mut [f32],
    ) {
        let (h, w, half) = (self.height, self.width, self.half());
        let pixel = 2 * channels;
        let row = pixel * half;
        self.columns(pool, spectrum, channels, 1.0, work);
        let work = &*work;
        let scale = 1.0 / ((h * w) as f32).sqrt();
        let dst = Shared::new(out);
        pool.split_by(h.div_ceil(2), 1, 2, |pairs, scratch| {
            let temp_len = self.rows.temp_len(pixel);
            let (buf, _) = scratch.floats(2 * w * pixel + pixel + temp_len);
            let (z, rest) = buf.split_at_mut(w * pixel);
            let (rows, rest) = rest.split_at_mut(w * pixel);
            let (zeros, temp) = rest.split_at_mut(pixel);
            zeros.fill(0.0);
            let zeros = &*zeros;
            for pair in pairs {
                let y = 2 * pair;
                let second = y + 1 < h;
                let a = &work[y * row..][..row];
                for (k, p) in z.chunks_exact_mut(pixel).enumerate() {
                    let (j, mirror) = if k < half { (k, false) } else { (w - k, true) };
                    // The first column and, for an even width, the last hold
                    // real values.
                    let real = j == 0 || 2 * j == w;
                    let b = if second {
                        &work[(y + 1) * row + j * pixel..][..pixel]
                    } else {
                        zeros
                    };
                    merge(&a[j * pixel..][..pixel], b, mirror, real, p);
                }
                self.rows.run(z, 1.0, pixel, rows, temp);
                for (xx, p) in rows.chunks_exact(pixel).enumerate() {
                    let (re, im) = p.split_at(channels);
                    // SAFETY: tasks own disjoint pairs of rows.
                    let first = unsafe { dst.part((y * w + xx) * channels, channels) };
                    add_scaled(first, re, scale);
                    if second {
                        // SAFETY: as above.
                        let next = unsafe { dst.part(((y + 1) * w + xx) * channels, channels) };
                        add_scaled(next, im, scale);
                    }
                }
            }
        });
    }

    /// The complex transform along the columns of a spectrum, with
    /// `exp(sign 2 pi i ...)`; the forward direction also applies the whole
    /// "ortho" scale. A task copies a column in, transforms it in cache, and
    /// copies its transform back.
    fn columns(&self, pool: &mut Pool, src: &[f32], channels: usize, sign: f32, out: &mut [f32]) {
        let (h, half) = (self.height, self.half());
        let pixel = 2 * channels;
        let scale = if sign < 0.0 {
            1.0 / ((self.height * self.width) as f32).sqrt()
        } else {
            1.0
        };
        let dst = Shared::new(out);
        pool.split_by(half, 1, 2, |range, scratch| {
            let temp_len = self.columns.temp_len(pixel);
            let (buf, _) = scratch.floats(2 * h * pixel + temp_len);
            let (gathered, rest) = buf.split_at_mut(h * pixel);
            let (transformed, temp) = rest.split_at_mut(h * pixel);
            for k in range {
                for (y, p) in gathered.chunks_exact_mut(pixel).enumerate() {
                    p.copy_from_slice(&src[(y * half + k) * pixel..][..pixel]);
                }
                self.columns.run(gathered, sign, pixel, transformed, temp);
                for (ky, p) in transformed.chunks_exact(pixel).enumerate() {
                    // SAFETY: tasks own disjoint columns.
                    let o = unsafe { dst.part((ky * half + k) * pixel, pixel) };
                    o.copy_from_slice(p);
                    if scale != 1.0 {
                        scaled(o, scale);
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(len: usize, seed: f64) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f64 * 0.7 + seed).sin() * 2.0) as f32)
            .collect()
    }

    #[test]
    fn spectrum_matches_definition() {
        let mut pool = Pool::new(3, "test").expect("worker threads");
        for &(h, w, c) in &[(4, 6, 3), (5, 7, 2), (16, 12, 2), (9, 10, 2), (1, 8, 1)] {
            let x = values(h * w * c, 0.3);
            let fourier = Fourier::new(h, w);
            let half = fourier.half();
            let mut work = vec![0.0; h * half * 2 * c];
            let mut spectrum = vec![0.0; h * half * 2 * c];
            fourier.forward(&mut pool, &x, c, &mut work, &mut spectrum);
            let norm = 1.0 / ((h * w) as f64).sqrt();
            for ky in 0..h {
                for k in 0..half {
                    for ch in 0..c {
                        let (mut re, mut im) = (0.0f64, 0.0f64);
                        for y in 0..h {
                            for xx in 0..w {
                                let angle = -std::f64::consts::TAU
                                    * ((ky * y) as f64 / h as f64 + (k * xx) as f64 / w as f64);
                                let v = f64::from(x[(y * w + xx) * c + ch]);
                                re += v * angle.cos();
                                im += v * angle.sin();
                            }
                        }
                        let at = (ky * half + k) * 2 * c;
                        assert!((f64::from(spectrum[at + ch]) - re * norm).abs() < 1e-5);
                        assert!((f64::from(spectrum[at + c + ch]) - im * norm).abs() < 1e-5);
                    }
                }
            }
            let mut back = vec![1.0; h * w * c];
            fourier.inverse(&mut pool, &spectrum, c, &mut work, &mut back);
            for (b, v) in back.iter().zip(&x) {
                assert!((b - 1.0 - v).abs() < 1e-5, "{b} vs 1 + {v}");
            }
        }
    }
}
