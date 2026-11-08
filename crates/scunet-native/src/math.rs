//! The engine's elementwise functions, and attention over one 8 x 8 window
//! of the Swin blocks.

pub use native_engine::attention::HEAD_DIM;
pub use native_engine::math::*;

/// Tokens of an 8 x 8 window.
pub const WINDOW_TOKENS: usize = 64;
/// Floats of workspace [`attention`] needs.
pub const ATTENTION_WORK: usize = native_engine::attention::work_len(WINDOW_TOKENS);

/// [`native_engine::attention::attention`] over a window of 64 tokens.
pub fn attention(
    qkv: &[f32],
    c: usize,
    head: usize,
    bias: &[f32],
    work: &mut [f32],
    out: &mut [f32],
) {
    native_engine::attention::attention::<WINDOW_TOKENS>(qkv, c, head, bias, work, out);
}
