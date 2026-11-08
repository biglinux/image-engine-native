//! The weights file as the GPU kernels read it: weight matrices in half
//! floats, padded to whole blocks of 64 columns, and everything else in
//! floats, all in one buffer the layers point into.

use std::path::Path;

use native_engine::weight_file::WeightFile;
use native_vulkan::Words;

use crate::Error;
use crate::decoder::{BRANCH, FullResolution, INTER, LEVELS, SIZES};
use crate::math::HEAD_DIM;
use crate::swin::{STAGES, bias_tables};

/// Parameters: weight matrices in half floats, the rest in floats, each
/// array starting at an even offset in half floats.
#[derive(Default)]
pub(super) struct Params(pub(super) Words);

impl Params {
    /// Floats, at the returned offset in floats.
    fn put(&mut self, values: &[f32]) -> usize {
        let words: Vec<u16> = values
            .iter()
            .flat_map(|v| {
                let bits = v.to_bits();
                [bits as u16, (bits >> 16) as u16]
            })
            .collect();
        let n = words.len();
        self.0.push(words, n, n) / 2
    }

    /// A `[k][n]` matrix in half floats, its rows padded to a multiple of 64,
    /// at the returned offset in half floats.
    fn matrix(&mut self, values: Vec<u16>, n: usize) -> usize {
        self.0.push(values, n, n.next_multiple_of(64))
    }
}

/// A product's weights `[k][n]` and bias, as offsets in half floats and
/// floats.
pub(super) struct Linear {
    pub(super) k: usize,
    pub(super) n: usize,
    pub(super) weights: usize,
    pub(super) bias: Option<usize>,
}

impl Linear {
    fn load(
        file: &mut WeightFile,
        params: &mut Params,
        name: &str,
        k: usize,
        n: usize,
        bias: bool,
    ) -> Result<Self, Error> {
        let w = file
            .take_weights(&format!("{name}.weight"), &[k, n])?
            .into_f16();
        let bias = if bias {
            Some(file.take(&format!("{name}.bias"), &[n])?)
        } else {
            None
        };
        Ok(Self::new(params, w, k, n, bias.as_deref()))
    }

    fn new(params: &mut Params, w: Vec<u16>, k: usize, n: usize, bias: Option<&[f32]>) -> Self {
        Self {
            k,
            n,
            weights: params.matrix(w, n),
            bias: bias.map(|b| params.put(b)),
        }
    }
}

/// A layer normalisation's scale and shift.
#[derive(Clone, Copy)]
pub(super) struct Norm {
    pub(super) gamma: usize,
    pub(super) beta: usize,
}

impl Norm {
    fn load(
        file: &mut WeightFile,
        params: &mut Params,
        name: &str,
        c: usize,
    ) -> Result<Self, Error> {
        Ok(Self {
            gamma: params.put(&file.take(&format!("{name}.weight"), &[c])?),
            beta: params.put(&file.take(&format!("{name}.bias"), &[c])?),
        })
    }
}

pub(super) struct SwinBlock {
    pub(super) norm1: Norm,
    pub(super) qkv: Linear,
    /// `swin::bias_tables` of the block.
    pub(super) bias: usize,
    pub(super) proj: Linear,
    pub(super) norm2: Norm,
    pub(super) fc1: Linear,
    pub(super) fc2: Linear,
}

pub(super) struct Stage {
    pub(super) channels: usize,
    pub(super) blocks: Vec<SwinBlock>,
    pub(super) norm: Norm,
    pub(super) merge: Option<(Norm, Linear)>,
}

pub(super) struct Branch {
    pub(super) size: usize,
    pub(super) offsets: Linear,
    pub(super) weight: Linear,
}

pub(super) struct Aspp {
    pub(super) branches: Vec<Branch>,
    pub(super) pool: Linear,
    /// The 1x1 convolution's rows for the four deformable branches.
    pub(super) conv1: Linear,
    /// Its rows for the pooling branch, with its bias.
    pub(super) conv1_pool: Linear,
}

pub(super) struct DecoderBlock {
    pub(super) conv_in: Linear,
    pub(super) aspp: Aspp,
    pub(super) conv_out: Linear,
    /// A 3x3 convolution to 16 channels and a 1x1 one to the gate's logit.
    pub(super) gate: Option<(Linear, Linear)>,
}

pub(super) struct Patches {
    pub(super) conv1: Linear,
    pub(super) conv_out: Linear,
}

/// The layers, as offsets into the parameters.
pub(super) struct Layers {
    pub(super) embed: Linear,
    pub(super) embed_norm: Norm,
    pub(super) stages: Vec<Stage>,
    pub(super) squeeze: DecoderBlock,
    pub(super) blocks: Vec<DecoderBlock>,
    pub(super) laterals: Vec<Linear>,
    pub(super) patches: Vec<Patches>,
    pub(super) head: Linear,
    /// `full.comp`'s parameters.
    pub(super) full: usize,
}

impl Layers {
    /// The layers of the weights file at `path` and the parameters they
    /// point into.
    pub(super) fn load(path: &Path) -> Result<(Self, Params), Error> {
        let mut file = WeightFile::read(path, b"BIREFNET")?;
        let mut params = Params::default();
        let p = &mut params;
        let f = &mut file;

        let embed = Linear::load(f, p, "patch_embed", 48, 96, true)?;
        let embed_norm = Norm::load(f, p, "patch_embed.norm", 96)?;
        let mut stages = Vec::new();
        for (s, &(c, depth)) in STAGES.iter().enumerate() {
            let heads = c / HEAD_DIM;
            let mut blocks = Vec::new();
            for b in 0..depth {
                let n = format!("stage{s}.block{b}");
                let position = f.take(&format!("{n}.position_bias"), &[heads, 49, 49])?;
                blocks.push(SwinBlock {
                    norm1: Norm::load(f, p, &format!("{n}.norm1"), c)?,
                    qkv: Linear::load(f, p, &format!("{n}.qkv"), c, 3 * c, true)?,
                    bias: p.put(&bias_tables(position, b % 2 == 1)),
                    proj: Linear::load(f, p, &format!("{n}.proj"), c, c, true)?,
                    norm2: Norm::load(f, p, &format!("{n}.norm2"), c)?,
                    fc1: Linear::load(f, p, &format!("{n}.fc1"), c, 4 * c, true)?,
                    fc2: Linear::load(f, p, &format!("{n}.fc2"), 4 * c, c, true)?,
                });
            }
            let merge = if s < 3 {
                Some((
                    Norm::load(f, p, &format!("stage{s}.merge.norm"), 4 * c)?,
                    Linear::load(f, p, &format!("stage{s}.merge"), 4 * c, 2 * c, false)?,
                ))
            } else {
                None
            };
            let norm = Norm::load(f, p, &format!("stage{s}.norm"), c)?;
            stages.push(Stage {
                channels: c,
                blocks,
                norm,
                merge,
            });
        }

        let squeeze = DecoderBlock::load(f, p, "squeeze", 2880, 1536, false)?;
        let mut blocks = Vec::new();
        for (i, &(cin, cout)) in LEVELS.iter().enumerate() {
            blocks.push(DecoderBlock::load(
                f,
                p,
                &format!("block{}", 4 - i),
                cin,
                cout,
                i < 3,
            )?);
        }
        let mut laterals = Vec::new();
        for (n, c) in [(4, 768), (3, 384), (2, 192)] {
            laterals.push(Linear::load(f, p, &format!("lateral{n}"), c, c, true)?);
        }
        let mut patches = Vec::new();
        for (n, cin, cout) in [(5, 3072, 192), (4, 768, 192), (3, 192, 96), (2, 48, 48)] {
            let name = format!("patches{n}");
            patches.push(Patches {
                conv1: Linear::load(f, p, &format!("{name}.conv1"), 9 * cin, INTER, true)?,
                conv_out: Linear::load(f, p, &format!("{name}.conv_out"), 9 * INTER, cout, true)?,
            });
        }
        let head = Linear::load(f, p, "head", 96, 1, true)?;
        let full_resolution = FullResolution::load(f)?;
        file.finish()?;
        let mut composed = full_resolution.composed.clone();
        composed.push(full_resolution.composed_bias);
        let full = p.put(&composed);
        p.put(&full_resolution.conv1);
        p.put(&full_resolution.bias1);
        let mut conv_out = full_resolution.conv_out.clone();
        conv_out.push(full_resolution.bias_out);
        p.put(&conv_out);

        let layers = Self {
            embed,
            embed_norm,
            stages,
            squeeze,
            blocks,
            laterals,
            patches,
            head,
            full,
        };
        Ok((layers, params))
    }
}

impl Aspp {
    fn load(file: &mut WeightFile, params: &mut Params, name: &str) -> Result<Self, Error> {
        let mut branches = Vec::new();
        for (i, &size) in SIZES.iter().enumerate() {
            let taps = size * size;
            let k = taps * INTER;
            let branch = format!("{name}.branch{i}");
            branches.push(Branch {
                size,
                offsets: Linear::load(
                    file,
                    params,
                    &format!("{branch}.offset"),
                    k,
                    3 * taps,
                    true,
                )?,
                weight: Linear::load(file, params, &branch, k, BRANCH, true)?,
            });
        }
        let pool = Linear::load(file, params, &format!("{name}.pool"), INTER, BRANCH, true)?;
        let mut w = file
            .take_weights(&format!("{name}.conv1.weight"), &[5 * BRANCH, INTER])?
            .into_f16();
        let b = file.take(&format!("{name}.conv1.bias"), &[INTER])?;
        let pooled = w.split_off(4 * BRANCH * INTER);
        Ok(Self {
            branches,
            pool,
            conv1: Linear::new(params, w, 4 * BRANCH, INTER, None),
            conv1_pool: Linear::new(params, pooled, BRANCH, INTER, Some(&b)),
        })
    }
}

impl DecoderBlock {
    fn load(
        file: &mut WeightFile,
        params: &mut Params,
        name: &str,
        cin: usize,
        cout: usize,
        gate: bool,
    ) -> Result<Self, Error> {
        let gate = if gate {
            Some((
                Linear::load(file, params, &format!("{name}.gdt"), 9 * cout, 16, true)?,
                Linear::load(file, params, &format!("{name}.gdt_attn"), 16, 1, true)?,
            ))
        } else {
            None
        };
        Ok(Self {
            conv_in: Linear::load(
                file,
                params,
                &format!("{name}.conv_in"),
                9 * cin,
                INTER,
                true,
            )?,
            aspp: Aspp::load(file, params, &format!("{name}.aspp"))?,
            conv_out: Linear::load(
                file,
                params,
                &format!("{name}.conv_out"),
                9 * INTER,
                cout,
                true,
            )?,
            gate,
        })
    }
}
