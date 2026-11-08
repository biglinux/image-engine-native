//! Reading a weights file: the named tensors a model's `export_weights.py`
//! writes, after eight bytes naming the model.
//!
//! The file is untrusted input: every length is checked against what is left
//! of the file before anything is allocated for it.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::Error;
use crate::gemm::Weights;

const VERSION: u32 = 1;

struct Tensor {
    shape: Vec<usize>,
    values: Weights,
}

pub struct WeightFile {
    tensors: HashMap<String, Tensor>,
}

impl WeightFile {
    /// Read the weights file at `path`, which must start with `magic`.
    pub fn read(path: &Path, magic: &[u8; 8]) -> Result<Self, Error> {
        let io = |e| Error::Io(path.display().to_string(), e);
        let file = File::open(path).map_err(io)?;
        let len = file.metadata().map_err(io)?.len();
        Self::parse(BufReader::new(file), len, magic).map_err(|e| match e {
            FileError::Io(e) => io(e),
            FileError::Format(msg) => Error::Weights(msg),
        })
    }

    /// Read a weights file of `len` bytes, one tensor at a time so that no copy of the
    /// whole file is ever held.
    fn parse(reader: impl Read, len: u64, magic: &[u8; 8]) -> Result<Self, FileError> {
        let mut at = Cursor { reader, left: len };
        if &at.take::<8>()? != magic {
            let kind = String::from_utf8_lossy(magic);
            return Err(FileError::Format(format!(
                "not a {} weights file",
                kind.trim_end_matches('\0')
            )));
        }
        let version = u32::from_le_bytes(at.take()?);
        if version != VERSION {
            return Err(FileError::Format(format!(
                "weights file version {version}, expected {VERSION}"
            )));
        }
        let count = u32::from_le_bytes(at.take()?);
        let mut tensors = HashMap::new();
        let mut bytes = Vec::new();
        for _ in 0..count {
            let name_len = usize::from(u16::from_le_bytes(at.take()?));
            at.bytes(name_len, &mut bytes)?;
            let name = std::str::from_utf8(&bytes)
                .map_err(|_| FileError::Format("tensor name is not UTF-8".into()))?
                .to_owned();
            let [dtype, rank] = at.take()?;
            let width = match dtype {
                0 => 4,
                1 => 2,
                _ => return Err(FileError::Format(format!("{name}: unknown type {dtype}"))),
            };
            let mut shape = Vec::with_capacity(usize::from(rank));
            let mut count = 1usize;
            for _ in 0..rank {
                let dim = u32::from_le_bytes(at.take()?) as usize;
                count = count
                    .checked_mul(dim)
                    .ok_or_else(|| FileError::Format(format!("{name}: too large")))?;
                shape.push(dim);
            }
            at.bytes(
                count
                    .checked_mul(width)
                    .ok_or_else(|| FileError::Format(format!("{name}: too large")))?,
                &mut bytes,
            )?;
            let values = if width == 4 {
                Weights::F32(
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|&b| f32::from_le_bytes(b))
                        .collect(),
                )
            } else {
                Weights::F16(
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|&b| u16::from_le_bytes(b))
                        .collect(),
                )
            };
            if tensors
                .insert(name.clone(), Tensor { shape, values })
                .is_some()
            {
                return Err(FileError::Format(format!("{name}: stored twice")));
            }
        }
        if at.left != 0 {
            return Err(FileError::Format(
                "trailing bytes after the last tensor".into(),
            ));
        }
        Ok(Self { tensors })
    }

    /// Remove tensor `name`, which must have `shape`, as 32-bit floats.
    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, Error> {
        Ok(self.take_weights(name, shape)?.to_f32())
    }

    /// Remove tensor `name`, which must have `shape`, in the type it is stored in.
    pub fn take_weights(&mut self, name: &str, shape: &[usize]) -> Result<Weights, Error> {
        let tensor = self
            .tensors
            .remove(name)
            .ok_or_else(|| Error::Weights(format!("{name}: missing")))?;
        if tensor.shape != shape {
            return Err(Error::Weights(format!(
                "{name}: shape {:?}, expected {shape:?}",
                tensor.shape
            )));
        }
        Ok(tensor.values)
    }

    /// Names nothing took, which mean the file and the code disagree.
    pub fn finish(self) -> Result<(), Error> {
        match self.tensors.keys().next() {
            Some(name) => Err(Error::Weights(format!("{name}: not used by this model"))),
            None => Ok(()),
        }
    }
}

enum FileError {
    Io(std::io::Error),
    Format(String),
}

/// A reader that knows how many bytes are left, so that no length read from
/// the file is trusted before it is checked against what remains.
struct Cursor<R> {
    reader: R,
    left: u64,
}

impl<R: Read> Cursor<R> {
    fn reserve(&mut self, len: usize) -> Result<(), FileError> {
        let len = len as u64;
        if len > self.left {
            return Err(FileError::Format("weights file is truncated".into()));
        }
        self.left -= len;
        Ok(())
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], FileError> {
        self.reserve(N)?;
        let mut buf = [0; N];
        self.reader.read_exact(&mut buf).map_err(FileError::Io)?;
        Ok(buf)
    }

    /// The next `len` bytes, into `buf`.
    fn bytes(&mut self, len: usize, buf: &mut Vec<u8>) -> Result<(), FileError> {
        self.reserve(len)?;
        buf.resize(len, 0);
        self.reader.read_exact(buf).map_err(FileError::Io)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAGIC: &[u8; 8] = b"TESTFILE";

    fn file_bytes(entries: &[(&str, &[u32], &[f32])]) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        out.extend(VERSION.to_le_bytes());
        out.extend((entries.len() as u32).to_le_bytes());
        for (name, shape, data) in entries {
            out.extend((name.len() as u16).to_le_bytes());
            out.extend(name.as_bytes());
            out.push(0);
            out.push(shape.len() as u8);
            for d in *shape {
                out.extend(d.to_le_bytes());
            }
            for v in *data {
                out.extend(v.to_le_bytes());
            }
        }
        out
    }

    #[test]
    fn reads_named_tensors() {
        let bytes = file_bytes(&[("a", &[2, 1], &[1.0, -2.5]), ("b", &[], &[7.0])]);
        let mut file = WeightFile::parse(&bytes[..], bytes.len() as u64, MAGIC)
            .ok()
            .unwrap();
        assert_eq!(file.take("a", &[2, 1]).unwrap(), [1.0, -2.5]);
        assert!(file.take("b", &[1]).is_err());
    }

    #[test]
    fn truncated_file_is_an_error() {
        let bytes = file_bytes(&[("a", &[4], &[1.0, 2.0, 3.0, 4.0])]);
        for cut in [3, 20, bytes.len() - 1] {
            assert!(
                WeightFile::parse(&bytes[..cut], cut as u64, MAGIC).is_err(),
                "cut at {cut}"
            );
        }
    }

    #[test]
    fn huge_dimensions_are_refused() {
        let bytes = file_bytes(&[("a", &[u32::MAX, u32::MAX, u32::MAX], &[])]);
        assert!(WeightFile::parse(&bytes[..], bytes.len() as u64, MAGIC).is_err());
    }
}
