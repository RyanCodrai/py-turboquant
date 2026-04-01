//! Encode vectors: normalize, rotate, quantize, bit-pack.

use ndarray::{Array2, ArrayView2, s};
use rayon::prelude::*;

/// Encode n vectors of dimension dim.
/// Returns (packed_codes as flat Vec<u8>, norms as Vec<f32>).
pub fn encode(
    vectors: &[f32],
    n: usize,
    dim: usize,
    rotation: &[f32], // dim x dim, row-major
    boundaries: &[f32],
    bit_width: usize,
) -> (Vec<u8>, Vec<f32>) {
    let mut norms = vec![0.0f32; n];
    let mut unit_flat = vec![0.0f32; n * dim];

    // 1. Extract norms and normalize
    for i in 0..n {
        let row = &vectors[i * dim..(i + 1) * dim];
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        norms[i] = norm;
        let inv_norm = if norm > 1e-10 { 1.0 / norm } else { 0.0 };
        for j in 0..dim {
            unit_flat[i * dim + j] = row[j] * inv_norm;
        }
    }

    // 2. Rotate: rotated = unit @ rotation.T (BLAS-accelerated via ndarray)
    let unit_mat = ArrayView2::from_shape((n, dim), &unit_flat).unwrap();
    let rot_mat = ArrayView2::from_shape((dim, dim), rotation).unwrap();
    let rotated_mat = unit_mat.dot(&rot_mat.t());
    let rotated = rotated_mat.as_slice().unwrap();

    // 3. Quantize: for each boundary, codes += (rotated > boundary)
    let mut codes = vec![0u8; n * dim];
    for b in boundaries {
        for idx in 0..n * dim {
            if rotated[idx] > *b {
                codes[idx] += 1;
            }
        }
    }

    // 4. Bit-pack into bit-plane format
    let packed = pack_codes(&codes, n, dim, bit_width);

    (packed, norms)
}

/// Pack quantized codes into bit-plane format.
fn pack_codes(codes: &[u8], n: usize, dim: usize, bits: usize) -> Vec<u8> {
    let bytes_per_plane = dim / 8;
    let bytes_per_row = bits * bytes_per_plane;
    let mut packed = vec![0u8; n * bytes_per_row];

    for i in 0..n {
        for j in 0..dim {
            let code = codes[i * dim + j];
            let byte_pos = j / 8;
            let bit_pos = 7 - (j % 8);
            for p in 0..bits {
                if code & (1 << p) != 0 {
                    packed[i * bytes_per_row + p * bytes_per_plane + byte_pos] |= 1 << bit_pos;
                }
            }
        }
    }

    packed
}
