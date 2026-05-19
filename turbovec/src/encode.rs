//! Encode vectors: normalize, rotate, quantize, bit-pack, compute per-vector scale.
//!
//! For each vector `v` with rotated unit form `u` and reconstructed
//! centroid vector `x_hat`, the stored scale is `||v|| / <u, x_hat>` —
//! the RaBitQ-style length-renormalization correction adapted to
//! turbovec's Lloyd-Max codebook. Applying this scale at the final
//! score-multiplication site in the SIMD kernel gives an unbiased
//! estimator of `<v, q>` (the biased version would have multiplied
//! by `||v||` alone, leaving the systematic shrinkage `<u, x_hat> < 1`
//! uncompensated). When quantization is perfect (`x_hat = u`),
//! `<u, x_hat> = 1` and `scale` reduces to `||v||`.

use ndarray::ArrayView2;
use rayon::prelude::*;

/// Encode n vectors of dimension dim.
/// Returns (packed_codes as flat Vec<u8>, scales as Vec<f32>).
pub fn encode(
    vectors: &[f32],
    n: usize,
    dim: usize,
    rotation: &[f32],
    boundaries: &[f32],
    centroids: &[f32],
    bit_width: usize,
) -> (Vec<u8>, Vec<f32>) {
    let mut norms = vec![0.0f32; n];
    let mut unit_flat = vec![0.0f32; n * dim];

    // Normalize. Rows are independent so Rayon splits them across cores.
    norms.par_iter_mut()
        .zip(unit_flat.par_chunks_mut(dim))
        .enumerate()
        .for_each(|(i, (norm, unit_row))| {
            let row = &vectors[i * dim..(i + 1) * dim];
            let n_val = simd_norm(row);
            *norm = n_val;
            let inv = if n_val > 1e-10 { 1.0 / n_val } else { 0.0 };
            simd_scale(row, inv, unit_row);
        });

    // Rotate. ndarray calls into Accelerate (macOS) or OpenBLAS (Linux),
    // both of which handle their own threading internally.
    let unit_mat = ArrayView2::from_shape((n, dim), &unit_flat).unwrap();
    let rot_mat = ArrayView2::from_shape((dim, dim), rotation).unwrap();
    let rotated_mat = unit_mat.dot(&rot_mat.t());
    let rotated = rotated_mat.as_slice().unwrap();

    // Quantize + scale + pack, fused into one pass per row.
    // No intermediate codes allocation — codes live on the stack
    // and get packed immediately.
    let bytes_per_plane = dim / 8;
    let bytes_per_row = bit_width * bytes_per_plane;
    let mut packed = vec![0u8; n * bytes_per_row];
    let mut scales = vec![0.0f32; n];

    packed.par_chunks_mut(bytes_per_row)
        .zip(scales.par_iter_mut())
        .enumerate()
        .for_each(|(i, (packed_row, scale))| {
            let rot_row = &rotated[i * dim..(i + 1) * dim];
            *scale = fused_quantize_scale_pack(
                rot_row, boundaries, centroids, norms[i],
                packed_row, dim, bit_width, bytes_per_plane,
            );
        });

    (packed, scales)
}

// ─── Norm and scale (aarch64) ────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn simd_norm(row: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let dim = row.len();
    let chunks = dim / 4;
    let mut acc = unsafe { vdupq_n_f32(0.0) };

    unsafe {
        for c in 0..chunks {
            let v = vld1q_f32(row.as_ptr().add(c * 4));
            acc = vfmaq_f32(acc, v, v);
        }
        let mut sum = vaddvq_f32(acc);
        for j in (chunks * 4)..dim {
            sum += row[j] * row[j];
        }
        sum.sqrt()
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn simd_scale(row: &[f32], scale: f32, out: &mut [f32]) {
    use std::arch::aarch64::*;
    let dim = row.len();
    let chunks = dim / 4;
    let sv = unsafe { vdupq_n_f32(scale) };

    unsafe {
        for c in 0..chunks {
            let v = vld1q_f32(row.as_ptr().add(c * 4));
            vst1q_f32(out.as_mut_ptr().add(c * 4), vmulq_f32(v, sv));
        }
        for j in (chunks * 4)..dim {
            out[j] = row[j] * scale;
        }
    }
}

// ─── Norm and scale (fallback) ───────────────────────────────────────────────

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn simd_norm(row: &[f32]) -> f32 {
    row.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn simd_scale(row: &[f32], scale: f32, out: &mut [f32]) {
    for j in 0..row.len() {
        out[j] = row[j] * scale;
    }
}

// ─── Fused quantize + scale + pack (aarch64) ────────────────────────────────

/// Process one row: quantize against boundaries, accumulate the
/// centroid inner product for the scale correction, and pack the
/// resulting codes into bit-plane layout. Nothing hits the heap.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fused_quantize_scale_pack(
    rot_row: &[f32],
    boundaries: &[f32],
    centroids: &[f32],
    norm: f32,
    packed_row: &mut [u8],
    dim: usize,
    bits: usize,
    bytes_per_plane: usize,
) -> f32 {
    use std::arch::aarch64::*;

    let mut inner = 0.0f64;
    let chunks = dim / 8;

    unsafe {
        for c in 0..chunks {
            let offset = c * 8;
            let vals_lo = vld1q_f32(rot_row.as_ptr().add(offset));
            let vals_hi = vld1q_f32(rot_row.as_ptr().add(offset + 4));

            // Count how many boundaries each coordinate exceeds.
            // Linear scan is the right choice here: every lane compares against
            // the same boundary value, so it vectorises perfectly. A binary search
            // would need per-lane conditional indexing which breaks SIMD.
            let mut acc_lo = vdupq_n_u32(0);
            let mut acc_hi = vdupq_n_u32(0);

            for &b in boundaries {
                let bv = vdupq_n_f32(b);
                acc_lo = vaddq_u32(acc_lo, vshrq_n_u32::<31>(vcgtq_f32(vals_lo, bv)));
                acc_hi = vaddq_u32(acc_hi, vshrq_n_u32::<31>(vcgtq_f32(vals_hi, bv)));
            }

            // Now pull the counts out. This only happens once per 8 coordinates,
            // not once per boundary.
            let counts: [u8; 8] = [
                vgetq_lane_u32::<0>(acc_lo) as u8,
                vgetq_lane_u32::<1>(acc_lo) as u8,
                vgetq_lane_u32::<2>(acc_lo) as u8,
                vgetq_lane_u32::<3>(acc_lo) as u8,
                vgetq_lane_u32::<0>(acc_hi) as u8,
                vgetq_lane_u32::<1>(acc_hi) as u8,
                vgetq_lane_u32::<2>(acc_hi) as u8,
                vgetq_lane_u32::<3>(acc_hi) as u8,
            ];

            // Centroid lookup for the scale correction.
            for k in 0..8 {
                inner += rot_row[offset + k] as f64 * centroids[counts[k] as usize] as f64;
            }

            // Pack 8 codes into one byte per bit-plane.
            // Load the 8 counts, mask each bit-plane, weight by position,
            // then horizontal-add gives the packed byte directly.
            let codes_vec = vld1_u8(counts.as_ptr());
            let weights: [u8; 8] = [128, 64, 32, 16, 8, 4, 2, 1];
            let wv = vld1_u8(weights.as_ptr());

            for p in 0..bits {
                let mask = vdup_n_u8(1u8 << p);
                let hit = vcgt_u8(vand_u8(codes_vec, mask), vdup_n_u8(0));
                packed_row[p * bytes_per_plane + offset / 8] = vaddv_u8(vand_u8(hit, wv));
            }
        }

        // Tail elements when dim isn't a multiple of 8.
        for j in (chunks * 8)..dim {
            let mut code = 0u8;
            for &b in boundaries {
                if rot_row[j] > b { code += 1; }
            }
            inner += rot_row[j] as f64 * centroids[code as usize] as f64;
            let byte_pos = j / 8;
            let bit_pos = 7 - (j % 8);
            for p in 0..bits {
                if code & (1 << p) != 0 {
                    packed_row[p * bytes_per_plane + byte_pos] |= 1 << bit_pos;
                }
            }
        }
    }

    let inner = inner.max(1e-10) as f32;
    norm / inner
}

// ─── Fused quantize + scale + pack (fallback) ───────────────────────────────

/// Same logic, scalar. x86_64 still gets Rayon across rows.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn fused_quantize_scale_pack(
    rot_row: &[f32],
    boundaries: &[f32],
    centroids: &[f32],
    norm: f32,
    packed_row: &mut [u8],
    dim: usize,
    bits: usize,
    bytes_per_plane: usize,
) -> f32 {
    let mut inner = 0.0f64;

    for j in 0..dim {
        let mut code = 0u8;
        for &b in boundaries {
            if rot_row[j] > b { code += 1; }
        }
        inner += rot_row[j] as f64 * centroids[code as usize] as f64;

        let byte_pos = j / 8;
        let bit_pos = 7 - (j % 8);
        for p in 0..bits {
            if code & (1 << p) != 0 {
                packed_row[p * bytes_per_plane + byte_pos] |= 1 << bit_pos;
            }
        }
    }

    let inner = inner.max(1e-10) as f32;
    norm / inner
}
