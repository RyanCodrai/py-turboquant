//! Correctness tests for the rotation matrix generator.
//!
//! The rotation is a deterministic orthogonal matrix derived from a
//! fixed crate-wide seed via QR decomposition. These tests verify the
//! mathematical invariants the downstream quantizer depends on:
//!
//! - The matrix is orthogonal: `R^T R = I`.
//! - Rotation preserves vector L2 norms.
//! - Generation is deterministic across calls.
//! - `R^T` recovers the input (round-trip via the transpose).
//!
//! Tolerances here are generous enough for f32 + faer QR roundoff.

use turbovec::rotation::make_rotation_matrix;

fn mat_mul(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut acc = 0.0f32;
            for k in 0..dim {
                acc += a[i * dim + k] * b[k * dim + j];
            }
            out[i * dim + j] = acc;
        }
    }
    out
}

fn transpose(m: &[f32], dim: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            t[j * dim + i] = m[i * dim + j];
        }
    }
    t
}

fn mat_vec(m: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    for i in 0..dim {
        let mut acc = 0.0f32;
        for j in 0..dim {
            acc += m[i * dim + j] * v[j];
        }
        out[i] = acc;
    }
    out
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn rand_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let mut out = Vec::with_capacity(dim);
    for _ in 0..dim {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (((state >> 32) as u32) & 0x007FFFFF) | 0x3F800000;
        let uniform = f32::from_bits(bits) - 1.0;
        out.push(uniform * 2.0 - 1.0);
    }
    out
}

#[test]
fn orthogonal_across_dims() {
    for &dim in &[32usize, 64, 128, 256] {
        let r = make_rotation_matrix(dim);
        let rt = transpose(&r, dim);
        let product = mat_mul(&rt, &r, dim);
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j { 1.0 } else { 0.0 };
                let got = product[i * dim + j];
                assert!(
                    (got - expected).abs() < 1e-3,
                    "R^T R[{}][{}] = {}, expected {} (dim={})",
                    i,
                    j,
                    got,
                    expected,
                    dim
                );
            }
        }
    }
}

#[test]
fn preserves_norm() {
    for &dim in &[32usize, 64, 128] {
        let r = make_rotation_matrix(dim);
        for seed in 0..5u64 {
            let x = rand_vec(dim, seed);
            let y = mat_vec(&r, &x, dim);
            let nx = l2_norm(&x);
            let ny = l2_norm(&y);
            assert!(
                (nx - ny).abs() / nx < 1e-3,
                "norm changed: |Rx|={} vs |x|={} (dim={}, seed={})",
                ny,
                nx,
                dim,
                seed
            );
        }
    }
}

#[test]
fn deterministic_for_same_dim() {
    let r1 = make_rotation_matrix(128);
    let r2 = make_rotation_matrix(128);
    assert_eq!(r1.len(), r2.len());
    for (i, (a, b)) in r1.iter().zip(r2.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "rotation[{}] differs across calls: {} vs {}",
            i,
            a,
            b
        );
    }
}

#[test]
fn inverse_round_trip_via_transpose() {
    let dim = 128;
    let r = make_rotation_matrix(dim);
    let rt = transpose(&r, dim);
    for seed in 0..5u64 {
        let x = rand_vec(dim, seed);
        let rx = mat_vec(&r, &x, dim);
        let x_hat = mat_vec(&rt, &rx, dim);
        for (j, (orig, back)) in x.iter().zip(x_hat.iter()).enumerate() {
            assert!(
                (orig - back).abs() < 1e-3,
                "R^T R x [{}] = {} vs original {} (seed={})",
                j,
                back,
                orig,
                seed
            );
        }
    }
}

#[test]
fn size_matches_dim_squared() {
    for &dim in &[16usize, 64, 256] {
        let r = make_rotation_matrix(dim);
        assert_eq!(r.len(), dim * dim);
    }
}
