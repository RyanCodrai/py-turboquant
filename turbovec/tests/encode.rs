//! End-to-end correctness tests for the encoding pipeline.
//!
//! `encode::encode` is the direct entry point for normalize ->
//! rotate -> quantize -> bit-pack. Going through it (rather than
//! through `TurboQuantIndex`) lets us verify the low-level output
//! shape and the stored-norm invariant without reaching into private
//! state.

extern crate blas_src;

use turbovec::codebook::codebook;
use turbovec::encode::encode;
use turbovec::rotation::make_rotation_matrix;

fn make_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..(n * dim) {
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
fn produces_expected_shape() {
    for &bit_width in &[2usize, 4] {
        let dim = 128;
        let n = 17;
        let rotation = make_rotation_matrix(dim);
        let (boundaries, _) = codebook(bit_width, dim);
        let vectors = make_vectors(n, dim, 0);

        let (packed, norms) = encode(&vectors, n, dim, &rotation, &boundaries, bit_width);

        let bytes_per_row = bit_width * (dim / 8);
        assert_eq!(
            packed.len(),
            n * bytes_per_row,
            "wrong packed length for bits={}, dim={}",
            bit_width,
            dim
        );
        assert_eq!(norms.len(), n);
    }
}

#[test]
fn preserves_input_norms() {
    let dim = 128;
    let n = 10;
    let rotation = make_rotation_matrix(dim);
    let (boundaries, _) = codebook(4, dim);
    let vectors = make_vectors(n, dim, 0);

    let (_, norms) = encode(&vectors, n, dim, &rotation, &boundaries, 4);

    for i in 0..n {
        let input_norm = vectors[i * dim..(i + 1) * dim]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!(
            (input_norm - norms[i]).abs() < 1e-4,
            "norm mismatch at i={}: input={}, stored={}",
            i,
            input_norm,
            norms[i]
        );
    }
}

#[test]
fn deterministic_output() {
    let dim = 128;
    let n = 5;
    let rotation = make_rotation_matrix(dim);
    let (boundaries, _) = codebook(4, dim);
    let vectors = make_vectors(n, dim, 0);

    let (p1, n1) = encode(&vectors, n, dim, &rotation, &boundaries, 4);
    let (p2, n2) = encode(&vectors, n, dim, &rotation, &boundaries, 4);

    assert_eq!(p1, p2);
    assert_eq!(n1, n2);
}

#[test]
fn handles_zero_vector() {
    // A zero-norm vector must not produce NaN codes.
    let dim = 128;
    let rotation = make_rotation_matrix(dim);
    let (boundaries, _) = codebook(4, dim);
    let zeros = vec![0.0f32; dim];

    let (packed, norms) = encode(&zeros, 1, dim, &rotation, &boundaries, 4);

    assert_eq!(norms[0], 0.0);
    // All codes should be finite (specifically, 0 or mid-codebook).
    // The crucial invariant is no NaN bytes in the packed output
    // (Vec<u8> can't hold NaN, but we assert length here for sanity).
    let bytes_per_row = 4 * (dim / 8);
    assert_eq!(packed.len(), bytes_per_row);
}
