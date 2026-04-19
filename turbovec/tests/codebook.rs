//! Correctness tests for Lloyd-Max codebook generation.
//!
//! After orthogonal rotation, each coordinate of a unit vector on the
//! sphere S^{d-1} follows Beta((d-1)/2, (d-1)/2) on [-1, 1]. The
//! Lloyd-Max iteration produces the MMSE scalar quantizer for that
//! distribution. These tests verify structural invariants at
//! representative (bits, dim) pairs that the encoding pipeline relies
//! on.

use turbovec::codebook::codebook;

#[test]
fn centroids_strictly_ascending() {
    for &bits in &[2usize, 3, 4] {
        for &dim in &[256usize, 768, 1536] {
            let (_, centroids) = codebook(bits, dim);
            for i in 0..centroids.len() - 1 {
                assert!(
                    centroids[i] < centroids[i + 1],
                    "centroids not ascending at bits={}, dim={}: c[{}]={} >= c[{}]={}",
                    bits,
                    dim,
                    i,
                    centroids[i],
                    i + 1,
                    centroids[i + 1]
                );
            }
        }
    }
}

#[test]
fn boundaries_strictly_between_centroids() {
    for &bits in &[2usize, 3, 4] {
        for &dim in &[256usize, 1536] {
            let (boundaries, centroids) = codebook(bits, dim);
            assert_eq!(boundaries.len(), centroids.len() - 1);
            for i in 0..boundaries.len() {
                assert!(
                    boundaries[i] > centroids[i],
                    "boundary[{}] = {} not > centroid[{}] = {} (bits={}, dim={})",
                    i,
                    boundaries[i],
                    i,
                    centroids[i],
                    bits,
                    dim
                );
                assert!(
                    boundaries[i] < centroids[i + 1],
                    "boundary[{}] = {} not < centroid[{}] = {} (bits={}, dim={})",
                    i,
                    boundaries[i],
                    i + 1,
                    centroids[i + 1],
                    bits,
                    dim
                );
            }
        }
    }
}

#[test]
fn level_counts_correct() {
    for &bits in &[2usize, 3, 4] {
        let (boundaries, centroids) = codebook(bits, 1536);
        assert_eq!(
            centroids.len(),
            1 << bits,
            "expected 2^{} = {} centroids, got {}",
            bits,
            1 << bits,
            centroids.len()
        );
        assert_eq!(
            boundaries.len(),
            (1 << bits) - 1,
            "expected 2^{} - 1 = {} boundaries, got {}",
            bits,
            (1 << bits) - 1,
            boundaries.len()
        );
    }
}

#[test]
fn symmetric_about_zero() {
    for &bits in &[2usize, 3, 4] {
        for &dim in &[768usize, 1536] {
            let (_, centroids) = codebook(bits, dim);
            let n = centroids.len();
            for i in 0..n / 2 {
                let lo = centroids[i];
                let hi = centroids[n - 1 - i];
                assert!(
                    (lo + hi).abs() < 1e-4,
                    "asymmetric: c[{}]={} c[{}]={} (bits={}, dim={})",
                    i,
                    lo,
                    n - 1 - i,
                    hi,
                    bits,
                    dim
                );
            }
        }
    }
}

#[test]
fn deterministic_for_same_params() {
    let (b1, c1) = codebook(4, 1536);
    let (b2, c2) = codebook(4, 1536);
    assert_eq!(b1, b2);
    assert_eq!(c1, c2);
}

#[test]
fn centroids_within_unit_interval() {
    for &bits in &[2usize, 3, 4] {
        let (_, centroids) = codebook(bits, 1536);
        for (i, &c) in centroids.iter().enumerate() {
            assert!(
                c > -1.0 && c < 1.0,
                "centroid[{}] = {} outside (-1, 1) (bits={})",
                i,
                c,
                bits
            );
        }
    }
}
