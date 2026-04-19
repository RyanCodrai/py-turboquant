//! Statistical validation of quantizer distortion.
//!
//! Verifies that turbovec's Lloyd-Max codebook achieves the
//! quantization MSE predicted by the TurboQuant paper's Theorem 1.
//! Because Beta((d-1)/2, (d-1)/2) converges to N(0, 1/d) in high
//! dimensions, the Beta-on-sphere codebook's MSE should approach
//! `Theorem1(b) / d`, and its ratio to the Shannon lower bound
//! `2^{-2b} / d` should stay within the paper's ~2.7x claim.
//!
//! These tests catch a class of bug that no structural test can:
//! if Lloyd-Max ever failed to converge (wrong distribution,
//! truncated iteration, broken integration), the codebook would
//! still be well-formed but distortion would drift from the
//! theoretical optimum.

use statrs::distribution::{Beta, Continuous};
use turbovec::codebook::codebook;

/// Lloyd-Max MSE for an N(0, 1) source at b bits per dimension,
/// from the TurboQuant paper's Theorem 1 (equivalently, Max 1960).
/// Values match pyturboquant's `PAPER_MSE_VALUES` for cross-library
/// consistency.
const PAPER_MSE: &[(usize, f64)] = &[
    (2, 0.1175),
    (3, 0.03454),
    (4, 0.009497),
];

#[test]
fn codebook_mse_matches_paper_at_high_dim() {
    // At d=1536 the Beta((d-1)/2, (d-1)/2) distribution is very close
    // to N(0, 1/d), so the empirical Lloyd-Max MSE should equal
    // Theorem1(b) / d to within ~5%.
    let dim = 1536;

    for &(bits, paper_val) in PAPER_MSE {
        let (boundaries, centroids) = codebook(bits, dim);
        let mse = compute_codebook_mse(&boundaries, &centroids, dim);
        let expected = paper_val / dim as f64;
        let rel_err = (mse - expected).abs() / expected;
        assert!(
            rel_err < 0.05,
            "bits={}, dim={}: codebook MSE={:.3e} vs Theorem1/d={:.3e} (rel_err={:.3})",
            bits,
            dim,
            mse,
            expected,
            rel_err,
        );
    }
}

#[test]
fn codebook_mse_within_shannon_factor() {
    // The paper claims distortion within ~2.7x of the Shannon lower
    // bound 2^{-2b}. Validate the ratio stays below 3.0 across every
    // (bits, dim) pair turbovec cares about.
    for &bits in &[2usize, 3, 4] {
        for &dim in &[256usize, 768, 1536] {
            let (boundaries, centroids) = codebook(bits, dim);
            let mse = compute_codebook_mse(&boundaries, &centroids, dim);
            let shannon_bound = 2f64.powi(-2 * bits as i32) / dim as f64;
            let ratio = mse / shannon_bound;
            assert!(
                ratio < 3.0,
                "bits={}, dim={}: MSE/Shannon = {:.3} exceeds 3x paper bound",
                bits,
                dim,
                ratio,
            );
            // Sanity: MSE must be above the bound, not below (would
            // indicate a broken test or a result too good to be true).
            assert!(
                ratio > 1.0,
                "bits={}, dim={}: MSE/Shannon = {:.3} below Shannon lower bound",
                bits,
                dim,
                ratio,
            );
        }
    }
}

/// Analytical MSE of the given (boundaries, centroids) scalar
/// quantizer against Beta((d-1)/2, (d-1)/2) on [-1, 1].
///
/// For each region [b_{i-1}, b_i] with centroid c_i:
///     contribution = integral_{b_{i-1}}^{b_i} (x - c_i)^2 * p(x) dx
/// where p is the Beta PDF shifted from [0,1] to [-1,1].
fn compute_codebook_mse(boundaries: &[f32], centroids: &[f32], dim: usize) -> f64 {
    let a = (dim as f64 - 1.0) / 2.0;
    let beta = Beta::new(a, a).unwrap();

    let n = centroids.len();
    let mut edges = Vec::with_capacity(n + 1);
    edges.push(-1.0f64);
    edges.extend(boundaries.iter().map(|&b| b as f64));
    edges.push(1.0);

    let mut mse = 0.0f64;
    for i in 0..n {
        let lo = edges[i];
        let hi = edges[i + 1];
        let c = centroids[i] as f64;
        // pdf on [-1, 1]: beta.pdf((x + 1) / 2) / 2
        mse += simpson(
            |x: f64| (x - c).powi(2) * beta.pdf((x + 1.0) / 2.0) / 2.0,
            lo,
            hi,
            4000,
        );
    }
    mse
}

/// Composite Simpson's rule over `n` intervals. `n` is rounded down
/// to even; 4000 is enough for 8+ digits on the integrands here.
fn simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let n = n & !1;
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += if i % 2 == 0 { 2.0 * f(x) } else { 4.0 * f(x) };
    }
    sum * h / 3.0
}
