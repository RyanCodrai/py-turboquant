//! Random orthogonal rotation matrix generation.
//!
//! Generates a deterministic orthogonal matrix via QR decomposition of
//! a seeded Gaussian random matrix. The rotation makes each coordinate
//! of a unit vector follow a known Beta distribution.

use faer::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use crate::ROTATION_SEED;

/// Generate a dim x dim orthogonal matrix (deterministic, seeded).
/// Returns row-major flat Vec<f32> of length dim*dim.
pub fn make_rotation_matrix(dim: usize) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(ROTATION_SEED);

    // Generate random Gaussian matrix
    let mut g = faer::Mat::<f64>::zeros(dim, dim);
    for j in 0..dim {
        for i in 0..dim {
            g.write(i, j, rng.sample(StandardNormal));
        }
    }

    // Non-pivoted QR decomposition (deterministic)
    let qr = g.qr();
    let q_full = qr.compute_thin_q();
    let r = qr.compute_thin_r();

    // Sign correction: Q = Q * diag(sign(diag(R)))
    let mut q = q_full;
    for j in 0..dim {
        let sign = if r.read(j, j) >= 0.0 { 1.0 } else { -1.0 };
        if sign < 0.0 {
            for i in 0..dim {
                q.write(i, j, q.read(i, j) * sign);
            }
        }
    }

    // Convert to row-major f32
    let mut result = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            result[i * dim + j] = q.read(i, j) as f32;
        }
    }

    result
}
