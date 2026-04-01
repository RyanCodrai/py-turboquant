//! TurboQuant implementation for vector search.
//!
//! Compresses high-dimensional vectors to 2-4 bits per coordinate with
//! near-optimal distortion. Data-oblivious — no training required.
//!
//! ```rust
//! use turbovec::TurboQuantIndex;
//!
//! let mut index = TurboQuantIndex::new(1536, 4);
//! index.add(&vectors);
//! let results = index.search(&queries, 10);
//! index.write("index.tv").unwrap();
//! let loaded = TurboQuantIndex::load("index.tv").unwrap();
//! ```

pub mod codebook;
pub mod encode;
pub mod io;
pub mod pack;
pub mod rotation;
pub mod search;

use std::path::Path;

const ROTATION_SEED: u64 = 42;
const BLOCK: usize = 32;
const FLUSH_EVERY: usize = 256;

pub struct TurboQuantIndex {
    dim: usize,
    bit_width: usize,
    n_vectors: usize,
    packed_codes: Vec<u8>,
    norms: Vec<f32>,
    // Cached for search (lazily computed)
    blocked: Option<Vec<u8>>,
    n_blocks: usize,
    centroids: Option<Vec<f32>>,
    rotation: Option<Vec<f32>>,
}

pub struct SearchResults {
    pub scores: Vec<f32>,
    pub indices: Vec<i64>,
    pub nq: usize,
    pub k: usize,
}

impl SearchResults {
    pub fn scores_for_query(&self, qi: usize) -> &[f32] {
        &self.scores[qi * self.k..(qi + 1) * self.k]
    }

    pub fn indices_for_query(&self, qi: usize) -> &[i64] {
        &self.indices[qi * self.k..(qi + 1) * self.k]
    }
}

impl TurboQuantIndex {
    pub fn new(dim: usize, bit_width: usize) -> Self {
        assert!(
            bit_width >= 2 && bit_width <= 4,
            "bit_width must be 2, 3, or 4"
        );
        assert!(dim % 8 == 0, "dim must be a multiple of 8");

        Self {
            dim,
            bit_width,
            n_vectors: 0,
            packed_codes: Vec::new(),
            norms: Vec::new(),
            blocked: None,
            n_blocks: 0,
            centroids: None,
            rotation: None,
        }
    }

    pub fn add(&mut self, vectors: &[f32]) {
        let n = vectors.len() / self.dim;
        assert_eq!(
            vectors.len(),
            n * self.dim,
            "vectors length must be a multiple of dim"
        );

        let rotation = self.ensure_rotation();
        let (boundaries, _) = codebook::codebook(self.bit_width, self.dim);
        let (packed, norms) = encode::encode(vectors, n, self.dim, &rotation, &boundaries, self.bit_width);

        let bytes_per_row = self.dim * self.bit_width / 8;
        if self.n_vectors == 0 {
            self.packed_codes = packed;
            self.norms = norms;
        } else {
            self.packed_codes.extend_from_slice(&packed);
            self.norms.extend_from_slice(&norms);
        }
        self.n_vectors += n;
        self.blocked = None; // invalidate cache
    }

    pub fn search(&mut self, queries: &[f32], k: usize) -> SearchResults {
        let nq = queries.len() / self.dim;
        assert_eq!(queries.len(), nq * self.dim);

        self.ensure_cached();

        let rotation = self.rotation.as_ref().unwrap();
        let blocked = self.blocked.as_ref().unwrap();
        let centroids = self.centroids.as_ref().unwrap();
        let k = k.min(self.n_vectors);

        let (scores, indices) = search::search(
            queries,
            nq,
            rotation,
            blocked,
            centroids,
            &self.norms,
            self.bit_width,
            self.dim,
            self.n_vectors,
            self.n_blocks,
            k,
        );

        SearchResults {
            scores,
            indices,
            nq,
            k,
        }
    }

    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        io::write(path, self.bit_width, self.dim, self.n_vectors, &self.packed_codes, &self.norms)
    }

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let (bit_width, dim, n_vectors, packed_codes, norms) = io::load(path)?;
        Ok(Self {
            dim,
            bit_width,
            n_vectors,
            packed_codes,
            norms,
            blocked: None,
            n_blocks: 0,
            centroids: None,
            rotation: None,
        })
    }

    pub fn len(&self) -> usize {
        self.n_vectors
    }

    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn bit_width(&self) -> usize {
        self.bit_width
    }

    fn ensure_rotation(&mut self) -> Vec<f32> {
        if self.rotation.is_none() {
            self.rotation = Some(rotation::make_rotation_matrix(self.dim));
        }
        self.rotation.clone().unwrap()
    }

    fn ensure_cached(&mut self) {
        if self.rotation.is_none() {
            self.rotation = Some(rotation::make_rotation_matrix(self.dim));
        }
        if self.centroids.is_none() {
            let (_, c) = codebook::codebook(self.bit_width, self.dim);
            self.centroids = Some(c);
        }
        if self.blocked.is_none() {
            let (blocked, n_blocks) =
                pack::repack(&self.packed_codes, self.n_vectors, self.bit_width, self.dim);
            self.blocked = Some(blocked);
            self.n_blocks = n_blocks;
        }
    }
}
