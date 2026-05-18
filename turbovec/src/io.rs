//! Read/write TurboVec index files.
//!
//! Two formats live here:
//! * `.tv` — [`TurboQuantIndex`](crate::TurboQuantIndex) — 4-byte magic
//!   "TVPI" + version + bit_width/dim/n_vectors header + packed codes +
//!   per-vector scales.
//! * `.tvim` — [`IdMapIndex`](crate::IdMapIndex) — 4-byte magic "TVIM"
//!   + version + the same core-index payload + a trailing `slot_to_id`
//!   table of `u64` values.
//!
//! ## Format versioning
//!
//! Both formats are at version 2 as of turbovec 0.4.4. Version 1 (turbovec
//! ≤ 0.4.3) stored `||v||` in the per-vector slot; version 2 stores
//! `||v|| / <u_rot, x̂>` (the length-renormalized correction). The two are
//! the same on-disk shape but mean different things; loading a v1 file
//! under v2 code would silently produce wrong search scores, so we
//! refuse the load with a clear error pointing the caller at a rebuild.
//!
//! Version 1 `.tv` files had no magic — the file started with a bare
//! bit_width byte (2/3/4). Version 2 prepends magic + version, which
//! lets us detect either v2 or "looks like a v1 turbovec file" cleanly.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const TV_MAGIC: &[u8; 4] = b"TVPI";
const TV_VERSION: u8 = 2;
const TVIM_MAGIC: &[u8; 4] = b"TVIM";
const TVIM_VERSION: u8 = 2;

const REBUILD_HINT: &str =
    "Rebuild this index from the source vectors using turbovec 0.4.4 or later \
     (no in-place migration is provided; the format version 2 changes the meaning \
     of the per-vector scalar from ||v|| to a length-renormalization correction).";

/// `.tv` write — positional index.
pub fn write(
    path: impl AsRef<Path>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    packed_codes: &[u8],
    scales: &[f32],
) -> io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(TV_MAGIC)?;
    f.write_all(&[TV_VERSION])?;
    write_core(&mut f, bit_width, dim, n_vectors, packed_codes, scales)?;
    f.flush()?;
    Ok(())
}

/// `.tv` load — positional index.
pub fn load(path: impl AsRef<Path>) -> io::Result<(usize, usize, usize, Vec<u8>, Vec<f32>)> {
    let mut f = BufReader::new(File::open(path)?);

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TV_MAGIC {
        // Version 1 .tv files had no magic — first byte was the bit_width
        // (always 2, 3, or 4). If we see one of those as the first byte,
        // emit a targeted error rather than the generic "wrong magic"
        // message; otherwise treat it as a non-turbovec file.
        if (2..=4).contains(&magic[0]) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "this .tv file was written by turbovec ≤ 0.4.3 (format \
                     version 1). It is incompatible with turbovec 0.4.4+ \
                     because the per-vector scalar's meaning changed. {}",
                    REBUILD_HINT,
                ),
            ));
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a turbovec .tv file: wrong magic",
        ));
    }
    let mut version = [0u8; 1];
    f.read_exact(&mut version)?;
    if version[0] != TV_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported .tv format version: {} (this build expects version {})",
                version[0], TV_VERSION,
            ),
        ));
    }
    read_core(&mut f)
}

/// `.tvim` write — positional index plus the id-map side-tables.
pub fn write_id_map(
    path: impl AsRef<Path>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    packed_codes: &[u8],
    scales: &[f32],
    slot_to_id: &[u64],
) -> io::Result<()> {
    assert_eq!(
        slot_to_id.len(),
        n_vectors,
        "slot_to_id length {} does not match n_vectors {}",
        slot_to_id.len(),
        n_vectors,
    );

    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(TVIM_MAGIC)?;
    f.write_all(&[TVIM_VERSION])?;
    write_core(&mut f, bit_width, dim, n_vectors, packed_codes, scales)?;

    for &id in slot_to_id {
        f.write_all(&id.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

/// `.tvim` load — positional index plus the id-map side-tables.
pub fn load_id_map(
    path: impl AsRef<Path>,
) -> io::Result<(usize, usize, usize, Vec<u8>, Vec<f32>, Vec<u64>)> {
    let mut f = BufReader::new(File::open(path)?);

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TVIM_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a TVIM file: wrong magic",
        ));
    }
    let mut version = [0u8; 1];
    f.read_exact(&mut version)?;
    if version[0] == 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "this .tvim file was written by turbovec ≤ 0.4.3 (format \
                 version 1). It is incompatible with turbovec 0.4.4+ \
                 because the per-vector scalar's meaning changed. {}",
                REBUILD_HINT,
            ),
        ));
    }
    if version[0] != TVIM_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported TVIM version: {} (this build expects version {})",
                version[0], TVIM_VERSION,
            ),
        ));
    }

    let (bit_width, dim, n_vectors, packed_codes, scales) = read_core(&mut f)?;

    let mut slot_to_id = Vec::with_capacity(n_vectors);
    let mut buf = [0u8; 8];
    for _ in 0..n_vectors {
        f.read_exact(&mut buf)?;
        slot_to_id.push(u64::from_le_bytes(buf));
    }

    Ok((bit_width, dim, n_vectors, packed_codes, scales, slot_to_id))
}

const CORE_HEADER_SIZE: usize = 9;

/// Core header + packed codes + per-vector scales — shared by `.tv` and `.tvim`.
fn write_core<W: Write>(
    w: &mut W,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    packed_codes: &[u8],
    scales: &[f32],
) -> io::Result<()> {
    w.write_all(&[bit_width as u8])?;
    w.write_all(&(dim as u32).to_le_bytes())?;
    w.write_all(&(n_vectors as u32).to_le_bytes())?;
    w.write_all(packed_codes)?;
    for &s in scales {
        w.write_all(&s.to_le_bytes())?;
    }
    Ok(())
}

fn read_core<R: Read>(r: &mut R) -> io::Result<(usize, usize, usize, Vec<u8>, Vec<f32>)> {
    let mut header = [0u8; CORE_HEADER_SIZE];
    r.read_exact(&mut header)?;

    let bit_width = header[0] as usize;
    let dim = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
    let n_vectors = u32::from_le_bytes([header[5], header[6], header[7], header[8]]) as usize;

    let packed_bytes = (dim / 8) * bit_width * n_vectors;
    let mut packed_codes = vec![0u8; packed_bytes];
    r.read_exact(&mut packed_codes)?;

    let mut scales_bytes = vec![0u8; n_vectors * 4];
    r.read_exact(&mut scales_bytes)?;
    let scales: Vec<f32> = scales_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Ok((bit_width, dim, n_vectors, packed_codes, scales))
}
