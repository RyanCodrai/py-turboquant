// Force the linker to record libopenblas as a runtime dependency on Linux.
//
// Without this, the `.so` we produce references `cblas_sgemm` as an
// undefined symbol but has no `DT_NEEDED` entry pointing at openblas, so
// dynamic linking fails at Python import time:
//
//   ImportError: _turbovec.abi3.so: undefined symbol: cblas_sgemm
//
// The blas-src + openblas-src dependency chain doesn't reliably emit
// the link flag through pyo3's build, so we emit it directly. On macOS
// we rely on the Accelerate framework which `blas-src` wires up via its
// `accelerate` feature; no extra link flag needed there.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=openblas");
    }
}
