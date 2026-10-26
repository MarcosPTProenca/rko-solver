//! Build script that compiles the vendored RKO C++ reference only when requested.
//!
//! Objective:
//! - Keep the C++ engine opt-in: nothing is compiled, and no C++ toolchain or OpenMP is
//!   needed, unless `RKO_BUILD_CPP=1` is set in the environment.
//!
//! Technical approach:
//! - With the variable set on Linux, compile `cpp/rko_cpp.cpp` (which includes the vendored
//!   reference headers unchanged) with C++20, optimization 3 and OpenMP, link the GNU OpenMP
//!   runtime and emit `cfg(rko_cpp)`.
//! - Otherwise emit nothing but the cfg declaration, so the crate builds a stub.
//!
//! Flow (pseudocode):
//! 1. Declare the `rko_cpp` cfg and the environment dependency.
//! 2. If enabled and supported, compile and link; otherwise warn when asked on another OS.
//!
//! Important criteria:
//! - The vendored sources are never modified; behavior changes live in the driver file.
//!
//! Inputs/Outputs:
//! - Input: the `RKO_BUILD_CPP` environment variable and the target operating system.
//! - Output: a static library, a link directive and `cfg(rko_cpp)`, or nothing.

fn main() {
    println!("cargo:rustc-check-cfg=cfg(rko_cpp)");
    println!("cargo:rerun-if-env-changed=RKO_BUILD_CPP");
    println!("cargo:rerun-if-changed=cpp");
    println!("cargo:rerun-if-changed=vendor");
    if std::env::var_os("RKO_BUILD_CPP").is_none() {
        return;
    }
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        println!(
            "cargo:warning=RKO_BUILD_CPP is only supported on Linux; the C++ engine is disabled"
        );
        return;
    }
    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .opt_level(3)
        .warnings(false)
        .flag("-fopenmp")
        .flag("-ftls-model=initial-exec")
        .include("vendor/rko_cpp_v2/src")
        .file("cpp/rko_cpp.cpp")
        .compile("rko_cpp");
    println!("cargo:rustc-link-lib=gomp");
    println!("cargo:rustc-cfg=rko_cpp");
}
