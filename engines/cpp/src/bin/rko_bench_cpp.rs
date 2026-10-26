//! Command-line benchmark runner for the C++ reference engine on built-in problems.
//!
//! Objective:
//! - Run the vendored C++ algorithms on the same built-in problems, grids and seeds as
//!   `rko_bench`, so the two engines can be compared on identical objectives.
//!
//! Technical approach:
//! - Delegates to the shared `rko_core::bench` driver with `rko_cpp::run_portfolio` as the
//!   engine.
//!
//! Flow (pseudocode):
//! 1. Call the shared driver.
//!
//! Important criteria:
//! - Requires a build with `RKO_BUILD_CPP=1`; seconds must be at least 0.05 and `--threads`
//!   must be omitted.
//!
//! Inputs/Outputs:
//! - Input: the arguments documented in `rko_core::bench`.
//! - Output: the `RKO_BENCH_RESULT` line and a summary JSON object.

fn main() {
    rko_core::bench::run("rko_bench_cpp", rko_cpp::run_portfolio);
}
