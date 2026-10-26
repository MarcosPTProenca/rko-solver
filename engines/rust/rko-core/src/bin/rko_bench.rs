//! Command-line benchmark runner for the Rust engine on built-in problems.
//!
//! Objective:
//! - Expose `rko_core::bench` as an executable that runs the Rust engine.
//!
//! Technical approach:
//! - Delegates to the shared driver with `run_portfolio` as the engine.
//!
//! Flow (pseudocode):
//! 1. Call the shared driver.
//!
//! Important criteria:
//! - Arguments and output are those documented in `rko_core::bench`.
//!
//! Inputs/Outputs:
//! - Input: command-line arguments.
//! - Output: the `RKO_BENCH_RESULT` line and a summary JSON object.

fn main() {
    rko_core::bench::run("rko_bench", rko_core::engine::run_portfolio);
}
