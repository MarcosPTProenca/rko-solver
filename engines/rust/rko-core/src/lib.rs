//! Native random-key optimizer engine.
//!
//! Objective:
//! - Provide the performance-critical core of RKO: operators, metaheuristics,
//!   Q-learning control, the shared elite pool and a portfolio scheduler.
//!
//! Technical approach:
//! - Pure Rust with no dependencies; the Python extension is a thin binding layer.
//! - Every evaluation flows through `Search`, which enforces budgets and validates
//!   costs; every thread owns its `Rng`, so searches never contend on random state.
//!
//! Flow (pseudocode):
//! 1. Describe the problem with the `Problem` trait.
//! 2. Describe each search with a `mh::Spec` and run `engine::run_portfolio`.
//! 3. Read the incumbent and per-search reports from `PortfolioResult`.
//!
//! Important criteria:
//! - `unsafe` code is forbidden in this crate.
//! - Public items are documented; hot paths allocate nothing per evaluation.
//!
//! Inputs/Outputs:
//! - Input: a `Problem`, algorithm specifications and a time budget.
//! - Output: best random keys, their cost and per-search telemetry.

pub mod bench;
pub mod config;
pub mod engine;
pub mod local_search;
pub mod mh;
pub mod operators;
pub mod pool;
pub mod problem;
pub mod qlearning;
pub mod rng;
pub mod search;
pub mod seeding;
