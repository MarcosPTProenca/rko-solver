//! Repository-only initial-temperature experiment for offline simulated annealing.
//!
//! Objective:
//! - Estimate a local uphill acceptance temperature without changing public Python options.
//!
//! Technical approach:
//! - Solve the empirical mean Metropolis acceptance equation for finite positive deltas.
//! - Normalize by the largest delta and use its analytic upper bracket for bounded bisection.
//!
//! Flow (pseudocode):
//! 1. Reject invalid targets or empty/nonpositive/nonfinite observations.
//! 2. Bisect a normalized temperature bracket for 64 iterations.
//! 3. Rescale and return a finite positive temperature, or fall back.
//!
//! Important criteria:
//! - This targets local sampled uphill moves, not stationary Boltzmann acceptance.
//! - A fixed-seed screen must precede any public option or performance claim.
//!
//! Inputs/Outputs:
//! - Input: strictly positive cost deltas and a target acceptance probability.
//! - Output: an optional finite positive temperature.

use super::Context;
use crate::operators::shake;
use crate::search::Stop;

/// Number of independently shaken neighbors in one calibration.
pub const SAMPLE_COUNT: usize = 32;

/// Target acceptance probability for the repository-only offline experiment.
pub const TARGET_ACCEPTANCE: f64 = 0.8;

/// Bounded empirical temperature estimator; no objective calls occur in the solve.
pub struct SaTemperature {
    target: f64,
}

impl SaTemperature {
    /// Bind the estimator to a target probability; invalid targets make `solve` fall back.
    #[must_use]
    pub fn new(target: f64) -> Self {
        Self { target }
    }

    /// Sample independent neighbors of a fixed start through the budgeted search.
    ///
    /// `scratch` is caller-owned storage matching `start`; sampling never mutates `start`.
    /// Every sample counts against the evaluation/time budget and updates the search incumbent.
    ///
    /// # Errors
    ///
    /// Propagates `Stop` if the budget ends, cancellation occurs or a score is nonfinite.
    pub fn calibrate(
        &self,
        start: &[f64],
        start_cost: f64,
        widths: [f64; 2],
        context: &mut Context,
        scratch: &mut [f64],
    ) -> Result<Option<f64>, Stop> {
        let mut deltas = [0.0; SAMPLE_COUNT];
        let mut count = 0;
        for _ in 0..SAMPLE_COUNT {
            scratch.copy_from_slice(start);
            shake(scratch, widths[0], widths[1], context.rng);
            let delta = context.search.eval(scratch)? - start_cost;
            if delta.is_finite() && delta > 0.0 {
                deltas[count] = delta;
                count += 1;
            }
        }
        Ok(self.solve(&deltas[..count]))
    }

    /// Validate the target and every sampled uphill cost difference.
    fn usable(&self, deltas: &[f64]) -> bool {
        self.target > 0.0
            && self.target < 1.0
            && !deltas.is_empty()
            && deltas.iter().all(|delta| delta.is_finite() && *delta > 0.0)
    }

    /// Return a finite positive temperature, or `None` for unusable input or overflow.
    #[must_use]
    pub fn solve(&self, deltas: &[f64]) -> Option<f64> {
        if !self.usable(deltas) {
            return None;
        }
        let maximum = deltas.iter().copied().fold(0.0_f64, f64::max);
        let mut low = 0.0;
        let mut high = -1.0 / self.target.ln();
        for _ in 0..64 {
            let middle = 0.5 * (low + high);
            let acceptance = deltas
                .iter()
                .map(|delta| (-(delta / maximum) / middle).exp())
                .sum::<f64>()
                / deltas.len() as f64;
            if acceptance < self.target {
                low = middle;
            } else {
                high = middle;
            }
        }
        let temperature = maximum * (0.5 * (low + high));
        (temperature.is_finite() && temperature > 0.0).then_some(temperature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn acceptance(deltas: &[f64], temperature: f64) -> f64 {
        deltas
            .iter()
            .map(|delta| (-delta / temperature).exp())
            .sum::<f64>()
            / deltas.len() as f64
    }

    #[test]
    fn matches_the_local_target() {
        let deltas = [1.0, 2.0, 3.0, 0.5, 100.0];
        let temperature = SaTemperature::new(0.8).solve(&deltas).unwrap();
        assert!((acceptance(&deltas, temperature) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn invalid_inputs_fall_back() {
        for target in [0.0, 1.0, -0.1, f64::NAN, f64::INFINITY] {
            assert_eq!(SaTemperature::new(target).solve(&[1.0]), None);
        }
        let estimator = SaTemperature::new(0.8);
        for deltas in [&[][..], &[0.0], &[-1.0], &[f64::NAN], &[f64::INFINITY]] {
            assert_eq!(estimator.solve(deltas), None);
        }
        assert_eq!(estimator.solve(&[1.0, f64::NAN]), None);
    }

    #[test]
    fn positive_scaling_preserves_acceptance_for_generated_samples() {
        let estimator = SaTemperature::new(0.8);
        for seed in 0..128 {
            let mut rng = Rng::new(seed);
            let deltas: Vec<f64> = (0..32).map(|_| 0.01 + 100.0 * rng.f64()).collect();
            let base = estimator.solve(&deltas).unwrap();
            for factor in [1e-200, 1e-12, 1.0, 1e12, 1e200] {
                let scaled: Vec<f64> = deltas.iter().map(|delta| factor * delta).collect();
                let temperature = estimator.solve(&scaled).unwrap();
                assert!((temperature / (factor * base) - 1.0).abs() < 1e-12);
                assert!((acceptance(&scaled, temperature) - 0.8).abs() < 1e-12);
            }
        }
    }
}
