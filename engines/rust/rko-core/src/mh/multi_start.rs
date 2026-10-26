//! Random multi-start sampling.
//!
//! Objective:
//! - Baseline search that scores independent uniform random key vectors.
//!
//! Technical approach:
//! - Draw, evaluate and keep the best vector; every improvement enters the pool.
//!
//! Flow (pseudocode):
//! 1. Draw a random vector and score it.
//! 2. Publish strict improvements to the shared pool.
//! 3. Repeat until the budget ends.
//!
//! Important criteria:
//! - No parameters; complexity is one evaluation per draw.
//!
//! Inputs/Outputs:
//! - Input: a search `Context`.
//! - Output: `Result<(), Stop>`; the incumbent and pool carry the result.

use super::{Context, Spec};
use crate::operators::random_keys;
use crate::search::Stop;

/// Runs random multi-start until the search budget ends.
pub fn run(_spec: &Spec, context: &mut Context<'_, '_>) -> Result<(), Stop> {
    let n = context.search.dimension();
    let block = context.search.batch_size().max(1);
    let mut keys = vec![0.0; n];
    random_keys(&mut keys, context.rng);
    let mut best = context.search.eval(&keys)?;
    context.pool.insert(best, &keys, context.source);
    let mut rows = vec![0.0; n * block];
    let mut costs = vec![0.0; block];
    loop {
        random_keys(&mut rows, context.rng);
        context.search.eval_rows(&rows, &mut costs)?;
        for (row, &cost) in costs.iter().enumerate() {
            if cost < best {
                best = cost;
                context
                    .pool
                    .insert(best, &rows[row * n..(row + 1) * n], context.source);
            }
        }
    }
}
