//! Metaheuristic implementations and their shared run context.
//!
//! Objective:
//! - Group the search algorithms behind one calling convention so the portfolio
//!   scheduler can run them uniformly on worker threads.
//!
//! Technical approach:
//! - Each algorithm is a function over a `Context` bundling the budgeted `Search`,
//!   the shared `Pool`, a private `Rng`, and the algorithm's `Spec`.
//! - Algorithms never check the clock themselves: every candidate goes through
//!   `Search::eval`, whose `Stop` error unwinds the loop through `?`.
//! - `Adapter` wraps the optional Q-learning controller so every algorithm adopts
//!   online parameters with the same begin/finish protocol.
//!
//! Flow (pseudocode):
//! 1. The scheduler builds a `Context` per search.
//! 2. The algorithm loops until `Search::eval` returns `Stop`.
//! 3. The scheduler reads the incumbent and counters from the `Search`.
//!
//! Important criteria:
//! - A `Context` is owned by one thread; only the `Pool` is shared.
//! - Parameter vectors follow the axis order of the packaged TOML files.
//!
//! Inputs/Outputs:
//! - Input: a `Spec` (offline values or feasible online states) and a `Context`.
//! - Output: `Result<(), Stop>`; results live in the `Search` incumbent and the pool.

pub mod brkga;
pub mod brkga_cs;
pub mod ga;
pub mod grasp;
pub mod ils;
pub mod ipr;
pub mod lns;
pub mod multi_start;
mod population;
pub mod pso;
pub mod sa;
pub mod vns;

use crate::pool::Pool;
use crate::qlearning::{QLearning, Schedule};
use crate::rng::Rng;
use crate::search::{Search, Stop};

/// Algorithms implemented natively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    /// Simulated annealing.
    Sa,
    /// Iterated local search.
    Ils,
    /// Variable neighborhood search.
    Vns,
    /// Large neighborhood search.
    Lns,
    /// Biased random-key genetic algorithm.
    Brkga,
    /// BRKGA with community-based elite refinement.
    BrkgaCs,
    /// Particle swarm optimization.
    Pso,
    /// Genetic algorithm.
    Ga,
    /// Greedy randomized adaptive search.
    Grasp,
    /// Implicit path relinking.
    Ipr,
    /// Random multi-start.
    MultiStart,
}

impl Algorithm {
    /// Canonical display name, matching the Python registry.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Sa => "SA",
            Self::Ils => "ILS",
            Self::Vns => "VNS",
            Self::Lns => "LNS",
            Self::Brkga => "BRKGA",
            Self::BrkgaCs => "BRKGA_CS",
            Self::Pso => "PSO",
            Self::Ga => "GA",
            Self::Grasp => "GRASP",
            Self::Ipr => "IPR",
            Self::MultiStart => "MultiStart",
        }
    }

    /// Parses a name case-insensitively, accepting `-` or `_` separators.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_uppercase().replace('-', "_").as_str() {
            "SA" => Some(Self::Sa),
            "ILS" => Some(Self::Ils),
            "VNS" => Some(Self::Vns),
            "LNS" => Some(Self::Lns),
            "BRKGA" => Some(Self::Brkga),
            "BRKGA_CS" => Some(Self::BrkgaCs),
            "PSO" => Some(Self::Pso),
            "GA" => Some(Self::Ga),
            "GRASP" => Some(Self::Grasp),
            "IPR" => Some(Self::Ipr),
            "MULTISTART" | "MULTI_START" => Some(Self::MultiStart),
            _ => None,
        }
    }

    /// Every implemented algorithm in canonical order.
    #[must_use]
    pub fn all() -> [Self; 11] {
        [
            Self::Sa,
            Self::Ils,
            Self::Vns,
            Self::Lns,
            Self::Brkga,
            Self::BrkgaCs,
            Self::Pso,
            Self::Ga,
            Self::Grasp,
            Self::Ipr,
            Self::MultiStart,
        ]
    }
}

/// Parameters for one search: fixed values or feasible online states.
#[derive(Clone, Debug)]
pub struct Spec {
    /// Algorithm to run.
    pub algorithm: Algorithm,
    /// Offline parameter values in axis order.
    pub offline: Vec<f64>,
    /// Feasible online parameter vectors in enumeration order.
    pub states: Vec<Vec<f64>>,
    /// Whether Q-learning selects parameters online.
    pub adaptive: bool,
}

/// Everything one search needs while running.
pub struct Context<'a, 'b> {
    /// Budgeted evaluator and incumbent.
    pub search: &'a mut Search<'b>,
    /// Shared elite pool.
    pub pool: &'a Pool,
    /// Private random stream.
    pub rng: &'a mut Rng,
    /// Index recorded as the producer of pool entries.
    pub source: usize,
    /// Local-search sweep strategy: 1 sequential acceptance, 2 fixed-base best.
    pub strategy: u8,
}

impl Algorithm {
    /// Number of values in an online state vector for this algorithm.
    #[must_use]
    pub fn state_len(self) -> usize {
        match self {
            Self::Sa => 5,
            Self::Ils | Self::Vns => 2,
            Self::Lns | Self::Brkga | Self::BrkgaCs | Self::Pso => 4,
            Self::Ga | Self::Grasp => 3,
            Self::Ipr | Self::MultiStart => 0,
        }
    }

    fn offline_lens(self) -> &'static [usize] {
        match self {
            Self::Sa => &[5],
            Self::Ils | Self::Vns | Self::Grasp => &[2],
            Self::Lns | Self::Brkga | Self::Pso => &[4],
            Self::BrkgaCs => &[4, 5],
            Self::Ga => &[3],
            Self::Ipr | Self::MultiStart => &[0],
        }
    }

    /// Checks one parameter vector (`online` selects the GRASP axis layout).
    pub fn validate_values(self, values: &[f64], online: bool) -> Result<(), String> {
        let label = self.name();
        let expected = if online {
            &[self.state_len()][..]
        } else {
            self.offline_lens()
        };
        if !expected.contains(&values.len()) {
            return Err(format!(
                "{label}: expected {expected:?} parameter values, got {}",
                values.len()
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!("{label}: parameters must be finite"));
        }
        let unit = |value: f64| (0.0..=1.0).contains(&value);
        let ok = match self {
            Self::Sa => {
                values[0] >= 1.0
                    && values[1] > 0.0
                    && values[2] > 0.0
                    && values[2] < 1.0
                    && unit(values[3])
                    && unit(values[4])
                    && values[3] <= values[4]
            }
            Self::Ils => unit(values[0]) && unit(values[1]) && values[0] <= values[1],
            Self::Vns => values[0] >= 1.0 && values[1] >= 0.0,
            Self::Lns => {
                unit(values[0])
                    && unit(values[1])
                    && values[0] <= values[1]
                    && values[2] > 0.0
                    && values[3] > 0.0
                    && values[3] < 1.0
            }
            Self::Brkga | Self::BrkgaCs => {
                let population = values[0];
                let elite = (population * values[1]).floor().max(1.0);
                let mutants = if self == Self::Brkga {
                    (population * values[2]).floor()
                } else {
                    0.0
                };
                population >= 2.0
                    && unit(values[1])
                    && unit(values[2])
                    && unit(values[3])
                    && elite + mutants < population
                    && values.get(4).is_none_or(|sigma| unit(*sigma))
            }
            Self::Pso => {
                values[0] >= 1.0 && values[1] >= 0.0 && values[2] >= 0.0 && values[3] >= 0.0
            }
            Self::Ga => values[0] >= 3.0 && unit(values[1]) && unit(values[2]),
            Self::Grasp => {
                let (start, end) = if online {
                    (values[1], values[2])
                } else {
                    (values[0], values[1])
                };
                start > 0.0 && end > 0.0 && end <= start && (!online || unit(values[0]))
            }
            Self::Ipr | Self::MultiStart => true,
        };
        if ok {
            Ok(())
        } else {
            Err(format!(
                "{label}: parameter values {values:?} violate the algorithm's constraints"
            ))
        }
    }
}

impl Spec {
    /// Validates the offline vector or every online state before a run.
    pub fn validate(&self) -> Result<(), String> {
        if self.adaptive && self.algorithm.state_len() > 0 {
            if self.states.is_empty() {
                return Err(format!(
                    "{}: online parameters contain no feasible states",
                    self.algorithm.name()
                ));
            }
            for state in &self.states {
                self.algorithm.validate_values(state, true)?;
            }
            Ok(())
        } else {
            self.algorithm.validate_values(&self.offline, false)
        }
    }

    /// Runs the configured algorithm until the search budget is exhausted.
    pub fn run(&self, context: &mut Context<'_, '_>) -> Result<(), Stop> {
        match self.algorithm {
            Algorithm::Sa => sa::run(self, context),
            Algorithm::Ils => ils::run(self, context),
            Algorithm::Vns => vns::run(self, context),
            Algorithm::Lns => lns::run(self, context),
            Algorithm::Brkga => brkga::run(self, context),
            Algorithm::BrkgaCs => brkga_cs::run(self, context),
            Algorithm::Pso => pso::run(self, context),
            Algorithm::Ga => ga::run(self, context),
            Algorithm::Grasp => grasp::run(self, context),
            Algorithm::Ipr => ipr::run(self, context),
            Algorithm::MultiStart => multi_start::run(self, context),
        }
    }
}

/// Online parameter selection shared by all adaptive algorithms.
pub(crate) struct Adapter {
    controller: Option<QLearning>,
    state: usize,
    action: usize,
    schedule: Option<Schedule>,
    current: Vec<f64>,
}

impl Adapter {
    /// Builds the adapter, choosing a random initial state when adaptive.
    pub(crate) fn new(spec: &Spec, context: &mut Context<'_, '_>) -> Result<Self, Stop> {
        if !spec.adaptive {
            return Ok(Self {
                controller: None,
                state: 0,
                action: 0,
                schedule: None,
                current: spec.offline.clone(),
            });
        }
        let controller = QLearning::new(spec.states.clone(), context.search.budget(), context.rng)
            .map_err(|_| Stop::InvalidCost)?;
        let state = context.rng.below(controller.len());
        let current = controller.state(state).parameters.clone();
        Ok(Self {
            controller: Some(controller),
            state,
            action: 0,
            schedule: None,
            current,
        })
    }

    /// Current parameter vector.
    pub(crate) fn params(&self) -> &[f64] {
        &self.current
    }

    /// Whether Q-learning is active.
    pub(crate) fn adaptive(&self) -> bool {
        self.controller.is_some()
    }

    /// Chooses the next action and adopts its parameters; no-op when offline.
    pub(crate) fn begin(&mut self, context: &mut Context<'_, '_>) {
        if let Some(controller) = self.controller.as_mut() {
            let timing = controller.schedule(context.search.elapsed());
            self.action = controller.choose_action(self.state, timing.epsilon, context.rng);
            let target = controller.state(self.state).actions[self.action];
            self.current
                .clone_from(&controller.state(target).parameters);
            self.schedule = Some(timing);
        }
    }

    /// Rewards the last action and moves to its target state; no-op when offline.
    pub(crate) fn finish(&mut self, reward: f64) {
        if let (Some(controller), Some(timing)) = (self.controller.as_mut(), self.schedule) {
            let next = controller.state(self.state).actions[self.action];
            controller.update(self.state, self.action, reward, timing);
            self.state = next;
        }
    }
}
