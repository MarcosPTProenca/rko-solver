//! Q-learning controller for online parameter adaptation.
//!
//! Objective:
//! - Choose metaheuristic parameter vectors during a search by learning, over a
//!   discrete state graph, which single-axis moves lead to improvements.
//!
//! Technical approach:
//! - Each feasible Cartesian parameter combination is a state; actions connect states
//!   that differ in at most one axis (including staying put).
//! - Initial Q-values are uniform in `[0.005, 0.01)`; selection is epsilon-greedy with
//!   a cosine schedule that restarts every tenth of the search progress with a lower ceiling.
//! - Progress is `max(elapsed / horizon, evaluations / max_evaluations)` in `[0, 1]`, so the
//!   schedule is the same for a 2 s and a 100 s run and for runs bounded only by evaluations.
//! - Rewards are 1 for a strict improvement and otherwise `-min(1, (candidate - previous) /
//!   max(|previous|, |candidate|))`, which does not depend on the scale of the objective.
//! - The default (`progress_schedule` false) is the legacy whole-second window and the fixed
//!   reward scale of 1; the progress schedule and scale-free reward above are opt-in.
//!
//! Flow (pseudocode):
//! 1. Build states and distance-one actions from the filtered grid.
//! 2. Per decision: update epsilon/learning factor from progress, pick an action, adopt its state.
//! 3. After observing the outcome, apply the Bellman update to that action.
//!
//! Important criteria:
//! - The greedy action is the first maximum after every update.
//! - Learning factor decays linearly with progress to 0.1; discount factor is 0.8.
//! - Epsilon ceiling drops by 0.1 for every tenth of progress crossed, down to 0.1.
//!
//! Inputs/Outputs:
//! - Input: feasible parameter vectors, the time horizon, search progress and an `Rng`.
//! - Output: chosen state indices, learning factors and updated Q-values.

use crate::mh::Context;
use crate::mh::Spec;
use crate::rng::Rng;
use crate::search::Search;
use crate::search::Stop;

/// Number of epsilon restart windows per search horizon (each spans a tenth of progress).
const RESTART_WINDOWS: f64 = 10.0;

/// Smallest cost scale used as a divisor by the scale-free reward.
pub const REWARD_SCALE_FLOOR: f64 = f64::MIN_POSITIVE;

/// Parameter vector with its outgoing actions and Q-values.
#[derive(Clone, Debug)]
pub struct QState {
    /// Parameter values in axis order.
    pub parameters: Vec<f64>,
    /// Destination state of each action.
    pub actions: Vec<usize>,
    /// Q-value of each action.
    pub q: Vec<f64>,
    /// Largest Q-value among actions.
    pub max_q: f64,
    /// Index of the first action holding `max_q`.
    pub max_action: usize,
}

/// Exploration and learning schedule produced for one decision.
#[derive(Clone, Copy, Debug)]
pub struct Schedule {
    /// Exploration probability.
    pub epsilon: f64,
    /// Learning factor.
    pub learning: f64,
    /// Discount factor.
    pub discount: f64,
}

/// Q-learning state graph and schedule.
#[derive(Clone, Debug)]
pub struct QLearning {
    states: Vec<QState>,
    epsilon_max: f64,
    epsilon_min: f64,
    restart: usize,
    horizon: f64,
    window: usize,
    progress_schedule: bool,
}

impl QLearning {
    /// Enumerates the Cartesian product of `axes`, keeping combinations accepted by `keep`.
    #[must_use]
    pub fn grid_states(axes: &[Vec<f64>], keep: impl Fn(&[f64]) -> bool) -> Vec<Vec<f64>> {
        let mut states = Vec::new();
        if axes.is_empty() || axes.iter().any(Vec::is_empty) {
            return states;
        }
        let mut counters = vec![0_usize; axes.len()];
        loop {
            let candidate: Vec<f64> = counters
                .iter()
                .zip(axes)
                .map(|(&index, axis)| axis[index])
                .collect();
            if keep(&candidate) {
                states.push(candidate);
            }
            let mut position = axes.len();
            loop {
                if position == 0 {
                    return states;
                }
                position -= 1;
                counters[position] += 1;
                if counters[position] < axes[position].len() {
                    break;
                }
                counters[position] = 0;
            }
        }
    }

    /// Builds the graph over `parameters` (already feasible, in enumeration order).
    pub fn new(parameters: Vec<Vec<f64>>, horizon: f64, rng: &mut Rng) -> Result<Self, String> {
        if parameters.is_empty() {
            return Err("Online parameters contain no feasible states.".to_owned());
        }
        let mut states: Vec<QState> = parameters
            .iter()
            .map(|values| QState {
                parameters: values.clone(),
                actions: Vec::new(),
                q: Vec::new(),
                max_q: 0.0,
                max_action: 0,
            })
            .collect();
        for from in 0..states.len() {
            for to in 0..states.len() {
                let distance = parameters[from]
                    .iter()
                    .zip(&parameters[to])
                    .filter(|(left, right)| left != right)
                    .count();
                if distance <= 1 {
                    let initial = rng.uniform(0.005, 0.01);
                    let state = &mut states[from];
                    state.actions.push(to);
                    state.q.push(initial);
                    if initial > state.max_q {
                        state.max_q = initial;
                        state.max_action = state.actions.len() - 1;
                    }
                }
            }
        }
        Ok(Self {
            states,
            epsilon_max: 1.0,
            epsilon_min: 0.1,
            restart: 1,
            horizon,
            window: ((horizon * 0.1) as usize).max(1),
            progress_schedule: false,
        })
    }

    /// Builds the controller of one search from its spec, honouring the context's schedule mode.
    ///
    /// # Errors
    ///
    /// Returns `Stop::InvalidCost` when the spec holds no feasible online state.
    pub fn for_context(spec: &Spec, context: &mut Context<'_, '_>) -> Result<Self, Stop> {
        let controller = Self::new(spec.states.clone(), context.search.budget(), context.rng)
            .map_err(|_| Stop::InvalidCost)?;
        Ok(controller.with_progress_schedule(context.progress_schedule))
    }

    /// Selects the progress schedule and scale-free reward; false keeps the legacy mode.
    #[must_use]
    pub fn with_progress_schedule(mut self, progress_schedule: bool) -> Self {
        self.progress_schedule = progress_schedule;
        self
    }

    /// Whether the progress schedule and scale-free reward are active.
    #[must_use]
    pub fn progress_schedule(&self) -> bool {
        self.progress_schedule
    }

    /// Number of states.
    #[must_use]
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Whether the graph has no states.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Read access to a state.
    #[must_use]
    pub fn state(&self, index: usize) -> &QState {
        &self.states[index]
    }

    /// Computes the schedule of one decision from the search clock and evaluation counter.
    pub fn schedule(&mut self, search: &Search<'_>) -> Schedule {
        self.schedule_at(search.elapsed(), search.progress())
    }

    /// Computes epsilon, learning and discount factors for one decision.
    ///
    /// # Arguments
    ///
    /// * `elapsed` - Seconds since the search started; used only by the legacy schedule.
    /// * `progress` - Search progress in `[0, 1]`; values outside are clamped.
    pub fn schedule_at(&mut self, elapsed: f64, progress: f64) -> Schedule {
        if !self.progress_schedule {
            return self.legacy_schedule(elapsed);
        }
        let scaled = progress.clamp(0.0, 1.0) * RESTART_WINDOWS;
        while self.restart as f64 <= scaled {
            self.restart += 1;
            self.epsilon_max = (self.epsilon_max - 0.1).max(self.epsilon_min);
        }
        let phase = scaled.fract();
        let epsilon = self.epsilon_min
            + 0.5
                * (self.epsilon_max - self.epsilon_min)
                * (1.0 + (phase * std::f64::consts::PI).cos());
        Schedule {
            epsilon,
            learning: 1.0 - 0.9 * progress.clamp(0.0, 1.0),
            discount: 0.8,
        }
    }

    /// Earlier schedule: whole-second restart windows of a tenth of the horizon.
    fn legacy_schedule(&mut self, elapsed: f64) -> Schedule {
        self.window = ((self.horizon * 0.1) as usize).max(1);
        let epsilon = if elapsed >= (self.restart * self.window) as f64 {
            self.restart += 1;
            self.epsilon_max = (self.epsilon_max - 0.1).max(self.epsilon_min);
            self.epsilon_max
        } else {
            let phase = ((elapsed as usize) % self.window) as f64 / self.window as f64;
            self.epsilon_min
                + 0.5
                    * (self.epsilon_max - self.epsilon_min)
                    * (1.0 + (phase * std::f64::consts::PI).cos())
        };
        let learning = if self.horizon > 0.0 {
            1.0 - 0.9 * elapsed / self.horizon
        } else {
            1.0
        };
        Schedule {
            epsilon,
            learning,
            discount: 0.8,
        }
    }

    /// Selects an action index of `state` with an epsilon-greedy policy.
    pub fn choose_action(&self, state: usize, epsilon: f64, rng: &mut Rng) -> usize {
        let node = &self.states[state];
        if rng.f64() <= 1.0 - epsilon {
            node.max_action
        } else {
            rng.below(node.actions.len())
        }
    }

    /// Applies the Bellman update to `(state, action)`.
    pub fn update(&mut self, state: usize, action: usize, reward: f64, schedule: Schedule) {
        let next = self.states[state].actions[action];
        let next_max = self.states[next].max_q;
        let node = &mut self.states[state];
        let old = node.q[action];
        node.q[action] = old + schedule.learning * (reward + schedule.discount * next_max - old);
        let mut best = 0;
        for (index, &value) in node.q.iter().enumerate() {
            if value > node.q[best] {
                best = index;
            }
        }
        node.max_action = best;
        node.max_q = node.q[best];
    }

    /// Reward for strict improvement, or a scale-free penalty in `[-1, 0]` otherwise.
    ///
    /// The penalty is `-min(1, (candidate - previous) / max(|previous|, |candidate|))`, so it
    /// depends only on the relative worsening and is zero when the costs are equal.
    #[must_use]
    pub fn progress_reward(previous_best: f64, candidate: f64) -> f64 {
        if candidate < previous_best {
            return 1.0;
        }
        if candidate == previous_best {
            return 0.0;
        }
        let scale = previous_best
            .abs()
            .max(candidate.abs())
            .max(REWARD_SCALE_FLOOR);
        -((candidate - previous_best) / scale).min(1.0)
    }

    /// Earlier reward: improvement is 1, otherwise the difference over `max(|a|, |b|, 1)`.
    #[must_use]
    pub fn relative_reward(previous_best: f64, candidate: f64) -> f64 {
        if candidate < previous_best {
            return 1.0;
        }
        let scale = previous_best.abs().max(candidate.abs()).max(1.0);
        (previous_best / scale - candidate / scale).max(-1.0)
    }

    /// Reward under the active mode.
    #[must_use]
    pub fn reward(&self, previous_best: f64, candidate: f64) -> f64 {
        if self.progress_schedule {
            Self::progress_reward(previous_best, candidate)
        } else {
            Self::relative_reward(previous_best, candidate)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axes() -> Vec<Vec<f64>> {
        vec![vec![1.0, 2.0], vec![10.0, 20.0, 30.0]]
    }

    #[test]
    fn grid_enumeration_is_row_major_and_filtered() {
        let all = QLearning::grid_states(&axes(), |_| true);
        assert_eq!(all.len(), 6);
        assert_eq!(all[0], vec![1.0, 10.0]);
        assert_eq!(all[1], vec![1.0, 20.0]);
        assert_eq!(all[3], vec![2.0, 10.0]);
        let kept = QLearning::grid_states(&axes(), |state| state[0] * 10.0 <= state[1]);
        assert_eq!(kept.len(), 5);
    }

    #[test]
    fn actions_connect_distance_at_most_one() {
        let mut rng = Rng::new(1);
        let states = QLearning::grid_states(&axes(), |_| true);
        let controller = QLearning::new(states, 10.0, &mut rng).unwrap();
        let first = controller.state(0);
        assert_eq!(first.actions, vec![0, 1, 2, 3]);
        assert!(first.q.iter().all(|value| (0.005..0.01).contains(value)));
        assert!(QLearning::new(Vec::new(), 1.0, &mut rng).is_err());
    }

    #[test]
    fn update_moves_greedy_action_to_rewarded_one() {
        let mut rng = Rng::new(2);
        let states = QLearning::grid_states(&axes(), |_| true);
        let mut controller = QLearning::new(states, 10.0, &mut rng).unwrap();
        let schedule = controller.schedule_at(0.0, 0.0);
        controller.update(0, 2, 1.0, schedule);
        assert_eq!(controller.state(0).max_action, 2);
        assert!(controller.state(0).max_q > 0.5);
    }

    #[test]
    fn schedule_decays_learning_and_restarts_epsilon() {
        let mut rng = Rng::new(3);
        let states = QLearning::grid_states(&axes(), |_| true);
        let mut controller = QLearning::new(states, 10.0, &mut rng).unwrap();
        let start = controller.schedule_at(0.0, 0.0);
        assert!((start.epsilon - 1.0).abs() < 1e-12);
        assert!((start.learning - 1.0).abs() < 1e-12);
        let late = controller.schedule_at(9.5, 0.95);
        assert!((late.learning - 0.145).abs() < 1e-9);
        assert!(late.epsilon <= 1.0 && late.epsilon >= 0.1);
    }

    fn trace(horizon: f64, progress_schedule: bool) -> Vec<f64> {
        let mut rng = Rng::new(3);
        let states = QLearning::grid_states(&axes(), |_| true);
        let mut controller = QLearning::new(states, horizon, &mut rng)
            .unwrap()
            .with_progress_schedule(progress_schedule);
        (0..=20)
            .map(|step| {
                let share = f64::from(step) / 20.0;
                controller.schedule_at(horizon * share, share).epsilon
            })
            .collect()
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (left, right) in actual.iter().zip(expected) {
            assert!((left - right).abs() < 1e-9, "{actual:?} != {expected:?}");
        }
    }

    const PROGRESS_TRACE: [f64; 21] = [
        1.0, 0.55, 0.9, 0.5, 0.8, 0.45, 0.7, 0.4, 0.6, 0.35, 0.5, 0.3, 0.4, 0.25, 0.3, 0.2, 0.2,
        0.15, 0.1, 0.1, 0.1,
    ];

    #[test]
    fn progress_schedule_is_the_same_for_every_horizon() {
        for horizon in [2.0, 10.0, 100.0] {
            assert_close(&trace(horizon, true), &PROGRESS_TRACE);
        }
    }

    #[test]
    fn progress_schedule_follows_evaluation_counts_without_a_clock() {
        let mut rng = Rng::new(3);
        let states = QLearning::grid_states(&axes(), |_| true);
        let mut controller = QLearning::new(states, 1000.0, &mut rng)
            .unwrap()
            .with_progress_schedule(true);
        let counted: Vec<f64> = (0..=20)
            .map(|step| {
                controller
                    .schedule_at(0.0, f64::from(step * 20) / 400.0)
                    .epsilon
            })
            .collect();
        assert_close(&counted, &PROGRESS_TRACE);
        assert!((controller.schedule_at(0.0, 1.0).learning - 0.1).abs() < 1e-12);
        assert!((controller.schedule_at(0.0, 7.0).learning - 0.1).abs() < 1e-12);
    }

    #[test]
    fn legacy_schedule_keeps_the_whole_second_window() {
        let two = trace(2.0, false);
        assert_close(&two[..10], &[1.0; 10]);
        assert_close(&two[10..20], &[0.9; 10]);
        assert_close(
            &trace(10.0, false),
            &[
                1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3,
                0.2, 0.2, 0.1, 0.1, 0.1,
            ],
        );
        assert_close(&trace(100.0, false), &PROGRESS_TRACE);
    }

    #[test]
    fn reward_rules() {
        assert_eq!(QLearning::progress_reward(10.0, 9.0), 1.0);
        assert_eq!(QLearning::progress_reward(10.0, 10.0), 0.0);
        assert_eq!(QLearning::progress_reward(0.0, 0.0), 0.0);
        assert!(QLearning::progress_reward(10.0, 1000.0) >= -1.0);
        assert!(QLearning::progress_reward(-100.0, -50.0) < 0.0);
        assert_eq!(QLearning::progress_reward(0.0, 5.0), -1.0);
        assert_eq!(QLearning::progress_reward(1e308, -1e308), 1.0);
        assert_eq!(QLearning::progress_reward(-1e308, 1e308), -1.0);
    }

    #[test]
    fn penalty_does_not_depend_on_cost_scale() {
        let small = QLearning::progress_reward(1e-6, 1.5e-6);
        let large = QLearning::progress_reward(1e6, 1.5e6);
        assert!((small - large).abs() < 1e-12);
        assert!((small + 1.0 / 3.0).abs() < 1e-12);
        let legacy_small = QLearning::relative_reward(1e-6, 1.5e-6);
        assert!((legacy_small + 5e-7).abs() < 1e-12);
    }

    #[test]
    fn legacy_reward_keeps_the_fixed_scale() {
        assert_eq!(QLearning::relative_reward(0.0, 0.5), -0.5);
        assert!((QLearning::relative_reward(1.0, 3.0) + 2.0 / 3.0).abs() < 1e-12);
        let mut rng = Rng::new(3);
        let states = QLearning::grid_states(&axes(), |_| true);
        let controller = QLearning::new(states, 1.0, &mut rng)
            .unwrap()
            .with_progress_schedule(false);
        assert_eq!(controller.reward(0.0, 0.5), -0.5);
    }
}
