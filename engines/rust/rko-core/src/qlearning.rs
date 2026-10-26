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
//!   a cosine schedule that restarts every tenth of the horizon with a lower ceiling.
//! - Rewards are 1 for a strict improvement and otherwise a bounded relative penalty.
//!
//! Flow (pseudocode):
//! 1. Build states and distance-one actions from the filtered grid.
//! 2. Per decision: update epsilon/learning factor, pick an action, adopt its state.
//! 3. After observing the outcome, apply the Bellman update to that action.
//!
//! Important criteria:
//! - The greedy action is the first maximum after every update.
//! - Learning factor decays linearly to 0.1; discount factor is 0.8.
//!
//! Inputs/Outputs:
//! - Input: feasible parameter vectors, the time horizon and an `Rng`.
//! - Output: chosen state indices, learning factors and updated Q-values.

use crate::rng::Rng;

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
        })
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

    /// Computes epsilon, learning and discount factors at `elapsed` seconds.
    pub fn schedule(&mut self, elapsed: f64) -> Schedule {
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

    /// Reward for strict improvement, or a bounded relative penalty otherwise.
    #[must_use]
    pub fn relative_reward(previous_best: f64, candidate: f64) -> f64 {
        if candidate < previous_best {
            return 1.0;
        }
        let scale = previous_best.abs().max(candidate.abs()).max(1.0);
        (previous_best / scale - candidate / scale).max(-1.0)
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
        let schedule = controller.schedule(0.0);
        controller.update(0, 2, 1.0, schedule);
        assert_eq!(controller.state(0).max_action, 2);
        assert!(controller.state(0).max_q > 0.5);
    }

    #[test]
    fn schedule_decays_learning_and_restarts_epsilon() {
        let mut rng = Rng::new(3);
        let states = QLearning::grid_states(&axes(), |_| true);
        let mut controller = QLearning::new(states, 10.0, &mut rng).unwrap();
        let start = controller.schedule(0.0);
        assert!((start.epsilon - 1.0).abs() < 1e-12);
        assert!((start.learning - 1.0).abs() < 1e-12);
        let late = controller.schedule(9.5);
        assert!((late.learning - 0.145).abs() < 1e-9);
        assert!(late.epsilon <= 1.0 && late.epsilon >= 0.1);
    }

    #[test]
    fn reward_rules() {
        assert_eq!(QLearning::relative_reward(10.0, 9.0), 1.0);
        assert_eq!(QLearning::relative_reward(10.0, 10.0), 0.0);
        assert!(QLearning::relative_reward(10.0, 1000.0) >= -1.0);
        assert!(QLearning::relative_reward(-100.0, -50.0) < 0.0);
    }
}
