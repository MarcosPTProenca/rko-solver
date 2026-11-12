use super::*;
use crate::problem::Knapsack;

fn fixture() -> Knapsack {
    Knapsack::new(&[10, 20, 30], &[5, 6, 7], 12)
}

#[test]
fn tracks_best_and_counts() {
    let problem = fixture();
    let cancel = AtomicBool::new(false);
    let now = Instant::now();
    let mut search = Search::new(&problem, &cancel, now, now + Duration::from_secs(5), 0);
    assert_eq!(search.eval(&[0.9, 0.1, 0.1]).unwrap(), -10.0);
    assert_eq!(search.eval(&[0.9, 0.9, 0.1]).unwrap(), -30.0);
    assert_eq!(search.eval(&[0.1, 0.1, 0.1]).unwrap(), 0.0);
    assert_eq!(search.evaluations(), 3);
    assert_eq!(search.best_cost(), -30.0);
    assert_eq!(search.best_keys(), &[0.9, 0.9, 0.1]);
}

#[test]
fn annealing_trace_records_start_and_levels() {
    let problem = fixture();
    let cancel = AtomicBool::new(false);
    let mut search = Search::new(&problem, &cancel, Instant::now(), Instant::now(), 0);
    assert!(search.annealing().is_none());
    search.count_annealing_level();
    assert!(search.annealing().is_none());
    search.record_calibration(2.5e5, 40.0);
    search.begin_annealing(340.0, 3.4e-5, true);
    search.count_annealing_level();
    search.count_annealing_level();
    let trace = search.annealing().expect("trace after begin");
    assert_eq!(trace.t0, 340.0);
    assert_eq!(trace.cutoff, 3.4e-5);
    assert_eq!(trace.levels, 2);
    assert!(trace.calibrated);
    assert_eq!(trace.calibration, Some((2.5e5, 40.0)));
}

#[test]
fn first_evaluation_survives_expired_deadline() {
    let problem = fixture();
    let cancel = AtomicBool::new(false);
    let now = Instant::now();
    let mut search = Search::new(&problem, &cancel, now, now, 0);
    assert!(search.eval(&[0.9, 0.1, 0.1]).is_ok());
    let mut stopped = false;
    for _ in 0..16 {
        if search.eval(&[0.9, 0.1, 0.1]) == Err(Stop::TimeLimit) {
            stopped = true;
            break;
        }
    }
    assert!(stopped);
}

#[test]
fn cancellation_is_immediate() {
    let problem = fixture();
    let cancel = AtomicBool::new(true);
    let now = Instant::now();
    let mut search = Search::new(&problem, &cancel, now, now + Duration::from_secs(5), 0);
    assert_eq!(search.eval(&[0.9, 0.1, 0.1]), Err(Stop::Cancelled));
}

#[test]
fn probe_matches_eval_with_and_without_incremental_state() {
    use crate::problem::test_support::DeltaKnapsack;
    let knapsack = Knapsack::new(&[10, 20, 30, 15], &[5, 6, 7, 4], 12);
    let fast = DeltaKnapsack(knapsack.clone());
    let cancel = AtomicBool::new(false);
    let now = Instant::now();
    for incremental in [true, false] {
        let problem: &dyn Problem = if incremental { &fast } else { &knapsack };
        let mut search = Search::new(problem, &cancel, now, now + Duration::from_secs(5), 0);
        let mut keys = vec![0.9, 0.1, 0.9, 0.1];
        search.eval(&keys).unwrap();
        search.sync(&keys);
        assert_eq!(search.is_incremental(), incremental);
        let snapshot = keys.clone();
        let changes = [(1, 0.8), (3, 0.7)];
        let moved = search.probe(&mut keys, &changes).unwrap();
        assert_eq!(keys, snapshot, "probe must restore keys");
        let mut expected = snapshot.clone();
        expected[1] = 0.8;
        expected[3] = 0.7;
        assert_eq!(moved, knapsack.evaluate(&expected));
        search.commit(&mut keys, &changes);
        assert_eq!(keys, expected);
        assert_eq!(
            search.probe(&mut keys, &[]).unwrap(),
            knapsack.evaluate(&expected)
        );
        assert_eq!(search.evaluations(), 3);
    }
}

#[test]
fn evaluation_limit_is_exact_for_eval_probe_and_check() {
    let problem = fixture();
    let cancel = AtomicBool::new(false);
    let now = Instant::now();
    let mut search = Search::new(&problem, &cancel, now, now + Duration::from_secs(5), 0)
        .with_max_evaluations(3);
    let mut keys = vec![0.9, 0.9, 0.1];
    assert!(search.eval(&keys).is_ok());
    assert!(search.probe(&mut keys, &[(2, 0.8)]).is_ok());
    assert!(search.check().is_ok());
    assert!(search.eval(&keys).is_ok());
    assert_eq!(search.check(), Err(Stop::MaxEvaluations));
    assert_eq!(search.eval(&keys), Err(Stop::MaxEvaluations));
    assert_eq!(search.probe(&mut keys, &[]), Err(Stop::MaxEvaluations));
    assert_eq!(search.evaluations(), 3);
    assert_eq!(Stop::MaxEvaluations.label(), "max_evaluations");
}

#[test]
fn a_zero_limit_is_raised_to_one_and_cancellation_wins() {
    let problem = fixture();
    let cancel = AtomicBool::new(false);
    let now = Instant::now();
    let mut search = Search::new(&problem, &cancel, now, now + Duration::from_secs(5), 0)
        .with_max_evaluations(0);
    assert!(search.eval(&[0.9, 0.1, 0.1]).is_ok());
    assert_eq!(search.eval(&[0.9, 0.1, 0.1]), Err(Stop::MaxEvaluations));
    cancel.store(true, Ordering::Relaxed);
    assert_eq!(search.eval(&[0.9, 0.1, 0.1]), Err(Stop::Cancelled));
}

#[test]
fn a_block_is_truncated_at_the_limit_and_reports_the_stop() {
    use crate::problem::test_support::BatchKnapsack;
    let problem = BatchKnapsack::new(fixture(), 4);
    let cancel = AtomicBool::new(false);
    let now = Instant::now();
    let mut search = Search::new(&problem, &cancel, now, now + Duration::from_secs(5), 0)
        .with_max_evaluations(6);
    let rows = [0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9];
    let mut out = [0.0; 4];
    assert!(search.eval_batch(&rows, &mut out).is_ok());
    assert_eq!(search.evaluations(), 4);
    assert_eq!(
        search.eval_batch(&rows, &mut out),
        Err(Stop::MaxEvaluations)
    );
    assert_eq!(search.evaluations(), 6);
    assert_eq!(out[..2], [-10.0, -30.0]);
    assert_eq!(
        search.eval_batch(&rows, &mut out),
        Err(Stop::MaxEvaluations)
    );
    assert_eq!(search.evaluations(), 6);
    assert_eq!(search.best_cost(), -30.0);
}

#[test]
fn history_compaction_keeps_cap_and_endpoints() {
    struct Descending;
    impl Problem for Descending {
        fn dimension(&self) -> usize {
            1
        }
        fn evaluate(&self, keys: &[f64]) -> f64 {
            -keys[0]
        }
    }
    let problem = Descending;
    let cancel = AtomicBool::new(false);
    let now = Instant::now();
    let mut search = Search::new(&problem, &cancel, now, now + Duration::from_secs(5), 8);
    for step in 0..100 {
        search.eval(&[f64::from(step) / 100.0]).unwrap();
    }
    search.finish_history();
    assert!(search.history().len() <= 8);
    assert_eq!(search.history().first().unwrap().evaluations, 1);
    assert_eq!(search.history().last().unwrap().cost, -0.99);
    assert_eq!(search.history_observations(), 101);
}
