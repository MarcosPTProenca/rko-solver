"""Held-out bake-off of parameter tuners for the RKO metaheuristics.

Objective:
- Compare ways of choosing the fixed (offline) parameters of an algorithm, under one equal
  budget of solver runs and one held-out evaluation, and report where each one wins or loses.

Technical approach:
- `spaces` defines each algorithm's tunable parameters on a shared unit cube, `instances`
  generates and calibrates problem instances, `objective` meters solver runs, `tuners` holds the
  methods, `protocol` runs the sessions and `analysis` summarizes the held-out results.

Flow (pseudocode):
1. Generate training and held-out instances per family and calibrate their scale.
2. Give each tuner the same budget of runs on the training instances.
3. Score every returned configuration on the held-out instances and summarize.

Important criteria:
- Held-out instances never reach a tuner; the default and a random configuration are scored
  by the same code as the tuned ones.

Inputs/Outputs:
- Input: algorithm, instance family, tuner, repetition and budget.
- Output: JSON lines of held-out scores and a Markdown summary.
"""
