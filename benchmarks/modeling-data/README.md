# Practical model input provenance

The numerical data and mathematical constraints are frozen in
`../modeling_cases.py`. Native construction and validation code are authored for
this repository; upstream solver calls are not copied into RKO.

| Cases | Primary source and attribution | Explicit adaptation |
| --- | --- | --- |
| Production LP | Jeffrey Kantor, [ND Pyomo Cookbook §2.1](https://jckantor.github.io/ND-Pyomo-Cookbook/notebooks/02.01-Production-Models-with-Linear-Constraints.html) | Declare implied finite bounds x≤40 and y≤80. Original objective and demand/labor constraints unchanged. |
| Integer production | Same production data | Require integer product units. Known optimum remains 2,600. |
| Production setup MILP | Original extension of the same production data | Add binary X activation, x≤40·activation and a 100-unit setup cost. Optimum 2,500. |
| Transportation LP | Antony Phillips and Stuart Mitchell, [PuLP Beer Distribution case study](https://coin-or.github.io/pulp/CaseStudies/a_transportation_problem.html), [example source](https://github.com/coin-or/pulp/blob/master/examples/BeerDistributionProblem.py) | Continuous-flow relaxation; declare each route's warehouse capacity as an implied finite upper bound. Original supply/minimum-demand rows and costs unchanged. Optimum 8,600. |
| Exact-demand transportation | Same Beer Distribution data | Replace minimum-demand rows by equalities as an explicit continuous-equality stress case. Optimum remains 8,600. |
| Binary Knapsack | [Bundled kp50 provenance](../../examples/instances/README.md), upstream RKO_Cpp_v2.0 revision `126e35873e50dc308795b58b8c75d37f8de4c857` | Unchanged numerical data; standard binary mathematical formulation rather than the handwritten greedy decoder. DP optimum 149. |
| Worker/task assignment | Google, [OR-Tools assignment example](https://developers.google.com/optimization/assignment/assignment_example) | Same five-worker/four-task cost matrix, worker upper bounds and task equalities, reconstructed in each supported frontend. Optimum 265. This does not add MPSolver/CP-SAT frontend support. |
| Inconsistent production | Original extension of the production data | Add x≥30 and x≤10 while leaving the variable box valid. The independent reference solver reports infeasible; RKO must only report no-feasible-found. |

Retained code/data attribution notices:

- `LICENSE-pyomo-cookbook.txt`: MIT, copyright 2019 Jeffrey Kantor.
- `LICENSE-pulp.txt`: MIT, Jean-Sebastien Roy and Stuart Anthony Mitchell.
- `LICENSE-or-tools.txt`: Apache License 2.0; Google code samples are covered by
  that license. The [Google page policy](https://developers.google.com/site-policies)
  identifies prose as CC BY 4.0; no prose has been copied into these fixtures.
- Bundled Knapsack retains its upstream MIT notice in
  `../../examples/instances/LICENSE`.

These are small public teaching/reference instances and explicitly labelled
extensions, **not industrial-scale application certification**. The coefficient
snapshot is independent of future changes to the linked websites. A campaign
metadata record includes every original coefficient/bound, adaptations, fixture
fingerprints, runtime versions and actual production/benchmark source hashes.
