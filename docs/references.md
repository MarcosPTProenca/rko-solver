# Scientific references

[Documentation index](index.md) · [Limitations](status.md) · [Convergence guarantees](guarantees.md)

This page lists the papers behind the algorithms, components and experiments in this repository, and
the files that implement or test each one, so that a reader can study the source of an idea and find
the code that uses it. It is a bibliography, not a validation: citing a paper here does **not** mean
the code reproduces that paper's results. See the [limitations](status.md) for what is and is not
established.

## How the records were checked

On 2026-10-30 every DOI below was looked up in the Crossref registry and every arXiv item on its
abstract page. Titles, authors, venues, volumes and pages are the registry values. Where the RKO
papers' own reference lists disagree with a registry, the registry value is used and the difference
is noted. Several older publishers block automated access, so the text of those papers was **not
read** for this page: such an entry says what the paper is cited for by the RKO papers or by this
repository's module docstrings, and does not summarise the paper.

Each record carries one label: **Journal article**, **Conference paper**, **Book chapter**, **Book**,
**Preprint**, **Workshop paper**, **Unpublished proposal** or **No published reference**.

## The RKO framework

### RKO optimizer

- **Journal article.** A. A. Chaves, M. G. C. Resende, M. J. A. Schuetz, J. K. Brubaker,
  H. G. Katzgraber, E. F. de Arruda, R. M. A. Silva. "A Random-Key optimizer for combinatorial
  optimization." *Journal of Heuristics* 31(4), article 32, 2025.
  [DOI](https://doi.org/10.1007/s10732-025-09568-z) ·
  [arXiv 2411.04293](https://arxiv.org/abs/2411.04293) (v3, [CC BY 4.0](#local-copies)).
  Framework sections cited below: 4.1 elite pool, 4.2 shaking, 4.3 blending, 4.4 local search,
  4.5 metaheuristics, 4.6 Q-learning.
- **Reference implementation.** RKO C++ v2.0,
  [github.com/RKO-solver/RKO_Cpp_v2.0](https://github.com/RKO-solver/RKO_Cpp_v2.0), MIT license. Its
  README asks users to cite the paper above. The algorithm headers are vendored byte for byte in
  `engines/cpp/vendor/rko_cpp_v2/` (revision recorded in `NOTICE`) and compiled only with
  `RKO_BUILD_CPP=1`.

### RKO on mixed integer programs

- **Preprint.** A. A. Chaves, M. G. C. Resende, C. E. Schmidt, J. K. Brubaker, H. G. Katzgraber,
  M. J. A. Schuetz. "Applying a Random-Key Optimizer on Mixed Integer Programs." arXiv 2602.22173,
  2026. [arXiv](https://arxiv.org/abs/2602.22173) (v2, [CC BY 4.0](#local-copies)). No journal
  version was found.
- This is the closest published work to the model conversion in `docs/modeling.md` and
  `src/rko_solver/modeling/`. That implementation is this repository's own; the paper is related
  reading, not its specification.

### Random-key genetic algorithms, an overview

- **Preprint of a chapter.** M. A. Londe, L. S. Pessoa, C. E. Andrade, J. F. Gonçalves,
  M. G. C. Resende. "Random-key genetic algorithms: Principles and applications." arXiv 2506.02120,
  2025. [arXiv](https://arxiv.org/abs/2506.02120) (v2, [CC BY 4.0](#local-copies)). Background for
  [BRKGA](#biased-random-key-genetic-algorithm-brkga).

## Algorithms

Each entry lists the Python module, the Rust twin and the vendored C++ header. "Cited by the RKO
paper" refers to the arXiv version of the framework paper above.

### Biased random-key genetic algorithm (BRKGA)

- Implemented in `src/rko_solver/metaheuristics/brkga.py`,
  `engines/rust/rko-core/src/mh/brkga.rs`, `engines/cpp/vendor/rko_cpp_v2/src/MH/BRKGA.h`.
  Framework section 4.5.1.
- **Journal article.** J. F. Gonçalves, M. G. C. Resende. "Biased random-key genetic algorithms for
  combinatorial optimization." *Journal of Heuristics* 17(5), 487–525, 2011 (online 2010).
  [DOI](https://doi.org/10.1007/s10732-010-9143-1). The RKO paper's reference list gives issue 1; the
  registry gives issue 5.
- **Journal article.** J. C. Bean. "Genetic algorithms and random keys for sequencing and
  optimization." *ORSA Journal on Computing* 6(2), 154–160, 1994.
  [DOI](https://doi.org/10.1287/ijoc.6.2.154). The framework paper credits it with the first
  random-key genetic algorithm and with random-key encoding for permutation problems.
- Convergence of the mutant channel: see [guarantees](guarantees.md).

### BRKGA with community structure (BRKGA-CS)

- Implemented in `src/rko_solver/metaheuristics/brkga_cs.py`,
  `engines/rust/rko-core/src/mh/brkga_cs.rs`, `engines/cpp/vendor/rko_cpp_v2/src/MH/BRKGA_CS.h`.
- **No published reference.** Neither RKO paper cites a community-structure variant (both texts were
  searched), and the vendored C++ header, which calls the method BRKGA-QL in its own comment, names
  no source. The module docstring describes what the code does.

### Genetic algorithm (GA)

- Implemented in `src/rko_solver/metaheuristics/ga.py`, `engines/rust/rko-core/src/mh/ga.rs`,
  `engines/cpp/vendor/rko_cpp_v2/src/MH/GA.h`. Framework section 4.5.2 (tournament selection,
  blending crossover, mutation).
- **Book.** J. H. Holland. *Adaptation in Natural and Artificial Systems.* MIT Press, 1992. Cited by
  both RKO papers; existence confirmed in the
  [Open Library catalogue](https://openlibrary.org/search?q=holland+adaptation+in+natural+and+artificial+systems).
- **Book.** D. E. Goldberg. *Genetic Algorithms in Search, Optimization, and Machine Learning.*
  Addison-Wesley, 1989. Cited by the framework paper; confirmed in the
  [Open Library catalogue](https://openlibrary.org/search?q=goldberg+genetic+algorithms+in+search+optimization+and+machine+learning).

### GRASP

- Implemented in `src/rko_solver/metaheuristics/grasp.py`, `engines/rust/rko-core/src/mh/grasp.rs`,
  `engines/cpp/vendor/rko_cpp_v2/src/MH/GRASP.h`. Framework section 4.5.4.
- **Journal article.** T. A. Feo, M. G. C. Resende. "Greedy randomized adaptive search procedures."
  *Journal of Global Optimization* 6(2), 109–133, 1995.
  [DOI](https://doi.org/10.1007/BF01096763).
- **Journal article.** A. A. Chaves, M. G. C. Resende, R. M. A. Silva. "A random-key GRASP for
  combinatorial optimization." *Journal of Nonlinear and Variational Analysis* 8, 855–881, 2024.
  [DOI](https://doi.org/10.23952/jnva.8.2024.6.03) · [arXiv 2405.18681](https://arxiv.org/abs/2405.18681)
  (v2, [CC BY 4.0](#local-copies)). Cited by the framework paper.

### Iterated local search (ILS)

- Implemented in `src/rko_solver/metaheuristics/ils.py`, `engines/rust/rko-core/src/mh/ils.rs`,
  `engines/cpp/vendor/rko_cpp_v2/src/MH/ILS.h`. Framework section 4.5.5.
- **Book chapter.** H. R. Lourenço, O. C. Martin, T. Stützle. "Iterated Local Search." In *Handbook
  of Metaheuristics*, 320–353, Springer, 2003.
  [DOI](https://doi.org/10.1007/0-306-48056-5_11). Cited by both RKO papers; the two papers disagree
  on the volume's editors, so none are listed here.
- Uses [RVND](#random-variable-neighborhood-descent-rvnd). Acceptance is improvement-only, as the
  module docstring states, so it does not use the Metropolis criterion.

### Variable neighborhood search (VNS)

- Implemented in `src/rko_solver/metaheuristics/vns.py`, `engines/rust/rko-core/src/mh/vns.rs`,
  `engines/cpp/vendor/rko_cpp_v2/src/MH/VNS.h`. Framework section 4.5.6.
- **Journal article.** N. Mladenović, P. Hansen. "Variable neighborhood search." *Computers &
  Operations Research* 24(11), 1097–1100, 1997.
  [DOI](https://doi.org/10.1016/S0305-0548%2897%2900031-2).

### Large neighborhood search (LNS)

- Implemented in `src/rko_solver/metaheuristics/lns.py`, `engines/rust/rko-core/src/mh/lns.rs`,
  `engines/cpp/vendor/rko_cpp_v2/src/MH/LNS.h`. Framework section 4.5.8: the repair phase tries values
  taken from a [Farey sequence](#farey-local-search) for each removed key, and acceptance follows the
  [Metropolis criterion](#metropolis-acceptance-criterion).
- **Book chapter.** P. Shaw. "Using Constraint Programming and Local Search Methods to Solve Vehicle
  Routing Problems." *Lecture Notes in Computer Science*, 417–431, 1998.
  [DOI](https://doi.org/10.1007/3-540-49481-2_30). An early paper describing large neighbourhood
  search for vehicle routing.
- **Journal article.** S. Ropke, D. Pisinger. "A unified heuristic for a large class of Vehicle
  Routing Problems with Backhauls." *European Journal of Operational Research* 171(3), 750–775, 2006.
  [DOI](https://doi.org/10.1016/j.ejor.2004.09.004). The framework paper's citation for LNS.
- **Book chapter.** D. Pisinger, S. Ropke. "Large Neighborhood Search." In *Handbook of
  Metaheuristics*, 399–419, 2010.
  [DOI](https://doi.org/10.1007/978-1-4419-1665-5_13). The citation used by the mixed-integer-program
  paper instead.

### Particle swarm optimization (PSO)

- Implemented in `src/rko_solver/metaheuristics/pso.py`, `engines/rust/rko-core/src/mh/pso.rs`,
  `engines/cpp/vendor/rko_cpp_v2/src/MH/PSO.h`. Framework section 4.5.7.
- **Conference paper.** J. Kennedy, R. Eberhart. "Particle swarm optimization." *Proceedings of
  ICNN'95, International Conference on Neural Networks* 4, 1942–1948.
  [DOI](https://doi.org/10.1109/ICNN.1995.488968).

### Simulated annealing (SA)

- Implemented in `src/rko_solver/metaheuristics/sa.py`, `engines/rust/rko-core/src/mh/sa.rs`,
  `engines/cpp/vendor/rko_cpp_v2/src/MH/SA.h`. Framework section 4.5.3.
- **Journal article.** S. Kirkpatrick, C. D. Gelatt, M. P. Vecchi. "Optimization by Simulated
  Annealing." *Science* 220(4598), 671–680, 1983.
  [DOI](https://doi.org/10.1126/science.220.4598.671).
- Temperature control by Q-learning: see [Q-learning](#q-learning-parameter-control) and
  [SA control](sa-control.md).

### Implicit path relinking (IPR)

- Implemented in `src/rko_solver/metaheuristics/ipr.py`, `engines/rust/rko-core/src/mh/ipr.rs`,
  `engines/cpp/vendor/rko_cpp_v2/src/MH/IPR.h`.
- **Journal article.** C. E. Andrade, R. F. Toso, J. F. Gonçalves, M. G. C. Resende. "The Multi-Parent
  Biased Random-Key Genetic Algorithm with Implicit Path-Relinking and its real-world applications."
  *European Journal of Operational Research* 289(1), 17–30, 2021.
  [DOI](https://doi.org/10.1016/j.ejor.2019.11.037). The framework paper states that this paper
  introduced implicit path relinking. See also [IPR diagnosis](ipr-diagnosis.md).

### Multi-start

- Implemented in `src/rko_solver/metaheuristics/multi_start.py`,
  `engines/rust/rko-core/src/mh/multi_start.rs`, `engines/cpp/vendor/rko_cpp_v2/src/MH/MultiStart.h`.
- **Journal article.** R. Martí, M. G. C. Resende, C. C. Ribeiro. "Multi-start methods for
  combinatorial optimization." *European Journal of Operational Research* 226(1), 1–8, 2013.
  [DOI](https://doi.org/10.1016/j.ejor.2012.10.012). A general survey. Neither RKO paper cites
  multi-start.

## Shared components

### Elite pool, shaking and blending

- Elite pool in `src/rko_solver/components/solution_pool.py`, initial solutions in
  `src/rko_solver/components/initialization/`, shaking in `src/rko_solver/components/shaking/`,
  blending in `src/rko_solver/components/blending/`. Framework sections 4.1, 4.2 and 4.3:
  [arXiv 2411.04293](https://arxiv.org/abs/2411.04293).

### Swap local search

- `src/rko_solver/components/local_search/swap_ls.py`. Framework section 4.4.1:
  [arXiv 2411.04293](https://arxiv.org/abs/2411.04293).

### Farey local search

- `src/rko_solver/components/local_search/farey_ls.py`, Rust twin
  `engines/rust/rko-core/src/local_search.rs`. Framework section 4.4.2, which uses a Farey sequence
  of order 7: [arXiv 2411.04293](https://arxiv.org/abs/2411.04293). The vendored C++ header
  `engines/cpp/vendor/rko_cpp_v2/src/MH/LNS.h` builds the same kind of sequence for its repair step;
  the C++ README does not document it.

### Mirror local search

- `src/rko_solver/components/local_search/mirror_ls.py`. Framework section 4.4.3:
  [arXiv 2411.04293](https://arxiv.org/abs/2411.04293).

### Nelder-Mead simplex

- `src/rko_solver/components/local_search/nelder_mead_ls.py`. Framework section 4.4.4:
  [arXiv 2411.04293](https://arxiv.org/abs/2411.04293).
- **Journal article.** J. A. Nelder, R. Mead. "A Simplex Method for Function Minimization." *The
  Computer Journal* 7(4), 308–313, 1965. [DOI](https://doi.org/10.1093/comjnl/7.4.308).

### Random variable neighborhood descent (RVND)

- `src/rko_solver/components/local_search/rvnd.py`. The framework paper (section 4.4) cites
  Subramanian et al. (2010) for RVND and describes it as an extension of variable neighborhood
  descent by Mladenović and Hansen (see [VNS](#variable-neighborhood-search-vns)).
- **Journal article.** A. Subramanian, L. M. A. Drummond, C. Bentes, L. S. Ochi, R. Farias. "A
  parallel heuristic for the Vehicle Routing Problem with Simultaneous Pickup and Delivery."
  *Computers & Operations Research* 37(11), 1899–1911, 2010.
  [DOI](https://doi.org/10.1016/j.cor.2009.10.011). The publisher page could not be opened for this
  page, so its text was not read.
- **Journal article.** P. H. V. Penna, A. Subramanian, L. S. Ochi. "An Iterated Local Search
  heuristic for the Heterogeneous Fleet Vehicle Routing Problem." *Journal of Heuristics* 19(2),
  201–232, 2013. [DOI](https://doi.org/10.1007/s10732-011-9186-y). Its text gives an algorithmic
  definition (a random neighborhood order, with a neighborhood removed from the list when it yields
  no improvement) and says it extends the 2010 paper.

### Metropolis acceptance criterion

- Used by `src/rko_solver/metaheuristics/sa.py`, `lns.py` and `grasp.py`, as their module docstrings
  state. The framework paper names it for LNS and for SA's acceptance of worse solutions.
- **Journal article.** N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, E. Teller.
  "Equation of State Calculations by Fast Computing Machines." *The Journal of Chemical Physics*
  21(6), 1087–1092, 1953. [DOI](https://doi.org/10.1063/1.1699114).

### Q-learning parameter control

- `src/rko_solver/components/adaptive_control/q_learning.py`, Rust twin
  `engines/rust/rko-core/src/qlearning.rs`. Framework sections 4.6 and 5.4, with an epsilon-greedy
  policy.
- **Journal article.** C. J. C. H. Watkins, P. Dayan. "Q-learning." *Machine Learning* 8(3-4),
  279–292, 1992. [DOI](https://doi.org/10.1007/BF00992698). Cited by the framework paper; the
  mixed-integer-program paper does not mention Q-learning.
- **Book.** R. S. Sutton, A. G. Barto. *Reinforcement Learning: An Introduction*, second edition. MIT
  Press, 2018. Cited by the framework paper for the epsilon-greedy policy; listed here from that
  reference list and not independently verified.

## Learning components

These support the experiments in `benchmarks/README.md`. None of them is part of the installed
solver.

### Ranking model and order sampling

- `benchmarks/engines/priority_scorer.py`. A score vector defines a distribution over item orders;
  orders are sampled by adding Gumbel noise and sorting, and the log-probability gradient is checked
  in `tests/benchmarks/test_priority_scorer.py` against exact order probabilities and finite
  differences.
- **Journal article.** R. L. Plackett. "The Analysis of Permutations." *Applied Statistics* 24(2),
  starting on page 193, 1975. [DOI](https://doi.org/10.2307/2346567).
- **Book.** R. D. Luce. *Individual Choice Behavior.* 1959. Existence
  confirmed in the
  [Open Library catalogue](https://openlibrary.org/search?q=luce+individual+choice+behavior).
- **Conference paper.** W. Kool, H. van Hoof, M. Welling. "Stochastic Beams and Where to Find Them:
  The Gumbel-Top-k Trick for Sampling Sequences Without Replacement." ICML 2019.
  [arXiv 1903.06059](https://arxiv.org/abs/1903.06059).

### REINFORCE and the leave-one-out baseline

- `benchmarks/engines/priority_scorer.py` (`ReinforceTrainer`).
- **Journal article.** R. J. Williams. "Simple statistical gradient-following algorithms for
  connectionist reinforcement learning." *Machine Learning* 8(3-4), 229–256, 1992.
  [DOI](https://doi.org/10.1007/BF00992696).
- **Workshop paper.** W. Kool, H. van Hoof, M. Welling. "Buy 4 REINFORCE Samples, Get a Baseline for
  Free!" ICLR 2019 workshop on Deep Reinforcement Learning Meets Structured Prediction.
  [Author's page](https://wouterkool.github.io/publication/buy-4-samples-free-baseline/). The
  workshop record on OpenReview could not be opened for this page.

### Adam optimizer

- `benchmarks/engines/mlp_surrogate.py` (`MlpSurrogate.adam`), reused by the other learned models.
- **Preprint.** D. P. Kingma, J. Ba. "Adam: A Method for Stochastic Optimization." arXiv 1412.6980.
  [arXiv](https://arxiv.org/abs/1412.6980).

### Denoising autoencoder

- `benchmarks/engines/neural_key_adapter.py`.
- **Conference paper.** P. Vincent, H. Larochelle, Y. Bengio, P.-A. Manzagol. "Extracting and
  composing robust features with denoising autoencoders." *Proceedings of the 25th International
  Conference on Machine Learning*, 1096–1103, 2008.
  [DOI](https://doi.org/10.1145/1390156.1390294).

### Knapsack instance families

- `benchmarks/engines/priority_decoder_study.py` (`InstanceFactory`) generates uncorrelated, weakly
  correlated and strongly correlated Knapsack instances. Capacity is half of the total weight and the
  profit rules are simplified, so this follows the families of the paper and does not reproduce its
  generators.
- **Journal article.** D. Pisinger. "Where are the hard knapsack problems?" *Computers & Operations
  Research* 32(9), 2271–2284, 2005. [DOI](https://doi.org/10.1016/j.cor.2004.03.002).

### Five-constraint packing instances

- `benchmarks/engines/packing_problems.py` generates the five-constraint family used by the learned
  priority study.
- **No published reference.** The generator was written for this repository: weights uniform on 0 to
  1000, tightness 0.25, profits tied to the weights. It is not a published benchmark.

## Convergence

### Convergence of evolutionary and random search

- [Convergence guarantees](guarantees.md) states one elementary bound and tests it in
  `tests/test_brkga_survival_bound.py`. The papers below are classical analyses of related questions and are **not** the source of that page's constants or proof.
- **Journal article.** G. Rudolph. "Convergence analysis of canonical genetic algorithms." *IEEE
  Transactions on Neural Networks* 5(1), 96–101, 1994.
  [DOI](https://doi.org/10.1109/72.265964).
- **Journal article.** F. J. Solis, R. J.-B. Wets. "Minimization by Random Search Techniques."
  *Mathematics of Operations Research* 6(1), 19–30, 1981.
  [DOI](https://doi.org/10.1287/moor.6.1.19).
- **Journal article.** J. He, X. Yao. "Drift analysis and average time complexity of evolutionary
  algorithms." *Artificial Intelligence* 127(1), 57–85, 2001.
  [DOI](https://doi.org/10.1016/S0004-3702%2801%2900058-3).

### Decoder-aware bounds

- **No published reference.** The decoder-aware bound behind the guarantees page comes from an
  unpublished, machine-generated proposal listed under
  [Unpublished proposals](#decoder-aware-convergence-bounds-proposal). The page states only the
  elementary part and tests it.

## Related work not implemented here

### LLM-generated instance bias for BRKGA

- **Preprint.** C. Chacón Sartori, M. Isla Pino, P. Pinacho-Davidson, C. Blum. "LLM-Based Instance-Driven
  Heuristic Bias In the Context of a Biased Random Key Genetic Algorithm." arXiv 2509.09707, 2025.
  [arXiv](https://arxiv.org/abs/2509.09707) (CC BY-NC-SA 4.0, so linked and not copied). Related to
  `initial_keys` and to the priority decoders: it also steers a BRKGA with instance information. This
  repository does not implement it.

### Neural combinatorial optimization

- **Conference paper.** F. Berto and 32 co-authors. "RL4CO: an Extensive Reinforcement Learning for
  Combinatorial Optimization Benchmark." *Proceedings of the 31st ACM SIGKDD Conference on Knowledge
  Discovery and Data Mining*, 5278–5289, 2025.
  [DOI](https://doi.org/10.1145/3711896.3737433) · [arXiv 2306.17100](https://arxiv.org/abs/2306.17100)
  · [repository](https://github.com/ai4co/rl4co) (MIT). Named as a training base in the plan for the
  neural-decoder experiments; this repository does not use it.
- **Preprint.** F. Chalumeau, S. Surana, C. Bonnet, N. Grinsztajn, A. Pretorius, A. Laterre,
  T. D. Barrett. "Combinatorial Optimization with Policy Adaptation using Latent Space Search."
  [arXiv 2311.13569](https://arxiv.org/abs/2311.13569).
- **Preprint.** A. Hottung, Y.-D. Kwon, K. Tierney. "Efficient Active Search for Combinatorial
  Optimization Problems." [arXiv 2106.05126](https://arxiv.org/abs/2106.05126).
- **Conference paper.** R. Qiu, Z. Sun, Y. Yang. "DIMES: A Differentiable Meta Solver for Combinatorial
  Optimization Problems." NeurIPS 2022. [arXiv 2210.04123](https://arxiv.org/abs/2210.04123).
- **Preprint.** S. Surendran, A. Fermanian, S. Le Corff. "Latent Guided Sampling for Combinatorial
  Optimization." [arXiv 2506.03672](https://arxiv.org/abs/2506.03672). Its arXiv record gives a 2026
  International Conference on Machine Learning as the venue.

## Unpublished proposals

Three machine-generated method proposals were obtained by the maintainer in October 2026. They are
**not peer reviewed**, report no trained model, no benchmark and no run of this solver (only
small mechanics checks, as read on 2026-10-28), and a search of arXiv and the web on 2026-10-30
found no published version. They are not redistributed and not linked. Treat them as plans, not
literature.

### Q-RKO proposal

- **Unpublished proposal.** "Neural Decoders for Automatic Random-Key Optimization: a bidirectional
  quantile codec" (2026-10-27, 26 pages). This proposal is machine-generated and not peer reviewed.
- Tested here by `examples/priority_decoders.py`, `benchmarks/engines/priority_decoder_study.py`,
  `benchmarks/engines/packing_problems.py`, `benchmarks/engines/priority_scorer.py` and
  `benchmarks/engines/learned_priority_study.py`, which implement its residual-priority decoder, a
  quantile decoder and a per-family scorer trained with REINFORCE. The sequential per-decision policy
  with a global key block was **not** tested.

### RKO-SR proposal

- **Unpublished proposal.** "RKO-SR: certificate-aware random-key optimization for two-stage
  stochastic MIPs with recourse" (2026-10-28, 20 pages). This proposal is machine-generated and not peer reviewed. No
  code in this repository implements it.

### Decoder-aware convergence bounds proposal

- **Unpublished proposal.** "Decoder-aware convergence bounds for random-key metaheuristics"
  (2026-10-28, 21 pages). This proposal is machine-generated and not peer reviewed. [Convergence guarantees](guarantees.md)
  and its tests state and check the elementary bound `P(miss) <= (1 - mu)^(p + m*G)` only.

## Experiments without a source paper

| Module | What it tests | Source |
| --- | --- | --- |
| `benchmarks/engines/surrogate_seed_study.py`, `benchmarks/engines/mlp_surrogate.py` | Starting keys from a ridge or small neural cost surrogate fitted to sampled evaluations | No published reference; own design, motivated by the priority-decoder control |
| `benchmarks/engines/neural_key_adapter.py`, `benchmarks/engines/coupled_key_distribution.py`, `benchmarks/engines/neural_adapter_study.py` | Learned key distributions composed with the caller's decoder | No published reference; own design. Related ideas under [neural combinatorial optimization](#neural-combinatorial-optimization) |

Results are in `benchmarks/README.md`; all of these failed their pre-registered rules.

## Local copies

Copies of RKO-family papers whose arXiv license permits redistribution are kept in
`docs/references/papers/`. Each is the unchanged arXiv PDF, downloaded 2026-10-30. Authors and titles
are given in the entries above; the license is Creative Commons Attribution 4.0
([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). Papers under other licenses are linked and
not copied.

| File | arXiv | License | SHA-256 |
| --- | --- | --- | --- |
| [arxiv-2411.04293v3-random-key-optimizer.pdf](https://github.com/MarcosPTProenca/rko-solver/blob/main/docs/references/papers/arxiv-2411.04293v3-random-key-optimizer.pdf) | [2411.04293v3](https://arxiv.org/abs/2411.04293v3) | CC BY 4.0 | 93dcd4c9799d996cef8b48097726ac783fa60c505c86d5e800a4b032333339b7 |
| [arxiv-2602.22173v2-rko-mixed-integer-programs.pdf](https://github.com/MarcosPTProenca/rko-solver/blob/main/docs/references/papers/arxiv-2602.22173v2-rko-mixed-integer-programs.pdf) | [2602.22173v2](https://arxiv.org/abs/2602.22173v2) | CC BY 4.0 | db2835f174df199bc7603a016ea1787ec132062b5860f576d8db4beca9636b3c |
| [arxiv-2405.18681v2-random-key-grasp.pdf](https://github.com/MarcosPTProenca/rko-solver/blob/main/docs/references/papers/arxiv-2405.18681v2-random-key-grasp.pdf) | [2405.18681v2](https://arxiv.org/abs/2405.18681v2) | CC BY 4.0 | a67bc1e55a3321e7c709efe9b60ac1a01ffbb4ff20072c52766cdcc7cc7f48cb |
| [arxiv-2506.02120v2-random-key-genetic-algorithms.pdf](https://github.com/MarcosPTProenca/rko-solver/blob/main/docs/references/papers/arxiv-2506.02120v2-random-key-genetic-algorithms.pdf) | [2506.02120v2](https://arxiv.org/abs/2506.02120v2) | CC BY 4.0 | a417e7fe777f0fd512f11d60a29220c9941ae010bc723d01dc36d5e5a8c618d6 |

The journal version of the framework paper is not open access; the arXiv copy above is the version
the authors released under CC BY 4.0 and can differ from the published text.
