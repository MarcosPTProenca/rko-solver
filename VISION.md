# Vision

RKO exists to make high-performance, general-purpose metaheuristic optimization accessible without requiring users to implement or tune their own search algorithms.
It serves researchers, operations research practitioners and software engineers through a user-friendly framework for solving optimization problems across domains.
It owns exactly one thing: solving optimization problems through random-key search, independently of how users formulate them.

## The user defines the problem; RKO owns the search

A decoder and an objective function are sufficient but not mandatory, and supported mathematical modeling interfaces provide an alternative.
Random keys are the fundamental search representation, not a mandatory user-facing abstraction.
RKO reduces manual decoder engineering through automatic construction, structural recognition and feasibility-preserving techniques qualified against credible baselines for feasibility, quality and total cost.
Users retain the option to supply custom decoders and change algorithms or compatible backends without rewriting their problem.
The search core remains domain-neutral, and specialized techniques enter through explicit extension contracts rather than hardcoded domain logic.

## Autonomous search through composable components

RKO assumes responsibility for selecting, configuring and coordinating algorithms, operators, resource allocation and restart strategies.
Initialization, neighborhoods, local search, perturbation, acceptance, evaluation and cooperation are interchangeable through cohesive contracts that preserve random-key search and problem semantics.
The solver may compose generic or specialized strategies according to mathematical structure, problem class and observed search behavior.
Reinforcement learning may select, configure and combine qualified components when its benefits exceed its computational and architectural costs.
Cooperation earns its place against strong independent and single-algorithm baselines rather than being mandatory.
Automatic selection respects scientifically qualified applicability conditions and falls back to qualified generic strategies when specialization is unsupported or ineffective.
Automated decisions remain observable, and users can override them or run individual algorithms and fixed configurations.

## Python accessibility, technology-independent performance

Python is the primary public interface, and computational complexity belongs inside the solver rather than in user integration code.
No implementation language, backend or hardware architecture has permanent privilege.
Multiple backends coexist only when their measured workload benefits justify their total engineering cost, and they share public contracts and configuration semantics without unnecessary duplication.
RKO pursues leadership among random-key metaheuristic frameworks through solution quality, computational efficiency and ease of use.
RKO aims to match or exceed the original C++ RKO and relevant random-key implementations under comparable resources, while strong external metaheuristics and exact methods provide scientific context.
Backend selection and performance changes follow measured workload characteristics, preserve correctness and account for end-to-end solution quality.

## Scientific evidence precedes promotion

Scientific evaluation is a permanent development discipline, and proposed algorithmic, modeling and computational improvements carry the burden of demonstrating their value.
Experiments begin with falsifiable hypotheses, strong baselines including fixed configurations for adaptive methods, paired seeds, declared resources and predefined superiority or non-inferiority criteria.
Fixed-work efficiency and quality at equal wall time are assessed separately, including initialization, coordination and selection overhead where relevant.
General claims require representative problem families and independent confirmation instances, and specialized claims are limited to their qualified applicability conditions.
Qualification distinguishes individual components, selection policies and integrated solver behavior, including generalization, component interactions and regression protection outside the intended scope.
Aggregate gains do not excuse hidden material regressions in individual problem classes, and automatic deployment requires the relevant qualification and fallback evidence.
Versioned protocols, reference instances, raw results and reproducible analyses preserve negative and inconclusive outcomes alongside positive ones.
Software tests establish correctness, not scientific superiority, and novelty alone never justifies promotion.

## AI-assisted discovery, independent execution

Optional Self-Evolve explores decoders, operators, algorithms, parameter policies and portfolio strategies, including components specialized for particular problem classes.
Self-Evolve and reinforcement learning share component contracts so discovery and adaptive selection complement each other.
AI-generated candidates are evaluated in isolation against established baselines and preserve the original objective, constraints and validation contracts; only scientifically and architecturally qualified artifacts with declared applicability enter conventional optimization.
Conventional optimization remains independent of LLM providers, inference costs and external AI services.

## Correctness, stability and architectural integrity

Returned solutions and objectives are independently verifiable, with explicit evaluation budgets, stopping semantics and documented resource limits.
Randomness, numerical behavior, backend differences and reproducibility are documented without promising identical stochastic trajectories or unsupported feasibility, optimality or performance guarantees.
Stable public interfaces are commitments to users, and breaking changes require exceptional justification, explicit versioning and a documented migration path.
Public contracts and architectural changes are designed before implementation with compatibility, downstream impact and long-term maintenance in mind.
Cohesion, readability and low complexity take precedence over speculative abstractions, duplication, fragmented responsibilities and development speed.
Human-written and AI-generated code meet the same design, testing, review and maintainability standards.
Experimental capabilities declare their maturity and compatibility limits and may evolve without destabilizing established interfaces.

## Scope

RKO maintains one modular repository for its core, official extensions and scientific evaluation infrastructure.
The core remains lightweight and independently usable, with optional backends, modeling integrations, specialized components, reinforcement learning and Self-Evolve imposing no unnecessary dependencies or overhead when unused.
Package extras or separate distributions are justified by actual dependency, deployment or maintenance needs, not premature abstractions or service boundaries.
RKO is not an exact mathematical programming solver, a universal competitor to every optimization paradigm, an application platform, a hosted optimization service or a general-purpose AI framework.

A change aligns when it demonstrably improves modeling accessibility, qualified optimization outcomes, correctness, architectural integrity or reproducible scientific evaluation while preserving the random-key contract and user simplicity.
A change should be resisted when it bypasses extension contracts, hardcodes domain logic, destabilizes public interfaces without justification, adds unnecessary complexity or dependencies, exceeds the project's optimization responsibility or promotes unsupported claims.
