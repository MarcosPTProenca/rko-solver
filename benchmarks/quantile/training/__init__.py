"""Isolated training entry points for repository-only contextual quantile experiments.

Objective:
- Keep the neural training framework out of the installed solver's dependencies.

Technical approach:
- Run through the separately locked training project and exchange data-only weights.

Flow (pseudocode):
1. Select the isolated training environment explicitly.
2. Train a bounded contextual scoring model.
3. Export weights for ordinary NumPy model evaluation.

Important criteria:
- Ordinary solver imports never import this training package.

Inputs/Outputs:
- Input: explicit training-module invocation.
- Output: numerical weights and measured training evidence.
"""
