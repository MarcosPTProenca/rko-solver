"""Integration checks for optional service admission and Linux-owned job supervision.

Objective:
- Verify bounded HTTP behavior and real process cleanup independently of core optimization.

Technical approach:
- Use importable test adapters and actual spawn workers rather than mocked cancellation.

Flow (pseudocode):
1. Start an isolated private test service.
2. Exercise its public lifecycle and actual process boundaries.
3. Confirm terminal state and owned-work cleanup.

Important criteria:
- Tests never terminate unrelated host processes or require a Docker daemon.

Inputs/Outputs:
- Input: temporary private storage and bounded synthetic problem data.
- Output: deterministic assertions about service behavior.
"""
