"""Runtime helper-discovery regressions for WSL Windows-drive PATH entries.

Objective:
- Avoid OCI executable discovery traversing slow Windows-mounted PATH directories.
Technical approach:
- Inspect the real runner environment using a mocked administrative response only.
Flow (pseudocode):
1. Supply a mixed WSL PATH with Linux helpers and Windows drive entries.
2. Construct the runner through ordinary runtime-capability validation.
3. Require Linux discovery paths without mutating the operator environment.
Important criteria:
- No resource limit or effective isolation check is bypassed in production.
Inputs/Outputs:
- Input: controlled operator PATH and platform markers.
- Output: predictable child-only runtime discovery environment.
"""

import json
import os
import pytest
from rko_solver.experimental.self_evolve.sandbox import PodmanRunner


@pytest.mark.parametrize("wsl", [True, False])
def test_runtime_discovery_path_is_local_to_the_child(
    monkeypatch: pytest.MonkeyPatch, wsl: bool
) -> None:
    """Discard Windows-drive discovery paths on WSL, preserving ordinary Linux paths.

    Args:
        monkeypatch (pytest.MonkeyPatch): Controlled platform and administrative responses.
        wsl (bool): Whether inherited WSL markers identify Windows mount semantics.
    """
    path = os.pathsep.join(
        [
            "/usr/bin",
            "/mnt/c/Windows/System32",
            "/home/operator/bin",
            "/mnt/d/tools",
            "/mnt/cache/bin",
        ]
    )
    monkeypatch.setenv("PATH", path)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    if wsl:
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr("sys.platform", "linux")

    def administrative(runner: PodmanRunner, args: list[str], *, cwd: str | None = None) -> str:
        """Return fixed verified-capability inputs without launching any subprocess.

        Args:
            runner (PodmanRunner): Real constructor under test.
            args (list[str]): Administrative request.
            cwd (str | None): Optional private launch directory.

        Returns:
            str: Rootless cgroup evidence or immutable local image identity.
        """
        if args[0] == "info":
            return json.dumps(
                {
                    "host": {
                        "security": {"rootless": True},
                        "cgroupVersion": "v2",
                        "cgroupControllers": ["cpu", "memory", "pids"],
                    }
                }
            )
        return "sha256:" + "0" * 64

    monkeypatch.setattr(PodmanRunner, "_command", administrative)
    runner = PodmanRunner("trusted-fixture")
    expected = (
        os.pathsep.join(["/usr/bin", "/home/operator/bin", "/mnt/cache/bin"]) if wsl else path
    )
    assert runner.environment["PATH"] == expected
    assert os.environ["PATH"] == path
    assert "WSL_DISTRO_NAME" not in runner.environment
