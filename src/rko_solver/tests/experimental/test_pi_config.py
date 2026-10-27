"""Non-secret Pi configuration and private profile admission contracts.

Objective:
- Reject ambiguous access settings without reading credentials or changing user profiles.
Technical approach:
- Exercise bounded file loading and exact settings/privacy checks through PiConfig.
Flow (pseudocode):
1. Load explicit relative paths and exact supported provider/model settings.
2. Create a new private settings file without authentication material.
3. Reject unsafe settings, extra resources and unsupported configuration values.
Important criteria:
- These tests never start Pi, authenticate, execute candidate code or change an operator profile.
Inputs/Outputs:
- Input: test-owned JSON declarations and profile files.
- Output: validated metadata or fail-closed configuration errors.
"""

import json
import pytest
import sys
from pathlib import Path
from rko_solver.experimental.self_evolve.pi_config import PiConfig
from rko_solver.experimental.self_evolve.pi_config import PiError


def test_relative_config_load_and_profile_creation_are_non_overwriting(tmp_path: Path) -> None:
    """A declaration resolves relative to its file and never creates/copies credentials.

    Args:
        tmp_path (Path): Test-owned declaration and new profile parent.
    """
    path = tmp_path / "pi.json"
    path.write_text(
        json.dumps(
            {
                "executable": "bin/pi",
                "profile": "profile",
                "provider": "openai-codex",
                "model": "gpt-5.4",
                "seconds": 10.5,
            }
        )
    )
    config = PiConfig.load(path)
    assert config.seconds == 10.5
    assert config.executable == tmp_path / "bin" / "pi"
    assert config.profile == tmp_path / "profile"
    config.prepare_profile()
    assert not (config.profile / "auth.json").exists()
    settings = json.loads((config.profile / "settings.json").read_text())
    assert settings["cacheWarming"] == "off"
    assert settings["transport"] == "sse"
    assert settings["retry"]["provider"]["maxRetries"] == 0
    assert "providerOptions" not in settings
    with pytest.raises(FileExistsError):
        config.prepare_profile()
    assert config.descriptor()["pi_version"] == "1.3.0"


@pytest.mark.parametrize(
    "change",
    [
        {"provider": "openai"},
        {"model": "bad model"},
        {"model": "--flag"},
        {"seconds": True},
        {"seconds": 0},
        {"seconds": 601},
        {"seconds": "10"},
        {"api_key": "never-accept-a-key-field"},
        {"executable": None},
    ],
)
def test_invalid_or_secret_config_fields_are_rejected(
    tmp_path: Path, change: dict[str, object]
) -> None:
    """Only bounded explicit non-secret settings may influence the trusted executable.

    Args:
        tmp_path (Path): New declaration path.
        change (dict[str, object]): One invalid type, domain, provider or extra secret field.
    """
    data = {
        "executable": "pi",
        "profile": "profile",
        "provider": "openai-codex",
        "model": "gpt-5.4",
        "seconds": 10,
    }
    data.update(change)
    path = tmp_path / "pi.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        PiConfig.load(path)
    assert not (tmp_path / "profile").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Private live-profile policy is Linux/WSL.")
@pytest.mark.parametrize("unsafe", ["models.json", "mcp.json", "SYSTEM.md", "extensions"])
def test_custom_resources_cannot_enter_a_qualified_profile(tmp_path: Path, unsafe: str) -> None:
    """Resources and endpoint overrides cannot be admitted through inherited Pi discovery.

    Args:
        tmp_path (Path): Controller-created private profile.
        unsafe (str): Resource name that would broaden the approved proposer boundary.
    """
    config = PiConfig(tmp_path / "pi", tmp_path / "profile", "gpt-5.4")
    config.prepare_profile()
    auth = config.profile / "auth.json"
    auth.write_text("not a credential parsed by the controller")
    auth.chmod(0o600)
    config.verify_profile()
    (config.profile / unsafe).write_text("{}")
    with pytest.raises(PiError, match="profile"):
        config.verify_profile()
    assert auth.read_text() == "not a credential parsed by the controller"


@pytest.mark.skipif(sys.platform != "linux", reason="Private live-profile policy is Linux/WSL.")
def test_boolean_policy_fields_are_not_coerced_from_numeric_zero(tmp_path: Path) -> None:
    """JSON's false and zero must not be interchangeable in an executable policy declaration.

    Args:
        tmp_path (Path): Private minimal profile and malformed settings.
    """
    config = PiConfig(tmp_path / "pi", tmp_path / "profile", "gpt-5.4")
    config.prepare_profile()
    auth = config.profile / "auth.json"
    auth.write_text("{}")
    auth.chmod(0o600)
    path = config.profile / "settings.json"
    data = json.loads(path.read_text())
    data["retry"]["enabled"] = 0
    path.write_text(json.dumps(data))
    with pytest.raises(PiError, match="profile"):
        config.verify_profile()
