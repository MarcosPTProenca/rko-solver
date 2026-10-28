"""Pi subprocess transport contracts using a trusted local executable fixture.

Objective:
- Verify proposal generation without a provider request or generated-code execution.
Technical approach:
- Run a fake Pi executable through the real bounded process and RPC interfaces.
Flow (pseudocode):
1. Prepare an isolated profile and explicitly selected executable/model.
2. Exchange safe authentication metadata, model selection and one proposal.
3. Assert exact source bytes and nonduplicated usage without exposing ambient secrets.
Important criteria:
- The fixture is trusted test infrastructure, not candidate Python or live Pi qualification.
Inputs/Outputs:
- Input: temporary profile, executable and bounded development request.
- Output: pending source data and safe generation metadata.
"""

import json
import os
import pytest
import sys
import types
from dataclasses import replace
from pathlib import Path
from rko_solver.experimental.self_evolve.pi_config import PiConfig
from rko_solver.experimental.self_evolve.pi_config import PiError
from rko_solver.experimental.self_evolve.pi_process import PiProcess
from rko_solver.experimental.self_evolve.pi_transport import PiTransport

FAKE_PI = """import json, os, sys
from pathlib import Path
scenario = json.loads(Path(__file__).with_suffix('.json').read_text())
if scenario.get('device_id'):
    settings_path = Path(os.environ['PI_CODING_AGENT_DIR']) / 'settings.json'
    settings = json.loads(settings_path.read_text())
    settings['deviceId'] = scenario['device_id']
    settings_path.write_text(json.dumps(settings))
if '--version' in sys.argv:
    print(scenario.get('version', '1.3.0'))
    raise SystemExit(0)
if sys.argv[1:3] == ['auth', 'check']:
    default_auth = {'status':'ready','provider':'openai-codex','authType':'oauth'}
    print(' ' * scenario.get('auth_padding', 0) + json.dumps(scenario.get('auth', default_auth)))
    raise SystemExit(0)
assert '--no-tools' in sys.argv and '--no-extensions' in sys.argv
assert '--no-mcp' in sys.argv and '--no-context-files' in sys.argv
assert '--no-session' in sys.argv and '--no-approve' in sys.argv
assert not any(name in os.environ for name in ('OPENAI_API_KEY','NODE_OPTIONS','PI_SESSION_FILE'))
model = {'id':'fixture-model','provider':'openai-codex','api':'openai-codex-responses',
         'baseUrl':'https://chatgpt.com/backend-api'}
model.update(scenario.get('model_update', {}))
def emit(record):
    print(json.dumps(record), flush=True)
for line in sys.stdin.buffer:
    command = json.loads(line)
    kind = command['type']
    data = {}
    if kind == 'get_available_models': data = {'models':[model]}
    if kind == 'set_model': data = model
    if kind == 'get_state': data = {'model':model,'isStreaming':False,'isCompacting':False,
                                    'pendingMessageCount':0,'messageCount':0,'autoCompactionEnabled':False}
    if kind == 'get_state': data.update(scenario.get('state_update', {}))
    if kind == 'get_commands': data = {'commands':[]}
    if kind == 'prompt': data = {'disposition':'started'}
    reply = {'type':'response','id':command['id'],'command':kind,'success':True,'data':data}
    if not (kind == 'prompt' and scenario.get('early_events')): emit(reply)
    if kind == 'prompt':
        if scenario.get('interrupt_parent'):
            import signal, time
            os.kill(os.getppid(), signal.SIGINT)
            time.sleep(60)
        if scenario.get('hang'):
            import time
            time.sleep(60)
        if scenario.get('truncate'):
            sys.stdout.write('{"type":'); sys.stdout.flush(); break
        if scenario.get('flood'):
            os.write(2, b'x' * 70000); break
        assert scenario.get('marker', 'protected-never-sent') not in command['message']
        message = {'role':'assistant','provider':'openai-codex','model':'fixture-model',
                   'api':'openai-codex-responses',
                   'stopReason':'stop','content':[{'type':'text','text':json.dumps(scenario['candidate'])}],
                   'usage':{'input':11,'output':7,'cacheRead':2,'cacheWrite':0,'totalTokens':20}}
        message.update(scenario.get('message_update', {}))
        emit({'type':'agent_start'})
        system = {'role':'system','content':'','timestamp':1,'sections':{
            'preamble':sys.argv[sys.argv.index('--system-prompt') + 1],
            'cwd':'<cwd>\\n' + os.getcwd() + '\\n</cwd>'}}
        system.update(scenario.get('system_update', {}))
        emit({'type':'message_start','message':system})
        emit({'type':'message_end','message':system})
        emit({'type':'message_end','message':{'role':'user','content':'data'}})
        if 'event' in scenario: emit(scenario['event'])
        emit({'type':'message_update','usage':message['usage'],
              'assistantMessageEvent':{'type':'text_delta','contentIndex':0,'delta':'ignored'}})
        emit({'type':'message_end','message':message})
        if scenario.get('duplicate_message'): emit({'type':'message_end','message':message})
        emit({'type':'turn_end','message':message,'toolResults':[]})
        emit({'type':'agent_end','messages':[system,{'role':'user','content':'data'},message],
              'willRetry':False})
        if not scenario.get('omit_settled'):
            emit({'type':'agent_settled','aborted':scenario.get('aborted', False)})
        else: break
        if scenario.get('late_event'): emit({'type':'agent_start'})
        if scenario.get('bad_exit'): raise SystemExit(2)
        if scenario.get('early_events'): emit(reply)
"""


@pytest.fixture
def configured_pi(tmp_path: Path) -> PiConfig:
    """Prepare a trusted executable fixture and a new non-overwriting profile.

    Args:
        tmp_path (Path): Isolated test-owned files, never an operator profile.

    Returns:
        PiConfig: Explicit fake executable and model with a ten-second deadline.
    """
    executable = tmp_path / "fake-pi"
    executable.write_text(f"#!{sys.executable}\n" + FAKE_PI, encoding="utf-8")
    executable.chmod(0o700)
    executable.with_suffix(".json").write_text(
        json.dumps(
            {
                "candidate": {
                    "source": "def decode(keys, instance):\r\n    return []\r\n",
                    "hypothesis": "fixture",
                }
            }
        )
    )
    config = PiConfig(
        executable=executable, profile=tmp_path / "profile", model="fixture-model", seconds=10
    )
    config.prepare_profile()
    auth = config.profile / "auth.json"
    auth.write_text("{}")
    auth.chmod(0o600)
    return config


@pytest.mark.skipif(sys.platform != "linux", reason="Initial Pi process supervision is Linux/WSL.")
@pytest.mark.parametrize("device_id", [None, "b28a056e-d5bd-4a03-a2db-5e080138cf86"])
def test_one_proposal_preserves_source_and_counts_only_authoritative_usage(
    configured_pi: PiConfig, monkeypatch: pytest.MonkeyPatch, device_id: str | None
) -> None:
    """The real process transport returns source data, not execution or a free-cost claim.

    Args:
        configured_pi (PiConfig): Trusted fake Pi executable and isolated profile.
        monkeypatch (pytest.MonkeyPatch): Ambient credentials that must not reach the child.
        device_id (str | None): Optional UUID persisted by the fake Pi during metadata startup.
    """
    set_scenario(configured_pi, {"device_id": device_id})
    for name in ("OPENAI_API_KEY", "NODE_OPTIONS", "PI_SESSION_FILE"):
        monkeypatch.setenv(name, "never-inherit-this-secret")
    result = PiTransport(configured_pi).generate({"family": "knapsack", "task": "trusted fixture"})
    assert result["source"] == "def decode(keys, instance):\r\n    return []\r\n"
    assert result["hypothesis"] == "fixture"
    assert result["usage"]["totalTokens"] == 20
    assert result["monetary_cost"] is None
    assert result["provider"] == "openai-codex"
    assert "never-inherit-this-secret" == os.environ["OPENAI_API_KEY"]


@pytest.mark.skipif(sys.platform != "linux", reason="Initial Pi process supervision is Linux/WSL.")
def test_fast_events_before_prompt_acknowledgement_are_not_lost(configured_pi: PiConfig) -> None:
    """An early completed response still needs correlated acknowledgement and settlement.

    Args:
        configured_pi (PiConfig): Trusted executable emitting events before the RPC response.
    """
    set_scenario(configured_pi, {"early_events": True})
    result = PiTransport(configured_pi).generate({"family": "knapsack"})
    assert result["hypothesis"] == "fixture"
    assert result["usage"]["totalTokens"] == 20


def set_scenario(config: PiConfig, changes: dict[str, object]) -> None:
    """Change only trusted fake-executable responses at the external process seam.

    Args:
        config (PiConfig): Test-owned executable and profile.
        changes (dict[str, object]): Scripted metadata or event variations.
    """
    path = config.executable.with_suffix(".json")
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


@pytest.mark.skipif(sys.platform != "linux", reason="Initial Pi process supervision is Linux/WSL.")
@pytest.mark.parametrize(
    "changes",
    [
        {"version": "unsupported"},
        {"auth": {"status": "ready", "provider": "openai-codex", "authType": "api_key"}},
        {"auth": {"status": "not_ready", "provider": "openai-codex", "reason": "expired"}},
        {"model_update": {"id": "different-model"}},
        {"model_update": {"baseUrl": "https://unapproved.example"}},
        {"message_update": {"stopReason": "length"}},
        {"message_update": {"stopReason": "error", "errorMessage": "never-log-private-secret"}},
        {"message_update": {"provider": "other-provider"}},
        {"message_update": {"model": "other-model"}},
        {"message_update": {"api": "other-api"}},
        {"event": {"type": "message_update", "assistantMessageEvent": {"type": "error"}}},
        {"message_update": {"content": [{"type": "text", "text": None}]}},
        {"state_update": {"pendingMessageCount": False}},
        {"state_update": {"messageCount": 0.0}},
        {"system_update": {"toolsAdded": [{"name": "read"}]}},
        {"system_update": {"sections": {"preamble": "unexpected instructions"}}},
        {"system_update": {"content": "unexpected opaque prompt"}},
        {"event": {"type": "tool_execution_start", "args": {"secret": "never-log-private-secret"}}},
        {"event": {"type": "turn_end", "toolResults": [{"role": "toolResult"}]}},
        {"event": {"type": "agent_end", "messages": [{"role": "toolResult", "content": []}]}},
        {"event": {"type": "auto_retry_start"}},
        {"event": {"type": "compaction_start"}},
        {"truncate": True},
        {"omit_settled": True},
        {"aborted": True},
        {"late_event": True},
        {"bad_exit": True},
        {"flood": True},
    ],
)
def test_metadata_protocol_and_terminal_failures_never_return_a_candidate(
    configured_pi: PiConfig, changes: dict[str, object]
) -> None:
    """Unqualified access, hidden work or false completion cannot become usable source.

    Args:
        configured_pi (PiConfig): Trusted executable/profile seam.
        changes (dict[str, object]): One invalid external metadata or completion scenario.
    """
    set_scenario(configured_pi, changes)
    with pytest.raises(PiError) as caught:
        PiTransport(configured_pi).generate({"family": "knapsack"})
    assert "never-log-private-secret" not in str(caught.value)
    assert caught.value.unknown is False


@pytest.mark.skipif(sys.platform != "linux", reason="Initial Pi process supervision is Linux/WSL.")
def test_generation_deadline_confirms_owned_process_cleanup(configured_pi: PiConfig) -> None:
    """A hanging trusted harness cannot extend the declared deadline or remain alive.

    Args:
        configured_pi (PiConfig): Trusted executable/profile with a scripted sleeping request.
    """
    set_scenario(configured_pi, {"hang": True})
    processes = []
    with pytest.raises(PiError, match="deadline"):
        PiTransport(replace(configured_pi, seconds=1)).generate(
            {"family": "knapsack"}, processes.append
        )
    assert len(processes) == 3
    for pid in processes:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.skipif(sys.platform != "linux", reason="Initial Pi process supervision is Linux/WSL.")
def test_profile_policy_rejects_inherited_automation_without_overwriting_it(
    configured_pi: PiConfig,
) -> None:
    """A compatible profile is required; normal user configuration is never edited to fit.

    Args:
        configured_pi (PiConfig): Test-owned minimal profile before an unsafe setting mutation.
    """
    path = configured_pi.profile / "settings.json"
    data = json.loads(path.read_text())
    data["cacheWarming"] = "streaming"
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    with pytest.raises(PiError, match="profile"):
        PiTransport(configured_pi).generate({"family": "knapsack"})
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        configured_pi.prepare_profile()


@pytest.mark.skipif(sys.platform != "linux", reason="Initial Pi process supervision is Linux/WSL.")
def test_failed_response_keeps_available_reported_usage_without_raw_error(
    configured_pi: PiConfig,
) -> None:
    """A failed model response is not a free call or an excuse to persist private error text.

    Args:
        configured_pi (PiConfig): Fake Pi returning a failed authoritative assistant message.
    """
    set_scenario(
        configured_pi,
        {"message_update": {"stopReason": "error", "errorMessage": "never-log-private-secret"}},
    )
    with pytest.raises(PiError) as caught:
        PiTransport(configured_pi).generate({"family": "knapsack"})
    assert caught.value.usage == {
        "input": 11,
        "output": 7,
        "cacheRead": 2,
        "cacheWrite": 0,
        "totalTokens": 20,
    }
    assert "never-log-private-secret" not in str(caught.value)


@pytest.mark.skipif(sys.platform != "linux", reason="Initial Pi process supervision is Linux/WSL.")
def test_metadata_and_rpc_share_one_total_output_allowance(configured_pi: PiConfig) -> None:
    """Metadata commands do not each receive a new four-MiB output allocation.

    Args:
        configured_pi (PiConfig): Trusted executable emitting valid but padded metadata.
    """
    set_scenario(configured_pi, {"version": " " * 2200000 + "1.3.0", "auth_padding": 2200000})
    with pytest.raises(PiError, match="output_limit"):
        PiTransport(configured_pi).generate({"family": "knapsack"})


@pytest.mark.skipif(sys.platform != "linux", reason="Initial Pi process supervision is Linux/WSL.")
@pytest.mark.parametrize("early_events", [False, True])
def test_extra_completed_work_retains_usage_without_counting_queued_events_twice(
    configured_pi: PiConfig, early_events: bool
) -> None:
    """Extra completed reports fail admission without overwriting available prior usage.

    Args:
        configured_pi (PiConfig): Trusted executable with duplicate authoritative message events.
        early_events (bool): Whether completion events precede prompt acknowledgement.
    """
    set_scenario(configured_pi, {"duplicate_message": True, "early_events": early_events})
    with pytest.raises(PiError, match="extra_work") as caught:
        PiTransport(configured_pi).generate({"family": "knapsack"})
    assert caught.value.usage == {
        "input": 22,
        "output": 14,
        "cacheRead": 4,
        "cacheWrite": 0,
        "totalTokens": 40,
    }


@pytest.mark.skipif(sys.platform != "linux", reason="Initial Pi process supervision is Linux/WSL.")
def test_uncertain_cleanup_keeps_available_usage(
    configured_pi: PiConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cleanup exception after valid completion must not erase reported consumption.

    Args:
        configured_pi (PiConfig): Trusted complete-response executable.
        monkeypatch (pytest.MonkeyPatch): Failure injected only after actual child cleanup.
    """
    original = PiProcess.__exit__

    def uncertain_exit(
        process: PiProcess,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """Simulate loss of cleanup assurance after the real test process has been reaped.

        Args:
            process (PiProcess): Test-owned bounded subprocess.
            kind (type[BaseException] | None): Original exception type.
            error (BaseException | None): Original exception instance.
            traceback (TracebackType | None): Original traceback.

        Raises:
            PiError: For the RPC process only, after ordinary cleanup finishes.
        """
        original(process, kind, error, traceback)
        if process.rpc:
            raise PiError("cleanup", unknown=True)

    monkeypatch.setattr(PiProcess, "__exit__", uncertain_exit)
    with pytest.raises(PiError, match="cleanup") as caught:
        PiTransport(configured_pi).generate({"family": "knapsack"})
    assert caught.value.unknown
    assert caught.value.usage is not None
    assert caught.value.usage["totalTokens"] == 20


@pytest.mark.skipif(sys.platform != "linux", reason="Initial Pi process supervision is Linux/WSL.")
def test_successful_kill_signal_without_observed_group_exit_is_unknown(
    configured_pi: PiConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Signal delivery alone cannot certify that every owned descendant stopped.

    Args:
        configured_pi (PiConfig): Trusted executable that normally exits cleanly.
        monkeypatch (pytest.MonkeyPatch): Simulate a kernel group that remains observably active.
    """
    observations: list[list[int]] = []

    def active(groups: list[int]) -> bool:
        """Keep the external group observation live despite successful signal delivery.

        Args:
            groups (list[int]): Owned process group identities being inspected.

        Returns:
            bool: True to simulate delayed/uncertain descendant death.
        """
        observations.append(groups)
        return True

    monkeypatch.setattr(PiProcess, "groups_active", staticmethod(active), raising=False)
    with pytest.raises(PiError, match="cleanup") as caught:
        PiTransport(configured_pi).generate({"family": "knapsack"})
    assert caught.value.unknown
    assert len(observations) > 1
