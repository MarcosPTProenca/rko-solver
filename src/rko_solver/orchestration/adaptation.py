"""Readable adaptation modes and their legacy control-code mapping.

Objective:
- Expose one plain-language name for fixed versus adaptive parameter behaviour.

Technical approach:
- Map the public strings "fixed" and "adaptive" onto the legacy integer codes.
- Accept the legacy keyword only when it does not contradict the readable name.
- Keep one normalization function so every entry point rejects inputs identically.

Flow (pseudocode):
1. Read the optional adaptation string and the optional legacy control integer.
2. Reject unknown strings and contradictory pairs with one ValueError.
3. Return the integer code understood by the orchestration layer, or None.

Important criteria:
- "fixed" is the legacy offline mode: parameters chosen before the search starts.
- "adaptive" is the legacy online mode: Q-learning selects parameters during it.
- Nothing is silently coerced; unknown or conflicting values raise ValueError.

Inputs/Outputs:
- Input: adaptation (str | None) and control (int | None).
- Output: control (int | None), the internal mode code.
"""

from __future__ import annotations

from typing import Literal

Adaptation = Literal["fixed", "adaptive"]

ADAPTATION_TO_CONTROL: dict[str, int] = {"fixed": 0, "adaptive": 1}

CONTROL_TO_ADAPTATION: dict[int, Adaptation] = {0: "fixed", 1: "adaptive"}


def normalize_adaptation(
    adaptation: Adaptation | None = None,
    control: int | None = None,
) -> int | None:
    """Resolve the public adaptation name and the legacy control code into one mode.

    Args:
        adaptation (Adaptation | None): "fixed" or "adaptive"; None leaves the
            configured default in charge when no legacy code is supplied.
        control (int | None): Legacy 0 (fixed) or 1 (adaptive) code.

    Returns:
        int | None: Internal control code, or None when neither input was given.

    Raises:
        ValueError: If the adaptation name is unknown, the legacy code is not
            0 or 1, or both inputs describe different modes.
    """
    if control is not None and (type(control) is not int or control not in (0, 1)):
        raise ValueError("'control' must be 0 (fixed) or 1 (adaptive).")
    if adaptation is None:
        return control
    if not isinstance(adaptation, str) or adaptation not in ADAPTATION_TO_CONTROL:
        supported = ", ".join(sorted(ADAPTATION_TO_CONTROL))
        raise ValueError(f"'adaptation' must be one of: {supported}.")
    resolved = ADAPTATION_TO_CONTROL[adaptation]
    if control is not None and control != resolved:
        raise ValueError(
            f"'adaptation={adaptation}' contradicts control={control}; pass only 'adaptation'."
        )
    return resolved


def adaptation_name(control: int | None) -> Adaptation | None:
    """Translate an internal control code back into its public name.

    Args:
        control (int | None): Internal 0/1 mode code, or None when unknown.

    Returns:
        Adaptation | None: "fixed", "adaptive", or None when no mode is recorded.
    """
    if control is None:
        return None
    return CONTROL_TO_ADAPTATION.get(control)
