"""R12 regression tests — temperature is a single source of truth config → payload.

Before R12, every LLM call site in :mod:`ard.domain.text_anchor` passed a
hardcoded per-request ``temperature`` (``0.7`` for the input side, ``0.0`` for
the target side), so the configured values (``[input_generator].temperature``,
``[target_model].temperature``) never reached the wire.  These tests freeze the
fixed contract:

* ``[target_model].temperature`` defaults to **0.1** (user ruling) and
  ``[input_generator].temperature`` to **0.8**;
* changing the config value changes the temperature in the **actual request
  payload** — proven against a real :class:`ChatAPIClient` behind a recording
  ``httpx.MockTransport``, so no request ever leaves the process (no billed
  traffic).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from ard.backends import api_client as api_module
from ard.backends.api_client import ChatAPIClient, ChatAPIConfig
from ard.config import InputGeneratorConfig, TargetModelConfig, load_config
from ard.core.types import AnchorSpec, TurnSpec
from ard.domain.text_anchor import _generate_one_anchor

_CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "config.toml"

# A response body longer than the 8-char ``min_answer_chars`` gate, so the
# recorded target answer is accepted and no anchor is abandoned.
_ANSWER_TEXT = "The capital of France is Paris."


def _sse_data(payload: object) -> str:
    """One ``data:`` SSE line in the exact format the client's parser expects.

    A string payload (the literal ``[DONE]`` terminator) is emitted verbatim —
    it is protocol syntax, not a JSON value, so it must not be JSON-encoded.
    """
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_stream(content: str) -> list[str]:
    """A healthy streaming response: one content chunk ending with ``stop``."""
    return [
        _sse_data(
            {
                "id": "chatcmpl-recorded",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": None},
                        "finish_reason": None,
                    }
                ],
            }
        ),
        _sse_data(
            {
                "id": "chatcmpl-recorded",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "m",
                "choices": [
                    {"index": 0, "delta": {"content": content}, "finish_reason": "stop"}
                ],
            }
        ),
        _sse_data("[DONE]"),
    ]


def _make_spec(include_system_prompt: bool = False) -> AnchorSpec:
    """A single-turn spec; optionally carrying a system-prompt mode.

    With ``include_system_prompt=True`` the anchor's first input-generator
    request is the system-prompt generation (:func:`_generate_system_message`),
    which is the third of the four hardcoded call sites fixed in R12.
    """
    meta = {
        "language": "English",
        "knowledge_domain": "geography",
        "capability": "qa",
        "conversation_type": "single_turn",
    }
    if include_system_prompt:
        meta["system_prompt_mode"] = "task_constraint"
    return AnchorSpec(
        id=f"temp_001_{include_system_prompt}",
        anchor_meta=meta,
        turns=[
            TurnSpec(
                turn_index=0,
                role="user",
                generation_instruction="Ask about geography",
                is_final=True,
            )
        ],
        input_generator_id="input-gen",
    )


def _recording_transport(recorder: list[dict], content: str) -> httpx.MockTransport:
    """A transport that records every request body and replays *content*."""

    def handler(request: httpx.Request) -> httpx.Response:
        recorder.append(json.loads(request.read()))

        def body():
            for line in _sse_stream(content):
                yield line.encode("utf-8")

        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body()
        )

    return httpx.MockTransport(handler)


@pytest.fixture
def recording_clients(monkeypatch) -> tuple[list[dict], Callable[..., None]]:
    """Monkeypatch ``httpx.Client`` so every client records its request payloads.

    Returns ``(recorder, run)`` where ``run(input_temp, target_temp, ...)``
    runs a single anchor through real clients whose config temperatures are set
    from the caller, appending each request body to *recorder*.
    """
    recorder: list[dict] = []

    def run(
        *,
        input_temp: float,
        target_temp: float,
        include_system_prompt: bool = False,
    ) -> None:
        real_client = httpx.Client

        def patched_client(*args, **kwargs):
            kwargs.pop("timeout", None)
            transport = _recording_transport(recorder, _ANSWER_TEXT)
            return real_client(transport=transport, timeout=httpx.Timeout(None))

        monkeypatch.setattr(api_module.httpx, "Client", patched_client)

        input_client = ChatAPIClient(
            ChatAPIConfig(
                api_base="https://api.example.com",
                model_name="input-model",
                api_key="k",
                temperature=input_temp,
            )
        )
        target_client = ChatAPIClient(
            ChatAPIConfig(
                api_base="https://api.example.com",
                model_name="target-model",
                api_key="k",
                temperature=target_temp,
            )
        )
        _generate_one_anchor(
            _make_spec(include_system_prompt),
            input_client,
            target_client,
            "input-model",
            "target-model",
        )

    return recorder, run


# ── Config defaults (user ruling: teacher = 0.1) ─────────────────────────────


def test_config_defaults_target_0_1_and_input_0_8():
    """The merged, real ``configs/config.toml`` resolves teacher=0.1 / input=0.8."""
    config = load_config(_CONFIGS)
    assert config.input_generator.temperature == 0.8
    assert config.target_model.temperature == 0.1


def test_code_defaults_agree_with_toml_defaults():
    """Code-level Pydantic defaults are the same single truth as config.toml.

    Both defaults must match — two divergent defaults would silently pick which
    one wins depending on how the object is constructed (§1.4 单一真相源).
    """
    ig = InputGeneratorConfig(api_base="https://api.example.com", model_name="m", api_key="k")
    assert ig.temperature == 0.8
    tm = TargetModelConfig(api_base="https://api.example.com", model_name="m", api_key="k")
    assert tm.temperature == 0.1


# ── Config → payload flow (recording transport; no billed requests) ──────────


def test_config_temperature_reaches_input_and_target_payloads(recording_clients):
    """With the new defaults, the wire payload carries 0.8 (input) / 0.1 (target).

    The anchor generator passes **no** per-request temperature (R12 fix), so the
    payload temperature must come from the client config — the chain that made
    ``config.toml`` authoritative again.
    """
    recorder, run = recording_clients
    run(input_temp=0.8, target_temp=0.1)

    input_payloads = [p for p in recorder if p["model"] == "input-model"]
    target_payloads = [p for p in recorder if p["model"] == "target-model"]
    # Single-turn spec: one user-turn request on the input side, one answer on
    # the target side.
    assert len(input_payloads) == 1
    assert len(target_payloads) == 1
    assert input_payloads[0]["temperature"] == 0.8
    assert target_payloads[0]["temperature"] == 0.1


def test_changed_config_temperature_reaches_payloads(recording_clients):
    """Changing the config temperature changes the actual request payload.

    A different pair of values must flow all the way to the wire — this is the
    behavior the fixed code guarantees and the exact thing the hardcoded
    overrides used to break.
    """
    recorder, run = recording_clients
    run(input_temp=0.3, target_temp=0.7)

    input_payloads = [p for p in recorder if p["model"] == "input-model"]
    target_payloads = [p for p in recorder if p["model"] == "target-model"]
    assert len(input_payloads) == 1
    assert len(target_payloads) == 1
    assert input_payloads[0]["temperature"] == 0.3
    assert target_payloads[0]["temperature"] == 0.7


def test_system_prompt_generation_uses_input_generator_temperature(recording_clients):
    """The system-prompt generation call also follows ``[input_generator]``.

    That call site (third hardcoded ``0.7``) historically masked the configured
    input temperature just like the user-turn call did; it must now carry the
    same client-config value.
    """
    recorder, run = recording_clients
    run(input_temp=0.8, target_temp=0.1, include_system_prompt=True)

    input_payloads = [p for p in recorder if p["model"] == "input-model"]
    # system-prompt generation + the user turn → two input-side requests.
    assert len(input_payloads) == 2
    assert all(p["temperature"] == 0.8 for p in input_payloads)
