"""Tests for ARD — backends API client and image encoding."""

import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import httpx
import pytest

try:  # preferred: the installed package
    from ard.backends import api_client as api_module
    from ard.backends.api_client import (
        ARDEmptyContentError,
        ARDLogprobsError,
        ARDTimeoutError,
        ChatAPIClient,
        ChatAPIConfig,
        ChatRequest,
        ChatResult,
        ChatResultWithLogprobs,
        _build_payload,
        encode_image_to_base64,
        logprobs_failure_count,
        reasoning_stats,
        reset_logprobs_failure_count,
        reset_reasoning_stats,
    )
except ModuleNotFoundError:  # pragma: no cover - env without the full dep set
    # ``ard/__init__.py`` eagerly imports the pipeline, which needs numpy.
    # Load this one module directly from its source file so the SSE-parsing
    # tests stay runnable in a partially installed environment.
    _MODULE_PATH = (
        Path(__file__).resolve().parents[2] / "src" / "ard" / "backends" / "api_client.py"
    )
    _SPEC = importlib.util.spec_from_file_location("_ard_api_client_standalone", _MODULE_PATH)
    assert _SPEC is not None and _SPEC.loader is not None
    api_module = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = api_module  # dataclasses needs the module registered
    _SPEC.loader.exec_module(api_module)  # type: ignore[union-attr]

    ARDEmptyContentError = api_module.ARDEmptyContentError
    ARDLogprobsError = api_module.ARDLogprobsError
    ARDTimeoutError = api_module.ARDTimeoutError
    ChatAPIClient = api_module.ChatAPIClient
    ChatAPIConfig = api_module.ChatAPIConfig
    ChatRequest = api_module.ChatRequest
    ChatResult = api_module.ChatResult
    ChatResultWithLogprobs = api_module.ChatResultWithLogprobs
    _build_payload = api_module._build_payload
    encode_image_to_base64 = api_module.encode_image_to_base64
    logprobs_failure_count = api_module.logprobs_failure_count
    reasoning_stats = api_module.reasoning_stats
    reset_logprobs_failure_count = api_module.reset_logprobs_failure_count
    reset_reasoning_stats = api_module.reset_reasoning_stats

# Captured before any monkeypatching: the module under test creates its own
# ``httpx.Client``, so tests that replace ``httpx.Client`` must still be able
# to build a real client from this reference (no recursion).
_REAL_HTTPX_CLIENT = httpx.Client


# ── API Client types ────────────────────────────────────────────────────────


def test_chat_api_config_defaults():
    """ChatAPIConfig has sensible defaults."""
    c = ChatAPIConfig(api_base="https://api.example.com", model_name="m", api_key="k")
    assert c.temperature == 0.7
    assert c.max_tokens is None
    assert c.max_retries == 2
    assert c.chat_completions_url == "https://api.example.com/chat/completions"
    # §3.3: the pre-authorized fallback is OFF by default.
    assert c.allow_missing_logprobs is False


def test_chat_api_config_url_no_trailing_slash():
    """chat_completions_url handles trailing slash correctly."""
    c = ChatAPIConfig(api_base="https://api.example.com/v1/", model_name="m", api_key="k")
    assert c.chat_completions_url == "https://api.example.com/v1/chat/completions"


def test_chat_request():
    """ChatRequest stores messages and temperature."""
    req = ChatRequest(messages=[{"role": "user", "content": "hi"}], temperature=0.5)
    assert req.messages == [{"role": "user", "content": "hi"}]
    assert req.temperature == 0.5


def test_chat_result_success():
    """ChatResult stores success result."""
    r = ChatResult(content="hello", success=True)
    assert r.content == "hello"
    assert r.success is True
    assert r.error is None


def test_chat_result_failure():
    """ChatResult stores failure result."""
    r = ChatResult(content="", success=False, error="timeout")
    assert r.success is False
    assert r.error == "timeout"


def test_chat_result_with_logprobs():
    """ChatResultWithLogprobs stores logprobs."""
    lp = {"token_ids": [1, 2], "log_probs": [-0.1, -0.2]}
    r = ChatResultWithLogprobs(content="hi", success=True, logprobs=lp)
    assert r.logprobs == lp
    assert r.content == "hi"


def test_chat_client_creation():
    """ChatAPIClient can be instantiated."""
    config = ChatAPIConfig(api_base="https://api.example.com", model_name="m", api_key="k")
    client = ChatAPIClient(config)
    assert client._config == config


# ── _build_payload — enable_thinking regression (B3) ────────────────────────


def test_build_payload_enable_thinking_true():
    """_build_payload sends chat_template_kwargs when enable_thinking=True."""
    config = ChatAPIConfig(
        api_base="https://api.example.com", model_name="m", api_key="k",
        enable_thinking=True,
    )
    payload = _build_payload(config, [{"role": "user", "content": "hi"}], None)
    assert "chat_template_kwargs" in payload
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}


def test_build_payload_enable_thinking_false():
    """_build_payload sends chat_template_kwargs when enable_thinking=False (B3 regression)."""
    config = ChatAPIConfig(
        api_base="https://api.example.com", model_name="m", api_key="k",
        enable_thinking=False,
    )
    payload = _build_payload(config, [{"role": "user", "content": "hi"}], None)
    assert "chat_template_kwargs" in payload
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_build_payload_enable_thinking_none_is_rejected():
    """``enable_thinking=None`` is refused; the field is two-state, not tri-state.

    Measurement behind the decision (vLLM 0.22 + Qwen3.8-27B chat template,
    ``chat_template.jinja:46`` — ``{%- if enable_thinking is undefined or
    enable_thinking is true %}``):

    ==========================  =========  ===================================
    request shape               thinking   consequence
    ==========================  =========  ===================================
    key sent as ``false``       off        (25 prompt tokens measured)
    key sent as ``true``        on         (65 prompt tokens measured)
    key **absent**              on         the template's ``undefined`` branch
    key sent as ``null``        on         *and* the reasoning splitter stops
                                           splitting → reasoning prose arrives
                                           as ``message.content``
    ==========================  =========  ===================================

    So "do not send the key" is not a safe third state: it silently turns
    thinking on, and the only shape that additionally contaminates the answer is
    ``null``.  A tri-state would keep both footguns reachable through a legal
    API call, so the field is deliberately two-state and ``None`` (like any
    other non-bool) fails at the contract boundary — ``ChatAPIConfig`` is
    annotated ``bool``, so nothing can build this state through the normal path
    either (§2.1 契约即防呆, §2.2 显式即防呆: a non-optional annotation must
    actually hold).
    """
    with pytest.raises(TypeError, match="enable_thinking must be True or False"):
        ChatAPIConfig(
            api_base="https://api.example.com", model_name="m", api_key="k",
            enable_thinking=None,  # type: ignore[arg-type]
        )


def test_build_payload_always_sends_enable_thinking_as_bool():
    """The key is always present and always a real bool — no omitted-key state."""
    for value in (True, False):
        config = ChatAPIConfig(
            api_base="https://api.example.com", model_name="m", api_key="k",
            enable_thinking=value,
        )
        payload = _build_payload(config, [{"role": "user", "content": "hi"}], None)
        assert "chat_template_kwargs" in payload, "omitting the key enables thinking"
        sent = payload["chat_template_kwargs"]["enable_thinking"]
        assert sent is value
        assert isinstance(sent, bool)
        # The serialized form must never be JSON null.
        assert json.dumps(payload["chat_template_kwargs"]) == (
            '{"enable_thinking": true}' if value else '{"enable_thinking": false}'
        )


def test_build_payload_enable_thinking_default():
    """_build_payload sends chat_template_kwargs with default enable_thinking=False."""
    config = ChatAPIConfig(api_base="https://api.example.com", model_name="m", api_key="k")
    payload = _build_payload(config, [{"role": "user", "content": "hi"}], None)
    assert "chat_template_kwargs" in payload
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


# ── SSE chunk parsing — logprobs extraction (D1 regression) ─────────────────
#
# These fixtures reproduce the shape measured on a real vLLM 0.22 server
# (Qwen3.8-27B): log-probs live ONLY under ``choices[0].logprobs`` as
# ``{"content": [...]}``, streaming log-probs are INCREMENTAL (one entry per
# generated token), elements carry ``token`` (str) / ``logprob`` (float) /
# ``bytes`` / ``top_logprobs`` and no ``token_id``, and when logprobs is not
# requested the key is present with the value ``null``.

_LOGPROBS_TEST_CONFIG = ChatAPIConfig(
    api_base="https://api.example.com", model_name="m", api_key="k",
    max_retries=0,
)


def _sse_data(payload: dict | str) -> str:
    """Format one SSE ``data:`` line (JSON dict or the literal ``[DONE]``)."""
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chunk(
    delta: dict,
    *,
    logprobs: dict | None | str = "omit",
    finish_reason: str | None = None,
) -> dict:
    """Build one streaming chunk.

    ``logprobs="omit"`` builds a chunk *without* the key at all; ``None`` builds
    one carrying an explicit JSON ``null``.
    """
    choice: dict = {"index": 0, "delta": delta}
    if logprobs != "omit":
        choice["logprobs"] = logprobs
    choice["finish_reason"] = finish_reason
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1789264293,
        "model": "Qwen3.8-27B",
        "choices": [choice],
    }


def _entry(token: str, logprob: float) -> dict:
    """One log-prob entry exactly as vLLM emits it (no ``token_id`` field)."""
    return {
        "token": token,
        "logprob": logprob,
        "bytes": list(token.encode("utf-8")),
        "top_logprobs": [
            {"token": token, "logprob": logprob, "bytes": list(token.encode("utf-8"))}
        ],
    }


def _vllm_logprobs_stream() -> list[str]:
    """A three-chunk stream with incremental log-probs (1, then 2 tokens).

    Content is ``"hello world"``; the collected log_probs must have exactly
    three entries with the values below, in order.
    """
    return [
        _sse_data(_chunk({"role": "assistant", "content": ""}, logprobs=None)),
        _sse_data(_chunk({"content": "hello"}, logprobs={"content": [_entry("hello", -0.03)]})),
        _sse_data(_chunk({"content": " world"}, logprobs={"content": [
            _entry(" world", -0.5),
            _entry("!", -0.25),
        ]})),
        _sse_data(_chunk({"content": ""}, logprobs=None, finish_reason="stop")),
        _sse_data("[DONE]"),
    ]


def _sse_transport(lines: list[str], *, delay: float = 0.0) -> httpx.MockTransport:
    """A transport whose streaming POST replays *lines* as an SSE body.

    Headers are returned immediately; the body bytes start after *delay*
    seconds (models a slow prefill on the server side).
    """

    def handler(request: httpx.Request) -> httpx.Response:
        def body():
            if delay:
                time.sleep(delay)
            for line in lines:
                yield line.encode("utf-8")

        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body()
        )

    return httpx.MockTransport(handler)


@pytest.fixture
def inject_sse_transport(monkeypatch):
    """Force every ``httpx.Client`` built by the module under test to replay SSE.

    The client creates its own ``httpx.Client`` inside
    ``_send_streaming_request`` (that is where the timeout object is built),
    so the transport must be injected at that boundary.
    """

    def _inject(lines: list[str], *, delay: float = 0.0) -> None:
        transport = _sse_transport(lines, delay=delay)

        def fake_client(*args, **kwargs):
            kwargs.pop("timeout", None)
            return _REAL_HTTPX_CLIENT(transport=transport, timeout=httpx.Timeout(None))

        monkeypatch.setattr(api_module.httpx, "Client", fake_client)

    return _inject


def test_streaming_logprobs_extracted_from_choices(inject_sse_transport):
    """D1 regression: log-probs are read from ``choices[0].logprobs.content``.

    Reproduces the real wire shape: the chunk top level has NO ``logprobs``
    key at all, so the old ``chunk.get("logprobs")`` returned ``None`` for
    51/51 measured chunks and every anchor was written with empty log-probs.
    """
    reset_logprobs_failure_count()
    inject_sse_transport(_vllm_logprobs_stream())

    result = ChatAPIClient(_LOGPROBS_TEST_CONFIG).chat_with_logprobs(
        [{"role": "user", "content": "hi"}]
    )

    assert result["content"] == "hello world"
    # Incremental accumulation: 1 + 2 entries, in arrival order.
    assert result["logprobs"]["token_ids"] == ["hello", " world", "!"]
    assert result["logprobs"]["log_probs"] == [-0.03, -0.5, -0.25]
    assert logprobs_failure_count() == {}


def test_streaming_logprobs_not_taken_from_chunk_top_level(inject_sse_transport):
    """A chunk-top-level ``logprobs`` key must NOT be used (prompt-side trap).

    The top level of a *non-streaming* response carries ``prompt_logprobs``;
    picking the wrong level silently attaches the wrong token list.
    """
    reset_logprobs_failure_count()
    decoy = _chunk({"content": "hi"}, logprobs=None)
    decoy["logprobs"] = {"content": [_entry("WRONG", -99.0)]}  # top-level decoy
    inject_sse_transport([
        _sse_data(decoy),
        _sse_data(_chunk({}, logprobs=None, finish_reason="stop")),
        _sse_data("[DONE]"),
    ])

    with pytest.raises(ARDLogprobsError):
        ChatAPIClient(_LOGPROBS_TEST_CONFIG).chat_with_logprobs(
            [{"role": "user", "content": "hi"}]
        )


def test_null_logprobs_is_not_silent_success(inject_sse_transport):
    """``logprobs: null`` (requested flag not honoured) must not pass silently.

    ``is None`` is the correct empty check (§2.2): the key is present with a
    null value, so ``"logprobs" in choice`` is True and a truthiness check is
    ambiguous.  The response still parses, but the missing supervision signal
    fails the anchor instead of emitting ``{"token_ids": [], "log_probs": []}``.
    """
    reset_logprobs_failure_count()
    inject_sse_transport([
        _sse_data(_chunk({"role": "assistant", "content": ""}, logprobs=None)),
        _sse_data(_chunk({"content": "answer"}, logprobs=None)),
        _sse_data(_chunk({"content": ""}, logprobs=None, finish_reason="stop")),
        _sse_data("[DONE]"),
    ])

    with pytest.raises(ARDLogprobsError) as excinfo:
        ChatAPIClient(_LOGPROBS_TEST_CONFIG).chat_with_logprobs(
            [{"role": "user", "content": "hi"}]
        )

    assert excinfo.value.reason == "no_logprobs_observed"
    assert "no log-probs" in str(excinfo.value)
    assert logprobs_failure_count() == {"no_logprobs_observed": 1}


def test_missing_logprobs_key_raises_and_logs(inject_sse_transport, caplog):
    """A chunk that omits the ``logprobs`` key entirely fails the same way."""
    reset_logprobs_failure_count()
    inject_sse_transport([
        _sse_data(_chunk({"content": "answer"}, logprobs="omit")),
        _sse_data(_chunk({}, logprobs="omit", finish_reason="stop")),
        _sse_data("[DONE]"),
    ])

    with caplog.at_level(logging.ERROR, logger="ard.backends.api_client"):
        with pytest.raises(ARDLogprobsError) as excinfo:
            ChatAPIClient(_LOGPROBS_TEST_CONFIG).chat_with_logprobs(
                [{"role": "user", "content": "hi"}]
            )

    assert excinfo.value.reason == "key_missing"
    # Observable signal: an ERROR log line, not only an exception.
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert logprobs_failure_count() == {"key_missing": 1}


def test_allow_missing_logprobs_opt_in_degrades_with_warning(inject_sse_transport, caplog):
    """§3.3: the empty-payload fallback needs an explicit opt-in and warns."""
    reset_logprobs_failure_count()
    config = ChatAPIConfig(
        api_base="https://api.example.com", model_name="m", api_key="k",
        max_retries=0, allow_missing_logprobs=True,
    )
    inject_sse_transport([
        _sse_data(_chunk({"content": "answer"}, logprobs=None)),
        _sse_data(_chunk({}, logprobs=None, finish_reason="stop")),
        _sse_data("[DONE]"),
    ])

    with caplog.at_level(logging.WARNING, logger="ard.backends.api_client"):
        result = ChatAPIClient(config).chat_with_logprobs(
            [{"role": "user", "content": "hi"}]
        )

    assert result["logprobs"] == {"token_ids": [], "log_probs": []}
    assert any(r.levelno >= logging.WARNING for r in caplog.records)
    # Still counted — the user can see how many anchors degraded.
    assert logprobs_failure_count() == {"no_logprobs_observed": 1}


def test_incomplete_logprobs_raises_even_when_opt_in(inject_sse_transport):
    """A truncated token list is corrupt data — the opt-in does not cover it."""
    reset_logprobs_failure_count()
    config = ChatAPIConfig(
        api_base="https://api.example.com", model_name="m", api_key="k",
        max_retries=0, allow_missing_logprobs=True,
    )
    inject_sse_transport([
        _sse_data(_chunk({"content": "hello"}, logprobs={"content": [_entry("hello", -0.03)]})),
        _sse_data(_chunk({"content": " world"}, logprobs=None)),  # stream lost them
        _sse_data(_chunk({}, logprobs=None, finish_reason="stop")),
        _sse_data("[DONE]"),
    ])

    with pytest.raises(ARDLogprobsError) as excinfo:
        ChatAPIClient(config).chat_with_logprobs([{"role": "user", "content": "hi"}])

    assert "incomplete" in str(excinfo.value)
    # The reason is the *specific* one: delivery stopped mid-stream after entries
    # had already arrived. ``no_logprobs_observed`` would be the wrong diagnosis
    # here (something was observed) and would hide the truncation.
    assert logprobs_failure_count() == {"stream_stopped_delivering": 1}


def test_logprobs_key_vanishing_midstream_is_incomplete(inject_sse_transport):
    """The ``logprobs`` key itself disappearing mid-stream is also truncation.

    Same failure as a mid-stream ``null``, reached through a different schema
    violation: the key is simply absent from later chunks.  Both are "entries
    were delivered and then delivery stopped", so both must hard-fail, and
    neither may be excused by ``allow_missing_logprobs`` (that flag is about a
    response that never had log-probs at all — §3.4: a truncated token list is a
    defence, not a fallback).
    """
    reset_logprobs_failure_count()
    config = ChatAPIConfig(
        api_base="https://api.example.com", model_name="m", api_key="k",
        max_retries=0, allow_missing_logprobs=True,  # must NOT cover this
    )
    inject_sse_transport([
        _sse_data(_chunk({"content": "hello"}, logprobs={"content": [_entry("hello", -0.03)]})),
        _sse_data(_chunk({"content": " world"})),  # key absent entirely
        _sse_data(_chunk({"content": "!"}, logprobs={"content": []})),
        _sse_data(_chunk({}, logprobs=None, finish_reason="stop")),
        _sse_data("[DONE]"),
    ])

    with pytest.raises(ARDLogprobsError) as excinfo:
        ChatAPIClient(config).chat_with_logprobs([{"role": "user", "content": "hi"}])

    assert "incomplete" in str(excinfo.value)
    assert logprobs_failure_count() == {"stream_stopped_delivering": 1}


def test_malformed_logprob_entries_are_rejected_not_skipped(inject_sse_transport):
    """Entries without a usable ``logprob`` must fail, not shrink silently."""
    reset_logprobs_failure_count()
    bad = {"token": "x", "bytes": [120], "top_logprobs": []}  # no "logprob"
    inject_sse_transport([
        _sse_data(_chunk({"content": "x"}, logprobs={"content": [bad]})),
        _sse_data(_chunk({}, logprobs=None, finish_reason="stop")),
        _sse_data("[DONE]"),
    ])

    with pytest.raises(ARDLogprobsError) as excinfo:
        ChatAPIClient(_LOGPROBS_TEST_CONFIG).chat_with_logprobs(
            [{"role": "user", "content": "hi"}]
        )

    assert excinfo.value.reason == "entries_skipped"


# ── Timeout policy: no httpx read deadline, first-token wait wins (D2) ──────
#
# MockTransport cannot model a slow first byte: the mock handler is invoked
# synchronously by ``client.stream()`` and its whole body is drained before
# streaming starts, so the delay lands *before* the request returns instead of
# after the headers.  These tests therefore run a real (loopback) HTTP server
# that sends headers immediately and only then stalls the body.

_SLOW_SSE_LINES = [
    _sse_data(_chunk({"content": "late"}, logprobs="omit", finish_reason="stop")),
    _sse_data("[DONE]"),
]


class _SlowSSEServer:
    """A loopback HTTP server that returns headers now and the body later."""

    def __init__(self, first_byte_delay: float) -> None:
        import http.server
        import threading

        self.first_byte_delay = first_byte_delay
        self.requests_seen = 0
        self.client_gone = threading.Event()

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802 (http.server API)
                outer = self.server.ard_server  # type: ignore[attr-defined]
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)  # consume the request body
                outer.requests_seen += 1
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.flush()  # headers are on the wire now
                time.sleep(outer.first_byte_delay)  # ... server is busy (prefill)
                payload = "".join(_SLOW_SSE_LINES).encode("utf-8")
                block = f"{len(payload):x}\r\n".encode() + payload + b"\r\n"
                try:
                    self.wfile.write(block)
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except OSError:
                    # The client dropped the connection (timeout) — evidence
                    # that its reader was released rather than left blocked.
                    outer.client_gone.set()

            def log_message(self, *args) -> None:
                pass  # keep pytest output clean

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._server.ard_server = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address[:2]
        self.url = f"http://{host}:{port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def slow_sse_server():
    servers: list[_SlowSSEServer] = []

    def _make(first_byte_delay: float) -> _SlowSSEServer:
        server = _SlowSSEServer(first_byte_delay)
        servers.append(server)
        return server

    yield _make
    for server in servers:
        server.close()


def test_httpx_timeout_has_no_read_deadline(monkeypatch):
    """Strategy check: the httpx-level read deadline must be disabled.

    Captures the ``httpx.Timeout`` the module actually builds.
    """
    captured: dict = {}

    def recording_client(*args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return _REAL_HTTPX_CLIENT(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"choices": []})
            )
        )

    monkeypatch.setattr(api_module.httpx, "Client", recording_client)

    # The probe server returns a valid HTTP 200 with an empty choice list, i.e.
    # a response with no assistant content.  Since WP-F3 that is a typed
    # model-output failure rather than an empty string, so the call is expected
    # to raise — the point of this test is the timeout object, not the content.
    with pytest.raises(ARDEmptyContentError) as excinfo:
        ChatAPIClient(_LOGPROBS_TEST_CONFIG).chat([{"role": "user", "content": "hi"}])

    assert "no assistant content" in str(excinfo.value)
    timeout = captured["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read is None, f"read deadline must be None, got {timeout.read!r}"
    assert timeout.connect == _LOGPROBS_TEST_CONFIG.connect_timeout


def test_first_token_timeout_still_fires_and_releases_reader(slow_sse_server):
    """The first-token phase timeout is enforced by the app-level timer.

    Regression guard for the replacement of ``read=30.0``: the phase timeout
    must still bound the request, and it must also release the reader thread
    (the server observes the dropped connection).
    """
    server = slow_sse_server(5.0)
    config = ChatAPIConfig(
        api_base=server.url, model_name="m", api_key="k",
        max_retries=0, first_token_timeout=0.5, inter_token_timeout=5.0,
    )

    started = time.monotonic()
    with pytest.raises(ARDTimeoutError) as excinfo:
        ChatAPIClient(config).chat([{"role": "user", "content": "hi"}])
    elapsed = time.monotonic() - started

    assert "first token" in str(excinfo.value)
    assert "timeout=0s" in str(excinfo.value)  # 0.5 s formats as 0 s
    assert elapsed < 3.0, f"first-token timeout took {elapsed:.1f}s"
    # The server's write into a closed socket is the proof that closing the
    # response released the blocked recv().
    assert server.client_gone.wait(timeout=10.0), "client did not drop the connection"


@pytest.mark.slow
def test_first_token_wait_not_capped_at_30s(slow_sse_server):
    """Behavioral proof: a >30 s first-token wait is no longer cut off at 30 s.

    The server returns headers immediately and delays the first body byte by
    31 s — the measured production shape (headers at 0.342 s, first SSE byte at
    8.904 s for a 27k-token prompt; server prefill 8.51 s), scaled to just cross
    the removed 30 s deadline.  With ``read=30.0`` in place this request ended in
    ``httpx.ReadTimeout`` at ~30 s (the read timer measures exactly this wait);
    with the read deadline disabled the request must succeed, because
    ``first_token_timeout=90`` governs the phase.

    Runtime is dominated by the 31 s server-side delay — deselect with
    ``-m "not slow"`` when iterating locally.
    """
    server = slow_sse_server(31.0)
    config = ChatAPIConfig(
        api_base=server.url, model_name="m", api_key="k",
        max_retries=0, first_token_timeout=90.0, inter_token_timeout=15.0,
    )

    started = time.monotonic()
    content = ChatAPIClient(config).chat([{"role": "user", "content": "hi"}])
    elapsed = time.monotonic() - started

    assert content == "late"
    assert elapsed > 30.0, f"server delay was not exercised ({elapsed:.1f}s)"
    assert elapsed < 60.0, f"request took {elapsed:.1f}s"


# ── Image encoding ──────────────────────────────────────────────────────────


def test_encode_image_to_base64_png(tmp_path):
    """encode_image_to_base64 returns data URI for PNG."""
    # Create a minimal valid PNG (1x1 pixel)
    import struct
    import zlib

    def create_png(width, height):
        def chunk(chunk_type, data):
            c = chunk_type + data
            crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            return struct.pack(">I", len(data)) + c + crc

        header = b"\x89PNG\r\n\x1a\n"
        ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        raw = b""
        for y in range(height):
            raw += b"\x00" + b"\xff\x00\x00" * width
        idat = chunk(b"IDAT", zlib.compress(raw))
        iend = chunk(b"IEND", b"")
        return header + ihdr + idat + iend

    png_path = tmp_path / "test.png"
    png_path.write_bytes(create_png(1, 1))
    result = encode_image_to_base64(png_path)
    assert result.startswith("data:image/png;base64,")


def test_encode_image_to_base64_file_not_found():
    """encode_image_to_base64 raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        encode_image_to_base64("nonexistent.png")


def test_encode_image_to_base64_unsupported_format(tmp_path):
    """encode_image_to_base64 raises ValueError for unsupported format."""
    bad = tmp_path / "test.xyz"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Unsupported image format"):
        encode_image_to_base64(bad)

# ── Reasoning handling (WP-F3 regression) ───────────────────────────────────
#
# Real-server shape (vLLM 0.22.0 + Qwen3.8-27B): the reasoning field is
# ``delta.reasoning`` (NOT ``reasoning_content``), and while the model thinks
# ``delta.content`` stays ``null``.  The template treats "enable_thinking
# undefined" as ON, so a production run with ``enable_thinking = true`` streams
# a long reasoning phase first.  The old parser collected only ``content``, so
# the reasoning tokens vanished from the accounting and an exhausted budget
# produced an empty string that the dataset layer silently dropped.

_REASONING_TEST_CONFIG = ChatAPIConfig(
    api_base="https://api.example.com", model_name="m", api_key="k",
    max_retries=0,
)


def _reasoning_chunk(text: str, *, finish_reason: str | None = None) -> dict:
    """One thinking-phase chunk: ``content`` is ``null``, only ``reasoning`` set."""
    return _chunk({"role": "assistant", "reasoning": text, "content": None},
                  logprobs=None, finish_reason=finish_reason)


def _empty_content_only_chunk(*, finish_reason: str) -> dict:
    """A finish chunk with neither content nor reasoning (no-thinking case)."""
    return _chunk({"content": None}, logprobs=None, finish_reason=finish_reason)


def _reasoning_only_with_logprobs_stream() -> list[str]:
    """Same as above, but the server *did* deliver log-probs.

    Isolates the empty-content classification from the log-probs contract: with
    usable log-probs present, the only thing wrong with this response is that
    reasoning consumed the whole budget.
    """
    return [
        _sse_data(_chunk({"role": "assistant", "content": None}, logprobs=None)),
        _sse_data(_chunk({"role": "assistant", "reasoning": "thinking, thinking"},
                         logprobs={"content": [_entry("thinking", -0.01)]})),
        _sse_data(_chunk({"content": None}, logprobs={"content": []},
                         finish_reason="length")),
        _sse_data("[DONE]"),
    ]


def _reasoning_only_truncated_stream() -> list[str]:
    """Reasoning, then the budget runs out before any content: the production bug."""
    return [
        _sse_data(_chunk({"role": "assistant", "content": None}, logprobs=None)),
        _sse_data(_reasoning_chunk("Let me think about this. ")),
        _sse_data(_reasoning_chunk("Step one: the answer is 42, but I must check.")),
        _sse_data(_chunk({"content": None}, logprobs=None, finish_reason="length")),
        _sse_data("[DONE]"),
    ]


def _reasoning_then_answer_stream() -> list[str]:
    """Reasoning followed by a real answer: the healthy thinking case."""
    return [
        _sse_data(_chunk({"role": "assistant", "content": None}, logprobs=None)),
        _sse_data(_reasoning_chunk("The user asks for a number. ")),
        _sse_data(_reasoning_chunk("Six times seven is forty-two.")),
        _sse_data(_chunk({"content": "Six"}, logprobs=None)),
        _sse_data(_chunk({"content": " times seven"}, logprobs=None)),
        _sse_data(_chunk({"content": " is 42."}, logprobs=None, finish_reason="stop")),
        _sse_data("[DONE]"),
    ]


def test_reasoning_field_is_recognised_not_content(inject_sse_transport):
    """F3: ``delta.reasoning`` is counted; content collection ignores it."""
    reset_reasoning_stats()
    inject_sse_transport(_reasoning_then_answer_stream())

    content = ChatAPIClient(_REASONING_TEST_CONFIG).chat(
        [{"role": "user", "content": "what is 6*7"}]
    )

    # ① content is exactly the answer — no reasoning prose leaked into it.
    assert content == "Six times seven is 42."
    assert "The user asks" not in content
    assert "forty-two" not in content

    # ② the reasoning is observable: counters, no text.
    stats = reasoning_stats()
    assert stats["responses"] == 1
    assert stats["reasoning_responses"] == 1
    assert stats["reasoning_chars"] == len(
        "The user asks for a number. Six times seven is forty-two."
    )
    # Zero counts are not recorded (there was nothing to report): absence == 0.
    assert stats.get("reasoning_only_responses", 0) == 0
    assert stats.get("empty_content", 0) == 0


def test_reasoning_only_truncated_raises_and_warns(
    inject_sse_transport, caplog
):
    """F3 core bug: reasoning ate the budget → no content, must not be silent.

    Asserts the three required properties: the failure is *announced*
    (WARNING), *counted*, and *typed* (:exc:`ARDEmptyContentError`, distinct
    from a transport error).
    """
    reset_reasoning_stats()
    inject_sse_transport(_reasoning_only_truncated_stream())

    with caplog.at_level(logging.WARNING, logger=api_module.logger.name):
        with pytest.raises(ARDEmptyContentError) as excinfo:
            ChatAPIClient(_REASONING_TEST_CONFIG).chat([{"role": "user", "content": "hi"}])

    message = str(excinfo.value)
    # Distinguishable from a network/timeout failure: it says what happened.
    assert "max_tokens" in message
    assert "reasoning" in message
    assert isinstance(excinfo.value, RuntimeError)  # backwards-compatible base
    assert not isinstance(excinfo.value, ARDTimeoutError)

    # A WARNING is emitted (visible at the default logging level), not a debug toy.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Empty completion discarded" in r.getMessage() for r in warnings), caplog.text

    stats = reasoning_stats()
    assert stats["reasoning_responses"] == 1
    assert stats["reasoning_only_responses"] == 1
    assert stats["empty_content"] == 1
    assert stats["truncated_empty"] == 1
    assert stats["reasoning_chars"] > 0


def test_empty_content_without_reasoning_is_not_a_timeout(inject_sse_transport, caplog):
    """A content-less ``stop`` response is classified as a model-output failure.

    Guards the other half of the distinction: "empty" must not be reported as a
    transport problem, and it must never be retried as one.
    """
    reset_reasoning_stats()
    inject_sse_transport([
        _sse_data(_chunk({"role": "assistant", "content": None}, logprobs=None)),
        _sse_data(_empty_content_only_chunk(finish_reason="stop")),
        _sse_data("[DONE]"),
    ])

    with caplog.at_level(logging.WARNING, logger=api_module.logger.name):
        with pytest.raises(ARDEmptyContentError) as excinfo:
            ChatAPIClient(_REASONING_TEST_CONFIG).chat([{"role": "user", "content": "hi"}])

    assert "no assistant content" in str(excinfo.value)
    stats = reasoning_stats()
    # No reasoning at all: that key is never recorded (absence == 0).
    assert stats.get("reasoning_responses", 0) == 0
    assert stats["empty_content"] == 1
    # Not a truncation — a different cause, told apart explicitly.
    assert stats.get("truncated_empty", 0) == 0
    assert any("Empty completion discarded" in r.getMessage() for r in caplog.records)


def test_empty_content_is_not_retried(inject_sse_transport, monkeypatch):
    """The failure is deterministic, so it must not burn max_retries × backoff."""
    reset_reasoning_stats()
    calls: list[int] = []

    def counting_transport():
        lines = _reasoning_only_truncated_stream()

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)

            def body():
                for line in lines:
                    yield line.encode("utf-8")

            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=body()
            )

        return httpx.MockTransport(handler)

    real_inject = counting_transport()
    monkeypatch.setattr(
        api_module.httpx, "Client",
        lambda *a, **k: _REAL_HTTPX_CLIENT(transport=real_inject, timeout=httpx.Timeout(None)),
    )
    config = ChatAPIConfig(
        api_base="https://api.example.com", model_name="m", api_key="k", max_retries=2,
    )

    with pytest.raises(ARDEmptyContentError):
        ChatAPIClient(config).chat([{"role": "user", "content": "hi"}])

    assert len(calls) == 1, "an empty-content response must fail fast, not retry"


def test_chat_with_logprobs_empty_content_raises_before_logprobs_check(inject_sse_transport):
    """Reasoning-only truncation is reported as empty content, not as missing logprobs.

    Both are failures of the same response; the empty-content cause is the one
    that explains *why* the budget ran out, so it wins and it is not retried.
    """
    reset_reasoning_stats()
    inject_sse_transport(_reasoning_only_with_logprobs_stream())

    with pytest.raises(ARDEmptyContentError):
        ChatAPIClient(_REASONING_TEST_CONFIG).chat_with_logprobs(
            [{"role": "user", "content": "hi"}]
        )

    stats = reasoning_stats()
    assert stats["reasoning_only_responses"] == 1
    assert stats["truncated_empty"] == 1


def test_reasoning_content_fallback_field_is_supported(inject_sse_transport):
    """``delta.reasoning_content`` (DeepSeek spelling) is also recognised."""
    reset_reasoning_stats()
    inject_sse_transport([
        _sse_data(_chunk({"role": "assistant", "content": None}, logprobs=None)),
        _sse_data(_chunk({"reasoning_content": "thinking with the other spelling"},
                         logprobs=None)),
        _sse_data(_chunk({"content": "answer"}, logprobs=None, finish_reason="stop")),
        _sse_data("[DONE]"),
    ])

    content = ChatAPIClient(_REASONING_TEST_CONFIG).chat([{"role": "user", "content": "hi"}])

    assert content == "answer"
    stats = reasoning_stats()
    assert stats["reasoning_responses"] == 1
    assert stats["reasoning_chars"] == len("thinking with the other spelling")


def test_stats_dataclass_reports_reasoning_only():
    """ChatAPIStats flags a reasoning-only response without carrying its text."""
    stats = api_module.ChatAPIStats(
        content_chars=0, reasoning_chars=1234, reasoning_chunks=57, finish_reason="length"
    )
    assert stats.has_reasoning is True
    assert stats.empty_content is True
    assert "reasoning_chars=1234" in stats.describe()
    assert "1234" in stats.describe()
    # No field can hold reasoning text (count-only contract).
    assert all(
        not isinstance(getattr(stats, field), str) or field == "finish_reason"
        for field in stats.__slots__
    )


# ── enable_thinking type contract (WP-F3, §2.1/§2.2) ────────────────────────


@pytest.mark.parametrize("value", [None, 1, 0, "true", "yes", 1.0, [], {}])
def test_enable_thinking_rejects_non_bool(value):
    """Only True/False are accepted; null must never reach the wire.

    Real-server measurement: an explicit ``"enable_thinking": null`` still opens
    a ``<think>`` block (thinking ON) *and* disables the server's reasoning
    splitter, so reasoning prose leaks into ``message.content`` — the one
    outcome that silently corrupts the distilled dataset.
    """
    with pytest.raises(TypeError, match="enable_thinking must be True or False"):
        ChatAPIConfig(
            api_base="https://api.example.com", model_name="m", api_key="k",
            enable_thinking=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [True, False])
def test_enable_thinking_accepts_both_bools(value):
    """Both real bools are accepted and reach the payload as bools."""
    config = ChatAPIConfig(
        api_base="https://api.example.com", model_name="m", api_key="k",
        enable_thinking=value,
    )
    payload = _build_payload(config, [{"role": "user", "content": "hi"}], None)
    sent = payload["chat_template_kwargs"]["enable_thinking"]
    assert sent is value, f"expected identity with {value!r}, got {sent!r}"


def test_enable_thinking_rejected_before_any_request(monkeypatch):
    """The contract is enforced at construction, before a client can send NULL."""
    sent: list[dict] = []
    monkeypatch.setattr(
        api_module.httpx, "Client",
        lambda *a, **k: _REAL_HTTPX_CLIENT(
            transport=httpx.MockTransport(
                lambda request: sent.append(request.read()) or httpx.Response(200, content=b"")
            )
        ),
    )
    with pytest.raises(TypeError):
        ChatAPIClient(ChatAPIConfig(
            api_base="https://api.example.com", model_name="m", api_key="k",
            enable_thinking=None,  # type: ignore[arg-type]
        ))
    assert sent == []
