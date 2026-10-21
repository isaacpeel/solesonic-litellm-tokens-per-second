import json
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from types import SimpleNamespace

import pytest
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import convert_to_model_response_object
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    EmbeddingResponse,
    Function,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)

from solesonic_litellm_tokens_per_second import TokensPerSecondLogger

USER: UserAPIKeyAuth = UserAPIKeyAuth()


def fake_clock(times: Iterable[float]):
    remaining = iter(times)
    return lambda: next(remaining)


def role_chunk() -> ModelResponseStream:
    return ModelResponseStream(choices=[StreamingChoices(delta=Delta(role="assistant", content=None))])


def content_chunk(text: str) -> ModelResponseStream:
    return ModelResponseStream(choices=[StreamingChoices(delta=Delta(content=text))])


def reasoning_chunk(text: str) -> ModelResponseStream:
    return ModelResponseStream(choices=[StreamingChoices(delta=Delta(reasoning_content=text))])


def tool_call_chunk() -> ModelResponseStream:
    tool_call = ChatCompletionDeltaToolCall(
        id="call_1", type="function", index=0, function=Function(name="lookup", arguments='{"q":"x"}')
    )
    return ModelResponseStream(choices=[StreamingChoices(delta=Delta(tool_calls=[tool_call]))])


def usage_chunk(completion_tokens: int) -> ModelResponseStream:
    return ModelResponseStream(
        choices=[StreamingChoices(delta=Delta())],
        usage=Usage(prompt_tokens=25, completion_tokens=completion_tokens, total_tokens=25 + completion_tokens),
    )


async def as_stream(chunks: Sequence[object]) -> AsyncIterator[object]:
    for chunk in chunks:
        yield chunk


async def run_stream(
    chunks: Sequence[object], times: Iterable[float], request_data: dict | None = None
) -> list[object]:
    logger = TokensPerSecondLogger(clock=fake_clock(times))
    return [
        chunk
        async for chunk in logger.async_post_call_streaming_iterator_hook(
            user_api_key_dict=USER, response=as_stream(chunks), request_data=request_data or {}
        )
    ]


def reported_rate(chunks: Sequence[object]) -> float | None:
    usage = getattr(chunks[-1], "usage", None)
    return None if usage is None else usage.model_dump().get("tokens_per_second")


@pytest.mark.asyncio
async def test_rate_spans_first_to_last_output_chunk_excluding_first_token():
    chunks = [role_chunk(), content_chunk("Once"), content_chunk(" upon"), content_chunk(" a time"), usage_chunk(183)]

    out = await run_stream(chunks, times=[0.5, 2.5, 4.5])

    assert reported_rate(out) == pytest.approx(45.5)
    assert '"tokens_per_second":45.5' in out[-1].model_dump_json()


@pytest.mark.asyncio
async def test_reasoning_output_starts_the_window():
    chunks = [reasoning_chunk("Let me think"), content_chunk("Answer"), usage_chunk(41)]

    out = await run_stream(chunks, times=[1.0, 3.0])

    assert reported_rate(out) == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_tool_call_output_counts_as_generation():
    chunks = [tool_call_chunk(), tool_call_chunk(), usage_chunk(11)]

    out = await run_stream(chunks, times=[0.0, 0.5])

    assert reported_rate(out) == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_every_chunk_passes_through_unchanged_in_order():
    passthrough = b"data: not-a-model-response\n\n"
    chunks = [role_chunk(), passthrough, content_chunk("a"), content_chunk("b"), usage_chunk(3)]

    out = await run_stream(chunks, times=[0.0, 1.0])

    assert out == chunks
    assert reported_rate(out) == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_stream_without_usage_reports_nothing():
    chunks = [content_chunk("a"), content_chunk("b")]

    out = await run_stream(chunks, times=[0.0, 1.0])

    assert all(getattr(chunk, "usage", None) is None for chunk in out)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chunks", "times"),
    [
        pytest.param([content_chunk("a"), content_chunk("b"), usage_chunk(1)], [0.0, 1.0], id="single-token"),
        pytest.param([content_chunk("all at once"), usage_chunk(50)], [0.0], id="single-output-chunk"),
        pytest.param([role_chunk(), usage_chunk(0)], [], id="no-output"),
    ],
)
async def test_no_measurable_window_reports_nothing(chunks, times):
    out = await run_stream(chunks, times=times)

    assert reported_rate(out) is None


@pytest.mark.asyncio
async def test_cache_hit_reports_nothing():
    chunks = [content_chunk("a"), content_chunk("b"), usage_chunk(100)]
    cached = {"litellm_logging_obj": SimpleNamespace(model_call_details={"cache_hit": True})}

    out = await run_stream(chunks, times=[0.0, 0.001], request_data=cached)

    assert reported_rate(out) is None


LLAMA_CPP_TIMINGS = {
    "cache_n": 13,
    "prompt_n": 4,
    "prompt_ms": 34.809,
    "prompt_per_token_ms": 8.70225,
    "prompt_per_second": 114.91280990548422,
    "predicted_n": 117,
    "predicted_ms": 1334.334,
    "predicted_per_token_ms": 11.502879310344827,
    "predicted_per_second": 86.93475546602275,
    "draft_n": 99,
    "draft_n_accepted": 83,
}


def chat_response(completion_tokens: int = 117, **extra_fields: object) -> ModelResponse:
    """Built the way LiteLLM builds it from a provider body, so extra top-level fields survive as they do in the proxy."""
    response = convert_to_model_response_object(
        response_object={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hello"}}],
            "usage": {"prompt_tokens": 25, "completion_tokens": completion_tokens, "total_tokens": 25 + completion_tokens},
            **extra_fields,
        },
        model_response_object=ModelResponse(),
    )
    assert isinstance(response, ModelResponse)
    return response


def ollama_body(**fields: object) -> str:
    return json.dumps({"model": "llama3", "message": {"role": "assistant", "content": "Hello"}, "done": True, **fields})


OLLAMA_GENERATE_EXAMPLE = {
    "model": "gemma4",
    "created_at": "2025-10-17T23:14:07.414671Z",
    "response": "Hello! How can I help you today?",
    "done": True,
    "done_reason": "stop",
    "total_duration": 174560334,
    "load_duration": 101397084,
    "prompt_eval_count": 11,
    "prompt_eval_duration": 13074791,
    "eval_count": 18,
    "eval_duration": 52479709,
}


async def run_success(
    response: ModelResponse | EmbeddingResponse, model_call_details: Mapping[str, object] | None = None
) -> None:
    logging_obj = SimpleNamespace(model_call_details=dict(model_call_details or {}))
    result = await TokensPerSecondLogger().async_post_call_success_hook(
        data={"litellm_logging_obj": logging_obj}, user_api_key_dict=USER, response=response
    )
    assert result is None


def response_usage(response: ModelResponse) -> Usage:
    """LiteLLM attaches `usage` to a ModelResponse with setattr, so it is not a declared attribute."""
    usage = getattr(response, "usage", None)
    assert isinstance(usage, Usage)
    return usage


def non_streaming_rate(response: ModelResponse) -> float | None:
    return response_usage(response).model_dump().get("tokens_per_second")


LLAMA_CPP_NEMOTRON_TIMINGS = {
    "cache_n": 0,
    "prompt_n": 17,
    "prompt_ms": 1487.125,
    "prompt_per_token_ms": 87.4779411764706,
    "prompt_per_second": 11.431453307556527,
    "predicted_n": 120,
    "predicted_ms": 1147.821,
    "predicted_per_token_ms": 9.645554621848738,
    "predicted_per_second": 103.67470189167126,
    "draft_n": 96,
    "draft_n_accepted": 88,
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("completion_tokens", "timings", "expected_rate"),
    [
        pytest.param(117, LLAMA_CPP_TIMINGS, 86.93, id="sample-117-tokens"),
        pytest.param(120, LLAMA_CPP_NEMOTRON_TIMINGS, 103.67, id="nemotron-120-tokens"),
    ],
)
async def test_llama_cpp_rate_excludes_first_token_like_its_own_predicted_per_second(
    completion_tokens, timings, expected_rate
):
    response = chat_response(completion_tokens=completion_tokens, timings=timings)

    await run_success(response)

    assert non_streaming_rate(response) == pytest.approx(expected_rate)
    assert non_streaming_rate(response) == pytest.approx(round(timings["predicted_per_second"], 2))
    assert f'"tokens_per_second":{expected_rate}' in response_usage(response).model_dump_json()


@pytest.mark.asyncio
async def test_vllm_rate_uses_decode_time_not_its_prefill_inclusive_tokens_per_second():
    response = chat_response(
        completion_tokens=128,
        metrics={
            "time_to_first_token_ms": 85.2,
            "generation_time_ms": 1270.0,
            "queue_time_ms": 12.3,
            "mean_itl_ms": 9.1,
            "tokens_per_second": 103.2,
        },
    )

    await run_success(response)

    assert non_streaming_rate(response) == pytest.approx(100.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "original_response",
    [
        pytest.param(
            ollama_body(eval_count=18, eval_duration=52479709, prompt_eval_duration=13074791), id="api-chat"
        ),
        pytest.param(json.dumps(OLLAMA_GENERATE_EXAMPLE), id="api-generate-docs-example"),
    ],
)
async def test_ollama_rate_uses_eval_duration_only_from_the_logged_raw_body(original_response):
    response = chat_response(completion_tokens=18)

    await run_success(response, {"original_response": original_response})

    assert non_streaming_rate(response) == pytest.approx(342.99)


@pytest.mark.asyncio
async def test_no_backend_timing_reports_nothing_even_with_a_proxy_side_duration():
    response = chat_response()

    await run_success(response, {"llm_api_duration_ms": 1500.0, "original_response": '{"id": "chatcmpl-1"}'})

    assert non_streaming_rate(response) is None


@pytest.mark.asyncio
async def test_sglang_response_without_timing_reports_nothing():
    response = chat_response(
        completion_tokens=27, metadata={"weight_version": "default", "weight_versions": [{"version": "default"}]}
    )

    await run_success(response)

    assert non_streaming_rate(response) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "timings",
    [
        pytest.param({**LLAMA_CPP_TIMINGS, "predicted_ms": 0}, id="zero-duration"),
        pytest.param({**LLAMA_CPP_TIMINGS, "predicted_ms": -5.0}, id="negative-duration"),
        pytest.param({**LLAMA_CPP_TIMINGS, "predicted_ms": float("nan")}, id="nan-duration"),
        pytest.param({**LLAMA_CPP_TIMINGS, "predicted_ms": float("inf")}, id="infinite-duration"),
        pytest.param({**LLAMA_CPP_TIMINGS, "predicted_ms": "1334.334"}, id="string-duration"),
        pytest.param({**LLAMA_CPP_TIMINGS, "predicted_ms": True}, id="bool-duration"),
        pytest.param({**LLAMA_CPP_TIMINGS, "predicted_n": 1}, id="single-token"),
        pytest.param({**LLAMA_CPP_TIMINGS, "predicted_n": True}, id="bool-tokens"),
        pytest.param({**LLAMA_CPP_TIMINGS, "predicted_n": 117.0}, id="float-tokens"),
        pytest.param({"predicted_ms": 1334.334}, id="missing-tokens"),
        pytest.param("not a mapping", id="not-a-mapping"),
    ],
)
async def test_invalid_llama_cpp_timings_report_nothing(timings):
    response = chat_response(timings=timings)

    await run_success(response)

    assert non_streaming_rate(response) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("completion_tokens", "metrics"),
    [
        pytest.param(128, None, id="metrics-null-for-n-above-1"),
        pytest.param(128, {"generation_time_ms": None}, id="null-generation-time"),
        pytest.param(128, {"tokens_per_second": 103.2}, id="only-prefill-inclusive-rate"),
        pytest.param(1, {"generation_time_ms": 10.0}, id="single-token"),
    ],
)
async def test_unusable_vllm_metrics_report_nothing(completion_tokens, metrics):
    response = chat_response(completion_tokens=completion_tokens, metrics=metrics)

    await run_success(response)

    assert non_streaming_rate(response) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "original_response",
    [
        pytest.param(ollama_body(eval_count=18), id="missing-eval-duration"),
        pytest.param(ollama_body(eval_duration=52479709), id="missing-eval-count"),
        pytest.param(ollama_body(eval_count=0, eval_duration=52479709), id="zero-tokens"),
        pytest.param(ollama_body(eval_count=18, eval_duration=0), id="zero-duration"),
        pytest.param('{"eval_duration": 52479709, "eval_count": 18', id="truncated-json"),
        pytest.param('["eval_duration"]', id="json-not-an-object"),
        pytest.param({"eval_count": 18, "eval_duration": 52479709}, id="not-a-string"),
    ],
)
async def test_unusable_ollama_bodies_report_nothing(original_response):
    response = chat_response(completion_tokens=18)

    await run_success(response, {"original_response": original_response})

    assert non_streaming_rate(response) is None


@pytest.mark.asyncio
async def test_non_streaming_cache_hit_reports_nothing_despite_cached_timings():
    response = chat_response(timings=LLAMA_CPP_TIMINGS)
    response._hidden_params["cache_hit"] = True

    await run_success(response)

    assert non_streaming_rate(response) is None


@pytest.mark.asyncio
async def test_missing_logging_object_still_reads_response_timings():
    response = chat_response(timings=LLAMA_CPP_TIMINGS)

    await TokensPerSecondLogger().async_post_call_success_hook(data={}, user_api_key_dict=USER, response=response)

    assert non_streaming_rate(response) == pytest.approx(86.93)


class ExplodingLoggingObject:
    @property
    def model_call_details(self) -> dict:
        raise RuntimeError("broken logging object")


@pytest.mark.asyncio
async def test_hook_errors_never_reach_the_request():
    response = chat_response()

    result = await TokensPerSecondLogger().async_post_call_success_hook(
        data={"litellm_logging_obj": ExplodingLoggingObject()}, user_api_key_dict=USER, response=response
    )

    assert result is None
    assert non_streaming_rate(response) is None


@pytest.mark.asyncio
async def test_non_chat_responses_are_untouched():
    response = EmbeddingResponse(usage=Usage(prompt_tokens=5, completion_tokens=0, total_tokens=5))

    await run_success(response, {"original_response": ollama_body(eval_count=18, eval_duration=52479709)})

    assert response.usage is not None
    assert "tokens_per_second" not in response.usage.model_dump()
