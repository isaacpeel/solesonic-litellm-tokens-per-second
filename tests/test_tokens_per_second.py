from collections.abc import AsyncIterator, Iterable, Sequence
from types import SimpleNamespace

import pytest
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import ChatCompletionDeltaToolCall, Delta, Function, ModelResponseStream, StreamingChoices, Usage

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
