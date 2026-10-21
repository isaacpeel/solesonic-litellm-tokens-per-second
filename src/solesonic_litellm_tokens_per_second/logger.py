import json
import logging
import math
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from typing import Final

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import ModelResponse, ModelResponseStream

TOKENS_PER_SECOND_FIELD: Final = "tokens_per_second"

logger: Final = logging.getLogger(__name__)


def _carries_output(chunk: ModelResponseStream) -> bool:
    return any(
        choice.delta.content
        or getattr(choice.delta, "reasoning_content", None)
        or choice.delta.tool_calls
        or choice.delta.function_call
        for choice in chunk.choices
    )


def _is_cache_hit(request_data: dict) -> bool:
    logging_obj: Final = request_data.get("litellm_logging_obj")
    return bool(getattr(logging_obj, "model_call_details", {}).get("cache_hit"))


def tokens_per_second(completion_tokens: int | None, first_output_at: float, last_output_at: float) -> float | None:
    """The first output chunk opens the window, so its token is not part of the measured time."""
    elapsed: Final = last_output_at - first_output_at
    if completion_tokens is None or completion_tokens <= 1 or elapsed <= 0:
        return None
    return round((completion_tokens - 1) / elapsed, 2)


def _token_count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _positive_seconds(value: object, units_per_second: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    seconds: Final = value / units_per_second
    return seconds if math.isfinite(seconds) and seconds > 0 else None


def _rate(tokens: int | None, seconds: float | None) -> float | None:
    if tokens is None or tokens <= 0 or seconds is None:
        return None
    return round(tokens / seconds, 2)


def _vllm_rate(response: ModelResponse) -> float | None:
    """vLLM's `generation_time_ms` runs from the first output token to the last, so the first token opens the window."""
    metrics: Final = getattr(response, "metrics", None)
    usage: Final = getattr(response, "usage", None)
    if not isinstance(metrics, Mapping) or usage is None:
        return None
    completion_tokens: Final = _token_count(getattr(usage, "completion_tokens", None))
    return _rate(
        None if completion_tokens is None else completion_tokens - 1,
        _positive_seconds(metrics.get("generation_time_ms"), 1_000),
    )


def _llama_cpp_rate(response: ModelResponse) -> float | None:
    """llama.cpp's `predicted_ms` covers every generated token after the first."""
    timings: Final = getattr(response, "timings", None)
    if not isinstance(timings, Mapping):
        return None
    predicted_tokens: Final = _token_count(timings.get("predicted_n"))
    return _rate(
        None if predicted_tokens is None else predicted_tokens - 1,
        _positive_seconds(timings.get("predicted_ms"), 1_000),
    )


def _ollama_rate(model_call_details: Mapping[str, object]) -> float | None:
    """Ollama's `eval_duration` covers generation only; model loading and prompt evaluation have their own durations.

    LiteLLM's `ollama_chat` provider drops `eval_duration` from the response, so it is read from the raw body it logged.
    """
    raw_response: Final = model_call_details.get("original_response")
    if not isinstance(raw_response, str) or '"eval_duration"' not in raw_response:
        return None
    try:
        body: Final = json.loads(raw_response)
    except ValueError:
        return None
    if not isinstance(body, Mapping):
        return None
    return _rate(_token_count(body.get("eval_count")), _positive_seconds(body.get("eval_duration"), 1_000_000_000))


def non_streaming_tokens_per_second(
    response: ModelResponse, model_call_details: Mapping[str, object]
) -> float | None:
    """Generation speed as reported by the backend itself, or None when it reported no generation timing."""
    readers: Final[tuple[Callable[[], float | None], ...]] = (
        lambda: _vllm_rate(response),
        lambda: _llama_cpp_rate(response),
        lambda: _ollama_rate(model_call_details),
    )
    return next((rate for rate in (read() for read in readers) if rate is not None), None)


class TokensPerSecondLogger(CustomLogger):
    """Adds `usage.tokens_per_second` (generation speed) to chat completions.

    Streamed responses are timed at the proxy. Non-streaming responses use the backend's own generation timing and get
    no value when the backend reports none.
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        super().__init__()
        self._clock = clock

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncIterator[object],
        request_data: dict,
    ) -> AsyncGenerator[object, None]:
        cache_hit: Final = _is_cache_hit(request_data)
        first_output_at: float | None = None
        last_output_at: float | None = None
        async for chunk in response:
            if isinstance(chunk, ModelResponseStream) and not cache_hit:
                if _carries_output(chunk):
                    last_output_at = self._clock()
                    first_output_at = first_output_at if first_output_at is not None else last_output_at
                usage = getattr(chunk, "usage", None)
                if usage is not None and first_output_at is not None and last_output_at is not None:
                    rate = tokens_per_second(usage.completion_tokens, first_output_at, last_output_at)
                    if rate is not None:
                        setattr(usage, TOKENS_PER_SECOND_FIELD, rate)
            yield chunk

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: object,
    ) -> None:
        """Runs only for non-streaming responses. The proxy re-raises errors from this hook, so none may escape.

        `response` is widened from the base class's `LLMResponseTypes`, which LiteLLM defines as a plain union variable
        rather than a type alias; only `ModelResponse` is acted on.
        """
        try:
            usage: Final = getattr(response, "usage", None)
            if not isinstance(response, ModelResponse) or usage is None:
                return None
            if getattr(response, "_hidden_params", {}).get("cache_hit") or _is_cache_hit(data):
                return None
            model_call_details: Final = getattr(data.get("litellm_logging_obj"), "model_call_details", None)
            rate: Final = non_streaming_tokens_per_second(
                response, model_call_details if isinstance(model_call_details, Mapping) else {}
            )
            if rate is not None:
                setattr(usage, TOKENS_PER_SECOND_FIELD, rate)
        except Exception as error:
            logger.debug("tokens_per_second skipped for a non-streaming response: %s", error)
        return None


tokens_per_second_handler: Final = TokensPerSecondLogger()
