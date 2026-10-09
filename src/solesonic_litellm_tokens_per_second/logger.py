import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Final

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import ModelResponseStream

TOKENS_PER_SECOND_FIELD: Final = "tokens_per_second"


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


class TokensPerSecondLogger(CustomLogger):
    """Adds `usage.tokens_per_second` (generation speed, measured at the proxy) to streamed chat completions."""

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


tokens_per_second_handler: Final = TokensPerSecondLogger()
