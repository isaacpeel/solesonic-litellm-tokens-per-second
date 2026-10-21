# solesonic-litellm-tokens-per-second

A [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy) callback that adds `usage.tokens_per_second` to `/v1/chat/completions` responses, so clients get the model's generation speed directly instead of having to measure or derive it. Streamed responses always get it. Non-streaming responses get it when the backend reports its own generation timing (llama.cpp, vLLM, Ollama; see [Non-streaming responses](#non-streaming-responses))

It is a normal Python package that you install next to the proxy and register through LiteLLM's standard [custom callback](https://docs.litellm.ai/docs/observability/custom_callback) setting, `litellm_settings.callbacks`

## What clients get

```json
"usage": {
  "completion_tokens": 177,
  "prompt_tokens": 25,
  "total_tokens": 202,
  "tokens_per_second": 43.97
}
```

On a stream, the field rides on the final usage chunk, the same chunk that already carries the token counts (and `cost`, if you enabled it). On a non-streaming response it is in the response's `usage`. Either way, any OpenAI-compatible client that reads `usage` can read it, and it always means the same thing: generation speed, without queueing, prompt processing, or time to first token

## Why the proxy has to measure it

Most OpenAI-compatible servers do not report generation timing in their streamed responses by default, and LiteLLM's `x-litellm-response-duration-ms` header is sent when a stream opens, before any tokens are generated. On a 4 second stream it reads about 16 ms. The only place that sees every token leave is the proxy, so the callback times the stream there

## How it is measured

```
tokens_per_second = (completion_tokens - 1) / (time of last output chunk - time of first output chunk)
```

An output chunk is one that carries text, `reasoning_content`, or tool call content, so the role-only opening chunk and the trailing usage chunk do not count. The first token opens the window, which is why it is subtracted. Time to first token (queueing and prompt processing) is deliberately excluded, so the number is pure generation speed. Reasoning tokens are included, since the model generates them at the same speed and vLLM counts them in `completion_tokens`

The field is left out rather than set to a misleading value when the response was a cache hit, when there was one token or less, or when all output arrived in a single chunk

Measured against a live gemma-4-26b on vLLM, the callback reported 43.97 tok/s and a client independently timestamping the same stream's chunks measured 44.03 tok/s

## Non-streaming responses

A non-streaming response arrives all at once, so the proxy cannot see when generation started. Any duration the proxy can measure, including LiteLLM's own `llm_api_duration_ms`, also contains queueing, prompt processing, and the network. So the callback never times non-streaming responses itself. It uses only the timing the backend reports for its own generation, and leaves `tokens_per_second` out when there is none

| Backend | Timing it reads | Rate | What you need |
|---|---|---|---|
| llama.cpp server | `timings.predicted_n`, `timings.predicted_ms` | `(predicted_n - 1) / predicted_ms` | nothing, llama.cpp always sends `timings` |
| vLLM | `metrics.generation_time_ms`, `usage.completion_tokens` | `(completion_tokens - 1) / generation_time_ms` | start vLLM with `--enable-per-request-metrics` ([docs](https://docs.vllm.ai/en/stable/features/per_request_metrics/)) |
| Ollama | `eval_count`, `eval_duration` | `eval_count / eval_duration` | route the model through LiteLLM's `ollama_chat/` provider |

- **llama.cpp:** `predicted_ms` covers every generated token after the first, so the result equals llama.cpp's own `predicted_per_second`
- **vLLM:** `generation_time_ms` runs from the first output token to the last, the same window the streaming measurement uses. vLLM's own `metrics.tokens_per_second` includes prompt processing, so it is not used. vLLM sends no `metrics` when `n` is greater than 1
- **Ollama:** `eval_duration` is the time spent generating tokens. Model loading (`load_duration`) and prompt evaluation (`prompt_eval_duration`) are reported separately, so a cold model load never lowers the rate. LiteLLM's `ollama_chat` provider keeps `eval_count` but drops `eval_duration` from the response, so the callback reads both from the raw Ollama body that LiteLLM logs for the call
  - Models configured as `ollama/<model>` get no value. That provider calls `/api/generate`, which returns the same timing fields, but LiteLLM drops them and does not log the raw body. Switch those models to `ollama_chat/<model>`, which LiteLLM also recommends for chat completions:

    ```yaml
    model_list:
      - model_name: gemma4
        litellm_params:
          model: ollama_chat/gemma4
          api_base: http://localhost:11434
    ```

  - Ollama's OpenAI-compatible `/v1` endpoint (for example `openai/<model>` with an Ollama `api_base`) returns no timing, so it gets no value either

Other backends, such as SGLang, get no value until a reader for their own timing is added. Cached responses get no value either: a cached response still carries the original call's timing, which says nothing about the cache hit

## Install

You need a running LiteLLM proxy. The package does not declare `litellm` as a dependency on purpose: it runs inside your proxy's existing environment, and declaring it would let pip upgrade or downgrade the proxy you already run. It is tested against LiteLLM 1.106.2

### 1. Install the package where the proxy runs

#### Proxy running from a Python environment

Install into the same environment that runs `litellm`:

```bash
pip install "solesonic-litellm-tokens-per-second @ https://github.com/isaacpeel/solesonic-litellm-tokens-per-second/archive/refs/tags/v0.2.0.tar.gz"
```

#### Proxy running in Docker

Current LiteLLM images (the published `ghcr.io/berriai/litellm` images and images built from LiteLLM's own `Dockerfile`) keep the proxy's Python in `/app/.venv` and ship without `pip`, `uv`, or `git`. So `RUN pip install ...` and `git+https://` installs fail on them. Instead, borrow `uv` from its official image for the one install step and install from the release archive, which needs no `git`. [`examples/Dockerfile`](examples/Dockerfile) does exactly that:

```dockerfile
ARG LITELLM_IMAGE=ghcr.io/berriai/litellm:main-stable
ARG TPS_VERSION=v0.2.0

FROM ghcr.io/astral-sh/uv:0.11.7 AS uv

FROM ${LITELLM_IMAGE}
ARG TPS_VERSION
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    uv pip install --python /app/.venv/bin/python --no-cache \
    "solesonic-litellm-tokens-per-second @ https://github.com/isaacpeel/solesonic-litellm-tokens-per-second/archive/refs/tags/${TPS_VERSION}.tar.gz"
```

The `--mount` only exists for that `RUN`, so `uv` is not left in the final image. It needs BuildKit, which is the default builder in current Docker

You can confirm your image has this layout before building:

```bash
docker run --rm --entrypoint ls <your-litellm-image> /app/.venv/bin/python
```

If you use the published image, pin `LITELLM_IMAGE` to the tag you run and build:

```bash
docker build -f examples/Dockerfile --build-arg LITELLM_IMAGE=ghcr.io/berriai/litellm:<tag> -t litellm-with-tps .
```

If you build the proxy from your own LiteLLM checkout, build that image first, then pass its tag as `LITELLM_IMAGE`:

```bash
cd /path/to/litellm
docker build -t litellm-base .
docker build -f /path/to/solesonic-litellm-tokens-per-second/examples/Dockerfile --build-arg LITELLM_IMAGE=litellm-base -t litellm-with-tps .
```

Check that the package landed in the proxy's environment:

```bash
docker run --rm --entrypoint /app/.venv/bin/python litellm-with-tps -c "import solesonic_litellm_tokens_per_second; print('installed')"
```

### 2. Register the callback and turn on stream usage

Add both settings to your proxy `config.yaml` (see [`examples/config.yaml`](examples/config.yaml)):

```yaml
litellm_settings:
  callbacks:
    - solesonic_litellm_tokens_per_second.tokens_per_second_handler

general_settings:
  always_include_stream_usage: true
```

`always_include_stream_usage` matters. LiteLLM already asks the provider for usage on every stream for its own spend tracking, but it strips the usage chunk before the client sees it unless the client sent `stream_options.include_usage`. With this setting on, every streaming client gets the usage chunk, and therefore `tokens_per_second`, without changing anything on its side. If you leave it off, the callback still works for clients that send `"stream_options": {"include_usage": true}`

If your models live in the database and you do not use a config file yet, a config containing only these lines is enough. Pass it with `--config /path/to/config.yaml`

With Docker Compose, point the service at the image you built in step 1, mount the config, and add the flag to the command. Keep any `environment`, `volumes`, and `depends_on` entries you already have:

```yaml
services:
  litellm:
    image: litellm-with-tps
    volumes:
      - ./config.yaml:/app/config.yaml
    command: ["--port", "4000", "--config", "/app/config.yaml"]
```

If your Compose file currently builds the proxy with `build:` from a LiteLLM checkout, you can let Compose build both layers instead of running the two `docker build` commands by hand. Copy `examples/Dockerfile` next to your `docker-compose.yml` as `Dockerfile.tps`, then:

```yaml
services:
  litellm-base:
    build:
      context: .
      dockerfile: Dockerfile
    image: litellm-base
    profiles: ["build-only"]

  litellm:
    build:
      context: .
      dockerfile: Dockerfile.tps
      args:
        LITELLM_IMAGE: litellm-base
    volumes:
      - ./config.yaml:/app/config.yaml
    command: ["--port", "4000", "--config", "/app/config.yaml"]
```

```bash
docker compose build litellm-base
docker compose up -d --build litellm
```

### 3. Restart the proxy and check it

```bash
curl -sN http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "<your-model>", "stream": true, "messages": [{"role": "user", "content": "Write a 150 word story."}]}' \
  | grep -o '"usage":{.*'
```

The last usage line should contain `"tokens_per_second"`. If it has no usage at all, `always_include_stream_usage` is not set. If it has usage but no `tokens_per_second`, the callback did not load, so check the proxy startup log for an import error

For a non-streaming request to a backend from the [table above](#non-streaming-responses):

```bash
curl -s http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "<your-model>", "messages": [{"role": "user", "content": "Write a 150 word story."}]}' \
  | grep -o '"usage":{[^}]*}'
```

If the usage has no `tokens_per_second` but streaming works, the backend reported no generation timing. Check the backend's row in the table

## Reading it from a client

On a stream, the value is on `usage` in the last chunk that carries usage. Clients that keep unknown fields expose it directly. With the OpenAI Python SDK:

```python
for chunk in client.chat.completions.create(model="<your-model>", messages=messages, stream=True):
    if chunk.usage is not None:
        print(chunk.usage.model_extra.get("tokens_per_second"))
```

On a non-streaming response it is in the same place, on the response's `usage`:

```python
response = client.chat.completions.create(model="<your-model>", messages=messages)
print(response.usage.model_extra.get("tokens_per_second"))
```

With the OpenAI Java SDK (and Spring AI built on it), it is in `CompletionUsage._additionalProperties()` under `tokens_per_second`

## Limits

Each HTTP call is measured on its own. If your application makes several model calls for one user turn, such as tool calling rounds, each call carries its own rate, and the last call's value is the speed of the answer the user reads

For streams, the timing is taken at the proxy, so it includes the network hop between the proxy and the model server. When they run on the same host or network, that is negligible. Non-streaming values come from the backend's own clock and do not include it

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

The tests drive the real LiteLLM chunk types through the callback with an injected clock, and build non-streaming responses from provider JSON with LiteLLM's own response conversion, so every expected rate is exact

## License

MIT
