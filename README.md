# solesonic-litellm-tokens-per-second

A [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy) callback that adds `usage.tokens_per_second` to every streamed `/v1/chat/completions` response, so clients get the model's generation speed directly instead of having to measure or derive it

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

The field rides on the final usage chunk of the stream, the same chunk that already carries the token counts (and `cost`, if you enabled it), so any OpenAI-compatible client that reads `usage` can read it

## Why the proxy has to measure it

OpenAI-compatible servers such as vLLM do not report generation timing in their responses, and LiteLLM's `x-litellm-response-duration-ms` header is sent when a stream opens, before any tokens are generated. On a 4 second stream it reads about 16 ms. The only place that sees every token leave is the proxy, so the callback times the stream there

## How it is measured

```
tokens_per_second = (completion_tokens - 1) / (time of last output chunk - time of first output chunk)
```

An output chunk is one that carries text, `reasoning_content`, or tool call content, so the role-only opening chunk and the trailing usage chunk do not count. The first token opens the window, which is why it is subtracted. Time to first token (queueing and prompt processing) is deliberately excluded, so the number is pure generation speed. Reasoning tokens are included, since the model generates them at the same speed and vLLM counts them in `completion_tokens`

The field is left out rather than set to a misleading value when the response was a cache hit, when there was one token or less, or when all output arrived in a single chunk. Non-streaming responses are left untouched: the provider returns them all at once, so prompt processing and generation cannot be separated and any rate would mean something different

Measured against a live gemma-4-26b on vLLM, the callback reported 43.97 tok/s and a client independently timestamping the same stream's chunks measured 44.03 tok/s

## Install

You need a running LiteLLM proxy. The package does not declare `litellm` as a dependency on purpose: it runs inside your proxy's existing environment, and declaring it would let pip upgrade or downgrade the proxy you already run. It is tested against LiteLLM 1.104.2

### 1. Install the package where the proxy runs

If you run the proxy from a Python environment, install into that same environment:

```bash
pip install "git+https://github.com/isaacpeel/solesonic-litellm-tokens-per-second.git@v0.1.0"
```

If you run the official Docker image, extend it. [`examples/Dockerfile`](examples/Dockerfile) does this:

```dockerfile
ARG LITELLM_IMAGE=ghcr.io/berriai/litellm:main-stable
FROM ${LITELLM_IMAGE}
RUN pip install --no-cache-dir "git+https://github.com/isaacpeel/solesonic-litellm-tokens-per-second.git@v0.1.0"
```

Pin `LITELLM_IMAGE` to the tag you run in production, then build it:

```bash
docker build -f examples/Dockerfile -t litellm-with-tps .
```

If you build the proxy from your own LiteLLM checkout, add the same `RUN pip install ...` line to that Dockerfile

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

If your models live in the database and you do not use a config file yet, a config containing only these lines is enough. Pass it with `--config /path/to/config.yaml`. With Docker Compose, mount it and add the flag to the command:

```yaml
services:
  litellm:
    image: litellm-with-tps
    volumes:
      - ./config.yaml:/app/config.yaml
    command: ["--port", "4000", "--config", "/app/config.yaml"]
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

## Reading it from a client

The value is on `usage` in the last chunk that carries usage. Clients that keep unknown fields expose it directly. With the OpenAI Python SDK:

```python
for chunk in client.chat.completions.create(model="<your-model>", messages=messages, stream=True):
    if chunk.usage is not None:
        print(chunk.usage.model_extra.get("tokens_per_second"))
```

With the OpenAI Java SDK (and Spring AI built on it), it is in `CompletionUsage._additionalProperties()` under `tokens_per_second`

## Limits

Each HTTP call is measured on its own. If your application makes several model calls for one user turn, such as tool calling rounds, each call carries its own rate, and the last call's value is the speed of the answer the user reads

The timing is taken at the proxy, so it includes the network hop between the proxy and the model server. When they run on the same host or network, that is negligible

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

The tests drive the real LiteLLM chunk types through the callback with an injected clock, so every expected rate is exact

## License

MIT
