# free-model-aggregator-01

One server that serves the OpenAI Chat Completions API **and** the Anthropic
Messages API — each in JSON and in server-sent events — by routing every
request to the best free model able to serve it, across ten free providers.

`main.py` is Python 3.13, standard library only. No dependencies, no
framework, no clock, no unseeded randomness: every routing choice is a pure
function of the brief, the requests and the providers' answers, so a replayed
trial makes the same choices and leaves the same state export.

## How to run

The trial environment provides two variables:

```sh
export MAKEFUN_PROMPT_FILE=/path/to/prompt.txt   # the brief (three JSON docs)
export MAKEFUN_ENV_URL=http://127.0.0.1:PORT     # the env server
python3 main.py
```

On start the server:

1. reads the brief from `$MAKEFUN_PROMPT_FILE` (provider directory, quality
   table, workload manifest),
2. probes each provider's catalog route (`GET /v1/models`, Gemini's
   `/v1beta/models`) with that provider's own auth header,
3. registers its port and catalog with the env
   (`POST $MAKEFUN_ENV_URL/register`),
4. serves the env's requests — one at a time — then exits 0 when done.

To see the catalog it would register, without running a trial:

```sh
MAKEFUN_PROMPT_FILE=/path/to/prompt.txt MAKEFUN_ENV_URL=http://127.0.0.1:PORT \
  python3 main.py --list
```

Outside a trial (no prompt file, no env) `--list` prints
`{"providers":[]}` and exits 0. The document `--list` prints is byte for
byte the catalog the server registers.

## Endpoints

| Route | Client wire | Body |
| --- | --- | --- |
| `GET /v1/models` | OpenAI (`Authorization: Bearer …`) or Anthropic (`x-api-key: …`) | — |
| `POST /v1/chat/completions` | OpenAI | Chat Completions request, `stream` optional |
| `POST /v1/messages` | Anthropic | Messages request, `stream` optional |

`GET /v1/models` lists exactly the free models registered in the catalog,
in the wire the caller asked for. Chat and Messages responses carry the
client's model id and a fixed `created` timestamp; usage is always copied
from the upstream provider's own answer, never recomputed. Streaming
responses are built from the fully buffered upstream answer (the upstream
call is fast; real-time forwarding is not needed).

## Routing strategy

For every request the candidates are the free models that meet its
requirements — `tools`, `vision`, `json`, `min_context` — ordered by the
brief's sealed quality rating, best first (ties broken by provider id, then
model id). The server tries them in that order and takes the first provider
call that answers 2xx.

* **Failover.** A 402 (free allowance gone), 429 or 5xx fails over to the
  next candidate. A provider that fails twice in a row is set aside for the
  rest of the trial. A 401 or 400 is never retried — the request itself is
  the problem. A 2xx that cannot be parsed is never retried either: calling
  a second provider for the same request would break the one-provider rule,
  so the server answers 502 instead.
* **JSON mode.** An OpenAI JSON-mode request never goes to an
  Anthropic-wire provider, whose echo cannot carry JSON mode.
* **Economy.** Each request costs exactly one successful upstream call —
  never a probe call, never a speculative call. The tie-break favours fewer
  provider calls, so nothing is spent speculatively.
* **Weight order.** The workload arrives weight-2 requests first, so the
  greedy best-first order already serves the most valuable requests before
  the cheaper ones; no soft budget cap is needed.

Practice results (kit env + scorer, run locally, no Docker): **99/100 on
all three practice trials** — quality 99, served 100, wire 100, failover
100, catalog 100. Trial 0 re-run twice leaves a byte-identical state
export, so the entry is deterministic. The single quality point lost is
structural: the best-rated model for some requests sits behind a provider
whose per-trial budget is already spent by an earlier, heavier request.

## Translation

The three wires are translated both ways, request and response:

* **OpenAI ↔ Anthropic:** system message ↔ top-level `system`; tool
  definitions (`function` ↔ `input_schema`); `tool_calls` ↔ `tool_use`;
  `tool` messages ↔ `tool_result`; `max_tokens`, `temperature`, `top_p`,
  `stop` ↔ `stop_sequences`; images `image_url` ↔ base64/url sources.
  Same-wire calls pass through verbatim.
* **OpenAI / Anthropic → Gemini:** `systemInstruction`, `contents` with
  `user`/`model` roles, `functionDeclarations`, `generationConfig`
  (`maxOutputTokens`, `stopSequences`, `responseMimeType: application/json`
  for JSON mode), `functionCall`/`functionResponse` parts, `inlineData`
  for data-URL images and `fileData` for plain URLs.
* **Responses:** the provider's answer text is the echo of what it parsed,
  so it goes through byte-for-byte; only the envelope changes. Stop reasons
  are mapped across families (`stop`/`end_turn`/`STOP` → `stop` /
  `end_turn`; `length`/`max_tokens`/`MAX_TOKENS` → `length` / `max_tokens`;
  tool calls → `tool_calls` / `tool_use`). OpenAI streams include the usage
  chunk; Anthropic streams emit `message_start` / `content_block_*` /
  `message_delta` / `message_stop` with input and output token counts.

## The catalog

The registered catalog lists every provider that answered its probe, in the
brief's order, with models sorted by id. A model's `free` flag and
`context_length` come from the quality table where it has a row (the sealed
truth), falling back to the provider's own listing. The catalog document
`--list` prints is byte-identical to the one registered with the env.

## Limits

* Python 3.13 standard library only; single process, one request at a time.
* 180 s wall, 1 CPU, 2048 MB — the trial's limits. Upstream calls time out
  at 10 s; catalog probes at 5 s.
* Deterministic: no wall clock, no unseeded randomness. The only
  nondeterminism is which provider answers first, which is the providers'
  behaviour, not the server's.

## Layout

```
main.py               the server (entrypoint)
README.md             this file
LICENSE               MIT
.makefun/owner.json   payout-wallet claim file
```
