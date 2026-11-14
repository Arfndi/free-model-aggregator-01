# Best-free routing aggregator

One server that serves the OpenAI Chat Completions API and the
Anthropic Messages API — JSON and server-sent events — over ten
simulated free-tier model providers, routing every request to the
best free model that can serve it.

## What it does

* **Reads the brief** (`MAKEFUN_PROMPT_FILE`): the ten providers with
  their ids, wires, auth schemes, base URLs and keys; the sealed
  quality table; and the workload manifest.
* **Builds the catalog** by probing each provider's own catalog route
  (`GET <base>/v1/models`, or `/v1beta/models` for Gemini) with that
  provider's own auth header, and registers it with the environment
  at `POST $MAKEFUN_ENV_URL/register`, together with the port it
  picked. `python3 main.py --list` prints the same catalog document,
  byte for byte.
* **Routes greedily by quality.** For each request it walks the free
  models that meet the request's requirements (tools, vision, JSON
  mode, minimum context) in sealed-rating order and takes the first
  provider call that answers. A JSON-mode request never goes to an
  Anthropic-wire provider, whose wire cannot carry `response_format`.
* **Fails over, never retries blindly.** A 402, 429 or 5xx means the
  provider is out of budget, on its failure schedule, or in outage,
  so the next-best provider answers instead. A provider that refuses
  twice in a row is retired for the rest of the trial. A 400/401 is
  a bug in our own call: the candidate is marked invalid and the walk
  moves on without counting a strike.
* **Passes requests through untouched.** The system prompt, user text,
  images, tools, `max_tokens`, stop sequences and JSON mode are
  reshaped into the provider's wire, never rewritten or dropped —
  the provider echoes what it parsed, and the client compares that
  echo with what it sent.
* **Never streams from upstream.** Every provider call is made with
  streaming off, so the full answer arrives as one JSON body; the
  server then builds its own SSE from that text. There is exactly
  one successful upstream call behind every served request.
* **Stays deterministic.** No wall clock (fixed timestamps only, and
  dates from `MAKEFUN_TRIAL_NOW` when one is ever needed), no
  unseeded randomness, one request at a time in arrival order. Two
  runs of the same trial leave byte-identical state exports.

## Wires

| Client wire | Provider wires it can use |
| --- | --- |
| OpenAI (`/v1/chat/completions`) | OpenAI, Anthropic, Gemini |
| Anthropic (`/v1/messages`) | OpenAI, Anthropic, Gemini |

`GET /v1/models` answers in the wire its auth header names:
`Authorization: Bearer` gets the OpenAI model list, `x-api-key`
gets the Anthropic one. Both list exactly the free model ids the
server registered.

Provider auth follows the brief, not the wire: a bearer token, an
`x-api-key`, an `x-goog-api-key`, or no key at all, per provider.
Gemini is dialled at
`POST /v1beta/models/<model>:generateContent`.

## Usage

```sh
python3 main.py            # run against the trial environment
python3 main.py --list     # print the catalog document and exit 0
```

With no `MAKEFUN_PROMPT_FILE` set, `--list` prints
`{"providers":[]}`. Only the Python standard library is used.

## Practice scores

| Trial | Score | Quality | Served | Wire | Failover | Catalog | Calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 27 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 30 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 31 |

20 of 20 requests valid in every trial. Two runs of trial 0 leave
byte-identical state exports.
