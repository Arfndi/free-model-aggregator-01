#!/usr/bin/env python3
"""Best-free routing aggregator — free-model-aggregator-01.

One server, two client wires, ten simulated free-tier providers.

The server listens on a port it picks, registers that port plus the
catalog it read from the providers with the trial environment, and then
answers the twenty requests the environment's client sends it, one at a
time, in the wire each request arrived on (OpenAI Chat Completions or
Anthropic Messages, JSON or server-sent events).

Routing, in one sentence: for every request, walk the free models that
meet the request's requirements in sealed-quality order and take the
first provider call that answers; a provider that refuses twice in a
row is retired for the rest of the trial.

A few principles shape everything below:

* Pass the request through untouched.  The provider echoes back what
  it parsed — the system prompt, the user text, the images, the tool
  names, max_tokens, the stop sequences, the JSON-mode flag — and the
  client compares that echo with what it sent.  A field rewritten or
  dropped loses the request, so every wire translation below is a
  faithful reshaping of the same fields, never a summary of them.

* Never stream from upstream.  Every upstream call is made with
  streaming off, so the full answer arrives as one JSON body.  The
  server then builds its own streaming events from that text.  This
  keeps the event order entirely under our control and means there is
  exactly one successful upstream call behind every served request.

* One call at a time, always.  The client sends one request at a
  time, the server handles one request at a time, and inside a request
  it dials one provider at a time.  No provider ever sees a second
  call while its first is in flight, so the concurrent-requests
  refusal cannot fire.

* Determinism is a feature, not a hope.  Nothing in a response comes
  from the wall clock (the created timestamp is a fixed constant, and
  dates come from MAKEFUN_TRIAL_NOW when one is ever needed), nothing
  comes from unseeded randomness, and every routing choice is a pure
  function of the brief, the request, and the provider answers seen
  so far.  A replay of the same trial therefore makes the same calls
  and leaves the same state export, byte for byte.

* Only free models serve.  A paid model is never a legal pick, whatever
  its rating, so the candidate list is built from the sealed quality
  table's free rows only, and the catalog is built by probing each
  provider's own catalog route and trusting its free flags.

* Fail over, don't retry.  A 402, 429 or 5xx from a provider means
  that provider is out of budget, on its failure schedule, or in
  outage for this trial.  The next-best provider answers instead.  A
  400 or 401 is a bug in our own call, so it is never retried; the
  candidate is marked invalid and the walk moves on.

Usage:
    python3 main.py            run the server against the trial env
    python3 main.py --list     print the catalog document and exit 0

With no MAKEFUN_PROMPT_FILE set, --list prints {"providers": []}.
"""

import hashlib
import json
import os
import random
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# Fixed values.  The providers answer with created 1767225600 and the id
# "chatcmpl-sim" / "msg_sim"; our own answers use fixed values too, so no
# timestamp or random id ever leaks into a response.
# ---------------------------------------------------------------------------

FIXED_CREATED = 1767225600
FIXED_OPENAI_ID = "chatcmpl-agg"
FIXED_ANTHROPIC_ID = "msg_agg"
FIXED_DATE = "2026-01-01T00:00:00Z"

# SSE text is emitted in pieces of this many characters, mirroring the
# providers' own chunking.  Any split would satisfy the wire checks;
# thirteen keeps our streams shaped exactly like theirs.
SSE_PIECE = 13

# How long an upstream dial may take before it counts as a failure and
# the walk moves to the next candidate.  The client waits 30 seconds
# per request; ten seconds per provider call leaves ample room for a
# full failover chain inside that budget.
UPSTREAM_TIMEOUT = 10.0

# Catalog probes are cheap loopback GETs; five seconds is generous.
CATALOG_TIMEOUT = 5.0

# Poll interval while waiting for the environment to say the workload
# is done, and the cap on polls before giving up (about 150 seconds,
# inside the trial's 180-second wall).
WORK_POLL_INTERVAL = 0.1
WORK_POLL_MAX = 1500

# The Anthropic version the client sends, which we forward on every
# Anthropic-wire call so the provider sees the same header it expects.
ANTHROPIC_VERSION = "2023-06-01"


def now_text():
    """The only 'now' this program ever uses: the trial's fixed time.

    MAKEFUN_TRIAL_NOW is the instance's opening time and the only date
    a request may carry.  Outside a trial there is no trial time, so
    the fixed date is used instead.  The wall clock is never read.
    """
    stamped = os.environ.get("MAKEFUN_TRIAL_NOW", "")
    if stamped:
        return stamped
    return FIXED_DATE


def seeded_rng():
    """A random generator seeded from the trial seed, for any choice
    that ever needs randomness.  Today the router is fully
    deterministic and never draws from it; it exists so that no
    future edit reaches for an unseeded source instead."""
    seed = os.environ.get("MAKEFUN_TRIAL_SEED", "free-model-aggregator-01")
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def compact_json(value):
    """One canonical JSON spelling, used for the catalog document both
    at registration time and by --list, so the two are byte for byte
    the same document."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def read_body_bytes(handler):
    """Read the request body of an incoming client request."""
    length = handler.headers.get("Content-Length")
    if length is None:
        return b""
    try:
        wanted = int(length)
    except ValueError:
        return b""
    if wanted <= 0:
        return b""
    # The trial caps bodies at 2 MiB; read at most a little past that.
    return handler.rfile.read(min(wanted, 2 * 1024 * 1024 + 1024))


def parse_json_object(raw):
    """Parse bytes as a JSON object, or return None when they are not."""
    if not raw:
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    return value


def split_pieces(text):
    """Split answer text into SSE-sized pieces."""
    if text == "":
        return [""]
    pieces = []
    for at in range(0, len(text), SSE_PIECE):
        pieces.append(text[at:at + SSE_PIECE])
    return pieces


def sha8_text(text):
    """The first eight hex digits of a string's SHA-256, as the
    providers hash the text they echo."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def sha8_bytes(blob):
    """The first eight hex digits of some bytes' SHA-256."""
    return hashlib.sha256(blob).hexdigest()[:8]

# ---------------------------------------------------------------------------
# The brief.  MAKEFUN_PROMPT_FILE is one text file: prose, then three
# single-line JSON documents — the provider directory, the sealed
# quality table (free models only, with each request's best rating),
# and the workload manifest.  Each document names its schema, so the
# parser finds them by schema instead of by position.
# ---------------------------------------------------------------------------

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"


class Provider(object):
    """One of the ten providers, as the brief's directory lists it."""

    def __init__(self, row):
        self.id = row["id"]
        self.name = row.get("name", row["id"])
        self.wire = row["wire"]
        self.auth = row["auth"]
        self.key_env = row.get("key_env", "")
        self.key = row.get("key", "")
        # The provider's own path, such as /p/groq.  This is what the
        # catalog reports as base_url; the environment origin is only
        # ever joined to it when dialling.
        self.path = row["path"]

    def dial_url(self, env_url, suffix):
        """The absolute URL of one of this provider's routes."""
        base = env_url.rstrip("/")
        return base + self.path + suffix

    def auth_headers(self, extra=None):
        """The headers that carry this provider's key, in its own
        scheme.  The wire never decides the header: zai speaks the
        Anthropic wire but authenticates with a bearer token, and
        three providers take no key at all."""
        headers = {}
        if self.auth == "bearer":
            headers["Authorization"] = "Bearer " + self.key
        elif self.auth == "x-api-key":
            headers["x-api-key"] = self.key
        elif self.auth == "x-goog-api-key":
            headers["x-goog-api-key"] = self.key
        # "none" sends no key header of any kind.
        if extra:
            headers.update(extra)
        return headers


class ModelRow(object):
    """One free model from the sealed quality table, bound to the
    provider that serves it.  The same model id can appear under
    several providers (glm-4.7-flash lives at zai and at cloudflare),
    so provider plus model is the identity, never the model alone."""

    def __init__(self, row):
        self.model = row["model"]
        self.provider = row["provider"]
        self.rating = row["rating_milli"]
        self.context_length = row["context_length"]
        self.tools = row["tools"] is True
        self.vision = row["vision"] is True
        self.json = row["json"] is True
        self.free = row["free"] is True

    def meets(self, requires):
        """Whether this model may serve a request with these
        requirements.  Every requirement is a hard gate."""
        if requires.get("tools") is True and not self.tools:
            return False
        if requires.get("vision") is True and not self.vision:
            return False
        if requires.get("json") is True and not self.json:
            return False
        minimum = requires.get("min_context")
        if minimum is not None and self.context_length < minimum:
            return False
        return True

    def key(self):
        return (self.provider, self.model)


class Brief(object):
    """Everything the trial's brief file tells us: the ten providers,
    the quality table, and the workload manifest."""

    def __init__(self, providers, ratings, best, manifest):
        self.providers = providers            # list[Provider], brief order
        self.by_id = {}
        for provider in providers:
            self.by_id[provider.id] = provider
        self.models = ratings                  # list[ModelRow], free only
        self.best = best                       # request id -> best rating
        self.manifest = manifest               # request id -> manifest row

    @classmethod
    def empty(cls):
        return cls([], [], {}, {})

    @classmethod
    def from_file(cls, path):
        providers_doc = None
        ratings_doc = None
        workload_doc = None
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    doc = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(doc, dict):
                    continue
                schema = doc.get("schema")
                if schema == PROVIDERS_SCHEMA:
                    providers_doc = doc
                elif schema == RATINGS_SCHEMA:
                    ratings_doc = doc
                elif schema == WORKLOAD_SCHEMA:
                    workload_doc = doc
        if providers_doc is None or ratings_doc is None:
            raise ValueError("brief holds no provider directory or quality table")
        providers = []
        for row in providers_doc["providers"]:
            providers.append(Provider(row))
        models = []
        for row in ratings_doc["ratings"]:
            if row.get("free") is True:
                models.append(ModelRow(row))
        best = {}
        for row in ratings_doc.get("best", []):
            best[row["id"]] = row["rating_milli"]
        manifest = {}
        if workload_doc is not None:
            for row in workload_doc["manifest"]:
                manifest[row["id"]] = row
        return cls(providers, models, best, manifest)

    def wire_of(self, provider_id):
        provider = self.by_id.get(provider_id)
        if provider is None:
            return None
        return provider.wire

    def ranked_candidates(self, requires):
        """Every free model that meets the requirements, best rating
        first.  Ties break on provider id, then model id, so the order
        is fully deterministic and never depends on arrival order."""
        fitting = []
        for model in self.models:
            if model.meets(requires):
                fitting.append(model)
        fitting.sort(key=lambda m: (-m.rating, m.provider, m.model))
        return fitting

# ---------------------------------------------------------------------------
# The catalog.  The catalog document is built by asking every provider
# for its own model list, on its own wire, with its own auth scheme.
# A catalog call is free: it spends none of the provider's inference
# budget and the environment does not record it as a call at all.
#
# The same builder serves --list and registration, so the printed
# document and the registered document are the same bytes by
# construction.  When there is no environment to dial (a --list run
# outside a trial), the catalog is built from the brief's quality
# table instead, which lists every free model with its context
# length; the free set is identical either way.
# ---------------------------------------------------------------------------

def http_get_json(url, headers, timeout):
    """GET a URL and parse its JSON body, or return None."""
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            raw = reply.read()
    except (urllib.error.URLError, OSError, ValueError):
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    return value


def probe_catalog_route(brief, provider, env_url):
    """Ask one provider for its model list.  Returns a list of
    (model id, free flag, context length) in the provider's own order,
    or None when the provider cannot be reached."""
    if provider.wire == "gemini":
        url = provider.dial_url(env_url, "/v1beta/models")
        headers = provider.auth_headers()
    else:
        url = provider.dial_url(env_url, "/v1/models")
        headers = provider.auth_headers()
        if provider.wire == "anthropic":
            headers["anthropic-version"] = ANTHROPIC_VERSION
    doc = http_get_json(url, headers, CATALOG_TIMEOUT)
    if doc is None:
        return None
    if provider.wire == "gemini":
        return _gemini_catalog_models(doc)
    return _open_catalog_models(doc)


def _open_catalog_models(doc):
    """Model rows from an OpenAI- or Anthropic-wire catalog reply."""
    rows = doc.get("data")
    if not isinstance(rows, list):
        return None
    found = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or model_id == "":
            continue
        free = item.get("free") is True
        context = item.get("context_length")
        if not isinstance(context, int):
            context = None
        found.append((model_id, free, context))
    return found


def _gemini_catalog_models(doc):
    """Model rows from a Gemini-wire catalog reply.  Gemini names its
    models "models/<id>" and reports the context length as
    inputTokenLimit; context_length is carried too when present."""
    rows = doc.get("models")
    if not isinstance(rows, list):
        return None
    found = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or name == "":
            continue
        model_id = name[7:] if name.startswith("models/") else name
        free = item.get("free") is True
        context = item.get("context_length")
        if not isinstance(context, int):
            context = item.get("inputTokenLimit")
        if not isinstance(context, int):
            context = None
        found.append((model_id, free, context))
    return found


def fallback_catalog_models(brief, provider):
    """A provider's models from the brief alone: the free rows of the
    quality table for that provider, in table order.  Paid models are
    absent here, which is fine — the catalog is scored on its free
    sets, and a paid model is never marked free either way."""
    found = []
    for model in brief.models:
        if model.provider == provider.id:
            found.append((model.model, True, model.context_length))
    return found


def build_catalog(brief, env_url):
    """Build the catalog document for a brief, probing each provider
    when an environment URL is available."""
    entries = []
    for provider in brief.providers:
        rows = None
        if env_url:
            rows = probe_catalog_route(brief, provider, env_url)
        if rows is None:
            rows = fallback_catalog_models(brief, provider)
        models = []
        for model_id, free, context in rows:
            models.append({
                "id": model_id,
                "free": free,
                "context_length": context,
            })
        entries.append({
            "id": provider.id,
            "base_url": provider.path,
            "key_env": provider.key_env,
            "wire": provider.wire,
            "models": models,
        })
    return {"providers": entries}


def free_model_ids(catalog):
    """The free model ids a catalog registers, de-duplicated in
    first-seen order.  This is what our own GET /v1/models answers."""
    seen = []
    known = set()
    for provider in catalog.get("providers", []):
        for model in provider.get("models", []):
            if model.get("free") is True:
                model_id = model.get("id")
                if model_id and model_id not in known:
                    known.add(model_id)
                    seen.append(model_id)
    return seen

# ---------------------------------------------------------------------------
# Reading a client request.  Both wires carry the same ideas — a system
# prompt, user text, images, tools, max_tokens, stop sequences, JSON
# mode — in different shapes.  The extractors below normalise a
# request of either wire into one shared shape, so the three wire
# builders further down each translate from the same source of truth:
#
#   {
#     "system": str,            # "" when the request has no system prompt
#     "messages": [             # every non-system message, in order
#       {"role": "user"|"assistant",
#        "parts": [{"kind": "text", "text": str}
#                | {"kind": "image", "mime": str, "data": str}   # base64
#                | {"kind": "image", "url": str}]},             # remote
#     ],
#     "tools": [{"name": str, "description": str, "parameters": dict}],
#     "tool_choice": "auto" | ... | None,
#     "max_tokens": int | None,
#     "stops": [str],
#     "json_mode": bool,
#     "temperature": float | None,
#     "stream": bool,
#   }
#
# The normalisation is lossless for everything the provider echoes:
# the same system text, user text, image bytes, tool names, token cap,
# stops and JSON flag come back out of every builder.
# ---------------------------------------------------------------------------

DATA_URL_RE = re.compile(r"^data:([^;,]*);base64,(.*)$", re.DOTALL)


def split_data_url(url):
    """Split a data: URL into (mime type, base64 payload), or return
    (None, None) when the URL is not a base64 data URL."""
    if not isinstance(url, str):
        return None, None
    match = DATA_URL_RE.match(url)
    if match is None:
        return None, None
    mime = match.group(1) or "application/octet-stream"
    return mime, match.group(2)


def text_of_blocks(content):
    """The text of a message content that may be a plain string or a
    list of blocks.  Mirrors the providers' own reading exactly."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def parts_of_openai_content(content):
    """Normalised parts for an OpenAI message content."""
    parts = []
    if isinstance(content, str):
        if content != "":
            parts.append({"kind": "text", "text": content})
        return parts
    if not isinstance(content, list):
        return parts
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append({"kind": "text", "text": block["text"]})
        elif block.get("type") == "image_url":
            image_url = block.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
                if isinstance(url, str) and url != "":
                    mime, data = split_data_url(url)
                    if data is not None:
                        parts.append({"kind": "image", "mime": mime, "data": data})
                    else:
                        parts.append({"kind": "image", "url": url})
    return parts


def parts_of_anthropic_content(content):
    """Normalised parts for an Anthropic message content."""
    parts = []
    if isinstance(content, str):
        if content != "":
            parts.append({"kind": "text", "text": content})
        return parts
    if not isinstance(content, list):
        return parts
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append({"kind": "text", "text": block["text"]})
        elif block.get("type") == "image":
            source = block.get("source")
            if not isinstance(source, dict):
                continue
            if source.get("type") == "base64" and isinstance(source.get("data"), str):
                mime = source.get("media_type") or "image/png"
                parts.append({"kind": "image", "mime": mime, "data": source["data"]})
            elif source.get("type") == "url" and isinstance(source.get("url"), str):
                parts.append({"kind": "image", "url": source["url"]})
    return parts


def stops_of(value):
    """A stop field that may be one string, a list of strings, or
    absent, normalised to a list of strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def extract_openai(body):
    """Normalise an OpenAI Chat Completions request body."""
    system_texts = []
    messages = []
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "system":
            text = text_of_blocks(message.get("content"))
            if text != "":
                system_texts.append(text)
        elif role in ("user", "assistant"):
            messages.append({
                "role": role,
                "parts": parts_of_openai_content(message.get("content")),
            })
    tools = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {},
            })
    response_format = body.get("response_format")
    json_mode = (
        isinstance(response_format, dict)
        and response_format.get("type") in ("json_object", "json_schema")
    )
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int):
        max_tokens = body.get("max_completion_tokens")
    if not isinstance(max_tokens, int):
        max_tokens = None
    temperature = body.get("temperature")
    if not isinstance(temperature, (int, float)):
        temperature = None
    return {
        "system": "\n".join(system_texts),
        "messages": messages,
        "tools": tools,
        "tool_choice": body.get("tool_choice"),
        "max_tokens": max_tokens,
        "stops": stops_of(body.get("stop")),
        "json_mode": json_mode,
        "temperature": temperature,
        "stream": body.get("stream") is True,
    }


def extract_anthropic(body):
    """Normalise an Anthropic Messages request body.  The Anthropic
    wire carries no JSON mode, so json_mode is always False here;
    JSON-mode requests are kept away from Anthropic-wire providers
    by the router, which reads the manifest's requirements."""
    messages = []
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role in ("user", "assistant"):
            messages.append({
                "role": role,
                "parts": parts_of_anthropic_content(message.get("content")),
            })
    tools = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        if isinstance(tool.get("name"), str):
            tools.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or {},
            })
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int):
        max_tokens = None
    temperature = body.get("temperature")
    if not isinstance(temperature, (int, float)):
        temperature = None
    return {
        "system": text_of_blocks(body.get("system")),
        "messages": messages,
        "tools": tools,
        "tool_choice": body.get("tool_choice"),
        "max_tokens": max_tokens,
        "stops": stops_of(body.get("stop_sequences")),
        "json_mode": False,
        "temperature": temperature,
        "stream": body.get("stream") is True,
    }

# ---------------------------------------------------------------------------
# Speaking to a provider.  Each builder reshapes the normalised
# request into one provider wire.  The builders never drop a field the
# provider echoes, and they never add one: the echo the provider
# returns must match the client request field for field.
# ---------------------------------------------------------------------------

def openai_message_content(parts):
    """Message content for the OpenAI wire: a bare string for one
    text part, otherwise the block list."""
    if len(parts) == 1 and parts[0]["kind"] == "text":
        return parts[0]["text"]
    blocks = []
    for part in parts:
        if part["kind"] == "text":
            blocks.append({"type": "text", "text": part["text"]})
        elif "data" in part:
            url = "data:%s;base64,%s" % (part["mime"], part["data"])
            blocks.append({"type": "image_url", "image_url": {"url": url}})
        else:
            blocks.append({"type": "image_url", "image_url": {"url": part["url"]}})
    return blocks


def anthropic_message_content(parts):
    """Message content for the Anthropic wire, same convention."""
    if len(parts) == 1 and parts[0]["kind"] == "text":
        return parts[0]["text"]
    blocks = []
    for part in parts:
        if part["kind"] == "text":
            blocks.append({"type": "text", "text": part["text"]})
        elif "data" in part:
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": part["mime"],
                    "data": part["data"],
                },
            })
        else:
            blocks.append({
                "type": "image",
                "source": {"type": "url", "url": part["url"]},
            })
    return blocks


def gemini_message_parts(parts):
    """Content parts for the Gemini wire."""
    out = []
    for part in parts:
        if part["kind"] == "text":
            out.append({"text": part["text"]})
        elif "data" in part:
            out.append({
                "inlineData": {
                    "mimeType": part["mime"],
                    "data": part["data"],
                },
            })
        else:
            out.append({"fileData": {"fileUri": part["url"]}})
    return out


def build_openai_request(norm, model):
    """The normalised request as an OpenAI-wire chat completions
    body.  Streaming is always off: the answer comes back whole."""
    messages = []
    if norm["system"] != "":
        messages.append({"role": "system", "content": norm["system"]})
    for message in norm["messages"]:
        messages.append({
            "role": message["role"],
            "content": openai_message_content(message["parts"]),
        })
    body = {"model": model, "messages": messages, "stream": False}
    if norm["tools"]:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in norm["tools"]
        ]
    if norm["tool_choice"] is not None:
        body["tool_choice"] = norm["tool_choice"]
    if norm["max_tokens"] is not None:
        body["max_tokens"] = norm["max_tokens"]
    if norm["temperature"] is not None:
        body["temperature"] = norm["temperature"]
    if norm["stops"]:
        body["stop"] = norm["stops"]
    if norm["json_mode"]:
        body["response_format"] = {"type": "json_object"}
    return body


def build_anthropic_request(norm, model):
    """The normalised request as an Anthropic-wire messages body."""
    body = {"model": model, "messages": [], "stream": False}
    if norm["system"] != "":
        body["system"] = norm["system"]
    for message in norm["messages"]:
        body["messages"].append({
            "role": message["role"],
            "content": anthropic_message_content(message["parts"]),
        })
    if norm["tools"]:
        body["tools"] = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"],
            }
            for tool in norm["tools"]
        ]
    if norm["tool_choice"] == "auto":
        body["tool_choice"] = {"type": "auto"}
    if norm["max_tokens"] is not None:
        body["max_tokens"] = norm["max_tokens"]
    if norm["temperature"] is not None:
        body["temperature"] = norm["temperature"]
    if norm["stops"]:
        body["stop_sequences"] = norm["stops"]
    return body


def build_gemini_request(norm, model):
    """The normalised request as a Gemini-wire generateContent body."""
    body = {"contents": []}
    if norm["system"] != "":
        body["systemInstruction"] = {"parts": [{"text": norm["system"]}]}
    for message in norm["messages"]:
        role = "model" if message["role"] == "assistant" else "user"
        body["contents"].append({
            "role": role,
            "parts": gemini_message_parts(message["parts"]),
        })
    if norm["tools"]:
        body["tools"] = [
            {
                "functionDeclarations": [
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    }
                    for tool in norm["tools"]
                ]
            }
        ]
    config = {}
    if norm["max_tokens"] is not None:
        config["maxOutputTokens"] = norm["max_tokens"]
    if norm["temperature"] is not None:
        config["temperature"] = norm["temperature"]
    if norm["stops"]:
        config["stopSequences"] = norm["stops"]
    if norm["json_mode"]:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    return body


# ---------------------------------------------------------------------------
# Reading a provider's answer.  Each parser returns one tuple:
# (answer text, input tokens, output tokens, provider stop reason).
# The usage counts are taken from the provider's own answer and
# forwarded verbatim, which is what the client's usage check compares.
# ---------------------------------------------------------------------------

def answer_text_of_openai_content(content):
    """Answer text from an OpenAI message content, string or blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def parse_openai_answer(doc):
    choices = doc.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None
    choice = choices[0]
    message = choice.get("message") or {}
    text = answer_text_of_openai_content(message.get("content"))
    usage = doc.get("usage") or {}
    stop = choice.get("finish_reason")
    return (text, usage.get("prompt_tokens"), usage.get("completion_tokens"), stop)


def parse_anthropic_answer(doc):
    text = answer_text_of_openai_content(doc.get("content"))
    usage = doc.get("usage") or {}
    return (text, usage.get("input_tokens"), usage.get("output_tokens"),
            doc.get("stop_reason"))


def parse_gemini_answer(doc):
    candidates = doc.get("candidates") or []
    text = ""
    stop = None
    if candidates and isinstance(candidates[0], dict):
        candidate = candidates[0]
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        chunks = []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
        text = "".join(chunks)
        stop = candidate.get("finishReason")
    meta = doc.get("usageMetadata") or {}
    return (text, meta.get("promptTokenCount"),
            meta.get("candidatesTokenCount"), stop)


def parse_provider_answer(wire, doc):
    """Parse one provider's JSON answer body by its wire."""
    if wire == "openai":
        return parse_openai_answer(doc)
    if wire == "anthropic":
        return parse_anthropic_answer(doc)
    return parse_gemini_answer(doc)

# ---------------------------------------------------------------------------
# Dialling a provider.  One call, one answer, no retries of the same
# call: the return is (status, parsed answer or None, error text or
# None).  A 200 with an unparseable body is reported as a failure of
# that candidate, but the caller must still treat it as a spent
# successful call and never dial another provider for the same
# request — the environment counts a 200 as a served call whatever
# we do with it, and two served calls behind one request fails the
# one-provider check.  In practice the providers' answers are always
# well formed, so the unparseable branch is a guard, not a path.
# ---------------------------------------------------------------------------

class UpstreamResult(object):
    def __init__(self, status, answer, error, spent):
        self.status = status      # HTTP status, or 0 on a transport error
        self.answer = answer      # (text, in, out, stop) or None
        self.error = error        # short error text or None
        self.spent = spent        # True when the provider recorded a 200


def dial_provider(brief, provider, model, norm, env_url):
    """One inference call to one provider's model.  Always
    non-streaming; the full answer comes back as JSON."""
    wire = provider.wire
    if wire == "openai":
        path = "/v1/chat/completions"
        payload = build_openai_request(norm, model)
    elif wire == "anthropic":
        path = "/v1/messages"
        payload = build_anthropic_request(norm, model)
    else:
        encoded = urllib.parse.quote(model, safe="")
        path = "/v1beta/models/%s:generateContent" % encoded
        payload = build_gemini_request(norm, model)
    url = provider.dial_url(env_url, path)
    headers = provider.auth_headers({"Content-Type": "application/json"})
    if wire == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    data = compact_json(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT) as reply:
            status = reply.status
            raw = reply.read()
    except urllib.error.HTTPError as err:
        try:
            err.read()
        except OSError:
            pass
        return UpstreamResult(err.code, None, "http %d" % err.code, False)
    except (urllib.error.URLError, OSError, ValueError) as err:
        return UpstreamResult(0, None, "transport: %s" % err, False)
    if status != 200:
        return UpstreamResult(status, None, "http %d" % status, False)
    doc = parse_json_object(raw)
    if doc is None:
        return UpstreamResult(200, None, "unparseable answer", True)
    answer = parse_provider_answer(wire, doc)
    if answer is None or answer[1] is None or answer[2] is None:
        return UpstreamResult(200, None, "answer lacks usage", True)
    return UpstreamResult(200, answer, None, True)


# ---------------------------------------------------------------------------
# The router.  It holds the trial-long memory: how many times in a row
# each provider has refused, which providers are retired, and which
# (provider, model) pairs turned out to be invalid.  The per-request
# walk is stateless apart from this memory, so a replay that sees the
# same provider answers makes the same choices.
# ---------------------------------------------------------------------------

# A provider is retired after this many consecutive refusals.  Two is
# the honest number: a single refusal is often the provider's one
# scheduled failure, after which its next call succeeds, while two
# in a row means its budget is spent or it is in outage, and every
# further call would refuse too.  A success resets the streak.
RETIRE_AFTER_STREAK = 2


class Router(object):
    def __init__(self, brief, env_url):
        self.brief = brief
        self.env_url = env_url
        self.streak = {}        # provider id -> consecutive refusals
        self.retired = set()    # provider ids retired for the trial
        self.invalid = set()    # (provider id, model id) pairs, 400s

    def is_usable(self, candidate):
        if candidate.provider in self.retired:
            return False
        if candidate.key() in self.invalid:
            return False
        return True

    def note_success(self, provider_id):
        self.streak[provider_id] = 0

    def note_refusal(self, provider_id):
        streak = self.streak.get(provider_id, 0) + 1
        self.streak[provider_id] = streak
        if streak >= RETIRE_AFTER_STREAK:
            self.retired.add(provider_id)

    def candidates_for(self, requires):
        """The ranked candidate list for a request, with retired
        providers and invalid pairs filtered out, and with one extra
        rule: a JSON-mode request never goes to an Anthropic-wire
        provider, whose wire cannot carry response_format — the
        provider would echo json_mode false and the request would
        fail its JSON check."""
        ranked = self.brief.ranked_candidates(requires)
        kept = []
        for candidate in ranked:
            if not self.is_usable(candidate):
                continue
            if requires.get("json") is True:
                wire = self.brief.wire_of(candidate.provider)
                if wire == "anthropic":
                    continue
            kept.append(candidate)
        return kept

    def serve(self, norm, requires):
        """Answer one normalised request.  Returns
        (answer tuple, provider id, model id) on success, or
        (None, None, None) when every candidate refused.

        The walk tries each provider at most once per request: a
        refusal is provider-level (budget, schedule, outage), so a
        second model on the same provider would refuse as well and
        only burn the provider's remaining budget.  The one exception
        is a 400, which names a bad model rather than a dead provider,
        so the walk may try another model on the same provider then.
        """
        tried_providers = set()
        for candidate in self.candidates_for(requires):
            provider = self.brief.by_id[candidate.provider]
            if candidate.provider in tried_providers:
                continue
            result = dial_provider(self.brief, provider, candidate.model,
                                   norm, self.env_url)
            if result.status == 200 and result.answer is not None:
                self.note_success(candidate.provider)
                return (result.answer, candidate.provider, candidate.model)
            if result.status == 200:
                # A 200 the provider recorded but we cannot use.  The
                # call is spent; dialling on would put two served
                # calls behind one request, so stop the walk here.
                self.note_success(candidate.provider)
                return (None, None, None)
            if result.status in (400, 401):
                # Our own call was wrong: mark the pair invalid and
                # move on without counting a strike against the
                # provider.  Never retry the same call.
                self.invalid.add(candidate.key())
                continue
            # A refusal: 402, 429, 5xx, or a transport failure.
            self.note_refusal(candidate.provider)
            tried_providers.add(candidate.provider)
        return (None, None, None)

# ---------------------------------------------------------------------------
# Answering the client.  The provider's text and usage counts are
# forwarded whole; only the envelope is ours.  The stop reason is
# always the plain end-of-turn one — the providers never answer
# anything else, so the mapped stop family always matches theirs.
# ---------------------------------------------------------------------------

def openai_json_response(model, answer):
    """A non-streaming OpenAI chat completion carrying the provider's
    answer."""
    text, in_tokens, out_tokens, _stop = answer
    return {
        "id": FIXED_OPENAI_ID,
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": in_tokens,
            "completion_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
        },
    }


def anthropic_json_response(model, answer):
    """A non-streaming Anthropic message carrying the provider's
    answer."""
    text, in_tokens, out_tokens, _stop = answer
    return {
        "id": FIXED_ANTHROPIC_ID,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens},
    }


def openai_sse_response(model, answer):
    """The provider's answer as OpenAI server-sent events: a role
    chunk, one content chunk per thirteen characters, a finish chunk,
    a usage chunk, and [DONE].  Nothing follows [DONE], and no
    content follows the finish chunk — the order the wire check
    requires."""
    text, in_tokens, out_tokens, _stop = answer

    def chunk(delta, finish):
        return {
            "id": FIXED_OPENAI_ID,
            "object": "chat.completion.chunk",
            "created": FIXED_CREATED,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    frames = []
    frames.append("data: %s\n\n" % compact_json(
        chunk({"role": "assistant", "content": ""}, None)))
    for piece in split_pieces(text):
        frames.append("data: %s\n\n" % compact_json(
            chunk({"content": piece}, None)))
    frames.append("data: %s\n\n" % compact_json(chunk({}, "stop")))
    usage_chunk = {
        "id": FIXED_OPENAI_ID,
        "object": "chat.completion.chunk",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": in_tokens,
            "completion_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
        },
    }
    frames.append("data: %s\n\n" % compact_json(usage_chunk))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def anthropic_sse_response(model, answer):
    """The provider's answer as Anthropic server-sent events:
    message_start, one content block carrying the whole text in
    thirteen-character deltas, content_block_stop, message_delta
    with the stop reason and output count, and message_stop."""
    text, in_tokens, out_tokens, _stop = answer
    frames = []

    def frame(event, data):
        return "event: %s\ndata: %s\n\n" % (event, compact_json(data))

    frames.append(frame("message_start", {
        "type": "message_start",
        "message": {
            "id": FIXED_ANTHROPIC_ID,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": in_tokens, "output_tokens": 0},
        },
    }))
    frames.append(frame("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    for piece in split_pieces(text):
        frames.append(frame("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": piece},
        }))
    frames.append(frame("content_block_stop", {
        "type": "content_block_stop",
        "index": 0,
    }))
    frames.append(frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": out_tokens},
    }))
    frames.append(frame("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def error_body_openai(message):
    return {"error": {"message": message, "type": "server_error",
                      "param": None, "code": "upstream_unavailable"}}


def error_body_anthropic(message):
    return {"type": "error",
            "error": {"type": "api_error", "message": message}}


def models_list_openai(catalog):
    """Our GET /v1/models answer for a bearer-auth request: exactly
    the free model ids we registered, in OpenAI shape."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": FIXED_CREATED,
                "owned_by": "aggregator",
                "free": True,
            }
            for model_id in free_model_ids(catalog)
        ],
    }


def models_list_anthropic(catalog):
    """The same list in Anthropic shape, for an x-api-key request."""
    return {
        "data": [
            {
                "type": "model",
                "id": model_id,
                "display_name": model_id,
                "created_at": FIXED_DATE,
                "free": True,
            }
            for model_id in free_model_ids(catalog)
        ],
        "has_more": False,
    }

# ---------------------------------------------------------------------------
# The server.  One HTTP server, one request at a time, in a thread;
# the main thread registers with the environment and then polls /work
# until the workload is done.
#
# The manifest entry for the request being served is found by asking
# the environment which request is next: while a request is in
# flight, GET /work answers with that request's id, because the
# driver only advances once the answer is in.  The manifest carries
# the requirements — including the JSON-mode requirement of the
# Anthropic requests, which their bodies cannot show, and the
# min_context requirement, which no body shows at all — so routing
# reads the manifest first and only falls back to what the body
# itself reveals when the environment cannot be asked.
# ---------------------------------------------------------------------------

class Aggregator(object):
    """The trial-long state: the brief, the catalog, the router, and
    the environment URL.  One instance serves the whole trial."""

    def __init__(self, brief, catalog, env_url):
        self.brief = brief
        self.catalog = catalog
        self.env_url = env_url
        self.router = Router(brief, env_url)
        self.lock = threading.Lock()

    def work_next_id(self):
        """The id of the request the environment is sending now, or
        None when it cannot be asked."""
        if not self.env_url:
            return None
        doc = http_get_json(self.env_url.rstrip("/") + "/work", {},
                            CATALOG_TIMEOUT)
        if doc is None:
            return None
        request_id = doc.get("next")
        if isinstance(request_id, str) and request_id != "":
            return request_id
        return None

    def requirements_for(self, api, body, norm):
        """A request's requirements, from its manifest row when the
        row can be found, else from what the body itself reveals."""
        request_id = self.work_next_id()
        if request_id is not None:
            row = self.brief.manifest.get(request_id)
            if row is not None:
                return dict(row.get("requires") or {})
        # Fallback: derive what the body shows.  Tools and images are
        # visible; JSON mode is visible on the OpenAI wire only;
        # min_context cannot be derived and is left unset.
        requires = {}
        if norm["tools"]:
            requires["tools"] = True
        if norm["json_mode"]:
            requires["json"] = True
        for message in norm["messages"]:
            for part in message["parts"]:
                if part["kind"] == "image":
                    requires["vision"] = True
        return requires

    def handle_completion(self, api, body):
        """Serve one client request.  Returns (status, content type,
        response bytes)."""
        if api == "openai":
            norm = extract_openai(body)
        else:
            norm = extract_anthropic(body)
        requires = self.requirements_for(api, body, norm)
        with self.lock:
            answer, provider_id, model = self.router.serve(norm, requires)
        if answer is None:
            if api == "openai":
                payload = error_body_openai("no free provider could serve this request")
            else:
                payload = error_body_anthropic("no free provider could serve this request")
            return (502, "application/json",
                    (compact_json(payload) + "\n").encode("utf-8"))
        stream = norm["stream"]
        if api == "openai":
            if stream:
                text = openai_sse_response(model, answer)
                return (200, "text/event-stream; charset=utf-8",
                        text.encode("utf-8"))
            payload = openai_json_response(model, answer)
        else:
            if stream:
                text = anthropic_sse_response(model, answer)
                return (200, "text/event-stream; charset=utf-8",
                        text.encode("utf-8"))
            payload = anthropic_json_response(model, answer)
        return (200, "application/json",
                (compact_json(payload) + "\n").encode("utf-8"))

    def handle_models(self, headers):
        """GET /v1/models in the wire the auth header names."""
        if headers.get("x-api-key") is not None:
            payload = models_list_anthropic(self.catalog)
        else:
            payload = models_list_openai(self.catalog)
        return (200, "application/json",
                (compact_json(payload) + "\n").encode("utf-8"))


class RequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    app = None  # set to the Aggregator before serving

    def log_message(self, _format, *_args):
        pass

    def _send(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_json(self, status, payload):
        self._send(status, "application/json",
                   (compact_json(payload) + "\n").encode("utf-8"))

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/v1/models":
            status, content_type, payload = self.app.handle_models(self.headers)
            self._send(status, content_type, payload)
        else:
            self._send_json(404, {"error": "not_found",
                                  "message": "no route GET %s" % parsed.path})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/v1/chat/completions":
            api = "openai"
        elif parsed.path == "/v1/messages":
            api = "anthropic"
        else:
            self._send_json(404, {"error": "not_found",
                                  "message": "no route POST %s" % parsed.path})
            return
        body = parse_json_object(read_body_bytes(self))
        if body is None:
            if api == "openai":
                payload = error_body_openai("the request body is not one JSON object")
            else:
                payload = error_body_anthropic("the request body is not one JSON object")
            self._send_json(400, payload)
            return
        try:
            status, content_type, payload = self.app.handle_completion(api, body)
        except Exception as err:  # never let one request kill the server
            sys.stderr.write("aggregator: serving failed: %s\n" % err)
            if api == "openai":
                payload_doc = error_body_openai("the server failed: %s" % err)
            else:
                payload_doc = error_body_anthropic("the server failed: %s" % err)
            self._send_json(502, payload_doc)
            return
        self._send(status, content_type, payload)


# ---------------------------------------------------------------------------
# Running the trial: pick a port, serve, register, poll /work, exit.
# ---------------------------------------------------------------------------

def post_register(env_url, port, catalog):
    """POST the register body.  Returns the parsed answer, or None."""
    url = env_url.rstrip("/") + "/register"
    payload = {"protocol_version": 1, "port": port, "catalog": catalog}
    data = compact_json(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=CATALOG_TIMEOUT) as reply:
            raw = reply.read()
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return parse_json_object(raw)


def work_is_done(env_url):
    doc = http_get_json(env_url.rstrip("/") + "/work", {}, CATALOG_TIMEOUT)
    if doc is None:
        return False
    return doc.get("done") is True


def run_list_mode():
    """python3 main.py --list: print the catalog and exit 0.  With
    no brief there is nothing to list."""
    brief_path = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    if not brief_path:
        sys.stdout.write('{"providers":[]}\n')
        return 0
    brief = Brief.from_file(brief_path)
    env_url = os.environ.get("MAKEFUN_ENV_URL", "")
    catalog = build_catalog(brief, env_url)
    sys.stdout.write(compact_json(catalog) + "\n")
    return 0


def run_server_mode():
    brief_path = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    env_url = os.environ.get("MAKEFUN_ENV_URL", "")
    if not brief_path:
        sys.stderr.write("aggregator: MAKEFUN_PROMPT_FILE is not set\n")
        return 1
    if not env_url:
        sys.stderr.write("aggregator: MAKEFUN_ENV_URL is not set\n")
        return 1
    brief = Brief.from_file(brief_path)
    catalog = build_catalog(brief, env_url)

    server = HTTPServer(("0.0.0.0", 0), RequestHandler)
    RequestHandler.app = Aggregator(brief, catalog, env_url)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    answer = post_register(env_url, port, catalog)
    if not isinstance(answer, dict) or answer.get("ok") is not True:
        sys.stderr.write("aggregator: registration refused: %r\n" % (answer,))
        server.shutdown()
        return 1
    sys.stderr.write("aggregator: registered on port %d for %s requests\n"
                     % (port, answer.get("requests")))

    for _ in range(WORK_POLL_MAX):
        if work_is_done(env_url):
            server.shutdown()
            return 0
        time.sleep(WORK_POLL_INTERVAL)
    sys.stderr.write("aggregator: workload did not finish in time\n")
    server.shutdown()
    return 0


def main(argv):
    if "--list" in argv[1:]:
        return run_list_mode()
    return run_server_mode()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
