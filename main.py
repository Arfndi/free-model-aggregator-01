#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two client wires, ten free providers.

The server reads its brief from MAKEFUN_PROMPT_FILE, probes every provider's
catalog route, registers its port and catalog with the trial env, then serves
the env client's 20 requests: POST /v1/chat/completions (OpenAI wire) and
POST /v1/messages (Anthropic wire), each in JSON or server-sent events.

Routing: for each request the candidates are the free models that meet its
requirements (tools, vision, JSON mode, minimum context), ordered by the
brief's sealed quality rating, best first, ties by provider id then model id.
The first provider call that answers 2xx wins. A 402 (free allowance gone),
429 or 5xx fails over to the next candidate; a provider that refuses twice in
a row is set aside for the rest of the trial. A 401/400 is never retried --
the request itself is the problem. A JSON-mode OpenAI request never goes to
an Anthropic-wire provider, whose echo cannot carry JSON mode. Every choice
is a pure function of the brief, the requests and the providers' answers: no
wall clock, no unseeded randomness, so a replayed trial makes the same
choices.

`python3 main.py --list` prints the catalog document and exits 0.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
POLL_INTERVAL = 0.1
PROBE_TIMEOUT = 5.0
UPSTREAM_TIMEOUT = 10.0
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(value):
    return isinstance(value, dict)


def as_str(value):
    return value if isinstance(value, str) else ""


def as_number(value):
    if isinstance(value, (int, float)) and value == value:
        return value
    return None


def as_bool(value):
    return value if isinstance(value, bool) else None


# ---------------------------------------------------------------------------
# Brief parsing: the brief carries three JSON documents, found by `schema`.
# ---------------------------------------------------------------------------

def _object_end(text, start):
    """Index of the closing brace matching the `{` at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    """Every top-level JSON object in the text that carries a `schema`."""
    docs = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                docs.append(value)
        start = text.find("{", start + 1)
    return docs


class Provider:
    def __init__(self, row):
        self.id = as_str(row.get("id"))
        self.name = as_str(row.get("name")) or self.id
        self.wire = row.get("wire")
        self.auth = row.get("auth")
        base = as_str(row.get("path")) or as_str(row.get("base_path")) \
            or as_str(row.get("base_url")) or as_str(row.get("base"))
        self.base = base
        self.key_env = as_str(row.get("key_env"))
        self.key = as_str(row.get("key"))


class Brief:
    """The provider directory, the quality table and the workload manifest."""

    def __init__(self, path):
        self.providers = []
        self.ratings = {}      # (provider_id, model_id) -> row dict
        self.best = {}         # request id -> best rating_milli
        self.manifest = []     # workload rows, in arrival order
        self.error = None
        if not path:
            self.error = "MAKEFUN_PROMPT_FILE is not set"
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            self.error = "cannot read %s: %s" % (path, exc)
            return
        docs = _documents_in(text)
        directory = next((d for d in docs
                          if d.get("schema") == PROVIDERS_SCHEMA
                          and isinstance(d.get("providers"), list)), None)
        if directory is None:
            self.error = "the brief holds no %s document" % PROVIDERS_SCHEMA
            return
        for row in directory["providers"]:
            if not is_record(row):
                continue
            provider = Provider(row)
            if provider.id and provider.wire in ("openai", "anthropic", "gemini"):
                self.providers.append(provider)
        ratings_doc = next((d for d in docs
                            if d.get("schema") == RATINGS_SCHEMA
                            and isinstance(d.get("ratings"), list)), None)
        if ratings_doc is not None:
            for row in ratings_doc["ratings"]:
                if not is_record(row):
                    continue
                model = as_str(row.get("model"))
                provider = as_str(row.get("provider"))
                if model and provider:
                    self.ratings[(provider, model)] = row
            for row in ratings_doc.get("best") or []:
                if is_record(row) and as_str(row.get("id")):
                    rating = as_number(row.get("rating_milli"))
                    if rating is not None:
                        self.best[as_str(row.get("id"))] = rating
        workload_doc = next((d for d in docs
                             if d.get("schema") == WORKLOAD_SCHEMA
                             and isinstance(d.get("manifest"), list)), None)
        if workload_doc is not None:
            for row in workload_doc["manifest"]:
                if not is_record(row) or not as_str(row.get("id")):
                    continue
                requires = row.get("requires") if is_record(row.get("requires")) else {}
                min_context = requires.get("min_context")
                self.manifest.append({
                    "id": as_str(row.get("id")),
                    "api": "anthropic" if row.get("api") == "anthropic" else "openai",
                    "stream": row.get("stream") is True,
                    "weight": 2 if row.get("weight") == 2 else 1,
                    "requires": {
                        "tools": requires.get("tools") is True,
                        "vision": requires.get("vision") is True,
                        "json": requires.get("json") is True,
                        "min_context": min_context
                        if isinstance(min_context, (int, float)) else None,
                    },
                })


def rating_row(brief, provider_id, model_id):
    return brief.ratings.get((provider_id, model_id))


def model_is_free(brief, provider_id, model_id, listed_free):
    row = rating_row(brief, provider_id, model_id)
    if row is not None:
        return row.get("free") is True
    return listed_free is True


def model_caps(brief, provider_id, model_id, listed):
    """Capabilities of a model: the quality table first, the listing second."""
    row = rating_row(brief, provider_id, model_id)
    if row is not None:
        return {
            "context_length": as_number(row.get("context_length")),
            "tools": row.get("tools") is True,
            "vision": row.get("vision") is True,
            "json": row.get("json") is True,
            "rating": as_number(row.get("rating_milli")) or 0,
        }
    return {
        "context_length": listed.get("context_length"),
        "tools": listed.get("tools") is True,
        "vision": listed.get("vision") is True,
        "json": listed.get("json") is True,
        "rating": 0,
    }


def meets(requires, caps):
    if requires.get("tools") and caps["tools"] is not True:
        return False
    if requires.get("vision") and caps["vision"] is not True:
        return False
    if requires.get("json") and caps["json"] is not True:
        return False
    min_context = requires.get("min_context")
    if min_context is not None:
        if caps["context_length"] is None or caps["context_length"] < min_context:
            return False
    return True

# ---------------------------------------------------------------------------
# Provider catalog probe and the catalog document.
# ---------------------------------------------------------------------------

def join_base(env_url, base):
    base = (base or "").strip()
    if base.startswith("http://") or base.startswith("https://"):
        return base.rstrip("/")
    if not base:
        return env_url
    return env_url + ("" if base.startswith("/") else "/") + base.rstrip("/")


def auth_headers(provider):
    """The auth header this provider's own scheme asks for (never the wire's)."""
    if not provider.key:
        return {}
    if provider.auth == "bearer":
        return {"authorization": "Bearer %s" % provider.key}
    if provider.auth == "x-api-key":
        return {"x-api-key": provider.key}
    if provider.auth == "x-goog-api-key":
        return {"x-goog-api-key": provider.key}
    return {}


def _http_get(url, headers, timeout):
    request = urllib.request.Request(url, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, exc.read()
        except Exception:
            return exc.code, b""
    except Exception:
        return None, b""


def _listed_model(wire, row):
    """One model as a catalog route listed it, in that wire's spelling."""
    if not is_record(row):
        return None
    if wire == "gemini":
        model_id = as_str(row.get("name")).replace("models/", "", 1)
    else:
        model_id = as_str(row.get("id"))
    if not model_id:
        return None
    context = as_number(row.get("context_length"))
    if context is None:
        context = as_number(row.get("context_window"))
    if context is None:
        context = as_number(row.get("inputTokenLimit"))
    return {
        "id": model_id,
        "display_name": as_str(row.get("display_name"))
        or as_str(row.get("displayName")) or model_id,
        "free": as_bool(row.get("free")),
        "context_length": context,
        "tools": as_bool(row.get("tools")),
        "vision": as_bool(row.get("vision")),
        "json": as_bool(row.get("json")),
    }


def probe_provider(brief, provider):
    """GET the provider's catalog route in its own wire; None on any failure."""
    if not ENV_URL:
        # No env to probe (e.g. `--list` outside a trial): nothing to list.
        return None
    base = join_base(ENV_URL, provider.base)
    if provider.wire == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider))
    if provider.wire == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    status, body = _http_get(url, headers, PROBE_TIMEOUT)
    if status is None or status < 200 or status >= 300:
        log("probe %s: GET %s answered %s" % (provider.id, url, status))
        return None
    try:
        parsed = json.loads(body.decode("utf-8"))
    except ValueError:
        log("probe %s: catalog is not JSON" % provider.id)
        return None
    if not is_record(parsed):
        return None
    listing = parsed.get("models") if provider.wire == "gemini" else parsed.get("data")
    if not isinstance(listing, list):
        return None
    models = []
    for row in listing:
        model = _listed_model(provider.wire, row)
        if model is not None:
            models.append(model)
    return models


def build_catalog(brief, probes):
    """The one catalog: every provider that answered, every model it listed.

    A model's free flag comes from the quality table where it has a row (the
    sealed truth), else from the listing; its context length the same way.
    Providers are in the brief's order, models sorted by id.
    """
    providers = []
    for provider, models in probes:
        if models is None:
            continue
        seen = set()
        catalog_models = []
        for model in sorted(models, key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_row(brief, provider.id, model["id"])
            context = None
            if row is not None:
                context = as_number(row.get("context_length"))
            if context is None:
                context = model["context_length"]
            catalog_models.append({
                "id": model["id"],
                "free": model_is_free(brief, provider.id, model["id"], model["free"]),
                "context_length": context,
            })
        providers.append({
            "id": provider.id,
            "base_url": provider.base,
            "key_env": provider.key_env,
            "wire": provider.wire,
            "models": catalog_models,
        })
    return {"providers": providers}


def served_models(brief, probes):
    """The free models this server lists, in brief order, models by id."""
    out = []
    for provider, models in probes:
        if models is None:
            continue
        for model in sorted(models, key=lambda m: m["id"]):
            if not model_is_free(brief, provider.id, model["id"], model["free"]):
                continue
            out.append({
                "id": model["id"],
                "provider": provider.id,
                "display_name": model["display_name"],
            })
    return out


def openai_model_list(models):
    return {
        "object": "list",
        "data": [
            {"id": m["id"], "object": "model", "created": FIXED_CREATED,
             "owned_by": m["provider"]}
            for m in models
        ],
    }


def anthropic_model_list(models):
    return {
        "data": [
            {"type": "model", "id": m["id"], "display_name": m["display_name"],
             "created_at": FIXED_ANTHROPIC_CREATED_AT}
            for m in models
        ],
        "has_more": False,
    }


# ---------------------------------------------------------------------------
# Upstream calls.
# ---------------------------------------------------------------------------

def upstream_call(method, url, headers, body_bytes):
    """One HTTP call; (status, body) or (None, b'') on a transport failure."""
    request = urllib.request.Request(url, data=body_bytes, method=method,
                                     headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, exc.read()
        except Exception:
            return exc.code, b""
    except Exception:
        return None, b""


def upstream_json(method, url, headers, payload):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = dict(headers)
    headers["content-type"] = "application/json"
    return upstream_call(method, url, headers, body)

# ---------------------------------------------------------------------------
# Request translation between the three wires.
#
# The provider echoes what it parsed (system text, user text, tool names,
# max_tokens, stop sequences, image hash, JSON mode), and the client compares
# that echo with what it sent -- so every translation must carry each field
# across untouched. Fields the echo never carries (temperature, tool_choice)
# pass through where the wire supports them and are dropped where it does
# not.
# ---------------------------------------------------------------------------

def text_of(content):
    """The text of a string, or of a block array's text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block["text"] for block in content
        if is_record(block) and block.get("type") == "text"
        and isinstance(block.get("text"), str))


def _image_part(block):
    """An OpenAI image_url part as an Anthropic image block, or None."""
    if not is_record(block) or block.get("type") != "image_url":
        return None
    url = block.get("image_url")
    if not is_record(url) or not isinstance(url.get("url"), str):
        return None
    raw = url["url"]
    if raw.startswith("data:"):
        head, _, data = raw.partition(",")
        media = head[5:].split(";")[0] or "image/png"
        return {"type": "image", "source": {
            "type": "base64", "media_type": media, "data": data}}
    return {"type": "image", "source": {"type": "url", "url": raw}}


def _anthropic_image_to_openai(block):
    if not is_record(block) or block.get("type") != "image":
        return None
    source = block.get("source")
    if not is_record(source):
        return None
    if source.get("type") == "base64" and isinstance(source.get("data"), str):
        media = as_str(source.get("media_type")) or "image/png"
        return {"type": "image_url", "image_url": {
            "url": "data:%s;base64,%s" % (media, source["data"])}}
    if source.get("type") == "url" and isinstance(source.get("url"), str):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text else json.dumps(content)
    if content is None:
        return ""
    return json.dumps(content)


def _stop_list(body):
    stop = body.get("stop")
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        return [s for s in stop if isinstance(s, str)]
    return []


def _openai_tools(body):
    tools = []
    for raw in body.get("tools") or []:
        if not is_record(raw) or not is_record(raw.get("function")):
            continue
        fn = raw["function"]
        if not as_str(fn.get("name")):
            continue
        tools.append({
            "name": fn["name"],
            "description": as_str(fn.get("description")),
            "input_schema": fn.get("parameters")
            if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
        })
    return tools


def _anthropic_tools(body):
    tools = []
    for raw in body.get("tools") or []:
        if not is_record(raw) or not as_str(raw.get("name")):
            continue
        tools.append({
            "type": "function",
            "function": {
                "name": raw["name"],
                "description": as_str(raw.get("description")),
                "parameters": raw.get("input_schema")
                if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
            },
        })
    return tools


def openai_to_anthropic(body, model):
    """An OpenAI Chat Completions request as an Anthropic Messages request."""
    system_text = ""
    messages = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        if role == "system":
            text = text_of(raw.get("content"))
            if text:
                system_text = (system_text + "\n" + text) if system_text else text
            continue
        if role == "user":
            content = raw.get("content")
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                blocks = []
                for block in content:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        blocks.append({"type": "text", "text": block["text"]})
                    else:
                        image = _image_part(block)
                        if image is not None:
                            blocks.append(image)
                messages.append({"role": "user", "content": blocks})
            continue
        if role == "assistant":
            content = raw.get("content")
            blocks = []
            text = text_of(content)
            if text:
                blocks.append({"type": "text", "text": text})
            for call in raw.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                args = fn.get("arguments")
                parsed_args = {}
                if isinstance(args, str) and args.strip():
                    try:
                        parsed_args = json.loads(args)
                    except ValueError:
                        parsed_args = {}
                if not is_record(parsed_args):
                    parsed_args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": as_str(call.get("id")),
                    "name": as_str(fn.get("name")),
                    "input": parsed_args,
                })
            messages.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": as_str(raw.get("tool_call_id")),
                "content": _tool_result_text(raw.get("content")),
            }]})
    out = {"model": model, "messages": messages}
    if system_text:
        out["system"] = system_text
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stops = _stop_list(body)
    if stops:
        out["stop_sequences"] = stops
    tools = _openai_tools(body)
    if tools:
        out["tools"] = tools
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif is_record(choice) and choice.get("type") == "function" \
            and is_record(choice.get("function")):
        out["tool_choice"] = {"type": "tool",
                              "name": as_str(choice["function"].get("name"))}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as an OpenAI Chat Completions request."""
    messages = []
    system = text_of(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        content = raw.get("content")
        if raw.get("role") == "assistant":
            tool_calls = []
            for block in content if isinstance(content, list) else []:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": as_str(block.get("id")),
                    "type": "function",
                    "function": {
                        "name": as_str(block.get("name")),
                        "arguments": json.dumps(
                            block.get("input") if is_record(block.get("input")) else {}),
                    },
                })
            text = text_of(content)
            messages.append({
                "role": "assistant",
                "content": text if text else None,
                **({"tool_calls": tool_calls} if tool_calls else {}),
            })
            continue
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        parts = []
        saw_tool_result = False
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                saw_tool_result = True
                messages.append({
                    "role": "tool",
                    "tool_call_id": as_str(block.get("tool_use_id")),
                    "content": _tool_result_text(block.get("content")),
                })
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
            else:
                image = _anthropic_image_to_openai(block)
                if image is not None:
                    parts.append(image)
        if parts or not saw_tool_result:
            messages.append({"role": "user", "content": parts})
        elif saw_tool_result:
            # A user turn holding only a tool result still counts as a user
            # turn for the echo: keep an empty user message so the last
            # user text stays "" on both sides of the translation.
            messages.append({"role": "user", "content": ""})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stops = body.get("stop_sequences")
    if isinstance(stops, list) and stops:
        out["stop"] = [s for s in stops if isinstance(s, str)]
    tools = _anthropic_tools(body)
    if tools:
        out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function",
                                  "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out

def _gemini_text_parts(content):
    """OpenAI/Anthropic content as Gemini parts (text + images)."""
    parts = []
    if isinstance(content, str):
        if content:
            parts.append({"text": content})
        return parts
    if not isinstance(content, list):
        return parts
    for block in content:
        if not is_record(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append({"text": block["text"]})
        elif block.get("type") == "image_url" and is_record(block.get("image_url")):
            raw = block["image_url"].get("url")
            if not isinstance(raw, str):
                continue
            if raw.startswith("data:"):
                head, _, data = raw.partition(",")
                media = head[5:].split(";")[0] or "image/png"
                parts.append({"inlineData": {"mimeType": media, "data": data}})
            else:
                parts.append({"fileData": {"fileUri": raw, "mimeType": "image/png"}})
        elif block.get("type") == "image" and is_record(block.get("source")):
            source = block["source"]
            if source.get("type") == "base64" and isinstance(source.get("data"), str):
                parts.append({"inlineData": {
                    "mimeType": as_str(source.get("media_type")) or "image/png",
                    "data": source["data"]}})
            elif source.get("type") == "url" and isinstance(source.get("url"), str):
                parts.append({"fileData": {"fileUri": source["url"], "mimeType": "image/png"}})
    return parts


def _gemini_tools(openai_tools, anthropic_tools):
    """Tool definitions as Gemini functionDeclarations, or None."""
    declarations = []
    for raw in openai_tools or []:
        if not is_record(raw) or not is_record(raw.get("function")):
            continue
        fn = raw["function"]
        if not as_str(fn.get("name")):
            continue
        declarations.append({
            "name": fn["name"],
            "description": as_str(fn.get("description")),
            "parameters": fn.get("parameters")
            if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
        })
    for raw in anthropic_tools or []:
        if not is_record(raw) or not as_str(raw.get("name")):
            continue
        declarations.append({
            "name": raw["name"],
            "description": as_str(raw.get("description")),
            "parameters": raw.get("input_schema")
            if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
        })
    if not declarations:
        return None
    return [{"functionDeclarations": declarations}]


def _gemini_config(body, json_mode):
    config = {}
    max_tokens = body.get("max_tokens")
    if isinstance(max_tokens, (int, float)):
        config["maxOutputTokens"] = max_tokens
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    stops = _stop_list(body)
    if not stops and isinstance(body.get("stop_sequences"), list):
        stops = [s for s in body["stop_sequences"] if isinstance(s, str)]
    if stops:
        config["stopSequences"] = stops
    if json_mode:
        config["responseMimeType"] = "application/json"
    return config


def openai_to_gemini(body, model):
    """An OpenAI request as a Gemini generateContent body (+ stream flag)."""
    system_text = ""
    contents = []
    tool_call_names = {}
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        if role == "system":
            text = text_of(raw.get("content"))
            if text:
                system_text = (system_text + "\n" + text) if system_text else text
            continue
        if role == "user":
            parts = _gemini_text_parts(raw.get("content"))
            contents.append({"role": "user", "parts": parts})
            continue
        if role == "assistant":
            parts = _gemini_text_parts(raw.get("content"))
            for call in raw.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                args = fn.get("arguments")
                parsed_args = {}
                if isinstance(args, str) and args.strip():
                    try:
                        parsed_args = json.loads(args)
                    except ValueError:
                        parsed_args = {}
                if not is_record(parsed_args):
                    parsed_args = {}
                tool_call_names[as_str(call.get("id"))] = as_str(fn.get("name"))
                parts.append({"functionCall": {"name": fn.get("name"), "args": parsed_args}})
            contents.append({"role": "model", "parts": parts})
            continue
        if role == "tool":
            name = tool_call_names.get(as_str(raw.get("tool_call_id")), "unknown")
            contents.append({"role": "user", "parts": [{"functionResponse": {
                "name": name,
                "response": {"content": _tool_result_text(raw.get("content"))}}}]})
    payload = {"contents": contents}
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}
    tools = _gemini_tools(body.get("tools"), None)
    if tools is not None:
        payload["tools"] = tools
    fmt = body.get("response_format")
    json_mode = is_record(fmt) and fmt.get("type") in ("json_object", "json_schema")
    config = _gemini_config(body, json_mode)
    if config:
        payload["generationConfig"] = config
    return payload, body.get("stream") is True


def anthropic_to_gemini(body, model):
    """An Anthropic request as a Gemini generateContent body (+ stream flag)."""
    system_text = text_of(body.get("system"))
    contents = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        content = raw.get("content")
        if raw.get("role") == "assistant":
            parts = []
            text = text_of(content)
            if text:
                parts.append({"text": text})
            if isinstance(content, list):
                for block in content:
                    if not is_record(block) or block.get("type") != "tool_use":
                        continue
                    parts.append({"functionCall": {
                        "name": as_str(block.get("name")),
                        "args": block.get("input")
                        if is_record(block.get("input")) else {}}})
            contents.append({"role": "model", "parts": parts})
            continue
        parts = []
        if isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "tool_result":
                    parts.append({"functionResponse": {
                        "name": "unknown",
                        "response": {"content": _tool_result_text(block.get("content"))}}})
                    continue
                parts.extend(_gemini_text_parts([block]))
        elif isinstance(content, str) and content:
            parts.append({"text": content})
        contents.append({"role": "user", "parts": parts})
    payload = {"contents": contents}
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}
    tools = _gemini_tools(None, body.get("tools"))
    if tools is not None:
        payload["tools"] = tools
    config = _gemini_config(body, False)
    if config:
        payload["generationConfig"] = config
    return payload, body.get("stream") is True

# ---------------------------------------------------------------------------
# Response translation back to the client's wire.
#
# The provider's answer text is always the echo of what it parsed, and the
# client compares that echo with what it sent -- so the answer text goes
# through byte-for-byte, only the envelope changes. Usage is always copied
# from the provider's own answer, never recomputed.
# ---------------------------------------------------------------------------

def _provider_answer(body, wire):
    """(text, finish_reason, usage, tool calls, model) from a provider answer."""
    if not is_record(body):
        return None
    if wire == "openai":
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0]
        if not is_record(choice):
            return None
        message = choice.get("message")
        text = ""
        tool_calls = []
        if is_record(message):
            text = message.get("content")
            if not isinstance(text, str):
                text = ""
            for call in message.get("tool_calls") or []:
                if is_record(call):
                    tool_calls.append(call)
        finish = choice.get("finish_reason") or "stop"
        usage = body.get("usage") if is_record(body.get("usage")) else {}
        return {"text": text, "finish": finish, "usage": usage,
                "tool_calls": tool_calls, "model": as_str(body.get("model"))}
    if wire == "anthropic":
        content = body.get("content")
        text = ""
        tool_uses = []
        if isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    text = block["text"]
                elif block.get("type") == "tool_use":
                    tool_uses.append(block)
        stop_reason = body.get("stop_reason")
        raw_usage = body.get("usage") if is_record(body.get("usage")) else {}
        input_tokens = raw_usage.get("input_tokens", 0)
        output_tokens = raw_usage.get("output_tokens", 0)
        # Normalise to the same keys the OpenAI wire uses, so the response
        # builders can copy usage without knowing which wire answered.
        usage = {"input_tokens": input_tokens, "output_tokens": output_tokens,
                 "prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                 "total_tokens": input_tokens + output_tokens}
        return {"text": text, "finish": stop_reason or "end_turn",
                "usage": usage, "tool_calls": tool_uses,
                "model": as_str(body.get("model"))}
    # Gemini.
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    candidate = candidates[0]
    if not is_record(candidate):
        return None
    content = candidate.get("content")
    text = ""
    parts = content.get("parts") if is_record(content) else None
    if isinstance(parts, list):
        for part in parts:
            if is_record(part) and isinstance(part.get("text"), str):
                text = part["text"]
    finish = candidate.get("finishReason") or "STOP"
    usage = body.get("usageMetadata") if is_record(body.get("usageMetadata")) else {}
    usage = {"prompt_tokens": usage.get("promptTokenCount", 0),
             "completion_tokens": usage.get("candidatesTokenCount", 0),
             "total_tokens": usage.get("totalTokenCount", 0)}
    return {"text": text, "finish": finish, "usage": usage, "tool_calls": [],
            "model": as_str(body.get("modelVersion")) or ""}


STOP_FAMILY = {
    "stop": "stop", "end_turn": "stop", "STOP": "stop",
    "length": "length", "max_tokens": "length", "MAX_TOKENS": "length",
    "tool_calls": "tool_calls", "tool_use": "tool_calls", "TOOL_CALLS": "tool_calls",
    "content_filter": "content_filter", "refusal": "content_filter",
    "SAFETY": "content_filter",
}

ANTHROPIC_STOP_FAMILY = {
    "stop": "end_turn", "end_turn": "end_turn", "STOP": "end_turn",
    "length": "max_tokens", "max_tokens": "max_tokens", "MAX_TOKENS": "max_tokens",
    "tool_calls": "tool_use", "tool_use": "tool_use", "TOOL_CALLS": "tool_use",
    "content_filter": "refusal", "refusal": "refusal", "SAFETY": "refusal",
}


def _openai_finish(reason, tool_calls):
    if tool_calls:
        return "tool_calls"
    return STOP_FAMILY.get(reason, "stop")


def _anthropic_finish(reason, tool_calls):
    if tool_calls:
        return "tool_use"
    return ANTHROPIC_STOP_FAMILY.get(reason, "end_turn")


def openai_response(answer, client_model, request_id, upstream_status):
    """An upstream answer as an OpenAI Chat Completions response."""
    usage = answer["usage"]
    message = {"role": "assistant", "content": answer["text"]}
    if answer["tool_calls"]:
        message["tool_calls"] = answer["tool_calls"]
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": client_model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": _openai_finish(answer["finish"], answer["tool_calls"]),
        }],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


def anthropic_response(answer, client_model, request_id):
    """An upstream answer as an Anthropic Messages response."""
    usage = answer["usage"]
    content = []
    if answer["text"]:
        content.append({"type": "text", "text": answer["text"]})
    for call in answer["tool_calls"]:
        if is_record(call):
            content.append({
                "type": "tool_use",
                "id": as_str(call.get("id")),
                "name": as_str(call.get("name")),
                "input": call.get("input")
                if is_record(call.get("input")) else {},
            })
    return {
        "id": request_id,
        "type": "message",
        "role": "assistant",
        "model": client_model,
        "content": content,
        "stop_reason": _anthropic_finish(answer["finish"], answer["tool_calls"]),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
        },
    }


def _sse_json(payload):
    return "data: %s\n\n" % json.dumps(payload, separators=(",", ":"))


def openai_stream(answer, client_model, request_id):
    """The answer as an OpenAI SSE stream, ending with the usage chunk."""
    usage = answer["usage"]
    head = {"id": request_id, "object": "chat.completion.chunk",
            "created": FIXED_CREATED, "model": client_model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                         "finish_reason": None}]}
    content = {"id": request_id, "object": "chat.completion.chunk",
               "created": FIXED_CREATED, "model": client_model,
               "choices": [{"index": 0, "delta": {"content": answer["text"]},
                            "finish_reason": None}]}
    if answer["tool_calls"]:
        content["choices"][0]["delta"]["tool_calls"] = answer["tool_calls"]
    finish = {"id": request_id, "object": "chat.completion.chunk",
              "created": FIXED_CREATED, "model": client_model,
              "choices": [{"index": 0, "delta": {},
                           "finish_reason": _openai_finish(answer["finish"],
                                                           answer["tool_calls"])}]}
    usage_chunk = {"id": request_id, "object": "chat.completion.chunk",
                   "created": FIXED_CREATED, "model": client_model,
                   "choices": [],
                   "usage": {
                       "prompt_tokens": usage.get("prompt_tokens", 0),
                       "completion_tokens": usage.get("completion_tokens", 0),
                       "total_tokens": usage.get("total_tokens", 0),
                   }}
    return "".join([
        _sse_json(head),
        _sse_json(content),
        _sse_json(finish),
        _sse_json(usage_chunk),
        "data: [DONE]\n\n",
    ]).encode("utf-8")


def anthropic_stream(answer, client_model, request_id):
    """The answer as an Anthropic SSE stream, with usage and stop events."""
    usage = answer["usage"]
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    start = {
        "type": "message_start",
        "message": {"id": request_id, "type": "message", "role": "assistant",
                    "model": client_model, "content": [],
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0}},
    }
    block_start = {"type": "content_block_start", "index": 0,
                   "content_block": {"type": "text", "text": ""}}
    delta = {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": answer["text"]}}
    block_stop = {"type": "content_block_stop", "index": 0}
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": _anthropic_finish(answer["finish"],
                                                   answer["tool_calls"]),
                  "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }
    message_stop = {"type": "message_stop"}
    return "".join([
        "event: message_start\n" + _sse_json(start),
        "event: content_block_start\n" + _sse_json(block_start),
        "event: content_block_delta\n" + _sse_json(delta),
        "event: content_block_stop\n" + _sse_json(block_stop),
        "event: message_delta\n" + _sse_json(message_delta),
        "event: message_stop\n" + _sse_json(message_stop),
    ]).encode("utf-8")

# ---------------------------------------------------------------------------
# Routing.
# ---------------------------------------------------------------------------

class State:
    def __init__(self, brief, catalog, probes, probes_by_id):
        self.brief = brief
        self.catalog = catalog
        self.probes = probes              # [(Provider, models or None)]
        self.probes_by_id = probes_by_id  # id -> (Provider, models or None)
        self.models = served_models(brief, probes)
        self.manifest_index = 0
        self.calls = 0                    # successful provider calls
        self.attempts = 0                 # every upstream call made
        self._fail_streaks = {}
        self._exhausted = set()
        self._lock = threading.Lock()

    def request_row(self):
        if self.manifest_index < len(self.brief.manifest):
            return self.brief.manifest[self.manifest_index]
        return None

    def candidates(self, api, requires):
        """(Provider, model, rating) candidates, best rating first.

        Only free models that meet the request's requirements. A JSON-mode
        OpenAI request never goes to an Anthropic-wire provider, whose echo
        cannot carry JSON mode.
        """
        out = []
        for provider, models in self.probes:
            if models is None or provider.id in self._exhausted:
                continue
            for model in models:
                if not model_is_free(self.brief, provider.id, model["id"], model["free"]):
                    continue
                if api == "openai" and requires.get("json") \
                        and provider.wire == "anthropic":
                    continue
                caps = model_caps(self.brief, provider.id, model["id"], model)
                if not meets(requires, caps):
                    continue
                out.append((provider, model["id"], caps["rating"]))
        out.sort(key=lambda item: (-item[2], item[0].id, item[1]))
        return out

    def call(self, provider, url, headers, payload, stream, is_gemini):
        """One upstream attempt; returns (status, body) or (None, b'')."""
        with self._lock:
            self.attempts += 1
        if is_gemini:
            headers = dict(headers)
            if stream:
                url = url + ":streamGenerateContent?alt=sse"
            else:
                url = url + ":generateContent"
            status, body = upstream_json("POST", url, headers, payload)
        else:
            status, body = upstream_json(
                "POST", url, headers,
                {**payload, "stream": True} if stream else payload)
        if status is None:
            self._note_failure(provider.id)
            return None, b""
        if 200 <= status < 300:
            self._note_success(provider.id)
            with self._lock:
                self.calls += 1
            return status, body
        # A refusal does not consume an answer but does count against the
        # provider: two refusals in a row retire it for this trial. (The
        # budget check on the provider's side already counts refusals, so
        # this is a guard, not the throttle.)
        if status in (402, 429) or 500 <= status < 600:
            self._note_failure(provider.id)
            return status, body
        return status, body

    def _note_failure(self, provider_id):
        with self._lock:
            streak = self._fail_streaks.get(provider_id, 0) + 1
            self._fail_streaks[provider_id] = streak
            if streak >= 2:
                self._exhausted.add(provider_id)

    def _note_success(self, provider_id):
        with self._lock:
            self._fail_streaks[provider_id] = 0


def _parse_upstream_answer(body, provider, stream):
    if provider.wire == "gemini":
        # streamGenerateContent answers as SSE; take the last JSON payload.
        text = body.decode("utf-8", errors="replace")
        last = None
        for line in text.splitlines():
            if line.startswith("data:"):
                last = line[5:].strip()
        payload = text
        if stream and last:
            payload = last
        try:
            return _provider_answer(json.loads(payload), "gemini")
        except ValueError:
            return None
    try:
        return _provider_answer(json.loads(body.decode("utf-8")), provider.wire)
    except ValueError:
        return None


def route(state, api, body):
    """Serve one client request; returns (status, content_type, payload)."""
    row = state.request_row()
    if row is not None and row["api"] != api:
        # Manifest and arrival order disagree; trust the client's own wire
        # and the requirements the body itself carries.
        row = None
    with state._lock:
        state.manifest_index += 1
    requires = _body_requirements(body) if row is None else row["requires"]
    client_model = as_str(body.get("model")) or "unknown"
    stream = body.get("stream") is True
    for provider, model_id, _rating in state.candidates(api, requires):
        base = join_base(ENV_URL, provider.base)
        headers = {"accept": "application/json"}
        headers.update(auth_headers(provider))
        if provider.wire == "anthropic":
            headers["anthropic-version"] = ANTHROPIC_VERSION
        if provider.wire == "openai":
            if api == "anthropic":
                upstream_body = anthropic_to_openai(body, model_id)
            else:
                upstream_body = dict(body)
                upstream_body["model"] = model_id
            # Never ask the provider for a stream: the server buffers the
            # whole answer and builds the client's SSE itself. A streamed
            # upstream answer is SSE text, not the JSON this parser reads.
            upstream_body.pop("stream", None)
            upstream_body.pop("stream_options", None)
            url = base + "/v1/chat/completions"
            payload = upstream_body
        elif provider.wire == "anthropic":
            if api == "anthropic":
                upstream_body = dict(body)
                upstream_body["model"] = model_id
            else:
                upstream_body = openai_to_anthropic(body, model_id)
            upstream_body.pop("stream", None)
            url = base + "/v1/messages"
            payload = upstream_body
        else:
            if api == "anthropic":
                payload, _use_stream = anthropic_to_gemini(body, model_id)
            else:
                payload, _use_stream = openai_to_gemini(body, model_id)
            url = base + "/v1beta/models/" + model_id
        status, raw = state.call(provider, url, headers, payload, False,
                                 provider.wire == "gemini")
        if status is None:
            log("route %s: %s unreachable" % (provider.id, url))
            continue
        if status == 402 or status == 429 or 500 <= status < 600:
            log("route %s: refused with %s; failing over" % (provider.id, status))
            continue
        if status == 401 or status == 400:
            log("route %s: %s -- refusing to retry a bad request" % (provider.id, status))
            continue
        if not 200 <= status < 300:
            log("route %s: unexpected status %s; failing over" % (provider.id, status))
            continue
        answer = _parse_upstream_answer(raw, provider, False)
        if answer is None:
            # A 2xx the server cannot read is not a green light to call
            # another provider for this request: it would break the
            # one-provider rule. Return a plain 502 instead.
            log("route %s: 2xx answer unreadable; not retrying" % provider.id)
            return 502, "application/json", _error_json(
                "provider %s returned an unreadable answer" % provider.id)
        if api == "openai":
            if stream:
                payload_bytes = openai_stream(answer, client_model, _request_id())
            else:
                payload_bytes = _json_bytes(openai_response(
                    answer, client_model, _request_id(), status))
            content_type = "text/event-stream" if stream else "application/json"
        else:
            if stream:
                payload_bytes = anthropic_stream(answer, client_model, _request_id())
            else:
                payload_bytes = _json_bytes(anthropic_response(
                    answer, client_model, _request_id()))
            content_type = "text/event-stream" if stream else "application/json"
        return 200, content_type, payload_bytes
    log("route: every candidate refused the request")
    return 502, "application/json", _error_json("no provider could serve this request")


def _body_requirements(body):
    """Requirements derived from a request body, for manifest mismatches."""
    messages = body.get("messages") or []
    needs_vision = False
    for raw in messages:
        if not is_record(raw):
            continue
        content = raw.get("content")
        if isinstance(content, list):
            for block in content:
                if is_record(block) and block.get("type") in ("image_url", "image"):
                    needs_vision = True
    fmt = body.get("response_format")
    return {
        "tools": bool(body.get("tools")),
        "vision": needs_vision,
        "json": is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"),
        "min_context": None,
    }


def _json_bytes(payload):
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _error_json(message):
    return _json_bytes({"error": {"message": message, "type": "server_error"}})


_COUNTER = [0]
_COUNTER_LOCK = threading.Lock()


def _request_id():
    with _COUNTER_LOCK:
        _COUNTER[0] += 1
        return "req-aggregator-%06d" % _COUNTER[0]

# ---------------------------------------------------------------------------
# HTTP server.
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    state = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep stdout/stderr clean for the env
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            if self.headers.get("x-api-key"):
                payload = _json_bytes(anthropic_model_list(self.state.models))
            else:
                payload = _json_bytes(openai_model_list(self.state.models))
            self._send(200, "application/json", payload)
            return
        self._send(404, "application/json", _error_json("not found"))

    def do_POST(self):
        length = self.headers.get("Content-Length")
        try:
            length = int(length) if length else 0
        except ValueError:
            length = 0
        if length < 0 or length > MAX_BODY_BYTES:
            self._send(400, "application/json", _error_json("bad request body"))
            return
        body_bytes = self.rfile.read(length) if length else b""
        try:
            body = json.loads(body_bytes.decode("utf-8"))
        except ValueError:
            self._send(400, "application/json", _error_json("invalid JSON"))
            return
        if not is_record(body):
            self._send(400, "application/json", _error_json("request must be a JSON object"))
            return
        if self.path == "/v1/chat/completions":
            status, content_type, payload = route(self.state, "openai", body)
            self._send(status, content_type, payload)
        elif self.path == "/v1/messages":
            status, content_type, payload = route(self.state, "anthropic", body)
            self._send(status, content_type, payload)
        else:
            self._send(404, "application/json", _error_json("not found"))

    def _send(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        if payload:
            self.wfile.write(payload)


def serve(brief):
    """Probe, register, serve the 20 requests, then exit 0."""
    probes = []
    for provider in brief.providers:
        models = probe_provider(brief, provider)
        probes.append((provider, models))
        log("probe %s: %d models" % (
            provider.id, len(models) if models is not None else -1))
    catalog = build_catalog(brief, probes)
    probes_by_id = {p.id: (p, m) for p, m in probes}
    state = State(brief, catalog, probes, probes_by_id)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Handler.state = state
    port = server.server_address[1]
    register_payload = {"protocol_version": PROTOCOL_VERSION,
                        "port": port, "catalog": catalog}
    status, body = upstream_json(
        "POST", ENV_URL + "/register",
        {"content-type": "application/json"}, register_payload)
    if status != 200:
        log("register: env answered %s: %s" % (status, body.decode("utf-8", "replace")[:200]))
        sys.exit(1)
    log("registered on port %d" % port)
    requests = 20
    try:
        parsed = json.loads(body.decode("utf-8"))
        if is_record(parsed) and isinstance(parsed.get("requests"), int):
            requests = parsed["requests"]
    except ValueError:
        pass

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        while state.manifest_index < requests:
            time.sleep(POLL_INTERVAL)
    finally:
        server.shutdown()
        server.server_close()
    log("served %d requests, %d upstream calls, %d attempts"
        % (state.manifest_index, state.calls, state.attempts))
    return 0


def main(argv):
    if "--list" in argv:
        brief = Brief(PROMPT_FILE)
        if brief.error is not None or not ENV_URL:
            # No brief, or no env to probe: the brief carries no model
            # lists of its own, so there is nothing to list.
            sys.stdout.write('{"providers":[]}\n')
            if brief.error is not None:
                log("list: %s" % brief.error)
            return 0
        probes = [(provider, probe_provider(brief, provider))
                  for provider in brief.providers]
        catalog = build_catalog(brief, probes)
        sys.stdout.write(json.dumps(catalog, separators=(",", ":")) + "\n")
        return 0
    brief = Brief(PROMPT_FILE)
    if brief.error is not None:
        log(brief.error)
        return 1
    if not ENV_URL:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    return serve(brief)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
