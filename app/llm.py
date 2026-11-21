"""Chat-model client with automatic fallback.

Order: 1) primary OpenRouter model  2) OpenRouter free router  3) OpenAI (only if a key is set).
A provider is skipped if it errors (rate limit, outage, timeout) or returns an empty reply.
"""
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)

MODEL = os.getenv("LLM_MODEL", "openrouter/free")
REASONING = os.getenv("LLM_REASONING", "off")      # off | low | medium | high


def reasoning_body(setting):
    """OpenRouter reasoning setting: 'off' disables thinking (fastest), else an effort level."""
    if setting == "off":
        return {"reasoning": {"enabled": False}}
    return {"reasoning": {"effort": setting}}

_openrouter = OpenAI(base_url="https://openrouter.ai/api/v1",
                     api_key=os.getenv("OPENROUTER_API_KEY") or "missing",
                     timeout=60, max_retries=1)
_or_headers = {"HTTP-Referer": os.getenv("APP_URL", "http://localhost:8501"),
               "X-Title": "Real Estate Property Assistant"}
_or_body = reasoning_body(REASONING)

PROVIDERS = [{"name": f"OpenRouter ({MODEL})", "client": _openrouter, "model": MODEL,
              "headers": _or_headers, "body": _or_body}]
if MODEL != "openrouter/free":
    PROVIDERS.append({"name": "OpenRouter (openrouter/free)", "client": _openrouter,
                      "model": "openrouter/free", "headers": _or_headers, "body": reasoning_body("low")})
if os.getenv("OPENAI_API_KEY"):
    PROVIDERS.append({"name": "OpenAI fallback",
                      "client": OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60, max_retries=1),
                      "model": os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini"),
                      "headers": None, "body": None})


def _create(p, messages, temperature, max_tokens, stream=False):
    return p["client"].chat.completions.create(
        model=p["model"], messages=messages, temperature=temperature, max_tokens=max_tokens,
        extra_headers=p["headers"], extra_body=p["body"], stream=stream)


def chat(messages, temperature=0.2, max_tokens=2000):
    """Full reply as text, trying each provider in order."""
    last_error = None
    for p in PROVIDERS:
        try:
            text = _create(p, messages, temperature, max_tokens).choices[0].message.content
            if text:
                return text
            last_error = RuntimeError("empty reply")
        except Exception as e:
            last_error = e
        logger.warning("%s failed: %s - trying next provider", p["name"], last_error)
    raise RuntimeError(f"All model providers failed: {last_error}")


def stream_chat(messages, temperature=0.2, max_tokens=2000):
    """Reply piece by piece. Falls back only if a provider fails before sending any text."""
    last_error = None
    for p in PROVIDERS:
        started = False
        try:
            for event in _create(p, messages, temperature, max_tokens, stream=True):
                if event.choices and event.choices[0].delta.content:
                    started = True
                    yield event.choices[0].delta.content
            if started:
                return
            last_error = RuntimeError("empty reply")
        except Exception as e:
            if started:          # already showed part of an answer; can't switch mid-sentence
                raise
            last_error = e
        logger.warning("%s failed: %s - trying next provider", p["name"], last_error)
    raise RuntimeError(f"All model providers failed: {last_error}")


if __name__ == "__main__":
    print("providers:", [p["name"] for p in PROVIDERS])
    print("reply:", chat([{"role": "user", "content": "Reply with exactly: hello"}], max_tokens=1000))