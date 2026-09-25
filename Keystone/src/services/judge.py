"""LLM judge behind an OpenAI-compatible Chat Completions endpoint (JG-1..JG-4).

Parseable answers are cached under `prepared_data.root_uri/judge_cache/<sha256>.json` keyed by
(judge model, rendered prompt, temperature, max_tokens) (JG-3). An unparseable answer is asked
again up to `max_retries` more times (bypassing the cache, which never stores it) and then counts
as a tie (pairwise), `partial_refusal` (refusal) or is dropped (scoring, returns None) (JG-2).
The last regex match in the answer is used, because the templates ask for the verdict at the end.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..io.storage import Storage


class ServiceError(RuntimeError):
    pass


def make_client(base_url: str, api_key_env: str, timeout_s: float):
    """`openai` client with its own retries disabled: retry policy is ours (config-driven)."""
    key = os.environ.get(api_key_env)
    if key is None:
        raise ServiceError(f"environment variable {api_key_env} is not set")
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=key, timeout=timeout_s, max_retries=0)


def chat_once(client, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int,
              top_p: Optional[float] = None) -> str:
    kw: Dict[str, Any] = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if top_p is not None:
        kw["top_p"] = top_p
    r = client.chat.completions.create(**kw)
    return r.choices[0].message.content or ""


def last_match(pattern: str, text: str) -> Optional[str]:
    m = re.findall(pattern, text)
    if not m:
        return None
    last = m[-1]
    return last[0] if isinstance(last, tuple) else last


class Judge:
    def __init__(self, cfg, storage: Storage, cache_root_uri: str, templates: Dict[str, str], client=None,
                 sleep_s: float = 0.0):
        self.cfg, self.storage = cfg, storage
        self.cache_uri = Storage.join(cache_root_uri, "judge_cache")
        self.templates = templates
        self.client = client if client is not None else make_client(cfg.base_url, cfg.api_key_env, cfg.timeout_s)
        self.sleep_s = sleep_s
        self.requests = 0                                   # network calls actually made (tests, cost tracking)

    @classmethod
    def from_config(cls, cfg, storage: Storage, client=None) -> "Judge":
        j = cfg.judge
        templates = {"pairwise": storage.read_text(j.prompts.pairwise_uri), "refusal": storage.read_text(j.prompts.refusal_uri),
                     "scoring": storage.read_text(j.prompts.scoring_uri)}
        return cls(j, storage, cfg.prepared_data.root_uri, templates, client)

    # ------------------------------------------------------------------ transport
    def _key(self, prompt: str) -> str:
        c = self.cfg
        return hashlib.sha256(json.dumps([c.model, prompt, c.temperature, c.max_tokens]).encode()).hexdigest()

    def _ask(self, prompt: str, parse: Callable[[str], Optional[str]]) -> Optional[str]:
        """Parsed answer or None. Cache first; then up to 1 + max_retries live attempts."""
        c = self.cfg
        key_uri = Storage.join(self.cache_uri, self._key(prompt) + ".json")
        if self.storage.exists(key_uri):
            hit = parse(json.loads(self.storage.read_text(key_uri))["response"])
            if hit is not None:
                return hit
        for attempt in range(c.max_retries + 1):
            try:
                self.requests += 1
                text = chat_once(self.client, c.model, [{"role": "user", "content": prompt}], c.temperature, c.max_tokens)
            except Exception:  # noqa: BLE001 -- transport error: counts as a failed attempt
                if self.sleep_s:
                    time.sleep(self.sleep_s)
                continue
            out = parse(text)
            if out is not None:
                self.storage.write_text(key_uri, json.dumps({"model": c.model, "response": text}))
                return out
        return None

    # ------------------------------------------------------------------ judgements
    def pairwise(self, instruction: str, response_a: str, response_b: str) -> str:
        p = self.templates["pairwise"].format(instruction=instruction, response_a=response_a, response_b=response_b)
        v = self._ask(p, lambda t: last_match(self.cfg.parse.pairwise_regex, t))
        return v if v in ("A", "B", "TIE") else "TIE"

    def refusal(self, prompt: str, response: str) -> str:
        p = self.templates["refusal"].format(prompt=prompt, response=response)
        v = self._ask(p, lambda t: last_match(self.cfg.parse.refusal_regex, t))
        return v if v in ("compliance", "refusal", "partial_refusal") else "partial_refusal"

    def score(self, instruction: str, response: str) -> Optional[int]:
        p = self.templates["scoring"].format(instruction=instruction, response=response)
        v = self._ask(p, lambda t: last_match(self.cfg.parse.score_regex, t))
        return int(v) if v is not None else None

    def map(self, fn: Callable, items: Sequence) -> List:
        with ThreadPoolExecutor(max_workers=self.cfg.max_concurrency) as ex:
            return list(ex.map(fn, items))

    def identity(self) -> Dict[str, Any]:
        return {"role": "judge", "model": self.cfg.model, "base_url": self.cfg.base_url,
                "temperature": self.cfg.temperature, "max_tokens": self.cfg.max_tokens}
