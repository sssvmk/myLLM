"""Off-policy teacher behind an OpenAI-compatible endpoint (TC-1). Only text is used."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

from .judge import chat_once, make_client


class Teacher:
    def __init__(self, cfg, client=None, sleep_s: float = 0.0):
        self.cfg = cfg
        self.client = client if client is not None else make_client(cfg.base_url, cfg.api_key_env, cfg.timeout_s)
        self.sleep_s = sleep_s

    @classmethod
    def from_config(cls, cfg, client=None) -> "Teacher":
        return cls(cfg.teachers.offpolicy, client)

    def _one(self, messages: List[Dict[str, str]]) -> Optional[str]:
        c = self.cfg
        for _ in range(c.max_retries + 1):
            try:
                return chat_once(self.client, c.model, messages, c.temperature, c.max_tokens, top_p=c.top_p)
            except Exception:  # noqa: BLE001
                if self.sleep_s:
                    time.sleep(self.sleep_s)
        return None

    def complete(self, prompts: Sequence[List[Dict[str, str]]], samples_per_prompt: int) -> List[List[Optional[str]]]:
        """For each prompt (a message list), `samples_per_prompt` responses; None marks a request
        that failed after all retries."""
        jobs = [(i, m) for i, m in enumerate(prompts) for _ in range(samples_per_prompt)]
        with ThreadPoolExecutor(max_workers=self.cfg.max_concurrency) as ex:
            outs = list(ex.map(lambda j: self._one(j[1]), jobs))
        res: List[List[Optional[str]]] = [[] for _ in prompts]
        for (i, _), o in zip(jobs, outs):
            res[i].append(o)
        return res

    def identity(self) -> Dict[str, Any]:
        return {"role": "offpolicy_teacher", "model": self.cfg.model, "base_url": self.cfg.base_url,
                "temperature": self.cfg.temperature, "top_p": self.cfg.top_p, "max_tokens": self.cfg.max_tokens}
