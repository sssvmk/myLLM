"""IFEval scoring with lm-eval's own instruction checkers (B-5). `lm-eval` is imported only here
and only when the benchmark runs (importing it pulls NLTK data the first time)."""
from __future__ import annotations

from typing import Any, Dict, List


def score_ifeval(docs: List[Dict[str, Any]], responses: List[str]) -> Dict[str, float]:
    """docs: {key, prompt, instruction_id_list, kwargs}. Returns the four §11.2 metrics."""
    try:
        from lm_eval.tasks.ifeval import utils
    except ImportError as e:
        raise ImportError("IFEval needs the `lm-eval` package (IFEval instruction checkers only): pip install lm-eval") from e
    prompt_strict = prompt_loose = 0.0
    inst_strict: List[bool] = []
    inst_loose: List[bool] = []
    for doc, resp in zip(docs, responses):
        r = utils.process_results(doc, [resp])
        prompt_strict += float(r["prompt_level_strict_acc"])
        prompt_loose += float(r["prompt_level_loose_acc"])
        inst_strict += [bool(x) for x in r["inst_level_strict_acc"]]
        inst_loose += [bool(x) for x in r["inst_level_loose_acc"]]
    n = max(len(docs), 1)
    return {"ifeval_prompt_strict": prompt_strict / n, "ifeval_prompt_loose": prompt_loose / n,
            "ifeval_inst_strict": sum(inst_strict) / max(len(inst_strict), 1),
            "ifeval_inst_loose": sum(inst_loose) / max(len(inst_loose), 1)}
