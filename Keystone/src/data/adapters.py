"""Dataset adapters (DS-2, DS-3, DS-3a): one raw record in, one typed item out.

An adapter raises AdapterError(reason) for a record it cannot use; the registry counts drops per
reason (DS-5). Every item carries `texts`: the raw text fields of the record, which decontamination
(DP-4, "all text fields concatenated") compares against the evaluation n-grams. Items never hold
tokens except PackedBlock; tokenization happens in preparation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..rl.rewards.math import extract_ground_truth

ROLES = {"system", "user", "assistant"}

# CF-1 note: this label for appended test asserts is a fixed string, not a config value.
# Recorded in docs/implementation_notes.md; promote to config if it needs tuning.
TESTS_HEADER = "Your code should pass these tests:"


class AdapterError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


Message = Dict[str, str]


@dataclass
class TextDoc:
    text: str
    texts: List[str] = field(default_factory=list)


@dataclass
class PackedBlock:
    tokens: List[int]
    doc_start: Optional[List[bool]]
    texts: List[str] = field(default_factory=list)          # no text: decontamination not applicable


@dataclass
class Conversation:
    messages: List[Message]
    texts: List[str] = field(default_factory=list)


@dataclass
class PrefPair:
    prompt_messages: List[Message]
    chosen: str
    rejected: str
    texts: List[str] = field(default_factory=list)


@dataclass
class PromptItem:
    messages: List[Message]
    texts: List[str] = field(default_factory=list)


@dataclass
class RLMathItem:
    messages: List[Message]
    ground_truth: str
    texts: List[str] = field(default_factory=list)


@dataclass
class RLCodeItem:
    messages: List[Message]
    tests: str
    entry_point: Optional[str]
    texts: List[str] = field(default_factory=list)


@dataclass
class EvalItem:
    data: Dict[str, Any]
    texts: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- field access (DS-3)
_MISSING = object()


def get_path(record: Dict[str, Any], path: Optional[str]) -> Any:
    """Dotted path into a record. A null path means the field is absent for this dataset."""
    if path is None:
        return None
    cur: Any = record
    for seg in path.split("."):
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        else:
            raise AdapterError(f"missing_field:{path}")
    return cur


def _req_str(record, path: Optional[str], what: str) -> str:
    v = get_path(record, path)
    if not isinstance(v, str):
        raise AdapterError(f"not_a_string:{what}")
    if not v.strip():
        raise AdapterError(f"empty:{what}")
    return v


def _norm_messages(raw: Any, role_key: str, content_key: str, what: str) -> List[Message]:
    if not isinstance(raw, list) or not raw:
        raise AdapterError(f"bad_messages:{what}")
    out: List[Message] = []
    for m in raw:
        if not isinstance(m, dict) or role_key not in m or content_key not in m:
            raise AdapterError(f"bad_message_shape:{what}")
        role, content = m[role_key], m[content_key]
        if role not in ROLES:
            raise AdapterError(f"bad_role:{role}")
        if not isinstance(content, str):
            raise AdapterError("non_string_content")
        out.append({"role": role, "content": content})
    return out


def _check_chat(msgs: List[Message]) -> None:
    if not any(m["role"] == "user" for m in msgs):
        raise AdapterError("no_user_turn")
    if msgs[-1]["role"] != "assistant":
        raise AdapterError("no_final_assistant")
    if any(m["role"] == "assistant" and not m["content"].strip() for m in msgs):
        raise AdapterError("empty_assistant")


# --------------------------------------------------------------------------- adapters
def _text_field(ds, r):
    t = _req_str(r, ds.field_map["text"], "text")
    return TextDoc(t, [t])


def _packed_tokens(ds, r):
    toks = get_path(r, ds.field_map["tokens"])
    if not isinstance(toks, (list, tuple)) or not toks or not all(isinstance(t, int) for t in toks):
        raise AdapterError("bad_tokens")
    ds_path = ds.field_map.get("doc_start")
    flags = get_path(r, ds_path) if ds_path else None
    if flags is not None:
        if len(flags) != len(toks):
            raise AdapterError("doc_start_length_mismatch")
        flags = [bool(f) for f in flags]
    return PackedBlock(list(toks), flags)


def _messages_list(ds, r):
    fm = ds.field_map
    msgs = _norm_messages(get_path(r, fm["messages"]), fm["role_key"], fm["content_key"], "messages")
    _check_chat(msgs)
    return Conversation(msgs, [m["content"] for m in msgs])


def _instruction_output(ds, r):
    fm = ds.field_map
    instr = _req_str(r, fm["instruction"], "instruction")
    out = _req_str(r, fm["output"], "output")
    extra = get_path(r, fm.get("input")) if fm.get("input") else None
    user = instr if not (isinstance(extra, str) and extra.strip()) else f"{instr}\n\n{extra}"
    return Conversation([{"role": "user", "content": user}, {"role": "assistant", "content": out}],
                        [user, out])


def _chosen_rejected_messages(ds, r):
    fm = ds.field_map
    rk, ck = fm.get("role_key") or "role", fm.get("content_key") or "content"
    chosen = _norm_messages(get_path(r, fm["chosen"]), rk, ck, "chosen")
    rejected = _norm_messages(get_path(r, fm["rejected"]), rk, ck, "rejected")
    _check_chat(chosen)
    _check_chat(rejected)
    if chosen[:-1] != rejected[:-1]:
        raise AdapterError("prompt_mismatch")
    return PrefPair(chosen[:-1], chosen[-1]["content"], rejected[-1]["content"],
                    [m["content"] for m in chosen] + [rejected[-1]["content"]])


def _prompt_chosen_rejected_text(ds, r):
    fm = ds.field_map
    p, c, j = (_req_str(r, fm[k], k) for k in ("prompt", "chosen", "rejected"))
    return PrefPair([{"role": "user", "content": p}], c, j, [p, c, j])


def _prompt_text(ds, r):
    p = _req_str(r, ds.field_map["prompt"], "prompt")
    return PromptItem([{"role": "user", "content": p}], [p])


def _question_answer(ds, r):
    q = _req_str(r, ds.field_map["question"], "question")
    a = _req_str(r, ds.field_map["answer"], "answer")
    gt = extract_ground_truth(a, ds.answer_extraction, ds.answer_regex)
    if gt is None or not gt.strip():
        raise AdapterError("answer_extraction_failed")
    return RLMathItem([{"role": "user", "content": q}], gt.strip(), [q, a])


def _code_tests(ds, r):
    fm = ds.field_map
    prompt = _req_str(r, fm["prompt"], "prompt")
    tests = get_path(r, fm["tests"])
    if isinstance(tests, (list, tuple)):
        if not tests or not all(isinstance(t, str) for t in tests):
            raise AdapterError("bad_tests")
        lines = [t.strip() for t in tests]
    elif isinstance(tests, str) and tests.strip():
        lines = [ln.strip() for ln in tests.splitlines() if ln.strip().startswith("assert")]
    else:
        raise AdapterError("bad_tests")
    entry = get_path(r, fm["entry_point"]) if fm.get("entry_point") else None
    n = ds.prompt_include_tests
    shown = "\n".join(lines[:n])
    user = prompt if n == 0 or not shown else f"{prompt}\n\n{TESTS_HEADER}\n{shown}"
    test_prog = "\n".join(tests) if isinstance(tests, (list, tuple)) else tests
    return RLCodeItem([{"role": "user", "content": user}], test_prog, entry if isinstance(entry, str) else None,
                      [prompt, test_prog])


def _eval_mmlu(ds, r):
    fm = ds.field_map
    q = _req_str(r, fm["question"], "question")
    choices = get_path(r, fm["choices"])
    ans = get_path(r, fm["answer"])
    if not isinstance(choices, (list, tuple)) or len(choices) < 2 or not all(isinstance(c, str) for c in choices):
        raise AdapterError("bad_choices")
    if not isinstance(ans, int) or isinstance(ans, bool) or not 0 <= ans < len(choices):
        raise AdapterError("bad_answer_index")
    return EvalItem({"question": q, "choices": list(choices), "answer": int(ans)}, [q] + list(choices))


def _eval_gsm8k(ds, r):
    q, a = _req_str(r, ds.field_map["question"], "question"), _req_str(r, ds.field_map["answer"], "answer")
    return EvalItem({"question": q, "answer": a}, [q, a])


def _eval_math(ds, r):
    p, a = _req_str(r, ds.field_map["problem"], "problem"), _req_str(r, ds.field_map["answer"], "answer")
    return EvalItem({"problem": p, "answer": a}, [p, a])


def _eval_humaneval(ds, r):
    fm = ds.field_map
    p, t = _req_str(r, fm["prompt"], "prompt"), _req_str(r, fm["test"], "test")
    ep = _req_str(r, fm["entry_point"], "entry_point")
    return EvalItem({"prompt": p, "test": t, "entry_point": ep}, [p, t])


def _eval_ifeval(ds, r):
    fm = ds.field_map
    p = _req_str(r, fm["prompt"], "prompt")
    ids = get_path(r, fm["instruction_id_list"])
    kw = get_path(r, fm["kwargs"])
    if not isinstance(ids, (list, tuple)) or not ids:
        raise AdapterError("bad_instruction_id_list")
    return EvalItem({"key": get_path(r, fm["key"]), "prompt": p, "instruction_id_list": list(ids),
                     "kwargs": list(kw) if isinstance(kw, (list, tuple)) else kw}, [p])


def _eval_alpaca(ds, r):
    i, o = _req_str(r, ds.field_map["instruction"], "instruction"), _req_str(r, ds.field_map["reference_output"], "reference_output")
    return EvalItem({"instruction": i, "reference_output": o}, [i, o])


def _eval_safety(ds, r):
    p = _req_str(r, ds.field_map["prompt"], "prompt")
    label = get_path(r, ds.field_map["label"])
    return EvalItem({"prompt": p, "label": label, "is_safe": label == ds.safe_label}, [p])


ADAPTERS: Dict[str, Callable[[Any, Dict[str, Any]], Any]] = {
    "text_field": _text_field,
    "packed_tokens": _packed_tokens,
    "messages_list": _messages_list,
    "instruction_output": _instruction_output,
    "chosen_rejected_messages": _chosen_rejected_messages,
    "prompt_chosen_rejected_text": _prompt_chosen_rejected_text,
    "prompt_text": _prompt_text,
    "question_answer": _question_answer,
    "code_tests": _code_tests,
    "eval_mmlu": _eval_mmlu,
    "eval_gsm8k": _eval_gsm8k,
    "eval_math": _eval_math,
    "eval_humaneval": _eval_humaneval,
    "eval_ifeval": _eval_ifeval,
    "eval_alpaca": _eval_alpaca,
    "eval_safety": _eval_safety,
}


def adapt_record(ds, record: Dict[str, Any]):
    """Apply the dataset's adapter. Anything unexpected becomes a counted `malformed:<Type>` drop."""
    try:
        return ADAPTERS[ds.adapter](ds, record)
    except AdapterError:
        raise
    except Exception as e:  # noqa: BLE001 -- a bad record is a drop, never a crash (DS-5)
        raise AdapterError(f"malformed:{type(e).__name__}") from e
