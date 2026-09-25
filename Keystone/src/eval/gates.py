"""Promotion gates (GT-1..GT-3)."""
from __future__ import annotations

from typing import Any, Dict, List


def evaluate_gate(stage: str, current: Dict[str, Any], parent: Dict[str, Any], gates) -> Dict[str, Any]:
    """current/parent: eval_report dicts with a "metrics" mapping and a "mode" ("base"|"chat")."""
    cur, par = current["metrics"], parent["metrics"]
    checks: List[Dict[str, Any]] = []
    lower = set(gates.lower_is_better)

    for metric, tol in gates.tolerances.items():
        if metric not in cur or metric not in par or cur[metric] is None or par[metric] is None:
            continue
        drop = (cur[metric] - par[metric]) if metric in lower else (par[metric] - cur[metric])
        checks.append({"check": "tolerance", "metric": metric, "current": cur[metric], "parent": par[metric],
                       "tolerance": tol, "lower_is_better": metric in lower, "passed": drop <= tol})

    if "response_len_mean" in cur and "response_len_mean" in par and par["response_len_mean"]:
        ratio = cur["response_len_mean"] / par["response_len_mean"]
        checks.append({"check": "length_ratio", "metric": "response_len_mean", "current": cur["response_len_mean"],
                       "parent": par["response_len_mean"], "ratio": ratio,
                       "max": gates.thresholds.length_ratio_max, "passed": ratio <= gates.thresholds.length_ratio_max})

    for metric in gates.require_improvement.get(stage, []):
        if metric not in cur or metric not in par:
            checks.append({"check": "require_improvement", "metric": metric, "passed": False,
                           "reason": "metric missing from current or parent report"})
            continue
        delta = cur[metric] - par[metric]
        if metric in lower:
            delta = -delta
        checks.append({"check": "require_improvement", "metric": metric, "current": cur[metric],
                       "parent": par[metric], "min_improvement": gates.min_improvement,
                       "passed": delta >= gates.min_improvement})

    if current.get("mode") == "chat":
        th = gates.thresholds
        for metric, minimum in (("template_adherence", th.template_adherence_min),
                                ("safety_unsafe_refusal_rate", th.safety_unsafe_refusal_min),
                                ("safety_safe_compliance_rate", th.safety_safe_compliance_min)):
            if metric in cur and cur[metric] is not None:
                checks.append({"check": "threshold", "metric": metric, "current": cur[metric], "min": minimum,
                               "passed": cur[metric] >= minimum})

    return {"stage": stage, "passed": all(c["passed"] for c in checks), "checks": checks,
            "on_failure": gates.on_failure}
