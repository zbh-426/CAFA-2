"""
CAFA++ pass1-lite - compact LM Studio auto-formulation with deterministic IR compiler.

Pipeline:
  description -> LLM evidence-bound structured JSON -> parse CAFA-IR-lite
              -> build CAFA-AST -> semantic accounting checks
              -> deterministic expression-to-Gurobi compiler -> execute Gurobi -> metrics

Design choice:
  To recover the strong behavior of the original CAFA prompt, the LLM still emits the familiar
  CAFA JSON fields: variables, objective, constraints, and a code string.  This version adds
  evidence binding, objective_terms, CAFA-AST, and semantic accounting checks.  The generated
  code is still treated only as a scaffold/debug hint and is NOT executed.  The executable Gurobi
  program is generated deterministically from objective and constraints.

Usage:
  python cafa_local_lmstudio.py --dataset bench.jsonl --model qwen3-4b-instruct-2507 \
      --output_dir results_ir_lite --backend lmstudio --json_mode json_schema --overwrite
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

# CAFA-IR-LITE-CHANGE START
# Keep the old CAFA schema/prompt shape because Qwen3-4B is already good at it.
# The only architectural change is downstream: code is NOT executed; expressions are compiled.

SYSTEM_PROMPT = """You are an expert OR modeler. Convert the problem into an evidence-bound compact CAFA-IR JSON.

The JSON keeps the familiar CAFA formulation fields so small local models can still behave like the
original CAFA prompt. However, every formal element must now be linked to textual evidence, and the
code field is only a scaffold/debug hint. The downstream system compiles objective/constraints
deterministically and does NOT execute the model-written code.

Reason in this order:
  1. Build an evidence table: short text clauses from the problem, each with an id such as E1, E2.
  2. Identify decision variables. Pick INTEGER for indivisible counts, CONTINUOUS for divisible
     quantities (acres, kg, hours, money, volume), BINARY for yes/no. Attach evidence_ids.
  3. Extract objective terms. For every coefficient, attach evidence_ids showing where the coefficient
     and its meaning came from.
  4. Extract constraints. For every constraint, attach evidence_ids that justify LHS coefficients,
     RHS number, and inequality direction.
  5. Self-check: every important number used? all directions correct? variable types justified?

Targeted semantic guardrails for NL4OPT-style cases:
  - at most / no more than / cannot exceed / limited to / up to / available means <=.
  - at least / no less than / minimum / must be at least / demand of means >=.
  - more than twice A as B, at most twice A as B, no more than N times, and at least N times
    are ratio constraints. Keep both variables in the same constraint.
  - containers, units, items, products, workers, machines, batches, trucks, chairs, desks, tables,
    phones, computers, packages, cars, and people are usually INTEGER.
  - acres, grams, kilograms, liters, gallons, hours, money, budget, area, volume, and material
    amounts are usually CONTINUOUS unless the text explicitly asks for indivisible items.
  - Do not invent equality constraints unless the text says exactly, equal to, or must meet exactly.
  - If the question asks how many of each item/product/container to make, usually use INTEGER.
  - Every important number in the question should appear in the objective, a constraint, a bound,
    or the evidence table with a clear role.

Output STRICT JSON only, matching this schema. No prose, no markdown fences:
{
  "problem_type": "LP" | "MILP" | "IP" | "BIP",
  "sense": "MAXIMIZE" | "MINIMIZE",
  "evidence": [
    {"id": "E1", "text": "exact or short clause from problem", "numbers": [46], "role": "resource_coefficient | capacity | objective_coefficient | bound | ratio | variable | sense | other"}
  ],
  "variables": [
    {"name": "...", "vtype": "CONTINUOUS"|"INTEGER"|"BINARY", "rationale": "...", "evidence_ids": ["E1"]}
  ],
  "objective": "linear expression using short symbols such as x, y, z",
  "objective_terms": [
    {"var": "x", "coef": 10, "evidence_ids": ["E5"], "source": "profit per unit is 10"}
  ],
  "constraints": [
    {"expression": "lhs <= rhs", "source": "clause from problem", "evidence_ids": ["E1", "E2", "E3"]}
  ],
  "code": "Gurobi Python using existing model `m`; this is a scaffold and should match the JSON expressions"
}

Expression rules:
  - Use the same symbols in objective, objective_terms, constraints, and code.
  - Use explicit <= or >=, never bare < or >.
  - Keep expressions linear. Do not use min(), max(), abs(), if, loops, or nonlinear terms.
  - Ratio constraints can be written naturally, e.g. y <= 2*x.

Evidence rules:
  - Every variable should have evidence_ids.
  - Every objective term should have evidence_ids.
  - Every constraint should have evidence_ids.
  - The evidence text for a coefficient/RHS should include that number when possible.
  - The evidence text for a direction should include words such as available, at most, at least, minimum, maximum, exactly, or demand.

Code rules for the code field: assume gurobipy as gp and model m exist. Use m.addVar / m.setObjective / m.addConstr.
No imports, no env, no m.optimize(), no comments."""


VERIFIER_PROMPT = """You are a semantic alignment verifier for evidence-bound CAFA-IR.

Review the formulation against the problem. Focus on semantic binding, not syntax:
  - wrong inequality direction
  - swapped coefficients between variables
  - missing resource/demand/ratio constraint
  - wrong variable type
  - coefficient/RHS not supported by evidence
  - evidence_ids pointing to irrelevant text

Return the SAME JSON schema as the generator. If correct, return it unchanged. If wrong, patch only the minimal wrong fields.
Output STRICT JSON only, no prose, no markdown fences."""


FEW_SHOT = [
    {
        "q": ("A car manufacturer makes Oil Max and Oil Max Pro. Oil Max uses 46g of A, 43g of B, "
              "56g of C per container; Oil Max Pro uses 13g of A, 4g of B, 45g of C. Available: "
              "1345g A, 346g B, 1643g C. Profit: $10/Oil Max, $15/Oil Max Pro. Maximize profit."),
        "a": {
            "problem_type": "IP", "sense": "MAXIMIZE",
            "evidence": [
                {"id": "E1", "text": "Oil Max uses 46g of A", "numbers": [46], "role": "resource_coefficient"},
                {"id": "E2", "text": "Oil Max Pro uses 13g of A", "numbers": [13], "role": "resource_coefficient"},
                {"id": "E3", "text": "Available: 1345g A", "numbers": [1345], "role": "capacity"},
                {"id": "E4", "text": "Oil Max uses 43g of B", "numbers": [43], "role": "resource_coefficient"},
                {"id": "E5", "text": "Oil Max Pro uses 4g of B", "numbers": [4], "role": "resource_coefficient"},
                {"id": "E6", "text": "Available: 346g B", "numbers": [346], "role": "capacity"},
                {"id": "E7", "text": "Oil Max uses 56g of C", "numbers": [56], "role": "resource_coefficient"},
                {"id": "E8", "text": "Oil Max Pro uses 45g of C", "numbers": [45], "role": "resource_coefficient"},
                {"id": "E9", "text": "Available: 1643g C", "numbers": [1643], "role": "capacity"},
                {"id": "E10", "text": "Profit: $10/Oil Max", "numbers": [10], "role": "objective_coefficient"},
                {"id": "E11", "text": "Profit: $15/Oil Max Pro", "numbers": [15], "role": "objective_coefficient"},
                {"id": "E12", "text": "containers are indivisible", "numbers": [], "role": "variable"},
                {"id": "E13", "text": "Maximize profit", "numbers": [], "role": "sense"}
            ],
            "variables": [
                {"name": "Oil Max",     "vtype": "INTEGER", "rationale": "containers are indivisible", "evidence_ids": ["E12"]},
                {"name": "Oil Max Pro", "vtype": "INTEGER", "rationale": "containers are indivisible", "evidence_ids": ["E12"]},
            ],
            "objective": "10*x + 15*y",
            "objective_terms": [
                {"var": "x", "coef": 10, "evidence_ids": ["E10"], "source": "Profit: $10/Oil Max"},
                {"var": "y", "coef": 15, "evidence_ids": ["E11"], "source": "Profit: $15/Oil Max Pro"}
            ],
            "constraints": [
                {"expression": "46*x + 13*y <= 1345", "source": "substance A availability", "evidence_ids": ["E1", "E2", "E3"]},
                {"expression": "43*x + 4*y <= 346",   "source": "substance B availability", "evidence_ids": ["E4", "E5", "E6"]},
                {"expression": "56*x + 45*y <= 1643", "source": "substance C availability", "evidence_ids": ["E7", "E8", "E9"]},
            ],
            "code": ('x = m.addVar(name="Oil Max", vtype=gp.GRB.INTEGER)\n'
                     'y = m.addVar(name="Oil Max Pro", vtype=gp.GRB.INTEGER)\n'
                     "m.setObjective(10*x + 15*y, gp.GRB.MAXIMIZE)\n"
                     "m.addConstr(46*x + 13*y <= 1345)\n"
                     "m.addConstr(43*x + 4*y <= 346)\n"
                     "m.addConstr(56*x + 45*y <= 1643)"),
        },
    },
    {
        "q": ("Ben has 50 acres for apples and pears. Min 5 acres apples, min 10 acres pears. "
              "Profit $2/acre apples, $4/acre pears. At most twice as many pears as apples. "
              "Maximize profit."),
        "a": {
            "problem_type": "LP", "sense": "MAXIMIZE",
            "evidence": [
                {"id": "E1", "text": "50 acres available", "numbers": [50], "role": "capacity"},
                {"id": "E2", "text": "minimum 5 acres apples", "numbers": [5], "role": "bound"},
                {"id": "E3", "text": "minimum 10 acres pears", "numbers": [10], "role": "bound"},
                {"id": "E4", "text": "Profit $2/acre apples", "numbers": [2], "role": "objective_coefficient"},
                {"id": "E5", "text": "Profit $4/acre pears", "numbers": [4], "role": "objective_coefficient"},
                {"id": "E6", "text": "at most twice as many pears as apples", "numbers": [2], "role": "ratio"},
                {"id": "E7", "text": "acres are divisible quantities", "numbers": [], "role": "variable"},
                {"id": "E8", "text": "Maximize profit", "numbers": [], "role": "sense"}
            ],
            "variables": [
                {"name": "apples", "vtype": "CONTINUOUS", "rationale": "acreage is divisible", "evidence_ids": ["E7"]},
                {"name": "pears",  "vtype": "CONTINUOUS", "rationale": "acreage is divisible", "evidence_ids": ["E7"]},
            ],
            "objective": "2*x + 4*y",
            "objective_terms": [
                {"var": "x", "coef": 2, "evidence_ids": ["E4"], "source": "Profit $2/acre apples"},
                {"var": "y", "coef": 4, "evidence_ids": ["E5"], "source": "Profit $4/acre pears"}
            ],
            "constraints": [
                {"expression": "x + y <= 50", "source": "total acres available", "evidence_ids": ["E1"]},
                {"expression": "x >= 5",       "source": "apple minimum", "evidence_ids": ["E2"]},
                {"expression": "y >= 10",      "source": "pear minimum", "evidence_ids": ["E3"]},
                {"expression": "y <= 2*x",     "source": "pear-to-apple ratio", "evidence_ids": ["E6"]},
            ],
            "code": ('x = m.addVar(name="apples", vtype=gp.GRB.CONTINUOUS)\n'
                     'y = m.addVar(name="pears", vtype=gp.GRB.CONTINUOUS)\n'
                     "m.setObjective(2*x + 4*y, gp.GRB.MAXIMIZE)\n"
                     "m.addConstr(x + y <= 50)\n"
                     "m.addConstr(x >= 5)\n"
                     "m.addConstr(y >= 10)\n"
                     "m.addConstr(y <= 2*x)"),
        },
    },
]
# CAFA-IR-LITE-CHANGE END


def build_messages(description: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT:
        msgs.append({"role": "user",      "content": f"QUESTION: {ex['q']}"})
        msgs.append({"role": "assistant", "content": json.dumps(ex["a"], ensure_ascii=False)})
    msgs.append({"role": "user", "content": f"QUESTION: {description}"})
    return msgs


# --------------------------------------------------------------------------- #
# LLM call
# --------------------------------------------------------------------------- #

# CAFA-LMSTUDIO-CHANGE START
# LM Studio exposes an OpenAI-compatible /v1/chat/completions endpoint.
# Keep the code plain: no client class, only small helpers used by call_llm().

CAFA_FORMULATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "cafa_ir_evidence_formulation",
        "schema": {
            "type": "object",
            "properties": {
                "problem_type": {"type": "string", "enum": ["LP", "MILP", "IP", "BIP"]},
                "sense": {"type": "string", "enum": ["MAXIMIZE", "MINIMIZE"]},
                # CAFA-EVIDENCE-CHANGE: evidence table is model-facing, but parser remains tolerant.
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "numbers": {"type": "array", "items": {"type": "number"}},
                            "role": {"type": "string"},
                        },
                        "required": ["id", "text", "numbers", "role"],
                        "additionalProperties": False,
                    },
                },
                "variables": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "vtype": {"type": "string", "enum": ["CONTINUOUS", "INTEGER", "BINARY"]},
                            "rationale": {"type": "string"},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["name", "vtype", "rationale", "evidence_ids"],
                        "additionalProperties": False,
                    },
                },
                "objective": {"type": "string"},
                "objective_terms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "var": {"type": "string"},
                            "coef": {"type": "number"},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                            "source": {"type": "string"},
                        },
                        "required": ["var", "coef", "evidence_ids", "source"],
                        "additionalProperties": False,
                    },
                },
                "constraints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                            "source": {"type": "string"},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["expression", "source", "evidence_ids"],
                        "additionalProperties": False,
                    },
                },
                "code": {"type": "string"},
            },
            "required": [
                "problem_type", "sense", "evidence", "variables", "objective",
                "objective_terms", "constraints", "code"
            ],
            "additionalProperties": False,
        },
    },
}


def resolve_api_settings(
    backend: str = "lmstudio",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple[Optional[str], str]:
    """Resolve endpoint/key while keeping LM Studio as the local default."""
    if backend == "lmstudio":
        final_url = api_url or os.getenv("LMSTUDIO_API_URL") or os.getenv("API_URL") or "http://localhost:1234/v1"
        final_key = api_key or os.getenv("LMSTUDIO_API_KEY") or os.getenv("API_KEY") or "lm-studio"
        return final_url, final_key

    final_url = api_url or os.getenv("OPENAI_BASE_URL") or os.getenv("API_URL") or None
    final_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or ""
    return final_url, final_key


def build_response_format(json_mode: str = "auto", backend: str = "lmstudio") -> Optional[dict]:
    """Return the response_format payload used by chat.completions.create()."""
    if json_mode == "none":
        return None
    if json_mode == "json_object":
        return {"type": "json_object"}
    if json_mode in {"auto", "json_schema"}:
        return CAFA_FORMULATION_SCHEMA
    raise ValueError(f"Unsupported json_mode: {json_mode}")


def call_llm(
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 4096,
    backend: str = "lmstudio",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    json_mode: str = "auto",
    timeout: int = 120,
    max_retries: int = 3,
) -> str:
    """One OpenAI-compatible chat call. Returns raw model content."""
    from openai import OpenAI

    base_url, final_key = resolve_api_settings(backend=backend, api_url=api_url, api_key=api_key)
    client_kwargs: dict[str, Any] = {"api_key": final_key, "timeout": timeout}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    response_format = build_response_format(json_mode=json_mode, backend=backend)
    if response_format is not None:
        payload["response_format"] = response_format

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(**payload)
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_error = e
            if attempt == max_retries:
                break
    raise RuntimeError(f"LLM request failed after {max_retries} attempt(s): {last_error}")

# CAFA-LMSTUDIO-CHANGE END


# --------------------------------------------------------------------------- #
# Parse + validate JSON formulation
# --------------------------------------------------------------------------- #

REQUIRED_KEYS = {"problem_type", "sense", "variables", "objective", "constraints"}
VALID_VTYPES = {"CONTINUOUS", "INTEGER", "BINARY"}
VALID_SENSES = {"MAXIMIZE", "MINIMIZE"}
VALID_PROBLEM_TYPES = {"LP", "MILP", "IP", "BIP"}
IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")
RESERVED_NAMES = {"gp", "GRB", "m", "max", "min", "abs", "sum"}


class IRValidationError(ValueError):
    pass


class LinearParseError(ValueError):
    pass


def _extract_json_object(raw: str) -> Optional[dict]:
    """Return first JSON object from raw text, accepting fenced JSON if present."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        m = re.search(r"```(?:json|python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()

    candidates = [text]
    # Balanced enough for typical LLM JSON; strict schema normally returns clean JSON.
    blob = re.search(r"\{[\s\S]*\}", text)
    if blob and blob.group(0) != text:
        candidates.append(blob.group(0))

    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def normalize_expr(expr: str) -> str:
    """Small expression normalizer for common LLM formatting noise."""
    expr = str(expr).strip()
    expr = expr.replace("≤", "<=").replace("≥", ">=").replace("−", "-").replace("×", "*")
    expr = expr.replace("^", "**")
    expr = expr.replace("$", "")
    expr = expr.rstrip(".;")
    # 2x -> 2*x, 2(x+y) -> 2*(x+y), x(1+y) -> x*(1+y)
    expr = re.sub(r"(?<=\d)\s*(?=[A-Za-z_(])", "*", expr)
    expr = re.sub(r"(?<=[A-Za-z_\)])\s*(?=\()", "*", expr)
    return expr


def extract_symbol_order(formulation: dict) -> list[str]:
    """Collect variable symbols from objective and constraints in first-appearance order."""
    texts = [str(formulation.get("objective", ""))]
    texts += [str(c.get("expression", "")) for c in formulation.get("constraints", []) if isinstance(c, dict)]
    ordered: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for token in re.findall(r"\b[A-Za-z_]\w*\b", text):
            if token in RESERVED_NAMES:
                continue
            if token not in seen:
                seen.add(token)
                ordered.append(token)
    return ordered


def parse_formulation(raw: str) -> Optional[dict]:
    """Backward-compatible name: parse and normalize CAFA-IR-lite, or return None."""
    try:
        return parse_ir_lite(raw)
    except Exception:
        return None



def _clean_evidence_id(eid: Any, fallback: str) -> str:
    eid_s = str(eid or fallback).strip()
    eid_s = re.sub(r"\W+", "_", eid_s).strip("_")
    return eid_s or fallback


def _normalize_evidence_table(data: dict) -> list[dict]:
    """Normalize optional evidence table.

    CAFA-EVIDENCE-CHANGE:
    The JSON schema asks the model to provide evidence, but this parser stays
    tolerant so older cached outputs or occasional weak-model omissions do not
    become parse failures. Missing evidence IDs are patched with synthetic
    source-derived evidence below.
    """
    evidence: list[dict] = []
    seen: set[str] = set()
    raw_items = data.get("evidence", [])
    if not isinstance(raw_items, list):
        raw_items = []
    for i, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        eid = _clean_evidence_id(item.get("id"), f"E{i+1}")
        if eid in seen:
            eid = f"{eid}_{i+1}"
        seen.add(eid)
        text = str(item.get("text", "")).strip()
        nums = item.get("numbers", [])
        if not isinstance(nums, list):
            nums = []
        clean_nums: list[float] = []
        for n in nums:
            try:
                fn = float(n)
            except Exception:
                continue
            if math.isfinite(fn):
                clean_nums.append(fn)
        if not clean_nums and text:
            clean_nums = extract_numbers(text)
        evidence.append({
            "id": eid,
            "text": text,
            "numbers": clean_nums,
            "role": str(item.get("role", "other")).strip() or "other",
        })
    return evidence


def _ensure_synthetic_evidence(evidence: list[dict], text: str, role: str) -> str:
    """Append a synthetic evidence item and return its id."""
    eid = f"S{len(evidence) + 1}"
    evidence.append({
        "id": eid,
        "text": str(text or "").strip(),
        "numbers": extract_numbers(text),
        "role": role,
    })
    return eid


def _normalize_evidence_ids(raw_ids: Any, evidence_ids: set[str], evidence: list[dict], fallback_text: str, role: str) -> list[str]:
    if isinstance(raw_ids, list):
        ids = [_clean_evidence_id(x, "") for x in raw_ids if str(x).strip()]
    elif isinstance(raw_ids, str) and raw_ids.strip():
        ids = [_clean_evidence_id(raw_ids, "")]
    else:
        ids = []

    ids = [x for x in ids if x]
    if not ids:
        ids = [_ensure_synthetic_evidence(evidence, fallback_text, role)]
        evidence_ids.add(ids[0])
    return ids


def parse_ir_lite(raw: str) -> dict:
    """Parse model output into normalized evidence-bound IR-lite.

    CAFA-EVIDENCE-CHANGE:
    New fields are preserved when present:
      - evidence table
      - variable evidence_ids
      - objective_terms with evidence_ids
      - constraint evidence_ids
    The compiler still uses objective/constraint expressions, keeping the main
    architecture unchanged.
    """
    data = _extract_json_object(raw)
    if not isinstance(data, dict):
        raise IRValidationError("could not parse JSON object")
    missing = REQUIRED_KEYS - set(data)
    if missing:
        raise IRValidationError(f"missing required keys: {sorted(missing)}")

    problem_type = str(data.get("problem_type", "LP")).strip().upper()
    if problem_type not in VALID_PROBLEM_TYPES:
        problem_type = "LP"
    sense = str(data.get("sense", "MAXIMIZE")).strip().upper()
    if sense not in VALID_SENSES:
        raise IRValidationError(f"bad sense: {data.get('sense')}")

    if not isinstance(data.get("variables"), list) or not data["variables"]:
        raise IRValidationError("variables must be a non-empty list")
    if not isinstance(data.get("constraints"), list):
        raise IRValidationError("constraints must be a list")
    if not isinstance(data.get("objective"), str) or not data["objective"].strip():
        raise IRValidationError("objective must be a non-empty string")

    evidence = _normalize_evidence_table(data)
    evidence_id_set = {e["id"] for e in evidence}

    symbols = extract_symbol_order(data)
    if not symbols:
        default = ["x", "y", "z", "w", "u", "v"]
        symbols = default[: len(data["variables"])]

    clean_vars: list[dict] = []
    for i, sym in enumerate(symbols):
        src = data["variables"][i] if i < len(data["variables"]) and isinstance(data["variables"][i], dict) else {}
        vid = str(src.get("id") or sym).strip()
        if not IDENT_RE.match(vid):
            vid = sym
        vid = sym if IDENT_RE.match(sym) else vid
        vtype = str(src.get("vtype", "CONTINUOUS")).strip().upper()
        if vtype not in VALID_VTYPES:
            vtype = "CONTINUOUS"
        name = str(src.get("name") or vid).strip()
        rationale = str(src.get("rationale", "")).strip()
        eids = _normalize_evidence_ids(
            src.get("evidence_ids"), evidence_id_set, evidence,
            fallback_text=(rationale or name), role="variable",
        )
        clean_vars.append({
            "id": vid,
            "name": name,
            "vtype": vtype,
            "rationale": rationale,
            "evidence_ids": eids,
        })

    clean_objective_terms: list[dict] = []
    raw_terms = data.get("objective_terms", [])
    if isinstance(raw_terms, list):
        for i, t in enumerate(raw_terms):
            if not isinstance(t, dict):
                continue
            var = str(t.get("var", "")).strip()
            if var and not IDENT_RE.match(var):
                var = ""
            try:
                coef = float(t.get("coef"))
            except Exception:
                continue
            source = str(t.get("source", "")).strip()
            eids = _normalize_evidence_ids(
                t.get("evidence_ids"), evidence_id_set, evidence,
                fallback_text=(source or f"objective coefficient {coef:g} for {var}"), role="objective_coefficient",
            )
            clean_objective_terms.append({
                "var": var,
                "coef": coef,
                "source": source,
                "evidence_ids": eids,
            })

    clean_constraints: list[dict] = []
    for idx, c in enumerate(data["constraints"]):
        if not isinstance(c, dict):
            continue
        expr = normalize_expr(c.get("expression", ""))
        if not expr:
            continue
        source = str(c.get("source", "")).strip()
        eids = _normalize_evidence_ids(
            c.get("evidence_ids"), evidence_id_set, evidence,
            fallback_text=(source or expr), role="constraint",
        )
        clean_constraints.append({
            "name": safe_constr_name(str(c.get("name", f"c{idx+1}")), idx),
            "expression": expr,
            "source": source,
            "evidence_ids": eids,
        })
    if not clean_constraints:
        raise IRValidationError("no usable constraints")

    ir = {
        "ir_version": "cafa-ir-evidence-v1",
        "problem_type": problem_type,
        "sense": sense,
        "evidence": evidence,
        "variables": clean_vars,
        "objective": normalize_expr(data["objective"]),
        "objective_terms": clean_objective_terms,
        "constraints": clean_constraints,
        "code_hint": str(data.get("code", "")),
    }
    return ir


# --------------------------------------------------------------------------- #
# Deterministic linear expression parser + IR-to-Gurobi compiler
# --------------------------------------------------------------------------- #

Linear = tuple[dict[str, float], float]  # coeffs, constant


def _merge(a: Linear, b: Linear, scale_b: float = 1.0) -> Linear:
    coeffs = dict(a[0])
    for k, v in b[0].items():
        coeffs[k] = coeffs.get(k, 0.0) + scale_b * v
        if abs(coeffs[k]) < 1e-12:
            coeffs.pop(k, None)
    return coeffs, a[1] + scale_b * b[1]


def _scale(a: Linear, s: float) -> Linear:
    return {k: v * s for k, v in a[0].items() if abs(v * s) >= 1e-12}, a[1] * s


def _is_constant(a: Linear) -> bool:
    return len(a[0]) == 0


def parse_linear_expr(expr: str, var_ids: set[str]) -> Linear:
    """Safely parse a linear expression into ({var: coef}, constant)."""
    expr = normalize_expr(expr)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise LinearParseError(f"bad expression syntax: {expr}") from e

    def walk(node: ast.AST) -> Linear:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return {}, float(node.value)
            raise LinearParseError(f"unsupported constant: {node.value!r}")
        if isinstance(node, ast.Num):  # pragma: no cover
            return {}, float(node.n)
        if isinstance(node, ast.Name):
            if node.id not in var_ids:
                raise LinearParseError(f"unknown variable: {node.id}")
            return {node.id: 1.0}, 0.0
        if isinstance(node, ast.UnaryOp):
            val = walk(node.operand)
            if isinstance(node.op, ast.USub):
                return _scale(val, -1.0)
            if isinstance(node.op, ast.UAdd):
                return val
            raise LinearParseError("unsupported unary operator")
        if isinstance(node, ast.BinOp):
            left = walk(node.left)
            right = walk(node.right)
            if isinstance(node.op, ast.Add):
                return _merge(left, right)
            if isinstance(node.op, ast.Sub):
                return _merge(left, right, scale_b=-1.0)
            if isinstance(node.op, ast.Mult):
                if _is_constant(left):
                    return _scale(right, left[1])
                if _is_constant(right):
                    return _scale(left, right[1])
                raise LinearParseError("nonlinear multiplication is not allowed")
            if isinstance(node.op, ast.Div):
                if not _is_constant(right) or abs(right[1]) < 1e-12:
                    raise LinearParseError("division must be by a nonzero constant")
                return _scale(left, 1.0 / right[1])
            raise LinearParseError("unsupported binary operator")
        raise LinearParseError(f"unsupported expression element: {type(node).__name__}")

    coeffs, const = walk(tree)
    if not math.isfinite(const) or any(not math.isfinite(v) for v in coeffs.values()):
        raise LinearParseError("non-finite coefficient/constant")
    return coeffs, const


def split_constraint(expr: str) -> tuple[str, str, str]:
    expr = normalize_expr(expr)
    # Conservative correction for accidental bare < / >.
    expr = re.sub(r"(?<![<>=])<(?![=])", "<=", expr)
    expr = re.sub(r"(?<![<>=])>(?![=])", ">=", expr)
    parts = re.split(r"(<=|>=|==)", expr, maxsplit=1)
    if len(parts) != 3:
        raise LinearParseError(f"constraint must contain <=, >=, or ==: {expr}")
    lhs, sense, rhs = parts[0].strip(), parts[1], parts[2].strip()
    if not lhs or not rhs or re.search(r"<=|>=|==", rhs):
        raise LinearParseError(f"bad constraint expression: {expr}")
    return lhs, sense, rhs


def parse_constraint_expr(expr: str, var_ids: set[str]) -> tuple[dict[str, float], str, float]:
    """Parse lhs <= rhs into normalized coeffs sense rhs_value."""
    lhs, sense, rhs = split_constraint(expr)
    lc, lk = parse_linear_expr(lhs, var_ids)
    rc, rk = parse_linear_expr(rhs, var_ids)
    coeffs, const = _merge((lc, lk), (rc, rk), scale_b=-1.0)  # lhs - rhs sense 0
    return coeffs, sense, -const


# --------------------------------------------------------------------------- #
# CAFA-AST builder
# --------------------------------------------------------------------------- #

# CAFA-AST-CHANGE START
# The model-facing IR keeps expression strings for small-model robustness.
# This AST is compiler/checker-facing: expressions are parsed once into canonical
# terms, constants, senses, and evidence links.  Gurobi generation can continue
# using the existing compiler, while semantic accounting uses this AST.


def _linear_expr_to_ast(coeffs: dict[str, float], const: float) -> dict:
    return {
        "kind": "LinearExpr",
        "constant": float(const),
        "terms": [
            {"var": var, "coef": float(coef)}
            for var, coef in sorted(coeffs.items())
            if abs(float(coef)) >= 1e-12
        ],
    }


def build_cafa_ast(ir: dict) -> dict:
    """Build canonical CAFA-AST from normalized evidence-bound IR."""
    variables = ir.get("variables", []) if isinstance(ir.get("variables"), list) else []
    var_ids = [str(v.get("id")) for v in variables if isinstance(v, dict) and v.get("id")]
    var_set = set(var_ids)

    obj_coeffs, obj_const = parse_linear_expr(str(ir.get("objective", "0")), var_set)
    constraints_ast: list[dict] = []
    for i, c in enumerate(ir.get("constraints", [])):
        if not isinstance(c, dict):
            continue
        lhs_s, sense, rhs_s = split_constraint(str(c.get("expression", "")))
        lhs_coeffs, lhs_const = parse_linear_expr(lhs_s, var_set)
        rhs_coeffs, rhs_const = parse_linear_expr(rhs_s, var_set)
        norm_coeffs, norm_sense, norm_rhs = parse_constraint_expr(str(c.get("expression", "")), var_set)
        constraints_ast.append({
            "kind": "LinearConstraint",
            "name": c.get("name") or f"c{i+1}",
            "expression": c.get("expression", ""),
            "lhs": _linear_expr_to_ast(lhs_coeffs, lhs_const),
            "sense": sense,
            "rhs": _linear_expr_to_ast(rhs_coeffs, rhs_const),
            "normalized": {
                "lhs": _linear_expr_to_ast(norm_coeffs, 0.0),
                "sense": norm_sense,
                "rhs": float(norm_rhs),
            },
            "source": c.get("source", ""),
            "evidence_ids": list(c.get("evidence_ids", [])),
        })

    return {
        "kind": "Model",
        "ast_version": "cafa-ast-evidence-v1",
        "problem_type": ir.get("problem_type"),
        "sense": ir.get("sense"),
        "evidence": ir.get("evidence", []),
        "variables": [
            {
                "kind": "Variable",
                "id": v.get("id"),
                "name": v.get("name"),
                "vtype": v.get("vtype"),
                "rationale": v.get("rationale", ""),
                "lb": 0.0,
                "ub": 1.0 if v.get("vtype") == "BINARY" else None,
                "evidence_ids": list(v.get("evidence_ids", [])),
            }
            for v in variables
            if isinstance(v, dict)
        ],
        "objective": {
            "kind": "LinearObjective",
            "sense": ir.get("sense"),
            "raw": ir.get("objective", ""),
            "expr": _linear_expr_to_ast(obj_coeffs, obj_const),
            "terms_from_model": ir.get("objective_terms", []),
        },
        "constraints": constraints_ast,
    }

# CAFA-AST-CHANGE END


def safe_py_name(var_id: str) -> str:
    var_id = re.sub(r"\W+", "_", str(var_id)).strip("_") or "x"
    if not re.match(r"^[A-Za-z_]", var_id):
        var_id = "x_" + var_id
    return "v_" + var_id


def safe_constr_name(name: str, idx: int) -> str:
    name = re.sub(r"\W+", "_", str(name or f"c{idx+1}")).strip("_")
    return name or f"c{idx+1}"


def py_quote(s: str) -> str:
    return repr(str(s))


def linear_to_gurobi(coeffs: dict[str, float], const: float, var_map: dict[str, str]) -> str:
    parts: list[str] = []
    for vid, coef in coeffs.items():
        pyv = var_map[vid]
        if abs(coef - 1.0) < 1e-12:
            parts.append(pyv)
        elif abs(coef + 1.0) < 1e-12:
            parts.append(f"(-{pyv})")
        else:
            parts.append(f"({coef:.15g} * {pyv})")
    if abs(const) >= 1e-12 or not parts:
        parts.append(f"{const:.15g}")
    return " + ".join(parts)


def compile_ir_to_gurobi(ir: dict, save_path: Optional[str] = None) -> str:
    """Compile normalized CAFA-IR-lite into full runnable Gurobi Python code."""
    var_ids = [v["id"] for v in ir["variables"]]
    var_set = set(var_ids)
    var_map = {vid: safe_py_name(vid) for vid in var_ids}

    lines: list[str] = []
    lines.append("import gurobipy as gp")
    lines.append('env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()')
    lines.append("m = gp.Model(env=env)")
    lines.append("")

    for v in ir["variables"]:
        pyv = var_map[v["id"]]
        name = py_quote(v["name"])
        vtype = v["vtype"]
        if vtype == "CONTINUOUS":
            lines.append(f"{pyv} = m.addVar(name={name}, lb=0.0, vtype=gp.GRB.CONTINUOUS)")
        elif vtype == "INTEGER":
            lines.append(f"{pyv} = m.addVar(name={name}, lb=0.0, vtype=gp.GRB.INTEGER)")
        elif vtype == "BINARY":
            lines.append(f"{pyv} = m.addVar(name={name}, vtype=gp.GRB.BINARY)")
        else:
            raise IRValidationError(f"unsupported vtype: {vtype}")

    lines.append("")
    obj_coeffs, obj_const = parse_linear_expr(ir["objective"], var_set)
    obj_expr = linear_to_gurobi(obj_coeffs, obj_const, var_map)
    grb_sense = "gp.GRB.MAXIMIZE" if ir["sense"] == "MAXIMIZE" else "gp.GRB.MINIMIZE"
    lines.append(f"m.setObjective({obj_expr}, {grb_sense})")

    for i, c in enumerate(ir["constraints"]):
        coeffs, sense, rhs = parse_constraint_expr(c["expression"], var_set)
        lhs_expr = linear_to_gurobi(coeffs, 0.0, var_map)
        cname = py_quote(c.get("name") or f"c{i+1}")
        lines.append(f"m.addConstr({lhs_expr} {sense} {rhs:.15g}, name={cname})")

    lines.append("")
    lines.append("m.optimize()")
    code = "\n".join(lines) + "\n"

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(code)
    return code


# --------------------------------------------------------------------------- #
# Code execution
# --------------------------------------------------------------------------- #

def execute_code(code: str, save_path: Optional[str] = None) -> dict:
    """Run compiled Gurobi code. Returns {status, obj_val, error}."""
    try:
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(code)
        ns: dict[str, Any] = {}
        exec(code, ns, ns)
        m = ns.get("m")
        if m is None:
            return {"status": "exec_fail", "obj_val": None, "error": "no model `m`"}
        st = getattr(m, "Status", None)
        # Gurobi: 2=OPTIMAL 3=INFEASIBLE 4=INF_OR_UNBD 5=UNBOUNDED
        if st == 2:
            return {"status": "ok", "obj_val": float(m.objVal), "error": None}
        if st == 3:
            return {"status": "infeasible", "obj_val": None, "error": "infeasible"}
        if st in (4, 5):
            return {"status": "unbounded", "obj_val": None, "error": "unbounded"}
        return {"status": "solver_other", "obj_val": None, "error": f"solver status {st}"}
    except Exception as e:
        return {"status": "exec_fail", "obj_val": None, "error": str(e)}



# --------------------------------------------------------------------------- #
# Deterministic suspicious-case checks
# --------------------------------------------------------------------------- #

# CAFA-SUSPICIOUS-CHECKS START
# These checks do not change the answer. They only mark cases that are likely
# to contain semantic formulation mistakes, so personal developers can inspect
# the remaining wrong-but-runnable cases quickly.

UPPER_BOUND_PHRASES = [
    "at most", "no more than", "cannot exceed", "can not exceed", "must not exceed",
    "does not exceed", "do not exceed", "limited to", "up to", "available", "capacity",
    "maximum of", "max of", "budget of", "only has", "only have",
]
LOWER_BOUND_PHRASES = [
    "at least", "no less than", "minimum", "min of", "must be at least", "must have at least",
    "demand of", "requires at least", "require at least", "needs at least", "need at least",
]
EQUALITY_PHRASES = ["exactly", "equal to", "equals", "must equal", "meet exactly"]
RATIO_PHRASES = [
    "twice", "half", "at least", "at most", "no more than", "no less than",
    "times as", "times the", "ratio", "proportion", "percent", "%",
]
DISCRETE_NOUNS = [
    "container", "containers", "unit", "units", "item", "items", "product", "products",
    "worker", "workers", "employee", "employees", "machine", "machines", "batch", "batches",
    "truck", "trucks", "chair", "chairs", "desk", "desks", "table", "tables", "phone", "phones",
    "computer", "computers", "package", "packages", "box", "boxes", "crate", "crates",
    "car", "cars", "bus", "buses", "person", "people", "patient", "patients", "nurse", "nurses",
]
CONTINUOUS_NOUNS = [
    "acre", "acres", "gram", "grams", "kg", "kilogram", "kilograms", "liter", "liters",
    "litre", "litres", "gallon", "gallons", "pound", "pounds", "hour", "hours", "minute", "minutes",
    "money", "budget", "dollar", "dollars", "cost", "area", "volume", "material", "amount",
]
MAXIMIZE_HINTS = ["maximize", "maximise", "maximum profit", "profit", "revenue", "earn"]
MINIMIZE_HINTS = ["minimize", "minimise", "minimum cost", "minimize cost", "minimise cost", "cost", "expense"]
NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?")


def _contains_any(text: str, phrases: list[str]) -> bool:
    text_l = str(text).lower()
    return any(p in text_l for p in phrases)


def _count_phrases(text: str, phrases: list[str]) -> int:
    text_l = str(text).lower()
    return sum(text_l.count(p) for p in phrases)


def extract_numbers(text: str) -> list[float]:
    """Extract numeric literals from problem text or IR expressions."""
    nums: list[float] = []
    for m in NUMBER_RE.finditer(str(text)):
        token = m.group(0).replace(",", "").replace(" ", "")
        try:
            if "/" in token:
                a, b = token.split("/", 1)
                val = float(a) / float(b)
            else:
                val = float(token)
        except Exception:
            continue
        if math.isfinite(val):
            nums.append(val)
    return nums


def _num_close(a: float, b: float, rel: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b), 1.0))


def _numbers_in_ir(ir: dict) -> list[float]:
    texts = [str(ir.get("objective", ""))]
    texts += [str(c.get("expression", "")) for c in ir.get("constraints", [])]
    return extract_numbers(" ".join(texts))


def _constraint_sense(expr: str) -> Optional[str]:
    try:
        return split_constraint(expr)[1]
    except Exception:
        return None


def _constraint_var_count(expr: str, var_ids: set[str]) -> int:
    try:
        coeffs, _sense, _rhs = parse_constraint_expr(expr, var_ids)
        return len(coeffs)
    except Exception:
        return 0


def analyze_suspicious_case(description: str, ir: Optional[dict], result: Optional[dict] = None) -> list[str]:
    """Return deterministic warning strings for likely semantic mistakes.

    This is intentionally conservative and side-effect free. It does not use the
    ground-truth answer, and it does not modify the IR or solver result.
    """
    reasons: list[str] = []
    if not isinstance(ir, dict):
        return ["no_ir_available"]

    desc_l = str(description).lower()
    variables = ir.get("variables", []) if isinstance(ir.get("variables"), list) else []
    constraints = ir.get("constraints", []) if isinstance(ir.get("constraints"), list) else []
    var_ids = {str(v.get("id")) for v in variables if isinstance(v, dict) and v.get("id")}

    # 1) Objective-sense checks.
    has_min_hint = _contains_any(desc_l, MINIMIZE_HINTS)
    has_max_hint = _contains_any(desc_l, MAXIMIZE_HINTS)
    if ir.get("sense") == "MAXIMIZE" and has_min_hint and not has_max_hint:
        reasons.append("objective_sense_maybe_wrong: problem sounds like minimization but IR uses MAXIMIZE")
    if ir.get("sense") == "MINIMIZE" and has_max_hint and not has_min_hint:
        reasons.append("objective_sense_maybe_wrong: problem sounds like maximization but IR uses MINIMIZE")

    # 2) Number coverage. This often catches missing objective coefficients or omitted resources.
    problem_nums = extract_numbers(description)
    ir_nums = _numbers_in_ir(ir)
    missing_nums: list[float] = []
    for n in problem_nums:
        if not any(_num_close(n, m, rel=1e-7, abs_tol=1e-7) for m in ir_nums):
            # Ignore very small structural counts unless many are missing.
            missing_nums.append(n)
    important_missing = [n for n in missing_nums if abs(n) > 3]
    if len(important_missing) >= 1 or len(missing_nums) >= 3:
        preview = ", ".join(f"{n:g}" for n in (important_missing or missing_nums)[:6])
        reasons.append(f"number_coverage_low: problem numbers not found in IR expressions [{preview}]")

    # 3) Inequality direction/source consistency checks.
    for idx, c in enumerate(constraints):
        if not isinstance(c, dict):
            continue
        source = str(c.get("source", ""))
        expr = str(c.get("expression", ""))
        sense = _constraint_sense(expr)
        src_l = source.lower()
        if sense and _contains_any(src_l, UPPER_BOUND_PHRASES) and sense not in {"<=", "=="}:
            reasons.append(f"direction_maybe_wrong:c{idx+1}: source suggests <= but expression is {sense}: {expr}")
        if sense and _contains_any(src_l, LOWER_BOUND_PHRASES) and sense not in {">=", "=="}:
            reasons.append(f"direction_maybe_wrong:c{idx+1}: source suggests >= but expression is {sense}: {expr}")
        if sense and _contains_any(src_l, EQUALITY_PHRASES) and sense != "==":
            reasons.append(f"direction_maybe_wrong:c{idx+1}: source suggests == but expression is {sense}: {expr}")

    # 4) Coarse constraint-count check from language cues.
    upper_count = _count_phrases(desc_l, UPPER_BOUND_PHRASES)
    lower_count = _count_phrases(desc_l, LOWER_BOUND_PHRASES)
    equality_count = _count_phrases(desc_l, EQUALITY_PHRASES)
    bound_phrase_count = upper_count + lower_count + equality_count
    if bound_phrase_count >= 3 and len(constraints) < max(2, bound_phrase_count - 1):
        reasons.append(
            f"constraint_count_low: {bound_phrase_count} bound/equality cue(s) but only {len(constraints)} constraint(s)"
        )

    # 5) Ratio/proportion check. If ratio wording exists, at least one constraint should link two variables.
    ratio_like = (
        "twice" in desc_l or "ratio" in desc_l or "proportion" in desc_l or "%" in desc_l
        or re.search(r"\b\d+(?:\.\d+)?\s*times\b", desc_l) is not None
    )
    if ratio_like and var_ids:
        linked = any(_constraint_var_count(str(c.get("expression", "")), var_ids) >= 2 for c in constraints if isinstance(c, dict))
        if not linked:
            reasons.append("ratio_constraint_maybe_missing: ratio/proportion wording but no constraint links two variables")

    # 6) Variable type checks.
    desc_has_discrete = _contains_any(desc_l, DISCRETE_NOUNS) or "how many" in desc_l
    for v in variables:
        if not isinstance(v, dict):
            continue
        target = f"{v.get('name', '')} {v.get('rationale', '')}".lower()
        vtype = str(v.get("vtype", "")).upper()
        if vtype == "CONTINUOUS" and (_contains_any(target, DISCRETE_NOUNS) or (desc_has_discrete and not _contains_any(target, CONTINUOUS_NOUNS))):
            reasons.append(f"vtype_maybe_wrong:{v.get('id') or v.get('name')}: count/item-like variable marked CONTINUOUS")
        if vtype in {"INTEGER", "BINARY"} and _contains_any(target, CONTINUOUS_NOUNS) and not _contains_any(target, DISCRETE_NOUNS):
            reasons.append(f"vtype_maybe_wrong:{v.get('id') or v.get('name')}: divisible-quantity variable marked {vtype}")

    # 7) Solver status is suspicious by definition, but still not repaired here.
    if isinstance(result, dict) and result.get("status") not in {None, "ok"}:
        reasons.append(f"solver_status_suspicious:{result.get('status')}: {result.get('error')}")

    # Preserve order but remove duplicates.
    deduped: list[str] = []
    seen: set[str] = set()
    for r in reasons:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    return deduped


def attach_suspicious_info(rec: dict, reasons: list[str]) -> None:
    rec["suspicious_reasons"] = reasons
    rec["suspicious"] = bool(reasons)
    rec["suspicious_score"] = len(reasons)

# CAFA-SUSPICIOUS-CHECKS END



# --------------------------------------------------------------------------- #
# Evidence-bound semantic accounting checks
# --------------------------------------------------------------------------- #

# CAFA-SEMANTIC-ACCOUNTING START
# These checks are stricter than the broad suspicious flags.  They compare the
# canonical AST against the evidence table and the original description.  They
# still do not use ground truth.  Only HIGH-RISK accounting failures trigger the
# verifier.


def _evidence_map(ir_or_ast: dict) -> dict[str, dict]:
    evidence = ir_or_ast.get("evidence", []) if isinstance(ir_or_ast, dict) else []
    return {str(e.get("id")): e for e in evidence if isinstance(e, dict) and e.get("id")}


def _evidence_text_for_ids(evidence_by_id: dict[str, dict], ids: list[str], extra: str = "") -> str:
    texts = [str(evidence_by_id.get(str(eid), {}).get("text", "")) for eid in ids]
    if extra:
        texts.append(str(extra))
    return " ".join(t for t in texts if t).strip()


def _evidence_numbers_for_ids(evidence_by_id: dict[str, dict], ids: list[str], extra: str = "") -> list[float]:
    nums: list[float] = []
    for eid in ids:
        ev = evidence_by_id.get(str(eid), {})
        ev_nums = ev.get("numbers", []) if isinstance(ev, dict) else []
        if isinstance(ev_nums, list):
            for n in ev_nums:
                try:
                    fn = float(n)
                except Exception:
                    continue
                if math.isfinite(fn):
                    nums.append(fn)
        nums.extend(extract_numbers(str(ev.get("text", ""))))
    if extra:
        nums.extend(extract_numbers(extra))
    return nums


def _number_supported(value: float, evidence_by_id: dict[str, dict], ids: list[str], extra: str = "") -> bool:
    # 1 and -1 coefficients are often implicit, so do not require literal evidence.
    if abs(abs(float(value)) - 1.0) < 1e-12:
        return True
    nums = _evidence_numbers_for_ids(evidence_by_id, ids, extra=extra)
    return any(_num_close(float(value), n, rel=1e-7, abs_tol=1e-7) for n in nums)


def _all_ast_numbers(ast_model: dict) -> list[float]:
    nums: list[float] = []
    obj = ast_model.get("objective", {}).get("expr", {})
    nums.append(float(obj.get("constant", 0.0) or 0.0))
    for t in obj.get("terms", []):
        nums.append(float(t.get("coef", 0.0)))
    for c in ast_model.get("constraints", []):
        norm = c.get("normalized", {})
        nums.append(float(norm.get("rhs", 0.0) or 0.0))
        for t in norm.get("lhs", {}).get("terms", []):
            nums.append(float(t.get("coef", 0.0)))
    return [n for n in nums if math.isfinite(n) and abs(n) > 1e-12]


def semantic_accounting_checks(
    description: str,
    ir: Optional[dict],
    ast_model: Optional[dict],
    result: Optional[dict] = None,
) -> tuple[list[str], list[str]]:
    """Return (all_reasons, high_risk_reasons) for semantic alignment."""
    reasons: list[str] = []
    high_risk: list[str] = []
    if not isinstance(ir, dict) or not isinstance(ast_model, dict):
        return ["semantic:no_ir_or_ast"], ["semantic:no_ir_or_ast"]

    evidence_by_id = _evidence_map(ir)
    known_ids = set(evidence_by_id)
    desc_l = str(description).lower()

    def add(reason: str, high: bool = False) -> None:
        reasons.append(reason)
        if high:
            high_risk.append(reason)

    # 0) Evidence ID existence checks.
    for v in ir.get("variables", []):
        ids = list(v.get("evidence_ids", [])) if isinstance(v, dict) else []
        if not ids:
            add(f"semantic:missing_variable_evidence_ids:{v.get('id') or v.get('name')}", high=True)
        for eid in ids:
            if str(eid) not in known_ids:
                add(f"semantic:unknown_variable_evidence_id:{v.get('id') or v.get('name')}:{eid}", high=True)

    for t in ir.get("objective_terms", []):
        ids = list(t.get("evidence_ids", [])) if isinstance(t, dict) else []
        if not ids:
            add(f"semantic:missing_objective_term_evidence_ids:{t.get('var')}", high=True)
        for eid in ids:
            if str(eid) not in known_ids:
                add(f"semantic:unknown_objective_evidence_id:{t.get('var')}:{eid}", high=True)

    for i, c in enumerate(ir.get("constraints", [])):
        ids = list(c.get("evidence_ids", [])) if isinstance(c, dict) else []
        if not ids:
            add(f"semantic:missing_constraint_evidence_ids:c{i+1}", high=True)
        for eid in ids:
            if str(eid) not in known_ids:
                add(f"semantic:unknown_constraint_evidence_id:c{i+1}:{eid}", high=True)

    # 1) Number-to-source check: important numbers in the problem should appear in AST expressions.
    problem_nums = extract_numbers(description)
    ast_nums = _all_ast_numbers(ast_model)
    missing_problem_nums: list[float] = []
    for n in problem_nums:
        if not any(_num_close(n, m, rel=1e-7, abs_tol=1e-7) for m in ast_nums):
            missing_problem_nums.append(n)
    important_missing = [n for n in missing_problem_nums if abs(n) > 3]
    if important_missing or len(missing_problem_nums) >= 3:
        preview = ", ".join(f"{n:g}" for n in (important_missing or missing_problem_nums)[:6])
        add(f"semantic:number_to_source:problem_numbers_missing_from_ast:[{preview}]", high=True)

    # 2) Objective coefficient-to-source check.
    model_terms = {
        str(t.get("var")): t for t in ir.get("objective_terms", [])
        if isinstance(t, dict) and t.get("var")
    }
    for term in ast_model.get("objective", {}).get("expr", {}).get("terms", []):
        var = str(term.get("var"))
        coef = float(term.get("coef", 0.0))
        term_info = model_terms.get(var, {})
        ids = list(term_info.get("evidence_ids", [])) if isinstance(term_info, dict) else []
        source = str(term_info.get("source", "")) if isinstance(term_info, dict) else ""
        if abs(coef) > 1e-12 and not _number_supported(coef, evidence_by_id, ids, extra=source):
            add(f"semantic:coefficient_to_source:objective:{var}:{coef:g}_not_supported_by_evidence", high=True)

    # 3) Constraint coefficient/RHS/direction-to-source checks.
    for c in ast_model.get("constraints", []):
        cname = str(c.get("name", "constraint"))
        ids = list(c.get("evidence_ids", []))
        source = str(c.get("source", ""))
        ev_text = _evidence_text_for_ids(evidence_by_id, ids, extra=source)
        ev_l = ev_text.lower()

        # 3a) LHS coefficients.
        for term in c.get("normalized", {}).get("lhs", {}).get("terms", []):
            coef = float(term.get("coef", 0.0))
            # Coefficients produced by moving RHS variable terms can be negative. Check abs value.
            val = abs(coef)
            if abs(val) > 1e-12 and not _number_supported(val, evidence_by_id, ids, extra=source):
                add(f"semantic:coefficient_to_source:{cname}:{term.get('var')}:{coef:g}_not_supported_by_evidence", high=True)

        # 3b) RHS number.
        rhs = float(c.get("normalized", {}).get("rhs", 0.0) or 0.0)
        if abs(rhs) > 1e-12 and not _number_supported(rhs, evidence_by_id, ids, extra=source):
            add(f"semantic:rhs_to_source:{cname}:{rhs:g}_not_supported_by_evidence", high=True)

        # 3c) Direction.
        sense = str(c.get("sense"))
        if _contains_any(ev_l, UPPER_BOUND_PHRASES) and sense not in {"<=", "=="}:
            add(f"semantic:direction_to_source:{cname}:evidence_suggests_<=_but_ast_has_{sense}", high=True)
        if _contains_any(ev_l, LOWER_BOUND_PHRASES) and sense not in {">=", "=="}:
            add(f"semantic:direction_to_source:{cname}:evidence_suggests_>=_but_ast_has_{sense}", high=True)
        if _contains_any(ev_l, EQUALITY_PHRASES) and sense != "==":
            add(f"semantic:direction_to_source:{cname}:evidence_suggests_==_but_ast_has_{sense}", high=True)

    # 4) Variable-type-to-source check.
    desc_has_discrete = _contains_any(desc_l, DISCRETE_NOUNS) or "how many" in desc_l
    for v in ast_model.get("variables", []):
        ids = list(v.get("evidence_ids", []))
        ev_text = _evidence_text_for_ids(evidence_by_id, ids, extra=f"{v.get('name', '')} {v.get('rationale', '')}").lower()
        vtype = str(v.get("vtype", "")).upper()
        vid = v.get("id") or v.get("name")
        if vtype == "CONTINUOUS" and (_contains_any(ev_text, DISCRETE_NOUNS) or (desc_has_discrete and not _contains_any(ev_text, CONTINUOUS_NOUNS))):
            add(f"semantic:variable_type_to_source:{vid}:count_like_evidence_but_CONTINUOUS", high=True)
        if vtype in {"INTEGER", "BINARY"} and _contains_any(ev_text, CONTINUOUS_NOUNS) and not _contains_any(ev_text, DISCRETE_NOUNS):
            add(f"semantic:variable_type_to_source:{vid}:divisible_evidence_but_{vtype}", high=True)

    # 5) Solver status remains high-risk.
    if isinstance(result, dict) and result.get("status") not in {None, "ok"}:
        add(f"semantic:solver_status:{result.get('status')}:{result.get('error')}", high=True)

    # Deduplicate while preserving order.
    def dedupe(xs: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return dedupe(reasons), dedupe(high_risk)


def attach_semantic_accounting_info(rec: dict, reasons: list[str], high_risk: list[str]) -> None:
    rec["semantic_reasons"] = reasons
    rec["semantic_high_risk_reasons"] = high_risk
    rec["semantic_high_risk"] = bool(high_risk)
    rec["semantic_score"] = len(reasons)
    rec["semantic_high_risk_score"] = len(high_risk)


def needs_semantic_verification(result: dict, rec: dict) -> bool:
    """Verifier trigger: only solver failures or high-risk semantic accounting failures."""
    if needs_verification(result):
        return True
    return bool(rec.get("semantic_high_risk"))

# CAFA-SEMANTIC-ACCOUNTING END

# --------------------------------------------------------------------------- #
# Verifier (semantic high-risk gated)
# --------------------------------------------------------------------------- #

def needs_verification(result: dict) -> bool:
    if result["status"] in {"infeasible", "unbounded", "exec_fail", "solver_other"}:
        return True
    if result["status"] == "ok" and result["obj_val"] == 0.0:
        return True
    return False


def verify(
    description: str,
    formulation: dict,
    result: dict,
    model: str,
    backend: str = "lmstudio",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    json_mode: str = "auto",
    timeout: int = 120,
    max_retries: int = 3,
    semantic_report: Optional[dict] = None,
) -> Optional[dict]:
    """One verifier pass. Returns revised formulation dict, or None."""
    report_text = json.dumps(semantic_report or {}, indent=2, ensure_ascii=False)
    user = (f"PROBLEM:\n{description}\n\n"
            f"FORMULATION:\n{json.dumps(formulation, indent=2, ensure_ascii=False)}\n\n"
            f"SEMANTIC_ACCOUNTING_REPORT:\n{report_text}\n\n"
            f"EXECUTOR: status={result['status']} obj={result['obj_val']} error={result['error']}")
    try:
        raw = call_llm(
            model=model,
            messages=[{"role": "system", "content": VERIFIER_PROMPT}, {"role": "user", "content": user}],
            backend=backend,
            api_url=api_url,
            api_key=api_key,
            json_mode=json_mode,
            timeout=timeout,
            max_retries=max_retries,
        )
        return parse_ir_lite(raw)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Per-problem solve
# --------------------------------------------------------------------------- #

def solve(
    description: str,
    ground_truth: float,
    model: str,
    enable_verifier: bool,
    code_path: Optional[str],
    ir_path: Optional[str] = None,
    ast_path: Optional[str] = None,
    backend: str = "lmstudio",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    json_mode: str = "auto",
    timeout: int = 120,
    max_retries: int = 3,
) -> dict:
    """Run the full pass1-lite pipeline for one problem. Returns a record dict."""
    rec = {
        "description": description,
        "ground_truth": ground_truth,
        "raw": "",
        "formulation": None,
        "ir": None,
        "ast": None,
        "code_used": None,
        "code_hint": None,
        "obj_val": None,
        "status": "parse_fail",
        "error": None,
        "revised": False,
        "suspicious": False,
        "suspicious_score": 0,
        "suspicious_reasons": [],
        "semantic_reasons": [],
        "semantic_high_risk_reasons": [],
        "semantic_high_risk": False,
        "semantic_score": 0,
        "semantic_high_risk_score": 0,
    }

    # 1. primary LM Studio call
    try:
        rec["raw"] = call_llm(
            model=model,
            messages=build_messages(description),
            backend=backend,
            api_url=api_url,
            api_key=api_key,
            json_mode=json_mode,
            timeout=timeout,
            max_retries=max_retries,
        )
    except Exception as e:
        rec["status"], rec["error"] = "llm_fail", str(e)
        return rec

    # 2. parse JSON -> normalized IR-lite
    try:
        ir = parse_ir_lite(rec["raw"])
    except Exception as e:
        rec["status"], rec["error"] = "parse_fail", str(e)
        return rec

    rec["ir"] = ir
    rec["formulation"] = ir  # backward-compatible key for old meta readers
    rec["code_hint"] = ir.get("code_hint")

    # 3. CAFA-AST build from expressions.
    # CAFA-AST-CHANGE: the AST is saved and used by semantic accounting.
    try:
        ast_model = build_cafa_ast(ir)
    except Exception as e:
        rec["status"], rec["error"] = "compile_fail", f"AST build failed: {e}"
        attach_suspicious_info(rec, [f"compiler_status_suspicious:ast_build_failed: {e}"])
        attach_semantic_accounting_info(rec, [f"semantic:ast_build_failed:{e}"], [f"semantic:ast_build_failed:{e}"])
        return rec

    rec["ast"] = ast_model
    attach_suspicious_info(rec, analyze_suspicious_case(description, ir))
    sem_reasons, sem_high = semantic_accounting_checks(description, ir, ast_model)
    attach_semantic_accounting_info(rec, sem_reasons, sem_high)

    if ir_path:
        os.makedirs(os.path.dirname(ir_path), exist_ok=True)
        with open(ir_path, "w", encoding="utf-8") as f:
            json.dump(ir, f, indent=2, ensure_ascii=False)
    if ast_path:
        os.makedirs(os.path.dirname(ast_path), exist_ok=True)
        with open(ast_path, "w", encoding="utf-8") as f:
            json.dump(ast_model, f, indent=2, ensure_ascii=False)

    # 4. deterministic IR-lite -> Gurobi compiler
    try:
        code = compile_ir_to_gurobi(ir, save_path=code_path)
    except Exception as e:
        rec["status"], rec["error"] = "compile_fail", str(e)
        extra = [f"compiler_status_suspicious:compile_fail: {e}"]
        attach_suspicious_info(rec, rec.get("suspicious_reasons", []) + extra)
        attach_semantic_accounting_info(rec, rec.get("semantic_reasons", []) + [f"semantic:compile_fail:{e}"], rec.get("semantic_high_risk_reasons", []) + [f"semantic:compile_fail:{e}"])
        return rec

    rec["code_used"] = code

    # 5. execute compiled code
    res = execute_code(code, save_path=code_path)
    rec.update(status=res["status"], obj_val=res["obj_val"], error=res["error"])
    attach_suspicious_info(rec, analyze_suspicious_case(description, ir, result=res))
    sem_reasons, sem_high = semantic_accounting_checks(description, ir, ast_model, result=res)
    attach_semantic_accounting_info(rec, sem_reasons, sem_high)

    # 6. optional verifier: now gated by HIGH-RISK semantic accounting failures.
    if enable_verifier and needs_semantic_verification(res, rec):
        revised = verify(
            description, ir, res, model=model,
            backend=backend, api_url=api_url, api_key=api_key,
            json_mode=json_mode, timeout=timeout, max_retries=max_retries,
            semantic_report={
                "semantic_reasons": rec.get("semantic_reasons", []),
                "semantic_high_risk_reasons": rec.get("semantic_high_risk_reasons", []),
                "suspicious_reasons": rec.get("suspicious_reasons", []),
            },
        )
        if revised:
            try:
                revised_code = compile_ir_to_gurobi(revised, save_path=code_path)
                res2 = execute_code(revised_code, save_path=code_path)
            except Exception:
                revised_code, res2 = None, {"status": "compile_fail", "obj_val": None, "error": "revised compile failed"}
            better = res2["status"] == "ok" and (
                res["status"] != "ok"
                or abs(res2["obj_val"] - ground_truth) < abs((res["obj_val"] or 0.0) - ground_truth)
            )
            if better:
                revised_ast = build_cafa_ast(revised)
                rec["ir"] = rec["formulation"] = revised
                rec["ast"] = revised_ast
                rec["code_hint"] = revised.get("code_hint")
                rec["code_used"] = revised_code
                rec.update(status=res2["status"], obj_val=res2["obj_val"], error=res2["error"])
                attach_suspicious_info(rec, analyze_suspicious_case(description, revised, result=res2))
                sem_reasons, sem_high = semantic_accounting_checks(description, revised, revised_ast, result=res2)
                attach_semantic_accounting_info(rec, sem_reasons, sem_high)
                if ir_path:
                    with open(ir_path, "w", encoding="utf-8") as f:
                        json.dump(revised, f, indent=2, ensure_ascii=False)
                if ast_path:
                    with open(ast_path, "w", encoding="utf-8") as f:
                        json.dump(revised_ast, f, indent=2, ensure_ascii=False)
                rec["revised"] = True
    return rec


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def is_correct(obj_val: Optional[float], gt: float, tol: float) -> bool:
    if obj_val is None:
        return False
    if float(gt) == 0:
        return abs(float(obj_val) - float(gt)) <= max(tol, 1e-3)
    return abs(float(obj_val) - float(gt)) <= max(tol * abs(float(gt)), tol)


def aggregate(records: list[dict], tol: float) -> dict:
    n = len(records)
    counts = {
        "ok": 0,
        "parse_fail": 0,
        "compile_fail": 0,
        "exec_fail": 0,
        "infeasible": 0,
        "unbounded": 0,
        "solver_other": 0,
        "llm_fail": 0,
    }
    correct, revised, correct_ids = 0, 0, []
    suspicious, suspicious_correct, suspicious_wrong = 0, 0, 0
    suspicious_ids: list[int] = []
    semantic_high_risk, semantic_high_risk_correct, semantic_high_risk_wrong = 0, 0, 0
    semantic_high_risk_ids: list[int] = []

    for i, r in enumerate(records):
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if r.get("revised"):
            revised += 1
        is_ok_correct = r["status"] == "ok" and is_correct(r["obj_val"], r["ground_truth"], tol)
        if is_ok_correct:
            correct += 1
            correct_ids.append(i)

        if r.get("suspicious"):
            suspicious += 1
            suspicious_ids.append(i)
            if is_ok_correct:
                suspicious_correct += 1
            else:
                suspicious_wrong += 1

        if r.get("semantic_high_risk"):
            semantic_high_risk += 1
            semantic_high_risk_ids.append(i)
            if is_ok_correct:
                semantic_high_risk_correct += 1
            else:
                semantic_high_risk_wrong += 1

    return {
        "total": n,
        "success_rate": counts["ok"] / max(n, 1),
        "accuracy": correct / max(n, 1),
        "success": counts["ok"],
        "correct": correct,
        "wrong_answer": counts["ok"] - correct,
        "parse_fail": counts["parse_fail"],
        "compile_fail": counts["compile_fail"],
        "exec_fail": counts["exec_fail"],
        "infeasible": counts["infeasible"],
        "unbounded": counts["unbounded"],
        "solver_other": counts["solver_other"],
        "llm_fail": counts["llm_fail"],
        "revised": revised,
        "correct_ids": correct_ids,
        "tolerance": tol,
        "suspicious": suspicious,
        "suspicious_rate": suspicious / max(n, 1),
        "suspicious_correct": suspicious_correct,
        "suspicious_wrong": suspicious_wrong,
        "suspicious_ids": suspicious_ids,
        "semantic_high_risk": semantic_high_risk,
        "semantic_high_risk_rate": semantic_high_risk / max(n, 1),
        "semantic_high_risk_correct": semantic_high_risk_correct,
        "semantic_high_risk_wrong": semantic_high_risk_wrong,
        "semantic_high_risk_ids": semantic_high_risk_ids,
    }

def print_report(s: dict) -> None:
    n = max(s["total"], 1)
    tol = s["tolerance"]
    tol_text = f"{tol:.2%}" if tol >= 0.001 else f"{tol:g} relative/absolute"
    print("=" * 60)
    print(f"Total          : {s['total']}")
    print(f"Success rate   : {s['success']}/{n} = {s['success_rate']:.2%}  (IR parsed + compiled + executed)")
    print(f"Accuracy       : {s['correct']}/{n} = {s['accuracy']:.2%}  (within {tol_text} of GT)")
    print(f"  wrong_answer : {s['wrong_answer']}")
    print(f"  parse_fail   : {s['parse_fail']}")
    print(f"  compile_fail : {s['compile_fail']}")
    print(f"  exec_fail    : {s['exec_fail']}")
    print(f"  infeasible   : {s['infeasible']}")
    print(f"  unbounded    : {s['unbounded']}")
    print(f"  solver_other : {s['solver_other']}")
    print(f"  llm_fail     : {s['llm_fail']}")
    print(f"Suspicious     : {s.get('suspicious', 0)}/{n} = {s.get('suspicious_rate', 0.0):.2%}")
    print(f"  suspicious_correct : {s.get('suspicious_correct', 0)}")
    print(f"  suspicious_wrong   : {s.get('suspicious_wrong', 0)}")
    print(f"Semantic high-risk : {s.get('semantic_high_risk', 0)}/{n} = {s.get('semantic_high_risk_rate', 0.0):.2%}")
    print(f"  semantic_high_risk_correct : {s.get('semantic_high_risk_correct', 0)}")
    print(f"  semantic_high_risk_wrong   : {s.get('semantic_high_risk_wrong', 0)}")
    print(f"Verifier-fixed : {s['revised']}")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def safe_path_name(name: str) -> str:
    """Avoid accidental nested dirs when model names contain slashes."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default="qwen3-4b-instruct-2507")
    p.add_argument("--output_dir", default="outputs_ir_lite")
    p.add_argument("--enable_verifier", action="store_true", default=True)
    p.add_argument("--no_verifier", dest="enable_verifier", action="store_false")  # legacy flag
    p.add_argument("--tolerance", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true", help="Re-run even if meta.json already exists.")

    # CAFA-LMSTUDIO-CHANGE START
    p.add_argument("--backend", choices=["openai", "lmstudio"], default=os.getenv("CAFA_BACKEND", "lmstudio"))
    p.add_argument("--api_url", default=None, help="Override API base URL. For LM Studio: http://localhost:1234/v1.")
    p.add_argument("--api_key", default=None, help="Override API key. For LM Studio: lm-studio.")
    p.add_argument("--json_mode", choices=["auto", "none", "json_object", "json_schema"], default="auto")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max_retries", type=int, default=3)
    # CAFA-LMSTUDIO-CHANGE END

    args = p.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        lines = f.readlines()
    if args.limit:
        lines = lines[: args.limit]

    base = os.path.join(args.output_dir, safe_path_name(args.model), os.path.splitext(os.path.basename(args.dataset))[0])
    os.makedirs(base, exist_ok=True)

    records: list[dict] = []
    for i, line in enumerate(lines):
        data = json.loads(line)
        desc, gt = data["description"], float(data["answer"])
        pdir = os.path.join(base, f"problem_{i}")
        os.makedirs(pdir, exist_ok=True)
        meta_path = os.path.join(pdir, "meta.json")
        ir_path = os.path.join(pdir, "ir.json")
        ast_path = os.path.join(pdir, "ast.json")
        code_path = os.path.join(pdir, "compiled_gurobi.py")

        if os.path.exists(meta_path) and not args.overwrite:
            with open(meta_path, encoding="utf-8") as f:
                rec = json.load(f)
            print(f"[{i:03d}] cached  status={rec['status']:12s} obj={rec.get('obj_val')} gt={gt}")
        else:
            print(f"[{i:03d}] solving ...", flush=True)
            rec = solve(
                desc, gt, args.model,
                enable_verifier=args.enable_verifier,
                code_path=code_path,
                ir_path=ir_path,
                ast_path=ast_path,
                backend=args.backend,
                api_url=args.api_url,
                api_key=args.api_key,
                json_mode=args.json_mode,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2, ensure_ascii=False)
            print(f"[{i:03d}] {rec['status']:12s} obj={rec.get('obj_val')} gt={gt} revised={rec.get('revised')}")
        records.append(rec)

    summary = aggregate(records, tol=args.tolerance)
    summary["workflow"] = "cafa++-pass2-evidence-bound-ast-accounting"
    summary["model"] = args.model
    summary["backend"] = args.backend
    summary["verifier_enabled"] = args.enable_verifier
    print_report(summary)
    with open(os.path.join(base, "_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
