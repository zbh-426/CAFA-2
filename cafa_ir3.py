"""
CAFA++ pass1-lite - compact LM Studio auto-formulation with deterministic IR compiler.

Pipeline:
  description -> LLM structured JSON -> parse CAFA-IR-lite
              -> deterministic expression-to-Gurobi compiler -> execute Gurobi -> metrics

Design choice:
  To recover the strong behavior of the original CAFA prompt, the LLM still emits the familiar
  CAFA JSON fields: variables, objective, constraints, and a code string.  However, in this
  pass1-lite compiler version, the generated code is treated only as a scaffold/debug hint and is
  NOT executed.  The executable Gurobi program is generated deterministically from objective and
  constraints.

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

SYSTEM_PROMPT = """You are an expert OR modeler. Convert the problem into a compact CAFA-IR JSON.

The JSON should look like the original CAFA formulation JSON. It includes a code field only as a
modeling scaffold, but the downstream system will compile objective/constraints deterministically.

Reason in this order:
  1. Identify decision variables. Pick INTEGER for indivisible counts, CONTINUOUS for divisible
     quantities (acres, kg, hours, money, volume), BINARY for yes/no. Match real-world meaning.
  2. Extract the objective. Watch for hidden coefficients.
  3. Extract every constraint. Watch directions: at most/no more than -> <=, at least/no less than -> >=.
  4. Self-check: every number used? all directions correct? variable types real-world correct?

Output STRICT JSON only, matching this schema. No prose, no markdown fences:
{
  "problem_type": "LP" | "MILP" | "IP" | "BIP",
  "sense": "MAXIMIZE" | "MINIMIZE",
  "variables": [{"name": "...", "vtype": "CONTINUOUS"|"INTEGER"|"BINARY", "rationale": "..."}],
  "objective": "linear expression using short symbols such as x, y, z",
  "constraints": [{"expression": "lhs <= rhs", "source": "clause from problem"}],
  "code": "Gurobi Python using existing model `m`; this is a scaffold and should match the JSON expressions"
}

Expression rules:
  - Use the same symbols in objective, constraints, and code.
  - Use explicit <= or >=, never bare < or >.
  - Keep expressions linear. Do not use min(), max(), abs(), if, loops, or nonlinear terms.
  - Ratio constraints can be written naturally, e.g. y <= 2*x.

Code rules for the code field: assume gurobipy as gp and model m exist. Use m.addVar / m.setObjective / m.addConstr.
No imports, no env, no m.optimize(), no comments."""


VERIFIER_PROMPT = ("Review the formulation against the problem. Look for: wrong inequality direction, "
                   "swapped coefficients, missing constraint, wrong variable type. "
                   "Return the SAME JSON schema. If correct, return it unchanged. If wrong, fix it.")


FEW_SHOT = [
    {
        "q": ("A car manufacturer makes Oil Max and Oil Max Pro. Oil Max uses 46g of A, 43g of B, "
              "56g of C per container; Oil Max Pro uses 13g of A, 4g of B, 45g of C. Available: "
              "1345g A, 346g B, 1643g C. Profit: $10/Oil Max, $15/Oil Max Pro. Maximize profit."),
        "a": {
            "problem_type": "IP", "sense": "MAXIMIZE",
            "variables": [
                {"name": "Oil Max",     "vtype": "INTEGER", "rationale": "containers are indivisible"},
                {"name": "Oil Max Pro", "vtype": "INTEGER", "rationale": "containers are indivisible"},
            ],
            "objective": "10*x + 15*y",
            "constraints": [
                {"expression": "46*x + 13*y <= 1345", "source": "substance A"},
                {"expression": "43*x + 4*y <= 346",   "source": "substance B"},
                {"expression": "56*x + 45*y <= 1643", "source": "substance C"},
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
            "variables": [
                {"name": "apples", "vtype": "CONTINUOUS", "rationale": "acreage is divisible"},
                {"name": "pears",  "vtype": "CONTINUOUS", "rationale": "acreage is divisible"},
            ],
            "objective": "2*x + 4*y",
            "constraints": [
                {"expression": "x + y <= 50", "source": "total acres"},
                {"expression": "x >= 5",       "source": "apple minimum"},
                {"expression": "y >= 10",      "source": "pear minimum"},
                {"expression": "y <= 2*x",     "source": "pear-to-apple ratio"},
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
        "name": "cafa_ir_lite_formulation",
        "schema": {
            "type": "object",
            "properties": {
                "problem_type": {"type": "string", "enum": ["LP", "MILP", "IP", "BIP"]},
                "sense": {"type": "string", "enum": ["MAXIMIZE", "MINIMIZE"]},
                "variables": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "vtype": {"type": "string", "enum": ["CONTINUOUS", "INTEGER", "BINARY"]},
                            "rationale": {"type": "string"},
                        },
                        "required": ["name", "vtype", "rationale"],
                        "additionalProperties": False,
                    },
                },
                "objective": {"type": "string"},
                "constraints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                            "source": {"type": "string"},
                        },
                        "required": ["expression", "source"],
                        "additionalProperties": False,
                    },
                },
                "code": {"type": "string"},
            },
            "required": ["problem_type", "sense", "variables", "objective", "constraints", "code"],
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


def parse_ir_lite(raw: str) -> dict:
    """Parse model output into a normalized IR-lite dict.

    This is intentionally forgiving. If the JSON is valid and has objective/constraints,
    parser success should be high; algebra problems are left for compile_fail diagnostics.
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

    symbols = extract_symbol_order(data)
    if not symbols:
        # Last-resort fallback: create x, y, z by variable count. Compiler will fail if expressions are unusable.
        default = ["x", "y", "z", "w", "u", "v"]
        symbols = default[: len(data["variables"])]

    clean_vars: list[dict] = []
    for i, sym in enumerate(symbols):
        src = data["variables"][i] if i < len(data["variables"]) and isinstance(data["variables"][i], dict) else {}
        vid = str(src.get("id") or sym).strip()
        if not IDENT_RE.match(vid):
            vid = sym
        # Use expression symbol as the compiler id; keep src id only if it equals the expression symbol.
        vid = sym if IDENT_RE.match(sym) else vid
        vtype = str(src.get("vtype", "CONTINUOUS")).strip().upper()
        if vtype not in VALID_VTYPES:
            vtype = "CONTINUOUS"
        clean_vars.append({
            "id": vid,
            "name": str(src.get("name") or vid).strip(),
            "vtype": vtype,
            "rationale": str(src.get("rationale", "")).strip(),
        })

    clean_constraints: list[dict] = []
    for idx, c in enumerate(data["constraints"]):
        if not isinstance(c, dict):
            continue
        expr = normalize_expr(c.get("expression", ""))
        if not expr:
            continue
        clean_constraints.append({
            "name": safe_constr_name(str(c.get("name", f"c{idx+1}")), idx),
            "expression": expr,
            "source": str(c.get("source", "")).strip(),
        })
    if not clean_constraints:
        raise IRValidationError("no usable constraints")

    ir = {
        "ir_version": "cafa-ir-lite-v2",
        "problem_type": problem_type,
        "sense": sense,
        "variables": clean_vars,
        "objective": normalize_expr(data["objective"]),
        "constraints": clean_constraints,
        "code_hint": str(data.get("code", "")),  # ignored by compiler; useful for debugging/model scaffold
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
# Verifier (kept as later-work hook; disabled by default)
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
) -> Optional[dict]:
    """One verifier pass. Returns revised formulation dict, or None."""
    user = (f"PROBLEM:\n{description}\n\n"
            f"FORMULATION:\n{json.dumps(formulation, indent=2)}\n\n"
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
        "code_used": None,
        "code_hint": None,
        "obj_val": None,
        "status": "parse_fail",
        "error": None,
        "revised": False,
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
    if ir_path:
        os.makedirs(os.path.dirname(ir_path), exist_ok=True)
        with open(ir_path, "w", encoding="utf-8") as f:
            json.dump(ir, f, indent=2, ensure_ascii=False)

    # 3. deterministic IR-lite -> Gurobi compiler
    try:
        code = compile_ir_to_gurobi(ir, save_path=code_path)
    except Exception as e:
        rec["status"], rec["error"] = "compile_fail", str(e)
        return rec

    rec["code_used"] = code

    # 4. execute compiled code
    res = execute_code(code, save_path=code_path)
    rec.update(status=res["status"], obj_val=res["obj_val"], error=res["error"])

    # 5. optional verifier, kept for future use and disabled by default
    if enable_verifier and needs_verification(res):
        revised = verify(
            description, ir, res, model=model,
            backend=backend, api_url=api_url, api_key=api_key,
            json_mode=json_mode, timeout=timeout, max_retries=max_retries,
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
                rec["ir"] = rec["formulation"] = revised
                rec["code_hint"] = revised.get("code_hint")
                rec["code_used"] = revised_code
                rec.update(status=res2["status"], obj_val=res2["obj_val"], error=res2["error"])
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
    for i, r in enumerate(records):
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if r.get("revised"):
            revised += 1
        if r["status"] == "ok" and is_correct(r["obj_val"], r["ground_truth"], tol):
            correct += 1
            correct_ids.append(i)
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
    p.add_argument("--enable_verifier", action="store_true", default=False)
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
    summary["workflow"] = "cafa++-pass1-lite-expression-compiler"
    summary["model"] = args.model
    summary["backend"] = args.backend
    summary["verifier_enabled"] = args.enable_verifier
    print_report(summary)
    with open(os.path.join(base, "_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
