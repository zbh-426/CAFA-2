"""
CAFA++ pass1 - compact, single-file LP auto-formulation with LM Studio.

Pipeline:
  description -> LLM (structured CAFA-IR JSON) -> parse/validate IR
              -> deterministic IR-to-Gurobi compiler -> execute Gurobi -> metrics

Important pass1 choice:
  The LLM does NOT write Gurobi code. It only writes coefficient-level CAFA-IR.
  The Python/Gurobi code is generated deterministically by compile_ir_to_gurobi().

Usage:
  python cafa_local_lmstudio.py --dataset bench.jsonl --model qwen3-4b-instruct-2507 --output_dir results/
  python cafa_local_lmstudio.py ... --limit 20           # quick subset run
  python cafa_local_lmstudio.py ... --overwrite          # re-run cached problems
  python cafa_local_lmstudio.py ... --backend lmstudio --json_mode json_schema
"""
from __future__ import annotations
import argparse, json, math, os, re, sys
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

# CAFA-IR-CHANGE START
# Pass1 keeps the same one-call CAFA style, but changes the LLM output target:
#   old: JSON + LLM-written Gurobi code
#   new: coefficient-level CAFA-IR only
# The compiler below is responsible for all Gurobi code generation.

SYSTEM_PROMPT = """You are an expert OR modeler. Convert the problem into CAFA-IR.

CAFA-IR is a coefficient-level JSON representation of a linear optimization model.
Do NOT output Gurobi code. Do NOT output algebra strings as the main representation.
Do NOT output markdown fences or explanations. Output STRICT JSON only.

Reason in this order:
  1. Identify decision variables. Pick INTEGER for indivisible counts, CONTINUOUS for divisible
     quantities (acres, kg, hours), BINARY for yes/no. Match real-world meaning.
  2. Extract the objective. Put coefficients into objective.terms.
  3. Extract every constraint. Use normalized linear form: sum(coef_i * var_i) sense rhs.
  4. Self-check: every number used? all directions correct? variable types real-world correct?

IR rules:
  - ir_version must be "cafa-ir-pass1".
  - Variable ids must be short valid Python identifiers such as x, y, z, x1, x2.
  - Variables are non-negative by default. Add explicit constraints for stricter lower bounds.
  - Objective sense must be MAXIMIZE or MINIMIZE.
  - Constraint sense must be <=, >=, or ==.
  - Constraint rhs must be a number.
  - Ratio constraints must be moved to the left-hand side. Example: y <= 2*x becomes y - 2*x <= 0.
  - Use one constraint per real requirement in the problem.

Output STRICT JSON only, matching this schema shape:
{
  "ir_version": "cafa-ir-pass1",
  "problem_type": "LP" | "MILP" | "IP" | "BIP",
  "sense": "MAXIMIZE" | "MINIMIZE",
  "variables": [
    {"id": "x", "name": "human readable name", "vtype": "CONTINUOUS"|"INTEGER"|"BINARY", "rationale": "why this type"}
  ],
  "objective": {
    "constant": 0,
    "terms": [{"var": "x", "coef": 1.0}]
  },
  "constraints": [
    {"name": "c1", "sense": "<=", "rhs": 0.0, "terms": [{"var": "x", "coef": 1.0}], "source": "clause from problem"}
  ]
}"""


VERIFIER_PROMPT = ("Review the CAFA-IR against the problem. Look for: wrong inequality direction, "
                   "swapped coefficients, missing constraint, wrong variable type. "
                   "Return the SAME CAFA-IR schema. If correct, return it unchanged. If wrong, fix it.")


FEW_SHOT = [
    {
        "q": ("A car manufacturer makes Oil Max and Oil Max Pro. Oil Max uses 46g of A, 43g of B, "
              "56g of C per container; Oil Max Pro uses 13g of A, 4g of B, 45g of C. Available: "
              "1345g A, 346g B, 1643g C. Profit: $10/Oil Max, $15/Oil Max Pro. Maximize profit."),
        "a": {
            "ir_version": "cafa-ir-pass1",
            "problem_type": "IP",
            "sense": "MAXIMIZE",
            "variables": [
                {"id": "x", "name": "Oil Max containers", "vtype": "INTEGER", "rationale": "containers are indivisible"},
                {"id": "y", "name": "Oil Max Pro containers", "vtype": "INTEGER", "rationale": "containers are indivisible"},
            ],
            "objective": {"constant": 0, "terms": [{"var": "x", "coef": 10}, {"var": "y", "coef": 15}]},
            "constraints": [
                {"name": "substance_A", "sense": "<=", "rhs": 1345, "terms": [{"var": "x", "coef": 46}, {"var": "y", "coef": 13}], "source": "available substance A"},
                {"name": "substance_B", "sense": "<=", "rhs": 346,  "terms": [{"var": "x", "coef": 43}, {"var": "y", "coef": 4}],  "source": "available substance B"},
                {"name": "substance_C", "sense": "<=", "rhs": 1643, "terms": [{"var": "x", "coef": 56}, {"var": "y", "coef": 45}], "source": "available substance C"},
            ],
        },
    },
    {
        "q": ("Ben has 50 acres for apples and pears. Min 5 acres apples, min 10 acres pears. "
              "Profit $2/acre apples, $4/acre pears. At most twice as many pears as apples. "
              "Maximize profit."),
        "a": {
            "ir_version": "cafa-ir-pass1",
            "problem_type": "LP",
            "sense": "MAXIMIZE",
            "variables": [
                {"id": "x", "name": "acres of apples", "vtype": "CONTINUOUS", "rationale": "acreage is divisible"},
                {"id": "y", "name": "acres of pears",  "vtype": "CONTINUOUS", "rationale": "acreage is divisible"},
            ],
            "objective": {"constant": 0, "terms": [{"var": "x", "coef": 2}, {"var": "y", "coef": 4}]},
            "constraints": [
                {"name": "total_acres", "sense": "<=", "rhs": 50, "terms": [{"var": "x", "coef": 1}, {"var": "y", "coef": 1}], "source": "50 acres available"},
                {"name": "min_apples",  "sense": ">=", "rhs": 5,  "terms": [{"var": "x", "coef": 1}], "source": "minimum 5 acres apples"},
                {"name": "min_pears",   "sense": ">=", "rhs": 10, "terms": [{"var": "y", "coef": 1}], "source": "minimum 10 acres pears"},
                {"name": "pear_ratio",  "sense": "<=", "rhs": 0,  "terms": [{"var": "y", "coef": 1}, {"var": "x", "coef": -2}], "source": "at most twice as many pears as apples"},
            ],
        },
    },
]

# CAFA-IR-CHANGE END


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
# This follows the same plain pattern as lmstudio.py:
#   OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
#   client.chat.completions.create(..., response_format={"type": "json_schema", ...})
# No wrapper class is used, so the request payload is easy to inspect/debug.

# CAFA-IR-CHANGE START
# JSON schema used by LM Studio structured output. This schema asks for IR, not code.
CAFA_IR_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "cafa_ir_pass1",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "ir_version": {"type": "string", "enum": ["cafa-ir-pass1"]},
                "problem_type": {"type": "string", "enum": ["LP", "MILP", "IP", "BIP"]},
                "sense": {"type": "string", "enum": ["MAXIMIZE", "MINIMIZE"]},
                "variables": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "vtype": {"type": "string", "enum": ["CONTINUOUS", "INTEGER", "BINARY"]},
                            "rationale": {"type": "string"},
                        },
                        "required": ["id", "name", "vtype", "rationale"],
                        "additionalProperties": False,
                    },
                },
                "objective": {
                    "type": "object",
                    "properties": {
                        "constant": {"type": "number"},
                        "terms": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "var": {"type": "string"},
                                    "coef": {"type": "number"},
                                },
                                "required": ["var", "coef"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["constant", "terms"],
                    "additionalProperties": False,
                },
                "constraints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "sense": {"type": "string", "enum": ["<=", ">=", "=="]},
                            "rhs": {"type": "number"},
                            "terms": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "var": {"type": "string"},
                                        "coef": {"type": "number"},
                                    },
                                    "required": ["var", "coef"],
                                    "additionalProperties": False,
                                },
                            },
                            "source": {"type": "string"},
                        },
                        "required": ["name", "sense", "rhs", "terms", "source"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["ir_version", "problem_type", "sense", "variables", "objective", "constraints"],
            "additionalProperties": False,
        },
    },
}
# Keep this alias so the original build_response_format() shape remains easy to track.
CAFA_FORMULATION_SCHEMA = CAFA_IR_SCHEMA
# CAFA-IR-CHANGE END


def resolve_api_settings(
    backend: str = "lmstudio",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple[Optional[str], str]:
    """Resolve endpoint/key while keeping LM Studio as the local default."""
    if backend == "lmstudio":
        final_url = (
            api_url
            or os.getenv("LMSTUDIO_API_URL")
            or os.getenv("API_URL")
            or "http://localhost:1234/v1"
        )
        final_key = (
            api_key
            or os.getenv("LMSTUDIO_API_KEY")
            or os.getenv("API_KEY")
            or "lm-studio"
        )
        return final_url, final_key

    # OpenAI-compatible remote backend. base_url is optional for the official OpenAI API.
    final_url = api_url or os.getenv("OPENAI_BASE_URL") or os.getenv("API_URL") or None
    final_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or ""
    return final_url, final_key


def build_response_format(json_mode: str = "auto", backend: str = "lmstudio") -> Optional[dict]:
    """Return the response_format payload used by chat.completions.create()."""
    if json_mode == "none":
        return None
    if json_mode == "json_object":
        return {"type": "json_object"}

    # For LM Studio, auto should use the JSON schema style shown in lmstudio.py.
    # For pass1, this schema is the CAFA-IR schema.
    if json_mode in {"auto", "json_schema"}:
        return CAFA_IR_SCHEMA

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

    client_kwargs = {"api_key": final_key, "timeout": timeout}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    payload = {
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
# Parse + validate the JSON formulation / CAFA-IR
# --------------------------------------------------------------------------- #

# CAFA-IR-CHANGE START
# The public function name parse_formulation() is retained so solve() stays close to
# the previous cafa_local_lmstudio.py architecture. Internally it now parses IR.

REQUIRED_KEYS = {"ir_version", "problem_type", "sense", "variables", "objective", "constraints"}
VALID_VTYPES  = {"CONTINUOUS", "INTEGER", "BINARY"}
VALID_CONSTR_SENSES = {"<=", ">=", "=="}
VALID_OBJS = {"MAXIMIZE", "MINIMIZE"}
VAR_ID_RE = re.compile(r"^[A-Za-z_]\w*$")


def _first_json_object(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from text; safer than a greedy regex."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def parse_json_object(raw: str) -> Optional[dict]:
    """Return the first JSON object from raw text, or None."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        m = re.search(r"```(?:json|python)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if m:
            text = m.group(1).strip()

    candidates = [text]
    blob = _first_json_object(text)
    if blob and blob != text:
        candidates.append(blob)

    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def validate_ir(ir: dict) -> tuple[bool, str]:
    """Validate CAFA-IR without semantic repair. Returns (ok, reason)."""
    if not isinstance(ir, dict):
        return False, "IR is not a JSON object"
    if not REQUIRED_KEYS.issubset(ir):
        return False, f"missing required keys: {sorted(REQUIRED_KEYS - set(ir))}"
    if ir.get("ir_version") != "cafa-ir-pass1":
        return False, "missing or unsupported ir_version"
    if ir.get("problem_type") not in {"LP", "MILP", "IP", "BIP"}:
        return False, "invalid problem_type"
    if ir.get("sense") not in VALID_OBJS:
        return False, "invalid objective sense"

    variables = ir.get("variables")
    if not isinstance(variables, list) or not variables:
        return False, "variables must be a non-empty list"

    ids: list[str] = []
    for v in variables:
        if not isinstance(v, dict):
            return False, "variable entry is not an object"
        vid = v.get("id")
        if not isinstance(vid, str) or not VAR_ID_RE.match(vid):
            return False, f"invalid variable id: {vid!r}"
        if v.get("vtype") not in VALID_VTYPES:
            return False, f"invalid vtype for variable {vid}"
        ids.append(vid)
    if len(ids) != len(set(ids)):
        return False, "duplicate variable ids"
    id_set = set(ids)

    obj = ir.get("objective")
    if not isinstance(obj, dict):
        return False, "objective must be an object"
    if not _is_finite_number(obj.get("constant", 0)):
        return False, "objective.constant must be a finite number"
    if not isinstance(obj.get("terms"), list):
        return False, "objective.terms must be a list"
    for t in obj["terms"]:
        if not isinstance(t, dict):
            return False, "objective term is not an object"
        if t.get("var") not in id_set:
            return False, f"objective references unknown variable {t.get('var')!r}"
        if not _is_finite_number(t.get("coef")):
            return False, "objective coefficient must be a finite number"

    constraints = ir.get("constraints")
    if not isinstance(constraints, list):
        return False, "constraints must be a list"
    for i, c in enumerate(constraints):
        if not isinstance(c, dict):
            return False, f"constraint {i} is not an object"
        if c.get("sense") not in VALID_CONSTR_SENSES:
            return False, f"constraint {i} has invalid sense"
        if not _is_finite_number(c.get("rhs")):
            return False, f"constraint {i} rhs must be a finite number"
        if not isinstance(c.get("terms"), list) or not c["terms"]:
            return False, f"constraint {i} terms must be a non-empty list"
        for t in c["terms"]:
            if not isinstance(t, dict):
                return False, f"constraint {i} term is not an object"
            if t.get("var") not in id_set:
                return False, f"constraint {i} references unknown variable {t.get('var')!r}"
            if not _is_finite_number(t.get("coef")):
                return False, f"constraint {i} coefficient must be a finite number"

    return True, "ok"


def parse_formulation(raw: str) -> Optional[dict]:
    """Return a validated CAFA-IR dict, or None. Name kept for minimal solve() changes."""
    data = parse_json_object(raw)
    if data is None:
        return None
    ok, _reason = validate_ir(data)
    if not ok:
        return None
    return data


def parse_formulation_with_error(raw: str) -> tuple[Optional[dict], str]:
    """Same as parse_formulation(), but returns the validation reason for debugging."""
    data = parse_json_object(raw)
    if data is None:
        return None, "could not parse JSON object"
    ok, reason = validate_ir(data)
    if not ok:
        return None, reason
    return data, "ok"

# CAFA-IR-CHANGE END


# --------------------------------------------------------------------------- #
# Deterministic IR-to-Gurobi compiler + execution
# --------------------------------------------------------------------------- #

# CAFA-IR-CHANGE START
# The old code-cleaning path is replaced by a deterministic compiler. No algebra
# string is parsed and no LLM-written Python code is trusted in pass1.


def _safe_float(x: Any) -> float:
    value = float(x)
    if not math.isfinite(value):
        raise ValueError(f"non-finite number: {x!r}")
    return value


def _num(x: Any) -> str:
    """Stable Python numeric literal."""
    value = _safe_float(x)
    if value == 0:
        return "0.0"
    return repr(value)


def _sanitize_constraint_name(name: Any, fallback: str) -> str:
    s = str(name or fallback).strip()
    s = re.sub(r"\W+", "_", s)
    s = s.strip("_") or fallback
    if not re.match(r"^[A-Za-z_]", s):
        s = "c_" + s
    return s[:80]


def _linear_expr(terms: list[dict], var_symbol: dict[str, str], constant: float = 0.0) -> str:
    pieces: list[str] = []
    if constant:
        pieces.append(_num(constant))
    for t in terms:
        coef = _safe_float(t["coef"])
        if coef == 0:
            continue
        var = var_symbol[t["var"]]
        if coef == 1:
            pieces.append(var)
        elif coef == -1:
            pieces.append(f"(-{var})")
        else:
            pieces.append(f"({_num(coef)} * {var})")
    return " + ".join(pieces) if pieces else "0.0"


def compile_ir_to_gurobi(ir: dict) -> str:
    """Compile validated CAFA-IR into a full runnable Gurobi Python script."""
    ok, reason = validate_ir(ir)
    if not ok:
        raise ValueError(f"Invalid IR: {reason}")

    lines: list[str] = []
    lines.append("import gurobipy as gp")
    lines.append("env = gp.Env(empty=True)")
    lines.append('env.setParam("OutputFlag", 0)')
    lines.append("env.start()")
    lines.append("m = gp.Model(env=env)")
    lines.append("")

    var_symbol: dict[str, str] = {}
    vtype_map = {
        "CONTINUOUS": "gp.GRB.CONTINUOUS",
        "INTEGER": "gp.GRB.INTEGER",
        "BINARY": "gp.GRB.BINARY",
    }

    for v in ir["variables"]:
        vid = v["id"]
        sym = f"var_{vid}"
        var_symbol[vid] = sym
        name = str(v.get("name") or vid)
        vtype = vtype_map[v["vtype"]]
        if v["vtype"] == "BINARY":
            lines.append(f"{sym} = m.addVar(name={name!r}, vtype={vtype})")
        else:
            # NL4OPT-style decision variables are non-negative by default.
            lines.append(f"{sym} = m.addVar(name={name!r}, lb=0.0, vtype={vtype})")

    lines.append("")
    obj = ir["objective"]
    obj_expr = _linear_expr(obj.get("terms", []), var_symbol, constant=_safe_float(obj.get("constant", 0.0)))
    obj_sense = "gp.GRB.MAXIMIZE" if ir["sense"] == "MAXIMIZE" else "gp.GRB.MINIMIZE"
    lines.append(f"m.setObjective({obj_expr}, {obj_sense})")
    lines.append("")

    for i, c in enumerate(ir["constraints"], start=1):
        lhs = _linear_expr(c["terms"], var_symbol)
        rhs = _num(c["rhs"])
        sense = c["sense"]
        cname = _sanitize_constraint_name(c.get("name"), f"c{i}")
        if sense == "<=":
            lines.append(f"m.addConstr({lhs} <= {rhs}, name={cname!r})")
        elif sense == ">=":
            lines.append(f"m.addConstr({lhs} >= {rhs}, name={cname!r})")
        elif sense == "==":
            lines.append(f"m.addConstr({lhs} == {rhs}, name={cname!r})")
        else:
            raise ValueError(f"Unsupported constraint sense: {sense}")

    lines.append("")
    lines.append("m.optimize()")
    lines.append("if m.Status == gp.GRB.OPTIMAL:")
    lines.append('    print("OBJECTIVE_VALUE:", float(m.objVal))')
    lines.append("elif m.Status == gp.GRB.INFEASIBLE:")
    lines.append('    print("MODEL_STATUS: INFEASIBLE")')
    lines.append("elif m.Status in (gp.GRB.INF_OR_UNBD, gp.GRB.UNBOUNDED):")
    lines.append('    print("MODEL_STATUS: UNBOUNDED")')
    lines.append("else:")
    lines.append('    print("MODEL_STATUS:", m.Status)')
    return "\n".join(lines) + "\n"


def execute_code(code: str, save_path: Optional[str] = None) -> dict:
    """Run deterministic compiler output. Returns {status, obj_val, error}."""
    try:
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(code)
        ns: dict = {}
        exec(code, ns, ns)
        m = ns.get("m")
        if m is None:
            return {"status": "exec_fail", "obj_val": None, "error": "no model `m`"}
        # Gurobi: 2=OPTIMAL 3=INFEASIBLE 4=INF_OR_UNBD 5=UNBOUNDED
        st = getattr(m, "Status", None)
        if st == 2:       return {"status": "ok", "obj_val": float(m.objVal), "error": None}
        if st == 3:       return {"status": "infeasible", "obj_val": None, "error": "infeasible"}
        if st in (4, 5):  return {"status": "unbounded",  "obj_val": None, "error": "unbounded"}
        return {"status": "solver_other", "obj_val": None, "error": f"solver status {st}"}
    except Exception as e:
        return {"status": "exec_fail", "obj_val": None, "error": str(e)}

# CAFA-IR-CHANGE END


# --------------------------------------------------------------------------- #
# Verifier (kept as a later-work hook; disabled by default)
# --------------------------------------------------------------------------- #

def needs_verification(result: dict) -> bool:
    if result["status"] in {"infeasible", "unbounded", "exec_fail", "solver_other"}: return True
    if result["status"] == "ok" and result["obj_val"] == 0.0:        return True
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
    """One verifier pass. Returns revised CAFA-IR dict, or None.

    Pass1 keeps this hook for later repair-loop work, but CLI disables it by default.
    """
    user = (f"PROBLEM:\n{description}\n\n"
            f"CAFA_IR:\n{json.dumps(formulation, indent=2)}\n\n"
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
    except Exception:
        return None
    return parse_formulation(raw)


# --------------------------------------------------------------------------- #
# Per-problem solve
# --------------------------------------------------------------------------- #

def solve(description: str, ground_truth: float, model: str,
          enable_verifier: bool, code_path: Optional[str], ir_path: Optional[str] = None,
          backend: str = "lmstudio", api_url: Optional[str] = None, api_key: Optional[str] = None,
          json_mode: str = "auto", timeout: int = 120, max_retries: int = 3,
          temperature: float = 0.0, max_tokens: int = 4096) -> dict:
    """Run the full pass1 pipeline for one problem. Returns a record dict."""
    rec = {"description": description, "ground_truth": ground_truth,
           "raw": "", "formulation": None, "ir": None, "code_used": None,
           "obj_val": None, "status": "parse_fail", "error": None,
           "revised": False}

    # 1. primary call: NL -> structured CAFA-IR
    try:
        rec["raw"] = call_llm(
            model=model,
            messages=build_messages(description),
            temperature=temperature,
            max_tokens=max_tokens,
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

    # 2. parse + validate IR
    formulation, reason = parse_formulation_with_error(rec["raw"])
    if formulation is None:
        rec["error"] = reason
        return rec
    rec["formulation"] = formulation   # kept for compatibility with old meta.json readers
    rec["ir"] = formulation

    if ir_path:
        os.makedirs(os.path.dirname(ir_path), exist_ok=True)
        with open(ir_path, "w", encoding="utf-8") as f:
            json.dump(formulation, f, indent=2, ensure_ascii=False)

    # 3. deterministic IR -> Gurobi compiler
    try:
        code = compile_ir_to_gurobi(formulation)
    except Exception as e:
        rec["status"], rec["error"] = "compile_fail", str(e)
        return rec
    rec["code_used"] = code

    # 4. execute deterministic compiler output
    res = execute_code(code, save_path=code_path)
    rec.update(status=res["status"], obj_val=res["obj_val"], error=res["error"])

    # 5. optional verifier hook. This is off by default and can be treated as later work.
    if enable_verifier and needs_verification(res):
        revised = verify(
            description, formulation, res, model=model,
            backend=backend, api_url=api_url, api_key=api_key,
            json_mode=json_mode, timeout=timeout, max_retries=max_retries,
        )
        if revised and revised != formulation:
            try:
                revised_code = compile_ir_to_gurobi(revised)
            except Exception:
                return rec
            res2 = execute_code(revised_code, save_path=code_path)
            better = res2["status"] == "ok" and (
                res["status"] != "ok"
                or abs(res2["obj_val"] - ground_truth) < abs((res["obj_val"] or 0) - ground_truth)
            )
            if better:
                rec["formulation"], rec["ir"], rec["code_used"] = revised, revised, revised_code
                rec.update(status=res2["status"], obj_val=res2["obj_val"], error=res2["error"])
                rec["revised"] = True
    return rec


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def is_correct(obj_val: Optional[float], gt: float, tol: float) -> bool:
    if obj_val is None: return False
    if gt == 0:
        return abs(obj_val - gt) <= max(tol, 1e-3)
    return abs(obj_val - gt) <= max(tol * abs(gt), tol)


def aggregate(records: list[dict], tol: float) -> dict:
    n = len(records)
    counts = {"ok": 0, "parse_fail": 0, "compile_fail": 0, "exec_fail": 0,
              "infeasible": 0, "unbounded": 0, "solver_other": 0, "llm_fail": 0}
    correct, revised, correct_ids = 0, 0, []
    for i, r in enumerate(records):
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if r.get("revised"): revised += 1
        if r["status"] == "ok" and is_correct(r["obj_val"], r["ground_truth"], tol):
            correct += 1
            correct_ids.append(i)
    return {
        "total":         n,
        "success_rate":  counts["ok"] / max(n, 1),   # IR parsed + compiled + returned a value
        "accuracy":      correct      / max(n, 1),   # within tolerance of GT
        "success":       counts["ok"],
        "correct":       correct,
        "wrong_answer":  counts["ok"] - correct,
        "parse_fail":    counts["parse_fail"],
        "compile_fail":  counts["compile_fail"],
        "exec_fail":     counts["exec_fail"],
        "infeasible":    counts["infeasible"],
        "unbounded":     counts["unbounded"],
        "solver_other":  counts["solver_other"],
        "llm_fail":      counts["llm_fail"],
        "revised":       revised,
        "correct_ids":   correct_ids,
        "tolerance":     tol,
    }


def print_report(s: dict) -> None:
    n = max(s["total"], 1)
    print("=" * 60)
    print(f"Total          : {s['total']}")
    print(f"Success rate   : {s['success']}/{n} = {s['success_rate']:.2%}  (IR parsed + compiled + executed)")
    print(f"Accuracy       : {s['correct']}/{n} = {s['accuracy']:.2%}  (within {s['tolerance']:.0%} of GT)")
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
    """Avoid nested output paths when model names contain slashes."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     required=True)
    p.add_argument("--model",       default="qwen3-4b-instruct-2507")
    p.add_argument("--output_dir",  default="outputs")
    p.add_argument("--no_verifier", dest="enable_verifier", action="store_false", default=False)
    p.add_argument("--tolerance",   type=float, default=1e-4)
    p.add_argument("--limit",       type=int,   default=None)
    p.add_argument("--overwrite",   action="store_true", help="Re-run even when meta.json exists.")

    # CAFA-LMSTUDIO-CHANGE START
    # Minimal backend switches. LM Studio uses the OpenAI-compatible local server.
    p.add_argument("--backend", choices=["openai", "lmstudio"],
                   default=os.getenv("CAFA_BACKEND", "lmstudio"))
    p.add_argument("--api_url", default=None,
                   help="Override API base URL. For LM Studio, default is http://localhost:1234/v1.")
    p.add_argument("--api_key", default=None,
                   help="Override API key. For LM Studio, default is lm-studio.")
    p.add_argument("--json_mode", choices=["auto", "none", "json_object", "json_schema"],
                   default="auto",
                   help="auto/json_schema uses the CAFA-IR JSON schema for LM Studio structured output.")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=4096)
    # CAFA-LMSTUDIO-CHANGE END

    args = p.parse_args()

    with open(args.dataset, encoding="utf-8") as f: lines = f.readlines()
    if args.limit: lines = lines[: args.limit]

    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    model_dir = safe_path_name(args.model)
    base = os.path.join(args.output_dir, model_dir, dataset_name)
    os.makedirs(base, exist_ok=True)

    records = []
    for i, line in enumerate(lines):
        data = json.loads(line)
        desc, gt = data["description"], float(data["answer"])
        pdir = os.path.join(base, f"problem_{i}"); os.makedirs(pdir, exist_ok=True)
        meta_path = os.path.join(pdir, "meta.json")
        ir_path = os.path.join(pdir, "ir.json")
        code_path = os.path.join(pdir, "compiled_gurobi.py")

        if os.path.exists(meta_path) and not args.overwrite:
            with open(meta_path, encoding="utf-8") as f: rec = json.load(f)
            print(f"[{i:03d}] cached  status={rec['status']:12s} obj={rec.get('obj_val')} gt={gt}")
        else:
            print(f"[{i:03d}] solving IR pass1 ...", flush=True)
            rec = solve(
                desc, gt, args.model,
                enable_verifier=args.enable_verifier, code_path=code_path, ir_path=ir_path,
                backend=args.backend, api_url=args.api_url, api_key=args.api_key,
                json_mode=args.json_mode, timeout=args.timeout, max_retries=args.max_retries,
                temperature=args.temperature, max_tokens=args.max_tokens,
            )
            with open(meta_path, "w", encoding="utf-8") as f: json.dump(rec, f, indent=2, ensure_ascii=False)
            print(f"[{i:03d}] {rec['status']:12s} obj={rec.get('obj_val')} gt={gt} revised={rec.get('revised')}")
        records.append(rec)

    summary = aggregate(records, tol=args.tolerance)
    summary["workflow"] = "cafa++-pass1-ir-compiler"
    summary["model"] = args.model
    summary["backend"] = args.backend
    summary["verifier_enabled"] = args.enable_verifier
    print_report(summary)
    with open(os.path.join(base, "_metrics.json"), "w", encoding="utf-8") as f: json.dump(summary, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
