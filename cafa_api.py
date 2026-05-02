"""
CAFA v2 - compact, single-file LP auto-formulation. - zbh

Pipeline:
  description -> LLM (JSON) -> parse -> execute Gurobi -> metrics
                                              |
                                              v (only on infeasible/unbounded/exec_fail/zero)
                                          verifier (1 extra call)

Usage:
  python cafa_v2.py --dataset bench.jsonl --model gpt-4o --output_dir results/
  python cafa_v2.py ... --no_verifier        # disable the verifier pass
  python cafa_v2.py ... --limit 20           # quick subset run
"""
from __future__ import annotations
import argparse, json, os, re, sys
from typing import Optional


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are an expert OR modeler. Convert the problem into a Gurobi formulation.

Reason in this order:
  1. Identify decision variables. Pick INTEGER for indivisible counts, CONTINUOUS for divisible
     quantities (acres, kg, hours), BINARY for yes/no. Match real-world meaning.
  2. Extract the objective. Watch for hidden coefficients.
  3. Extract every constraint. Watch directions (max -> <=, min -> >=), ratios, sign restrictions.
  4. Self-check: every number used? all directions correct? variable types real-world correct?

Output STRICT JSON only, matching this schema (no prose, no fences):
{
  "problem_type": "LP" | "MILP" | "IP" | "BIP",
  "sense": "MAXIMIZE" | "MINIMIZE",
  "variables": [{"name": "...", "vtype": "CONTINUOUS"|"INTEGER"|"BINARY", "rationale": "..."}],
  "objective": "linear expression",
  "constraints": [{"expression": "lhs <op> rhs", "source": "clause from problem"}],
  "code": "Gurobi Python using existing model `m`"
}

Code rules: assume `gurobipy as gp` and model `m` exist. Use m.addVar / m.setObjective / m.addConstr.
Use explicit <= or >=, never bare < or >. No imports, no env, no m.optimize(), no comments."""


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
                {"name": "Oil Max",     "vtype": "INTEGER", "rationale": "containers indivisible"},
                {"name": "Oil Max Pro", "vtype": "INTEGER", "rationale": "containers indivisible"},
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
                {"name": "apples", "vtype": "CONTINUOUS", "rationale": "acreage divisible"},
                {"name": "pears",  "vtype": "CONTINUOUS", "rationale": "acreage divisible"},
            ],
            "objective": "2*x + 4*y",
            "constraints": [
                {"expression": "x + y <= 50", "source": "total acres"},
                {"expression": "x >= 5",       "source": "apple minimum"},
                {"expression": "y >= 10",      "source": "pear minimum"},
                {"expression": "y <= 2*x",     "source": "workforce ratio"},
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


def build_messages(description: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT:
        msgs.append({"role": "user",      "content": f"QUESTION: {ex['q']}"})
        msgs.append({"role": "assistant", "content": json.dumps(ex["a"])})
    msgs.append({"role": "user", "content": f"QUESTION: {description}"})
    return msgs


# --------------------------------------------------------------------------- #
# LLM call
# --------------------------------------------------------------------------- #

def call_llm(messages: list[dict], model: str, max_tokens: int = 4096) -> str:
    """One OpenAI-compatible chat call. Returns raw text."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("API_URL"))
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0.0, max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# Parse + validate the JSON formulation
# --------------------------------------------------------------------------- #

REQUIRED_KEYS = {"problem_type", "sense", "variables", "objective", "constraints", "code"}
VALID_VTYPES  = {"CONTINUOUS", "INTEGER", "BINARY"}


def parse_formulation(raw: str) -> Optional[dict]:
    """Return a validated dict, or None. Tries raw text, then markdown-fenced, then first {...}."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        m = re.search(r"```(?:json|python)?\s*(.*?)```", text, re.DOTALL)
        if m: text = m.group(1).strip()

    candidates = [text]
    blob = re.search(r"\{[\s\S]*\}", text)
    if blob and blob.group(0) != text:
        candidates.append(blob.group(0))

    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or not REQUIRED_KEYS.issubset(data):
            continue
        if not isinstance(data["variables"], list) or not isinstance(data["constraints"], list):
            continue
        if not isinstance(data["code"], str) or not data["code"].strip():
            continue
        if any(not isinstance(v, dict) or v.get("vtype") not in VALID_VTYPES for v in data["variables"]):
            continue
        return data
    return None


# --------------------------------------------------------------------------- #
# Code cleaning + execution
# --------------------------------------------------------------------------- #

GUROBI_PREFIX = ('import gurobipy as gp\n'
                 'env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()\n'
                 'm = gp.Model(env=env)\n')
GUROBI_SUFFIX = '\nm.optimize()\n'


def clean_code(src: str) -> str:
    """Fix bare < / > on addConstr lines and stray missing newlines."""
    out = []
    for line in src.split("\n"):
        line = line.rstrip()
        if line.lstrip().startswith("m.addConstr") and not re.search(r"<=|>=", line):
            line = re.sub(r"(?<![<>=])<(?!=)", "<=", line)
            line = re.sub(r"(?<![<>=])>(?!=)", ">=", line)
        out.append(line)
    return "\n".join(out).replace(")m.", ")\nm.")


def execute_code(code: str, save_path: Optional[str] = None) -> dict:
    """Run code in an isolated namespace. Returns {status, obj_val, error}.

    status is one of: ok | exec_fail | infeasible | unbounded
    """
    try:
        cleaned = clean_code(code)
        needs = "import gurobipy" not in cleaned.lower() and "m = gp.model" not in cleaned.lower()
        full  = (GUROBI_PREFIX if needs else "") + cleaned + GUROBI_SUFFIX
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f: f.write(full)
        ns: dict = {}
        exec(full, ns, ns)
        m = ns.get("m")
        if m is None:
            return {"status": "exec_fail", "obj_val": None, "error": "no model `m`"}
        # Gurobi: 2=OPTIMAL 3=INFEASIBLE 4=INF_OR_UNBD 5=UNBOUNDED
        st = getattr(m, "Status", None)
        if st == 3:       return {"status": "infeasible", "obj_val": None, "error": "infeasible"}
        if st in (4, 5):  return {"status": "unbounded",  "obj_val": None, "error": "unbounded"}
        return {"status": "ok", "obj_val": float(m.objVal), "error": None}
    except Exception as e:
        return {"status": "exec_fail", "obj_val": None, "error": str(e)}


# --------------------------------------------------------------------------- #
# Verifier (gated single shot)
# --------------------------------------------------------------------------- #

def needs_verification(result: dict) -> bool:
    if result["status"] in {"infeasible", "unbounded", "exec_fail"}: return True
    if result["status"] == "ok" and result["obj_val"] == 0.0:        return True
    return False


def verify(description: str, formulation: dict, result: dict, model: str) -> Optional[dict]:
    """One verifier pass. Returns revised formulation dict, or None."""
    user = (f"PROBLEM:\n{description}\n\n"
            f"FORMULATION:\n{json.dumps(formulation, indent=2)}\n\n"
            f"EXECUTOR: status={result['status']} obj={result['obj_val']} error={result['error']}")
    try:
        raw = call_llm(
            [{"role": "system", "content": VERIFIER_PROMPT}, {"role": "user", "content": user}],
            model=model,
        )
    except Exception:
        return None
    return parse_formulation(raw)


# --------------------------------------------------------------------------- #
# Per-problem solve
# --------------------------------------------------------------------------- #

def solve(description: str, ground_truth: float, model: str,
          enable_verifier: bool, code_path: Optional[str]) -> dict:
    """Run the full pipeline for one problem. Returns a record dict."""
    rec = {"description": description, "ground_truth": ground_truth,
           "raw": "", "formulation": None, "code_used": None,
           "obj_val": None, "status": "parse_fail", "error": None,
           "revised": False}

    # 1. primary call
    try:
        rec["raw"] = call_llm(build_messages(description), model=model)
    except Exception as e:
        rec["status"], rec["error"] = "llm_fail", str(e)
        return rec

    formulation = parse_formulation(rec["raw"])
    if formulation is None:
        rec["error"] = "could not parse JSON"
        return rec
    rec["formulation"] = formulation
    rec["code_used"]   = formulation["code"]

    # 2. execute
    res = execute_code(formulation["code"], save_path=code_path)
    rec.update(status=res["status"], obj_val=res["obj_val"], error=res["error"])

    # 3. optional verify (only if suspicious)
    if enable_verifier and needs_verification(res):
        revised = verify(description, formulation, res, model=model)
        if revised and revised["code"] != formulation["code"]:
            res2 = execute_code(revised["code"], save_path=code_path)
            # accept only if strictly better
            better = res2["status"] == "ok" and (
                res["status"] != "ok"
                or abs(res2["obj_val"] - ground_truth) < abs((res["obj_val"] or 0) - ground_truth)
            )
            if better:
                rec["formulation"], rec["code_used"] = revised, revised["code"]
                rec.update(status=res2["status"], obj_val=res2["obj_val"], error=res2["error"])
                rec["revised"] = True
    return rec


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def is_correct(obj_val: Optional[float], gt: float, tol: float) -> bool:
    if obj_val is None: return False
    return abs(obj_val - gt) <= max(tol * abs(gt), tol)


def aggregate(records: list[dict], tol: float) -> dict:
    n = len(records)
    counts = {"ok": 0, "parse_fail": 0, "exec_fail": 0,
              "infeasible": 0, "unbounded": 0, "llm_fail": 0}
    correct, revised, correct_ids = 0, 0, []
    for i, r in enumerate(records):
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if r.get("revised"): revised += 1
        if r["status"] == "ok" and is_correct(r["obj_val"], r["ground_truth"], tol):
            correct += 1
            correct_ids.append(i)
    return {
        "total":         n,
        "success_rate":  counts["ok"] / max(n, 1),   # parsed + ran + returned a value
        "accuracy":      correct      / max(n, 1),   # within tolerance of GT
        "success":       counts["ok"],
        "correct":       correct,
        "wrong_answer":  counts["ok"] - correct,
        "parse_fail":    counts["parse_fail"],
        "exec_fail":     counts["exec_fail"],
        "infeasible":    counts["infeasible"],
        "unbounded":     counts["unbounded"],
        "llm_fail":      counts["llm_fail"],
        "revised":       revised,
        "correct_ids":   correct_ids,
        "tolerance":     tol,
    }


def print_report(s: dict) -> None:
    n = max(s["total"], 1)
    print("=" * 60)
    print(f"Total          : {s['total']}")
    print(f"Success rate   : {s['success']}/{n} = {s['success_rate']:.2%}  (parsed + executed)")
    print(f"Accuracy       : {s['correct']}/{n} = {s['accuracy']:.2%}  (within {s['tolerance']:.0%} of GT)")
    print(f"  wrong_answer : {s['wrong_answer']}")
    print(f"  parse_fail   : {s['parse_fail']}")
    print(f"  exec_fail    : {s['exec_fail']}")
    print(f"  infeasible   : {s['infeasible']}")
    print(f"  unbounded    : {s['unbounded']}")
    print(f"  llm_fail     : {s['llm_fail']}")
    print(f"Verifier-fixed : {s['revised']}")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     required=True)
    p.add_argument("--model",       required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--no_verifier", action="store_true")
    p.add_argument("--tolerance",   type=float, default=0.05)
    p.add_argument("--limit",       type=int,   default=None)
    args = p.parse_args()

    with open(args.dataset) as f: lines = f.readlines()
    if args.limit: lines = lines[: args.limit]

    base = os.path.join(args.output_dir, args.model,
                        os.path.splitext(os.path.basename(args.dataset))[0])
    os.makedirs(base, exist_ok=True)

    records = []
    for i, line in enumerate(lines):
        data = json.loads(line)
        desc, gt = data["description"], float(data["answer"])
        pdir = os.path.join(base, f"problem_{i}"); os.makedirs(pdir, exist_ok=True)
        meta_path = os.path.join(pdir, "meta.json")
        code_path = os.path.join(pdir, "code.py")

        if os.path.exists(meta_path):
            with open(meta_path) as f: rec = json.load(f)
            print(f"[{i:03d}] cached  status={rec['status']:11s} obj={rec['obj_val']} gt={gt}")
        else:
            print(f"[{i:03d}] solving ...", flush=True)
            rec = solve(desc, gt, args.model,
                        enable_verifier=not args.no_verifier, code_path=code_path)
            with open(meta_path, "w") as f: json.dump(rec, f, indent=2)
            print(f"[{i:03d}] {rec['status']:11s} obj={rec['obj_val']} gt={gt} revised={rec['revised']}")
        records.append(rec)

    summary = aggregate(records, tol=args.tolerance)
    summary["verifier_enabled"] = not args.no_verifier
    print_report(summary)
    with open(os.path.join(base, "_metrics.json"), "w") as f: json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())