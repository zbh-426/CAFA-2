import os
import json
from openai import OpenAI
import re
import numpy as np



def get_cafa_results(description:str, model:str):
    """Build the CAFA system prompt and optionally call OpenAI for code generation.

    Returns:
        str: Generated text when the API call succeeds, or the assembled prompt
        when falling back to offline inspection or debugging.
    """
    # Core system prompt, aligned with the notebook version.
    sys_prompt = (
        "You are an expert in optimization problems and domain specific language generation.\n"
        "Your task is to convert the textual optimization text into a piece of code.\n"
        "DO NOT ADD ANY COMMENTS OR EXPLANATION TO THE CODE. JUST OUTPUT THE CODE.\n"
        "Here are some examples that you should refer to:\n"
    )

    example = """
        QUESTION:
        A car manufacturer makes two types of car oils: Oil Max and Oil Max Pro. A container of Oil Max contains 46 grams of substance A, 43 grams of substance B and 56 grams of substance C. A container of Oil Max Pro contains 13 grams of substance A, 4 grams of substance B and 45 grams of substance C. The car manufacturer has 1345 grams of substance A, 346 grams of substance B, 1643 grams of substance C. In addition, the profit per container of Oil Max is $10 and the profit per container of Oil Max Pro is $15. How many containers of each of oil should the car manufacturer make to maximize profit?
        CODE:
        x = m.addVar(name="Oil Max", vtype=gp.GRB.INTEGER)
        y = m.addVar(name="Oil Max Pro", vtype=gp.GRB.INTEGER)
        m.setObjective(10 * x + 15 * y, gp.GRB.MAXIMIZE)
        m.addConstr(46 * x + 13 * y <= 1345)
        m.addConstr(43 * x + 4 * y <= 346)
        m.addConstr(56 * x + 45 * y <= 1643)

        QUESTION:
        Ben is growing apples and pears on his orchard. He has 50 acres available on which he must grow a minimum of 5 acres of apples and a minimum of 10 acres of pears to meet demands. The profit per apple is $2 and the profit per pear is $4. He prefers to grow more pears than apples but limitations in his workforce allow him to grow at most twice the amount of pears as apples. How many of each fruit should Ben grow in order to maximize his profit? What is that profit?
        CODE:
        x = m.addVar(name="apples", vtype=gp.GRB.INTEGER)
        y = m.addVar(name="pears", vtype=gp.GRB.INTEGER)
        m.setObjective(2 * x + 4 * y, gp.GRB.MAXIMIZE)
        m.addConstr(x + y <= 50)
        m.addConstr(x >= 5)
        m.addConstr(y >= 10)
        m.addConstr(y <= 2 * x)
    """

    prompt = sys_prompt + example + "\nPlease finish the task think step by step.\nQUESTION: " + description + "\nCODE:"
    try:
        client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("API_URL"),
        )
        
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt + example},
                {"role": "user", "content": f"QUESTION: {description}"},
            ],
            temperature=0,
            max_tokens=20000,
        )
        # Do not strip whitespace so the original indentation is preserved.
        response_text = resp.choices[0].message.content
        return response_text
    except Exception as e:
        return prompt
    
def extract_and_run_code(code: str, save_path: str = None) -> float:
    """Clean and execute generated code, following the notebook behavior.

    The routine:
    - normalizes malformed `<` and `>` constraints
    - wraps the code with a Gurobi model prefix and optimize suffix when needed
    - executes the program and reads `m.objVal`

    Returns `np.inf` when execution fails.
    """

    prefix = """
import gurobipy as gp
env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()
m = gp.Model(env=env)
"""
    suffix = """
m.optimize()
"""

    def clean_code(src: str) -> str:
        # Preserve leading indentation and only trim trailing whitespace.
        lines = []
        for line in src.split('\n'):
            # Keep leading spaces intact.
            line = line.rstrip()
            if line.lstrip().startswith('m.addConstr') and not re.findall(r'<=|>=', line):
                line = re.sub(r'(?<![<>=])<(?!=)', '<=', line)
                line = re.sub(r'(?<![<>=])>(?!=)', '>=', line)
            lines.append(line)
        out = '\n'.join(lines)
        out = out.replace(')m', ')\nm')
        return out

    try:
        # Remove Markdown fences while preserving indentation inside the block.
        if isinstance(code, str) and code.lstrip().startswith('```'):
            first = code.find('```')
            last = code.rfind('```')
            if first != -1 and last != -1 and last > first:
                inner = code[first+3:last]
                # Drop the language marker line such as ```python.
                if inner.lstrip().startswith('python'):
                    nl = inner.find('\n')
                    if nl != -1:
                        inner = inner[nl+1:]
                code = inner

        cleaned = clean_code(code)

        # Skip the prefix if the code already imports gurobipy or defines `m`.
        lower = cleaned.lower()
        needs_prefix = ('import gurobipy' not in lower) and ('m = gp.model' not in lower)

        if needs_prefix:
            full_code = prefix + '\n' + cleaned + '\n' + suffix
        else:
            # The code already defines the model; only append optimize().
            full_code = cleaned + '\n' + suffix

        # Optionally save the runnable code for manual inspection.
        if save_path:
            try:
                with open(save_path, 'w') as cf:
                    cf.write(full_code)
            except Exception:
                # Saving the file is optional; execution should still continue.
                pass

        ex_locals = {}
        # Use one dictionary for globals and locals so `m` remains visible.
        exec(full_code, ex_locals, ex_locals)

        # Read objVal from the model object.
        if 'm' in ex_locals:
            try:
                return float(ex_locals['m'].objVal)
            except Exception:
                return np.inf
        else:
            return np.inf
    except Exception:
        return np.inf
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to the benchmark file.")
    parser.add_argument("--model", type=str, required=True, help="Model to use for generation.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results.")
    args = parser.parse_args()

    benchmark_path = args.dataset
    with open(benchmark_path, "r") as f:
        lines = f.readlines()
        
    id_count = 0
    for line in lines:
        data = json.loads(line)
        
        description = data["description"]
        answer = data["answer"]
        
        dataset_name = os.path.splitext(os.path.basename(benchmark_path))[0]
        result_file = os.path.join(args.output_dir, args.model, dataset_name, f"problem_{id_count}")
        
        os.makedirs(result_file, exist_ok=True)
        print(f"Processing problem_{id_count}...")
        # Skip problems that already have metadata.
        if os.path.exists(os.path.join(result_file, 'meta.json')):
            print(f"meta.json already exists in {result_file}, skipping...")
            id_count += 1
            continue
        generated = get_cafa_results(description, args.model)

        out_path = os.path.join(result_file, 'meta.json')
        meta = {
            'description': description,
            'ground_truth': answer,
            'generated_raw': generated,
        }

        # Execute the output only when it already looks like model code.
        code_to_run = None
        if isinstance(generated, str) and ("m.addVar" in generated or "m.setObjective" in generated):
            code_to_run = generated
        else:
            code_to_run = None

        if code_to_run:
            code_path = os.path.join(result_file, 'code.py')
            obj = extract_and_run_code(code_to_run, save_path=code_path)
            meta['objVal'] = obj
        else:
            meta['objVal'] = None

        # Save metadata.
        with open(out_path, 'w') as wf:
            json.dump(meta, wf, indent=2)

        id_count += 1
        
    
    # Compute accuracy using the stored objVal and ground truth values.
    count = 0
    correct = 0
    
    ids = []
    
    
    for line in lines:
        data = json.loads(line)
        answer = data["answer"]
        
        result_file = f"results/gemini-2.5-pro/{os.path.basename(benchmark_path)}/problem_{count}"
        out_path = os.path.join(result_file, 'meta.json')
        
        with open(out_path, 'r') as rf:
            meta = json.load(rf)
        
        objVal = meta.get('objVal', None)
        if objVal is not None:
            if abs(float(objVal) - float(answer)) <= 0.05 * abs(float(answer)):
            #if abs(float(objVal) - float(answer)) <= 0.0001 * abs(float(answer)):
                correct += 1
                ids.append(count)
        
        count += 1
        
    print(f"Total problems: {count}, Correctly solved: {correct}, Accuracy: {correct/count:.2%}")
    
    
    
    # # Print the ids of correctly solved problems if needed.
    # with open(benchmark_path, "r") as f:
    #     lines = f.readlines()
    # print("IDs of correctly solved problems:")
    # for id in ids:
    #     print(f"{json.loads(lines[id])["id"]}")
