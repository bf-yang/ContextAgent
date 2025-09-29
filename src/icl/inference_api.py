import os
import json
import ast
import argparse
import random
from datetime import datetime
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

# ---- Project imports ----
import config
from tool_registry import functions, process_function_call
from utils import (
    azure_inference,
    ollama_inference,                 # kept in case you switch later
    parse_proactive_agent_results,
    convert_sets_to_lists,
    execute_tools_with_memory,        # two-stage tool executor
)
from openai import AzureOpenAI

random.seed(42)

# =========================
# Azure client
# =========================
def get_azure_client() -> AzureOpenAI:
    """
    Build an AzureOpenAI client. Prefer environment variables if present,
    otherwise fall back to your provided defaults.
    """
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    return AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)

# =========================
# Args & Mode
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["live", "sandbox"], default="sandbox")
    p.add_argument("--model_base", type=str, default="gpt-35-turbo",
                  help='e.g., "gpt-35-turbo", "gpt-4o-2", "gpt-4o-mini-2"')
    p.add_argument("--dataset", type=str, default="cab", help="cab, cab_lite, cab_ood")
    p.add_argument("--zs", type=str, default="false", help="false, true")
    p.add_argument("--personas", type=str, default="w_p", help="w_p, wo_p")
    p.add_argument("--think", type=str, default="w_t", help="w_t, wo_t")
    p.add_argument("--n_fewshot", type=int, default=10, help="Number of few-shot examples")
    return p.parse_args()

def apply_mode(mode: str):
    """
    Propagate global mode to all tools via config.
    - "live": real external calls
    - "sandbox": mocked responses
    """
    try:
        from config import set_mode
        set_mode(mode)
    except Exception:
        config.MODE = mode

# =========================
# Data Loading & Prompt
# =========================
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_prompt_base(zs_flag: str) -> str:
    path = "prompt/baselines/icl_zs.txt" if zs_flag == "true" else "prompt/baselines/icl_fs.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def sanitize_demonstrations(samples: Dict[str, Any], think: str, personas: str) -> Dict[str, Any]:
    """Drop/keep fields to match your SFT-style demos."""
    for k, ex in samples.items():
        for fld in ("Response", "Tool planning", "Action"):
            ex.pop(fld, None)
        if think == "wo_t":
            ex.pop("Thoughts", None)
        if personas == "wo_p":
            ex.pop("Personas", None)
    return samples

def build_system_prompt(base_prompt: str,
                        zs_flag: str,
                        train_ds: Dict[str, Any],
                        n_fewshot: int,
                        think: str,
                        personas: str) -> Tuple[str, List[str]]:
    keys = list(train_ds.keys())
    fewshot_keys = random.sample(keys, n_fewshot) if n_fewshot > 0 else []
    fewshots = {k: train_ds[k] for k in fewshot_keys}
    fewshots = sanitize_demonstrations(fewshots, think, personas)

    print("Selected samples:", fewshot_keys)
    if zs_flag == "true":
        return base_prompt, fewshot_keys

    examples_str = json.dumps(fewshots, ensure_ascii=False, indent=4)
    filled = base_prompt.replace("{Examples}", examples_str)
    return filled, fewshot_keys

# =========================
# Per-sample helpers
# =========================
def get_contextual_info(dataset_name: str, sample: Dict[str, Any]) -> str:
    if dataset_name in ("cab", "cab_ood"):
        return sample["Context information"]
    if dataset_name == "cab_lite":
        return sample["Rawdata Context"]
    return sample.get("Context information", "")

def parse_tool_spec(tools_str: str) -> List[Dict[str, Any]]:
    if tools_str == "None":
        return []
    try:
        parsed = ast.literal_eval(tools_str)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []

def run_tools(json_tool: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Two-stage executor + memory (no-params first, then param tools).
    Delegates to execute_tools_with_memory + process_function_call.
    """
    memory = {"now_iso": datetime.now().isoformat()}
    results_tool = execute_tools_with_memory(json_tool, process_function_call, memory)
    return results_tool, memory

def summarize_with_llm_azure(client: AzureOpenAI,
                             model_base: str,
                             contextual_info: str,
                             personas_str: str,
                             thoughts: str,
                             results_tool: Any,
                             personas_flag: str) -> str:
    """
    Summarize context + tool outputs via Azure model.
    """
    with open("prompt/prompt_summarize.txt", "r", encoding="utf-8") as f:
        prompt_summarize = f.read()

    if personas_flag == "wo_p":
        content = (
            "# Sensory Contexts:\n" + contextual_info +
            "\n\n# Thoughts:\n" + thoughts +
            "\n\n# Tool results:\n" + str(results_tool)
        )
    else:
        content = (
            "# Sensory Contexts:\n" + contextual_info +
            "\n\n# Persona Contexts:\n" + personas_str +
            "\n\n# Thoughts:\n" + thoughts +
            "\n\n# Tool results:\n" + str(results_tool)
        )

    messages = [
        {"role": "system", "content": prompt_summarize},
        {"role": "user", "content": content},
    ]

    if config.MODE == "sandbox":
        return "sandbox outputs."

    try:
        return azure_inference(client, model_base, messages, temperature=0.7, max_tokens=4096)
    except Exception as e:
        return f"[summarize error] {e}"

def run_single_sample(client: AzureOpenAI,
                      sample_key: str,
                      sample: Dict[str, Any],
                      args,
                      sys_prompt: str,
                      dataset_name: str) -> Dict[str, Any]:
    print("Sample ID:\n", sample_key)
    print("=" * 50)

    contextual_info = get_contextual_info(dataset_name, sample)
    personas_str = ".".join(sample.get("Personas", []))
    print("\033[1;36mSensory Context:\033[0m\n", contextual_info)
    print("=" * 50)

    # Proactive LLM planning (Azure)
    messages = [{"role": "system", "content": sys_prompt}]
    if args.personas == "w_p":
        messages.append({
            "role": "user",
            "content": "Sensory Context:\n" + contextual_info + "\nPersona Context:\n" + personas_str
        })
    else:
        messages.append({"role": "user", "content": contextual_info})

    try:
        planning = azure_inference(client, args.model_base, messages, temperature=0.7, max_tokens=4096)
    except Exception as e:
        planning = f"[azure inference error] {e}"

    print("\033[1;36mProactive LLM Agent Predictions:\033[0m\n", planning)
    print("=" * 50)

    # Parse plan → (thoughts, proactive_idx, proactive_score, actions, tools_str)
    thoughts, p_idx, p_score, actions, tools_str = parse_proactive_agent_results(planning)
    print("\033[1;36mThoughts:\033[0;36m\n", thoughts, "\033[0m")
    print("\033[1;34mProactive Index:\033[0;34m\n", p_idx, "\033[0m")
    print("\033[1;34mProactive Score:\033[0;34m\n", p_score, "\033[0m")
    print("\033[1;37mActions:\033[0;37m\n", actions, "\033[0m")
    print("\033[1;35mTools:\033[0;35m\n", tools_str, "\033[0m")

    # Execute tools
    if tools_str != "None":
        json_tool = parse_tool_spec(tools_str)
        results_tool, memory = run_tools(json_tool)
        print("=" * 50)
        print("\033[1;36mTool Results:\033[0m\n", results_tool)
        print("=" * 50)
        print("\033[1;36mMemory State:\033[0m\n", memory)

        # Summarize with Azure
        response = summarize_with_llm_azure(
            client=client,
            model_base=args.model_base,
            contextual_info=contextual_info,
            personas_str=personas_str,
            thoughts=thoughts,
            results_tool=results_tool,
            personas_flag=args.personas,
        )
        print("=" * 50)
        print("\033[1;36mProactive Response:\033[0m\n", response)
    else:
        results_tool = "None"
        response = "None"

    return {
        "thoughts": thoughts,
        "proactive_idx": p_idx,
        "proactive_score": p_score,
        "actions": actions,
        "tools": tools_str,
        "tools_results": results_tool,
        "response": response,
    }

def save_results_incremental(ds: Dict[str, Any], args) -> str:
    """Save the full dataset (with predictions) after each sample."""
    safe_ds = convert_sets_to_lists(ds)
    if args.zs == "true":
        path = f"results/{args.dataset}/predictions/icl/pred_{args.model_base}_zs.json"
    else:
        path = (
            f"results/{args.dataset}/predictions/icl/"
            f"pred_{args.model_base}_fs_{args.n_fewshot}_{args.personas}_{args.think}.json"
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_ds, f, ensure_ascii=False, indent=4)
    return path

# =========================
# Main Orchestration
# =========================
def main():
    args = parse_args()
    apply_mode(args.mode)
    client = get_azure_client()

    print(f"[main] MODE = {config.MODE}")
    print(f"[main] Loaded tools: {', '.join(sorted(functions.keys()))}")

    # Load base prompt & train set for few-shot examples
    base_prompt = load_prompt_base(args.zs)
    train_path = f"data/{args.dataset}/{args.dataset}_train.json"
    train_ds = load_json(train_path)
    sys_prompt, fewshot_keys = build_system_prompt(
        base_prompt, args.zs, train_ds, args.n_fewshot, args.think, args.personas
    )
    print("Prompt ready with few-shots:", fewshot_keys)

    # Load test set
    test_path = f"data/{args.dataset}/{args.dataset}_test.json"
    test_ds = load_json(test_path)

    # Progress bar + clear separators per sample
    keys_list = list(test_ds.keys())
    total = len(keys_list)
    sep = "=" * 80

    for idx, k in enumerate(tqdm(keys_list, total=total, desc="Evaluating samples", unit="sample"), start=1):
        tqdm.write(f"\n{sep}\n[Sample {idx}/{total}] ID: {k}\n{sep}")
        preds = run_single_sample(client, k, test_ds[k], args, sys_prompt, args.dataset)
        test_ds[k]["predictions"] = preds

        # Save incrementally (same behavior as your original script)
        save_path = save_results_incremental(test_ds, args)
        tqdm.write(f"[main] Saved: {save_path}")
        tqdm.write(f"{'-'*80}\n[Completed] {k} ({idx}/{total})\n{'-'*80}")

    print("\nAll samples finished ✅")

if __name__ == "__main__":
    main()
