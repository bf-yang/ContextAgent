# Evaluate SFT-based approaches (modular version, with apply_mode and two-stage tool execution)

import os
import sys
import json
import ast
import argparse
from typing import Dict, Any, List, Tuple
from datetime import datetime
from tqdm import tqdm

# Project imports
import config
from tool_registry import functions, process_function_call
from utils import (
    parse_proactive_agent_results,
    convert_sets_to_lists,
    execute_tools_with_memory,   # two-stage tool executor
)
from openai import OpenAI  # local SFT server, OpenAI-compatible API

# =========================
# Args & Mode
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["live", "sandbox"], default="sandbox")
    p.add_argument("--port", type=int, default=8009, help="port for your local SFT model server")
    p.add_argument("--model_base", type=str, default="qwen7b",
                  help='base SFT model on your local server: qwen7b, llama8b, deepseek7b')
    p.add_argument("--dataset", type=str, default="cab", help="cab, cab_lite, cab_ood")
    p.add_argument("--think", type=str, default="w_t", help="w_t, wo_t")
    p.add_argument("--personas", type=str, default="w_p", help="w_p, wo_p")
    return p.parse_args()

def apply_mode(mode: str):
    """Propagate global mode to all tools via config."""
    try:
        from config import set_mode
        set_mode(mode)
    except Exception:
        config.MODE = mode

# =========================
# Client & IO
# =========================
def get_sft_client(API_PORT=8009) -> OpenAI:
    """
    Build a local OpenAI-compatible client for your SFT model server.
    Controlled by environment variables:
      API_KEY  (default: "0")
      API_PORT (default: 8009)
    """
    api_key = os.environ.get("API_KEY", "0")
    port = int(os.environ.get("API_PORT", API_PORT))
    return OpenAI(api_key=api_key, base_url=f"http://localhost:{port}/v1")

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_prompt_base(think_flag: str) -> str:
    path = "prompt/baselines/prompt_sys_wo_t.txt" if think_flag == "wo_t" else "prompt/prompt_sys.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# =========================
# Planning / Tools / Summary
# =========================
def get_contextual_info(dataset_name: str, sample: Dict[str, Any]) -> str:
    if dataset_name in ("cab", "cab_ood"):
        return sample["Context information"]
    if dataset_name == "cab_lite":
        return sample["Rawdata Context"]
    return sample.get("Context information", "")

def sft_infer_planning(client: OpenAI,
                       model_name: str,
                       sys_prompt: str,
                       contextual_info: str,
                       personas_str: str,
                       personas_flag: str) -> str:
    """Call the local SFT model to produce the proactive planning output."""
    messages = [{"role": "system", "content": sys_prompt}]
    if personas_flag == "w_p":
        user = f"Sensory Context:\n{contextual_info}\nPersona Context:\n{personas_str}"
    else:
        user = f"Sensory Context:\n{contextual_info}"
    messages.append({"role": "user", "content": user})

    try:
        resp = client.chat.completions.create(messages=messages, model=model_name)
        return resp.choices[0].message.content
    except Exception as e:
        return f"[sft planning error] {e}"

def parse_tool_spec(tools_str: str) -> List[Dict[str, Any]]:
    if tools_str == "None":
        return []
    try:
        parsed = ast.literal_eval(tools_str)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []

def run_tools(json_tool: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Two-stage executor + memory (no-params first, then param tools)."""
    memory = {"now_iso": datetime.now().isoformat()}
    results_tool = execute_tools_with_memory(json_tool, process_function_call, memory)
    return results_tool, memory

def summarize_with_sft(client: OpenAI,
                       model_name: str,
                       contextual_info: str,
                       personas_str: str,
                       thoughts: str,
                       results_tool: Any,
                       personas_flag: str) -> str:
    """
    Optional: summarize with the same SFT model.
    If you prefer a stub for speed/cost, return a fixed string instead.
    """
    try:
        with open("prompt/prompt_summarize.txt", "r", encoding="utf-8") as f:
            prompt_summarize = f.read()
    except Exception:
        prompt_summarize = "Summarize the following context and tool results."

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

    # Sandbox mode: avoid a second model call and just return a stub
    if config.MODE == "sandbox":
        return "sandbox outputs."

    messages = [
        {"role": "system", "content": prompt_summarize},
        {"role": "user", "content": content},
    ]
    try:
        resp = client.chat.completions.create(messages=messages, model=model_name)
        return resp.choices[0].message.content
    except Exception as e:
        return f"[sft summarize error] {e}"

# =========================
# Per-sample
# =========================
def run_single_sample(client: OpenAI,
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
    print("\033[1;36mPersona Context:\033[0m\n", personas_str)
    print("=" * 50)

    # 1) Planning with SFT model
    planning = sft_infer_planning(
        client=client,
        model_name=args.model_base,
        sys_prompt=sys_prompt,
        contextual_info=contextual_info,
        personas_str=personas_str,
        personas_flag=args.personas,
    )
    print("\033[1;36mProactive LLM Agent Predictions:\033[0m\n", planning)
    print("=" * 50)

    # 2) Parse plan
    thoughts, p_idx, p_score, actions, tools_str = parse_proactive_agent_results(planning)
    print("\033[1;36mThoughts:\033[0;36m\n", thoughts, "\033[0m")
    print("\033[1;34mProactive Index:\033[0;34m\n", p_idx, "\033[0m")
    print("\033[1;34mProactive Score:\033[0;34m\n", p_score, "\033[0m")
    print("\033[1;37mActions:\033[0;37m\n", actions, "\033[0m")
    print("\033[1;35mTools:\033[0;35m\n", tools_str, "\033[0m")

    # 3) Tools
    if tools_str != "None":
        json_tool = parse_tool_spec(tools_str)
        results_tool, memory = run_tools(json_tool)
        print("=" * 50)
        print("\033[1;36mTool Results:\033[0m\n", results_tool)
        print("=" * 50)
        print("\033[1;36mMemory State:\033[0m\n", memory)

        # 4) Summarize with SFT (or stub in sandbox)
        response = summarize_with_sft(
            client=client,
            model_name=args.model_base,
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

# =========================
# Save
# =========================
def save_results_incremental(ds: Dict[str, Any], args) -> str:
    """Save the full dataset (with predictions) after each sample."""
    ds = convert_sets_to_lists(ds)
    path = f"results/{args.dataset}/predictions/sft/pred_{args.model_base}_{args.personas}_{args.think}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ds, f, ensure_ascii=False, indent=4)
    return path

# =========================
# Main Orchestration
# =========================
def main():
    args = parse_args()
    apply_mode(args.mode)
    client = get_sft_client(args.port)

    print(f"[main] MODE = {config.MODE}")
    print(f"[main] Loaded tools: {', '.join(sorted(functions.keys()))}")

    # Load prompt
    sys_prompt = load_prompt_base(args.think)
    print(sys_prompt)

    # Load test set
    dataset_name = args.dataset
    test_path = f"data/{dataset_name}/{dataset_name}_test.json"
    dataset = load_json(test_path)
    print(dataset.keys())

    # Progress bar + separators
    keys_list = list(dataset.keys())
    total = len(keys_list)
    sep = "=" * 80

    for idx, key in enumerate(tqdm(keys_list, total=total, desc="Evaluating samples", unit="sample"), start=1):
        print(f"\n{sep}\n[Sample {idx}/{total}] ID: {key}\n{sep}")
        preds = run_single_sample(client, key, dataset[key], args, sys_prompt, dataset_name)
        dataset[key]["predictions"] = preds

        save_path = save_results_incremental(dataset, args)
        print(f"[main] Saved: {save_path}")
        print(f"{'-'*80}\n[Completed] {key} ({idx}/{total})\n{'-'*80}")

    print("\nAll samples finished ✅")

if __name__ == "__main__":
    main()
