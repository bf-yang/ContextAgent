from __future__ import annotations
import ollama
from ollama import chat
import transformers
from bert_score import score
from sklearn.metrics import mean_squared_error
import numpy as np
import json, re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

def azure_inference(client,model_id,messages,temperature,max_tokens):
    response = client.chat.completions.create(
        model=model_id,
        messages=messages, 
        temperature=temperature,
        max_tokens=max_tokens
        )
    response = response.choices[0].message.content
    return response

def ollama_inference(model_id,messages):
    response = ollama.chat(model=model_id, messages=messages,
                           options={"num_ctx":40960})
    return response['message']['content']


def parse_proactive_agent_results(results):
    '''Parse the proactive agent's results'''
    thoughts,proactive_idx,proactive_score, actions, tools = "None","None","None","None","None"

    # Parse <think> <\think> 
    thoughts_match = re.search(r'<think>(.*?)<\\think>', results, re.DOTALL)  
    if thoughts_match:  
        thoughts = thoughts_match.group(1)  
    
    # Parse "Proactive index" 
    proactive_idx_match = re.search(r'"Proactive index": (\w+)', results)  
    if proactive_idx_match:  
        proactive_idx = proactive_idx_match.group(1)  
    
    # Parse "Proactive score"  
    proactive_score_match = re.search(r'"Proactive score": (\d+)', results)  
    if proactive_score_match:  
        proactive_score = proactive_score_match.group(1)  

    action_match = re.search(r"## Action:\s*(.*?)\s*## Tool Calling:", results, re.DOTALL)  
    if action_match:  
        actions = action_match.group(1)

    # Parse "Tool Calling"  
    tool_match = re.search(r"## Tool Calling:\s*(\[.*?\]|None)", results, re.DOTALL)  
    if tool_match:  
        tools = tool_match.group(1)
    return thoughts, proactive_idx, proactive_score, actions, tools

def convert_sets_to_lists(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: convert_sets_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(element) for element in obj]
    else:
        return obj

# Evaluation metrics
def calculate_accuracy(pred_list, gt_list, label):
    if pred_list and gt_list:
        correct = sum(p == g for p, g in zip(pred_list, gt_list))
        accuracy = correct / len(gt_list)
        # print(f"Accuracy for {label}: {accuracy:.2%}")
        # 计算 miss-needed 和 false-detection
        miss_needed = sum(1 for p, g in zip(pred_list, gt_list) if g == "true" and p == "false") / len(gt_list)
        false_detection = sum(1 for p, g in zip(pred_list, gt_list) if g == "false" and p == "true") / len(gt_list)
        incorrect_indices = [i for i, (p, g) in enumerate(zip(pred_list, gt_list)) if p != g]
        return accuracy, incorrect_indices, miss_needed, false_detection

    else:
        print(f"No data available for calculating accuracy for {label}.")
        return []

def calculate_regression_metrics(pred_list, gt_list, label):
    if pred_list and gt_list:
        mse = mean_squared_error(gt_list, pred_list)
        rmse = np.sqrt(mse)
        return mse, rmse
    else:
        print(f"No data available for calculating regression metrics for {label}.")

def calculate_set_metrics(pred_set, gt_set, label):
    intersection = pred_set & gt_set
    precision = len(intersection) / len(pred_set) if pred_set else 0
    recall = len(intersection) / len(gt_set) if gt_set else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"{label} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return precision, recall, f1


# ---------- tiny helpers ----------

def _strip_quotes_or_json(s: str):
    """Trim quotes; if JSON string, parse and return object; otherwise return cleaned string."""
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    try:
        return json.loads(s)
    except Exception:
        return s

def ensure_iso_date(d: datetime) -> str:
    return d.date().isoformat()

# ---------- normalizers for tool results ----------

# def norm_gps(result: Any) -> Dict[str, Any]:
#     """Normalize gps tool output into {city, address} minimal schema."""
#     if isinstance(result, str):
#         result2 = _strip_quotes_or_json(result)
#         if isinstance(result2, dict):
#             result = result2
#         else:
#             city = result.split(",")[0].strip() if result else None
#             return {"city": city, "address": result}
#     if isinstance(result, dict):
#         city = result.get("city")
#         address = result.get("address") or result.get("text") or city
#         if city:
#             return {"city": city, "address": address}
#         return {"text": json.dumps(result, ensure_ascii=False)}
#     return {"text": str(result)}

def _strip_unbalanced_quotes(text: str) -> str:
    """Remove any leading/trailing quote characters (single/double), even if unbalanced."""
    s = str(text).strip()
    s = re.sub(r'^[\'"]+', '', s)   # strip any leading quotes
    s = re.sub(r'[\'"]+$', '', s)   # strip any trailing quotes
    return s

# def norm_gps(result: Any) -> Dict[str, Any]:
#     """
#     Normalize GPS tool output into a minimal schema: {"city", "address"}.

#     This is intentionally robust ("defensive"):
#     - If the tool returned a JSON-encoded string or a quoted string like "\"Hong Kong\"",
#       we parse/clean it.
#     - If we only get a free-form string, we derive city from the first comma-separated token.
#     - If we get a dict, we sanitize its string fields (strip stray quotes).
#     """
#     # Case 1: result is a string -> try to parse JSON first; otherwise sanitize text.
#     if isinstance(result, str):
#         parsed = _strip_quotes_or_json(result)  # may return dict or a cleaned string
#         if isinstance(parsed, dict):
#             data = parsed
#         else:
#             cleaned_text = _strip_unbalanced_quotes(parsed)
#             city = cleaned_text.split(",")[0].strip() if cleaned_text else None
#             return {"city": city, "address": cleaned_text}

#     # Case 2: result is a dict -> standardize fields
#     elif isinstance(result, dict):
#         data = result

#     # Fallback: unknown type -> just expose as text
#     else:
#         return {"text": str(result)}

#     # From here on we have a dict in `data`
#     city = data.get("city")
#     address = data.get("address") or data.get("text") or city

#     # Sanitize string fields (remove stray/unbalanced quotes)
#     if isinstance(city, str):
#         city = _strip_unbalanced_quotes(city)
#     if isinstance(address, str):
#         address = _strip_unbalanced_quotes(address)

#     if city:
#         return {"city": city, "address": address}
#     return {"text": json.dumps(data, ensure_ascii=False)}

def norm_gps(result: Any) -> Dict[str, Any]:
    """
    Normalize GPS tool output into a minimal schema: {"city", "address"}.

    This is intentionally robust ("defensive"):
    - If the tool returned a JSON-encoded string or a quoted string like "\"Hong Kong\"",
      we parse/clean it.
    - If we only get a free-form string, we derive city from the first comma-separated token.
    - If we get a dict, we sanitize its string fields (strip stray quotes).
    """
    def _from_pipe_text(s: str) -> Dict[str, Any]:
        """Extract city/address from 'coarse | fine' without rule-based trimming."""
        # Clean stray quotes first
        s = _strip_unbalanced_quotes(s)
        if "|" not in s:
            # Fallback: city is the first comma-delimited token; address is the whole string
            city = s.split(",")[0].strip() if s else None
            return {"city": city, "address": s}
        coarse, fine = [part.strip() for part in s.split("|", 1)]
        # City = first token (before comma) of the coarse side
        city = coarse.split(",")[0].strip() if coarse else None
        address = fine or None
        return {"city": city, "address": address}

    # Case 1: result is a string
    if isinstance(result, str):
        parsed = _strip_quotes_or_json(result)  # may return dict or a cleaned string
        if isinstance(parsed, dict):
            data = parsed
        else:
            return _from_pipe_text(parsed)

    # Case 2: result is a dict
    elif isinstance(result, dict):
        data = result

    # Fallback: unknown type -> expose as text
    else:
        return {"text": str(result)}

    # From here on we have a dict in `data`
    city = data.get("city")
    address = data.get("address")

    # If dict only contains a pipe-delimited 'text', parse it
    text = data.get("text")
    if (not city or not address) and isinstance(text, str):
        parsed = _from_pipe_text(text)
        city = city or parsed.get("city")
        address = address or parsed.get("address")

    # Sanitize string fields
    if isinstance(city, str):
        city = _strip_unbalanced_quotes(city)
    if isinstance(address, str):
        address = _strip_unbalanced_quotes(address)

    if city:
        return {"city": city, "address": address or city}
    return {"text": json.dumps(data, ensure_ascii=False)}


def norm_now(result: Any) -> Dict[str, Any]:
    """Normalize time tool output into {now_iso} minimal schema."""
    if isinstance(result, str):
        s = _strip_quotes_or_json(result)
        if isinstance(s, str):
            m = re.search(r"Date:\s*(.+?)\s*Time:\s*([\d:]{5,8})", s)
            if m:
                try:
                    dt = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%B %d, %Y %H:%M:%S")
                    return {"now_iso": dt.isoformat()}
                except Exception:
                    return {"now_iso": s}
            return {"now_iso": s}
        result = s  # if parsed to dict
    if isinstance(result, dict):
        return {"now_iso": result.get("now_iso") or result.get("now") or json.dumps(result, ensure_ascii=False)}
    return {"now_iso": str(result)}

def norm_pass(result: Any) -> Dict[str, Any]:
    """Default normalizer: wrap string as {text}, keep dict as-is, convert others to string."""
    if isinstance(result, str):
        return {"text": _strip_quotes_or_json(result)}
    return result if isinstance(result, dict) else {"text": str(result)}

TOOL_NORMALIZERS = {
    "get_current_gps_coordinates": norm_gps,
    "get_current_datetime":        norm_now,
    "get_city_weather":            norm_pass,
    "check_agenda_time_conflict":  norm_pass,
}

# ---------- time phrase parsing ----------

# def parse_time_phrase(phrase: str, now_iso: Optional[str]) -> Optional[Dict[str, str]]:
#     """
#     Parse common time phrases -> {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}.
#     Return None if not matched.
#     """
#     if not isinstance(phrase, str):
#         return None
#     p = phrase.strip().lower()
#     try:
#         base = datetime.fromisoformat(now_iso) if now_iso else datetime.now()
#     except Exception:
#         base = datetime.now()

#     def this_week_range(dt: datetime):
#         monday = dt - timedelta(days=dt.weekday())
#         sunday = monday + timedelta(days=6)
#         return ensure_iso_date(monday), ensure_iso_date(sunday)

#     def this_weekend_range(dt: datetime):
#         sat = dt + timedelta(days=(5 - dt.weekday()) % 7)
#         sun = dt + timedelta(days=(6 - dt.weekday()) % 7)
#         return ensure_iso_date(sat), ensure_iso_date(sun)

#     if p == "today":
#         d = base; s = e = ensure_iso_date(d); return {"start_date": s, "end_date": e}
#     if p == "yesterday":
#         d = base - timedelta(days=1); s = e = ensure_iso_date(d); return {"start_date": s, "end_date": e}
#     if p == "tomorrow":
#         d = base + timedelta(days=1); s = e = ensure_iso_date(d); return {"start_date": s, "end_date": e}
#     if p == "this weekend":
#         s, e = this_weekend_range(base); return {"start_date": s, "end_date": e}
#     if p == "this week":
#         s, e = this_week_range(base); return {"start_date": s, "end_date": e}
#     return None

def parse_time_phrase(phrase: str, now_iso: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Parse common time phrases -> {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}.
    Returns None if not matched so callers can fall back.

    Supported (additive to your originals):
      - "today", "yesterday", "tomorrow"         (existing)
      - "this weekend", "this week"              (existing)
      - "this <weekday>", "next <weekday>"       (NEW; e.g., "next Saturday")
    """
    if not isinstance(phrase, str):
        return None
    p = phrase.strip().lower()
    try:
        base = datetime.fromisoformat(now_iso) if now_iso else datetime.now()
    except Exception:
        base = datetime.now()

    def this_week_range(dt: datetime):
        monday = dt - timedelta(days=dt.weekday())
        sunday = monday + timedelta(days=6)
        return ensure_iso_date(monday), ensure_iso_date(sunday)

    def this_weekend_range(dt: datetime):
        sat = dt + timedelta(days=(5 - dt.weekday()) % 7)
        sun = dt + timedelta(days=(6 - dt.weekday()) % 7)
        return ensure_iso_date(sat), ensure_iso_date(sun)

    # --- existing simple phrases ---
    if p == "today":
        d = base; s = e = ensure_iso_date(d); return {"start_date": s, "end_date": e}
    if p == "yesterday":
        d = base - timedelta(days=1); s = e = ensure_iso_date(d); return {"start_date": s, "end_date": e}
    if p == "tomorrow":
        d = base + timedelta(days=1); s = e = ensure_iso_date(d); return {"start_date": s, "end_date": e}
    if p == "this weekend":
        s, e = this_weekend_range(base); return {"start_date": s, "end_date": e}
    if p == "this week":
        s, e = this_week_range(base); return {"start_date": s, "end_date": e}

    # --- NEW: "this/next <weekday>" (no explicit time -> date range is that single day) ---
    # Examples: "next saturday", "this fri"
    wd_map = {
        "mon": 0, "monday": 0,
        "tue": 1, "tues": 1, "tuesday": 1,
        "wed": 2, "wednesday": 2,
        "thu": 3, "thurs": 3, "thursday": 3,
        "fri": 4, "friday": 4,
        "sat": 5, "saturday": 5,
        "sun": 6, "sunday": 6,
    }
    m = re.match(r"^\s*(this|next)?\s*(mon|monday|tue|tues|tuesday|wed|wednesday|thu|thurs|thursday|fri|friday|sat|saturday|sun|sunday)\s*$", p, flags=re.I)
    if m:
        qualifier = (m.group(1) or "").lower()
        wd_key = m.group(2).lower()
        target_idx = wd_map[wd_key]
        today_idx = base.weekday()
        days_ahead = (target_idx - today_idx) % 7

        # If "next" is asked, always push to the following week.
        # If "this" and it's today, we keep today (common expectation for date-only).
        if qualifier == "next":
            days_ahead += 7

        target_date = (base + timedelta(days=days_ahead)).date()
        s = e = ensure_iso_date(datetime(target_date.year, target_date.month, target_date.day))
        return {"start_date": s, "end_date": e}

    # Not a recognized phrase; let caller handle it
    return None


# ---------- parameters & memory plumbing ----------

def is_no_params(p: Any) -> bool:
    return (p in (None, "None")) or (isinstance(p, dict) and len(p) == 0)

def _is_placeholder(x: Any) -> bool:
    return isinstance(x, str) and x.startswith("$") and x.endswith(")")

def resolve_placeholder(token: str, memory: Dict[str, Any]):
    """
    Two kinds of placeholders:
      1) $RESULT/$RESULTS(tool.field or tool) -> read from norm::<tool>
      2) $CONTEXT("literal") -> return the literal inside quotes
    """
    if token.startswith("$RESULTS(") or token.startswith("$RESULT("):
        inner = token[token.find("(")+1:-1].strip()
        tool, dot, field = inner.partition(".")
        obs = memory.get(f"norm::{tool}")
        if not dot:
            return obs
        return obs.get(field) if isinstance(obs, dict) else None

    if token.startswith("$CONTEXT("):
        inner = token[len("$CONTEXT("):-1].strip()
        if (inner.startswith('"') and inner.endswith('"')) or (inner.startswith("'") and inner.endswith("'")):
            return inner[1:-1]
        raise ValueError(f"$CONTEXT expects a quoted literal, got: {token}")

    return token

def _coerce_time_if_needed(param_name: str, val: Any, memory: Dict[str, Any]):
    """If param is time-like and is a known phrase, convert to date range dict; else return as-is."""
    if isinstance(param_name, str) and param_name.lower() in ("time", "date", "datetime", "when", "time_range"):
        if isinstance(val, str):
            parsed = parse_time_phrase(val, memory.get("now_iso"))
            if parsed:
                memory["time_range"] = parsed
                return parsed
    return val

def fill_args(params: Optional[Dict[str, Any]], memory: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve placeholders, auto-fill from memory, coerce time phrases."""
    if params in (None, "None"):
        return {}
    out: Dict[str, Any] = {}
    for k, v in (params or {}).items():
        if _is_placeholder(v):
            val = resolve_placeholder(v, memory)
            out[k] = _coerce_time_if_needed(k, val, memory)
        elif v in (None, "", "None"):
            val = memory.get(k)
            out[k] = _coerce_time_if_needed(k, val, memory)
        else:
            out[k] = _coerce_time_if_needed(k, v, memory)
    return out

def write_memory(tool_name: str, obs_raw: Any, memory: Dict[str, Any]) -> None:
    """
    Persist both raw and normalized views; promote common aliases (city / now_iso) to top-level.
    """
    memory[f"raw::{tool_name}"] = obs_raw
    norm_fn = TOOL_NORMALIZERS.get(tool_name, norm_pass)
    obs = norm_fn(obs_raw)
    memory[f"norm::{tool_name}"] = obs
    if isinstance(obs, dict):
        if obs.get("city"):
            memory["city"] = obs["city"]
        if obs.get("now_iso"):
            memory["now_iso"] = obs["now_iso"]

def execute_tools_with_memory(json_tool: List[Dict[str, Any]], process_function_call, memory: Dict[str, Any]):
    """
    Two-phase executor:
      1) Run no-parameter tools -> write to memory (raw & norm)
      2) Run parameterized tools -> resolve placeholders/time phrases -> call -> write to memory
    Returns a list of {tool_name, tool_parameters, results} where 'results' prefers normalized view.
    """
    results_tool: List[Dict[str, Any]] = []

    # Phase 1: no-params tools
    for tool_call in json_tool:
        if not isinstance(tool_call, dict) or 'name' not in tool_call or 'parameters' not in tool_call:
            continue
        if is_no_params(tool_call['parameters']):
            name = tool_call['name']
            print("=" * 50)
            print("Calling Function (no-params): ", name)
            try:
                raw = process_function_call({"name": name, "parameters": {}})
                print("Function Results (raw): ", raw)
            except Exception as e:
                raw = {"error": str(e)}
                print("Function Error: ", raw)
            try:
                write_memory(name, raw, memory)
            except Exception as e:
                print("write_memory error:", e)
            norm = memory.get(f"norm::{name}", raw)
            if norm is not raw:
                print("Function Results (normalized): ", norm)
            results_tool.append({"tool_name": name, "tool_parameters": {}, "results": norm})

    # Phase 2: tools with parameters
    for tool_call in json_tool:
        if not isinstance(tool_call, dict) or 'name' not in tool_call or 'parameters' not in tool_call:
            print("Invalid tool call format:", tool_call)
            results_tool.append({"tool_name": 'error', "tool_parameters": 'error', "results": 'error'})
            continue
        if is_no_params(tool_call['parameters']):
            continue

        name = tool_call['name']
        try:
            resolved_params = fill_args(tool_call['parameters'], memory)
        except Exception as e:
            print("=" * 50)
            print("Param resolve error for:", name, "| error:", e)
            results_tool.append({"tool_name": name, "tool_parameters": tool_call['parameters'],
                                 "results": {"error": f"param_resolve_failed: {e}"}})
            continue

        print("=" * 50)
        print("Calling Function: ", name)
        print("Function Params (resolved): ", resolved_params)
        try:
            raw = process_function_call({"name": name, "parameters": resolved_params})
            print("Function Results (raw): ", raw)
        except Exception as e:
            raw = {"error": str(e)}
            print("Function Error: ", raw)

        try:
            write_memory(name, raw, memory)
        except Exception as e:
            print("write_memory error:", e)

        norm = memory.get(f"norm::{name}", raw)
        if norm is not raw:
            print("Function Results (normalized): ", norm)

        results_tool.append({"tool_name": name, "tool_parameters": resolved_params, "results": norm})

    return results_tool