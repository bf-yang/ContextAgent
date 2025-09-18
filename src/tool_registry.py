from __future__ import annotations

# --- make sure 'src' is importable when running "python src/tool_registry.py" ---
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent))  # adds .../src to sys.path

import json
import logging
import inspect
from importlib import import_module
from pkgutil import iter_modules
from typing import Dict, Callable, Any

# required: src/tools/__init__.py exists
import tools

# optional: print current mode if you have src/config.py
try:
    import config  # has MODE or is_sandbox()
except Exception:
    config = None

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("tool_registry")

# -------- auto-discover FUNCTIONS from tools/* ----------
functions: Dict[str, Callable[..., Any]] = {}
module_tools: Dict[str, list[str]] = {}  # module_name -> [tool_names]

for m in iter_modules(tools.__path__):
    mod_name = m.name
    if mod_name.startswith("_"):
        continue
    try:
        mod = import_module(f"tools.{mod_name}")
    except Exception as e:
        logger.warning("Failed to import tools.%s: %s", mod_name, e)
        continue

    funcs = getattr(mod, "FUNCTIONS", None)
    if not isinstance(funcs, dict):
        continue

    for tool_name, fn in funcs.items():
        if not callable(fn):
            logger.warning("Skip non-callable '%s' in tools.%s", tool_name, mod_name)
            continue
        if tool_name in functions:
            logger.warning("Duplicate tool name '%s': tools.%s overrides previous", tool_name, mod_name)
        functions[tool_name] = fn
        module_tools.setdefault(mod_name, []).append(tool_name)

def process_function_call(json_tool: dict) -> str:
    """
    Dispatch by tool name. Input: {"name": "...", "parameters": {...}|"None"|None}
    Return: JSON-serialized result (or {"error": "..."}).
    """
    try:
        name = json_tool["name"]
        params = json_tool.get("parameters", {}) or {}
        fn = functions.get(name)
        if fn is None:
            raise NotImplementedError(f"Function {name} is not implemented.")
        result = fn(**params) if params else fn()
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ------------ pretty print when run as a script ------------
if __name__ == "__main__":
    mode_txt = None
    if config is not None:
        try:
            # prefer MODE if present, fallback to is_sandbox()
            if hasattr(config, "MODE"):
                mode_txt = config.MODE
            elif hasattr(config, "is_sandbox") and callable(config.is_sandbox):
                mode_txt = "sandbox" if config.is_sandbox() else "live"
        except Exception:
            pass

    if mode_txt:
        print(f"[tool_registry] MODE: {mode_txt}")
    print(f"[tool_registry] Discovered {len(functions)} tools from {len(module_tools)} modules.")

    # List modules and their tools
    for mod_name in sorted(module_tools):
        names = sorted(module_tools[mod_name])
        print(f"  - tools.{mod_name} ({len(names)}): {', '.join(names)}")

    # Detailed list with signature & first line of docstring
    print("\n[tool_registry] Tool details:")
    for tname in sorted(functions):
        fn = functions[tname]
        try:
            sig = str(inspect.signature(fn))
        except Exception:
            sig = "(...)"
        doc = (inspect.getdoc(fn) or "").strip().splitlines()
        doc1 = doc[0] if doc else ""
        # find module short name (after 'tools.')
        modpath = getattr(fn, "__module__", "") or ""
        short_mod = modpath.split("tools.", 1)[-1] if "tools." in modpath else modpath
        print(f"  • {tname}{sig}  <- {short_mod}  {('- ' + doc1) if doc1 else ''}")
