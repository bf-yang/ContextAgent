from importlib import import_module
from pkgutil import iter_modules

# Merge all FUNCTIONS dicts exported by submodules
functions = {}

for m in iter_modules(__path__):  # Automatically discover *.py modules in the same directory
    if m.name.startswith("_"):
        continue
    mod = import_module(f"{__name__}.{m.name}")
    if hasattr(mod, "FUNCTIONS"):
        functions.update(getattr(mod, "FUNCTIONS"))

def process_function_call(json_tool):
    """
    Look up the function by name in functions and call it
    """
    import json as _json
    try:
        name = json_tool["name"]
        params = json_tool.get("parameters", {})
        if name not in functions:
            raise NotImplementedError(f"Function {name} is not implemented.")
        fn = functions[name]
        result = fn(**params) if params and params != "None" else fn()
        return _json.dumps(result)
    except Exception as e:
        return _json.dumps({"error": str(e)})
