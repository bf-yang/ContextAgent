from importlib import import_module
from pkgutil import iter_modules

# 合并所有子模块导出的 FUNCTIONS 字典
functions = {}

for m in iter_modules(__path__):  # 自动发现同目录下的 *.py 模块
    if m.name.startswith("_"):
        continue
    mod = import_module(f"{__name__}.{m.name}")
    if hasattr(mod, "FUNCTIONS"):
        functions.update(getattr(mod, "FUNCTIONS"))

def process_function_call(json_tool):
    """
    与你原来的处理器保持一致：按 name 在 functions 里找对应函数并调用
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
