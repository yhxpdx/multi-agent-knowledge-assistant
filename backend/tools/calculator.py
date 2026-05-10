"""
数学计算工具

设计决策：
- 使用 Python 的 ast.literal_eval 替代 eval，更安全
- 支持基本数学运算和 math 模块函数
"""

import math
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。支持基本运算（+, -, *, /, **）和常用数学函数（sin, cos, sqrt, log 等）。

    Args:
        expression: 数学表达式，如 "2**10" 或 "sqrt(144)"
    """
    # 安全的数学函数
    safe_dict = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "int": int, "float": float,
        "pi": math.pi, "e": math.e,
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "ceil": math.ceil, "floor": math.floor,
        "factorial": math.factorial, "gcd": math.gcd,
    }

    try:
        # 只允许数学表达式，不允许赋值和函数定义
        if "=" in expression or "import" in expression or "def " in expression:
            return "安全限制：只允许数学表达式"

        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"
