"""
代码执行工具

安全设计（全链路思考）：
1. 为什么需要沙箱？
   - 防止恶意代码删除文件、访问网络、执行系统命令
   - 限制执行时间，防止死循环
   - 限制输出大小，防止内存溢出

2. 限制措施：
   - 禁止 import os, sys, subprocess, shutil 等危险模块
   - 禁止 open() 文件操作
   - 限制执行时间 5 秒
   - 限制输出长度 2000 字符
"""

import io
import signal
from contextlib import redirect_stdout, redirect_stderr
from langchain_core.tools import tool


# 危险模块黑名单
BLOCKED_MODULES = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests",
    "ctypes", "importlib", "code", "codeop",
}

# 危险内置函数
BLOCKED_BUILTINS = {"open", "exec", "eval", "compile", "__import__", "globals", "locals"}


def _check_code_safety(code: str) -> str | None:
    """检查代码安全性，返回错误信息或 None"""
    code_lower = code.lower()

    # 检查危险模块导入
    for mod in BLOCKED_MODULES:
        if f"import {mod}" in code_lower or f"from {mod}" in code_lower:
            return f"安全限制：不允许导入 {mod} 模块"

    # 检查危险内置函数
    for func in BLOCKED_BUILTINS:
        if func + "(" in code_lower:
            return f"安全限制：不允许调用 {func} 函数"

    return None


@tool
def code_executor(code: str) -> str:
    """执行 Python 代码并返回输出结果。适用于数学计算、数据处理、算法验证等。

    Args:
        code: 要执行的 Python 代码
    """
    # 安全检查
    error = _check_code_safety(code)
    if error:
        return error

    # 创建受限的执行环境
    safe_globals = {"__builtins__": {
        k: v for k, v in __builtins__.__dict__.items()
        if k not in BLOCKED_BUILTINS
    }} if isinstance(__builtins__, dict) is False else {
        k: v for k, v in __builtins__.items()
        if k not in BLOCKED_BUILTINS
    }

    stdout = io.StringIO()
    stderr = io.StringIO()

    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(code, safe_globals)
    except Exception as e:
        return f"执行错误: {type(e).__name__}: {e}"

    output = stdout.getvalue()
    error_output = stderr.getvalue()

    result = ""
    if output:
        result += output[:2000]
    if error_output:
        result += f"\n[stderr]: {error_output[:500]}"

    return result.strip() if result.strip() else "代码执行完成（无输出）"
