"""
工具函数单元测试

测试覆盖：
1. calculator - 基本运算、数学函数、安全限制
2. code_executor - 正常执行、安全检查、异常处理
"""

import pytest
from backend.tools.calculator import calculator
from backend.tools.code_executor import code_executor, _check_code_safety


class TestCalculator:
    """计算器工具测试"""

    def test_basic_addition(self):
        result = calculator.invoke({"expression": "2 + 3"})
        assert "5" in result

    def test_basic_multiplication(self):
        result = calculator.invoke({"expression": "6 * 7"})
        assert "42" in result

    def test_power(self):
        result = calculator.invoke({"expression": "2 ** 10"})
        assert "1024" in result

    def test_sqrt(self):
        result = calculator.invoke({"expression": "sqrt(144)"})
        assert "12" in result

    def test_trig_functions(self):
        result = calculator.invoke({"expression": "sin(0)"})
        assert "0" in result

    def test_pi_constant(self):
        result = calculator.invoke({"expression": "pi"})
        assert "3.14" in result

    def test_complex_expression(self):
        result = calculator.invoke({"expression": "(2 + 3) * 4 ** 2"})
        assert "80" in result

    def test_division_by_zero(self):
        result = calculator.invoke({"expression": "1 / 0"})
        assert "错误" in result or "error" in result.lower()

    def test_security_block_assignment(self):
        """测试阻止赋值操作"""
        result = calculator.invoke({"expression": "x = 5"})
        assert "安全限制" in result

    def test_security_block_import(self):
        """测试阻止 import"""
        result = calculator.invoke({"expression": "import os"})
        assert "安全限制" in result

    def test_invalid_expression(self):
        result = calculator.invoke({"expression": "invalid_expr"})
        assert "错误" in result or "error" in result.lower()


class TestCodeExecutor:
    """代码执行工具测试"""

    def test_simple_print(self):
        result = code_executor.invoke({"code": "print('Hello, World!')"})
        assert "Hello, World!" in result

    def test_arithmetic(self):
        result = code_executor.invoke({"code": "print(2 + 3)"})
        assert "5" in result

    def test_list_operations(self):
        result = code_executor.invoke({"code": "nums = [1, 2, 3]\nprint(sum(nums))"})
        assert "6" in result

    def test_loop(self):
        result = code_executor.invoke({"code": "s = 0\nfor i in range(5):\n    s += i\nprint(s)"})
        assert "10" in result

    def test_no_output(self):
        result = code_executor.invoke({"code": "x = 42"})
        assert "无输出" in result

    def test_security_block_os_import(self):
        """测试阻止 os 模块导入"""
        result = code_executor.invoke({"code": "import os"})
        assert "安全限制" in result

    def test_security_block_subprocess(self):
        """测试阻止 subprocess 模块"""
        result = code_executor.invoke({"code": "import subprocess"})
        assert "安全限制" in result

    def test_security_block_open(self):
        """测试阻止 open 函数"""
        result = code_executor.invoke({"code": "open('/etc/passwd')"})
        assert "安全限制" in result

    def test_runtime_error(self):
        result = code_executor.invoke({"code": "print(1/0)"})
        assert "错误" in result or "Error" in result

    def test_syntax_error(self):
        result = code_executor.invoke({"code": "def("})
        assert "错误" in result or "Error" in result

    def test_output_truncation(self):
        """测试输出截断"""
        code = "print('x' * 5000)"
        result = code_executor.invoke({"code": code})
        assert len(result) <= 2100  # 2000 + 一些开销


class TestCodeSafety:
    """代码安全检查测试"""

    def test_safe_code(self):
        assert _check_code_safety("print('hello')") is None

    def test_dangerous_import_os(self):
        assert _check_code_safety("import os") is not None

    def test_dangerous_import_sys(self):
        assert _check_code_safety("from sys import path") is not None

    def test_dangerous_import_requests(self):
        assert _check_code_safety("import requests") is not None

    def test_dangerous_open(self):
        assert _check_code_safety("open('file.txt')") is not None

    def test_dangerous_exec(self):
        assert _check_code_safety("exec('code')") is not None
