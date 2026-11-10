"""Prompt formatting and code extraction for code reasoning benchmarks."""

import ast
import re
from typing import List, Optional


def extract_target_function_name(assertion_list: List[str]) -> Optional[str]:
    """
    Dynamically extracts the target function name from MBPP assertion strings.
    Handles standard assertions (assert func(...) == ...) as well as wrapper calls
    like math.isclose(func(...), ...).
    """
    for test_str in assertion_list:
        test_str = test_str.strip()
        if not test_str:
            continue
        try:
            tree = ast.parse(test_str)
            if not tree.body or not isinstance(tree.body[0], ast.Assert):
                continue
            test_node = tree.body[0].test

            # Pattern 1: assert func(...) == ... or similar comparison
            if isinstance(test_node, ast.Compare):
                left = test_node.left
                if isinstance(left, ast.Call):
                    if isinstance(left.func, ast.Name):
                        return left.func.id
                    elif isinstance(left.func, ast.Attribute):
                        return left.func.attr

            # Pattern 2: assert math.isclose(func(...), ...) or wrapper call
            elif isinstance(test_node, ast.Call):
                for arg in test_node.args:
                    if isinstance(arg, ast.Call):
                        if isinstance(arg.func, ast.Name):
                            return arg.func.id
                        elif isinstance(arg.func, ast.Attribute):
                            return arg.func.attr
                if isinstance(test_node.func, ast.Name):
                    return test_node.func.id
        except Exception:
            # Fallback regex if AST parsing encountered unconventional syntax
            pass

        # Regex fallback: assert <func_name>(...
        match = re.search(r"assert\s+(?:math\.isclose\s*\(\s*)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", test_str)
        if match:
            return match.group(1)

    return None


def build_mbpp_prompt(problem_desc: str, function_name: Optional[str] = None) -> str:
    """
    Constructs an instruction prompt that enforces the exact function signature
    to avoid benchmark failure due to signature mismatch.
    """
    clean_desc = problem_desc.strip()
    if function_name:
        return (
            f"Write a Python function named `{function_name}` to solve the following problem:\n"
            f"{clean_desc}\n\n"
            f"Requirements:\n"
            f"- Your function MUST be named `{function_name}`.\n"
            f"- Include any necessary imports.\n"
            f"- Output ONLY executable Python code enclosed inside a ```python ... ``` block without explanation."
        )
    return (
        f"Write a Python function to solve the following problem:\n"
        f"{clean_desc}\n\n"
        f"Requirements:\n"
        f"- Include any necessary imports.\n"
        f"- Output ONLY executable Python code enclosed inside a ```python ... ``` block without explanation."
    )


def extract_python_code(text: str) -> str:
    """
    Extracts pure Python code from model responses, handling markdown code fences
    and filtering conversational commentary.
    """
    # 1. Look for ```python ... ``` or ``` ... ```
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. Fallback: filter conversational prose
    lines = text.split("\n")
    code_lines = []
    for line in lines:
        stripped = line.strip()
        if (
            stripped.startswith("Here is")
            or stripped.startswith("Sure,")
            or stripped.startswith("Explanation:")
            or stripped.startswith("Note:")
        ):
            continue
        code_lines.append(line)

    return "\n".join(code_lines).strip()

