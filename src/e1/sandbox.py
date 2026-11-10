"""Guarded execution sandbox for verifying code generation against assertions."""

import subprocess
import sys
from typing import Dict, List, Optional


def sandbox_execute_code(
    code: str,
    assertions: List[str],
    test_imports: Optional[List[str]] = None,
    timeout_sec: int = 5,
) -> Dict[str, object]:
    """
    Executes generated code against MBPP unit test assertions in an isolated
    subprocess guarded by a strict timeout to prevent infinite loops and hangs.

    Returns a dict with:
      - status: 'PASS', 'FAIL (Assertion/Syntax)', 'TIMEOUT (Loop Guard)', 'ERROR (...)'
      - returncode: int or None
      - stdout: str
      - stderr: str
    """
    # Prepend required test harness imports (e.g. import math)
    import_header = "\n".join(test_imports) if test_imports else ""
    assert_str = "\n".join(assertions)
    full_script = f"{import_header}\n\n{code}\n\n{assert_str}\nprint('PASS_ALL')"

    try:
        proc = subprocess.run(
            [sys.executable, "-c", full_script],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if proc.returncode == 0 and "PASS_ALL" in proc.stdout:
            return {
                "status": "PASS",
                "returncode": 0,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        else:
            return {
                "status": "FAIL (Assertion/Syntax)",
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
    except subprocess.TimeoutExpired as e:
        return {
            "status": "TIMEOUT (Loop Guard)",
            "returncode": None,
            "stdout": e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or ""),
            "stderr": e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or ""),
        }
    except Exception as e:
        return {
            "status": f"ERROR ({e.__class__.__name__})",
            "returncode": None,
            "stdout": "",
            "stderr": str(e),
        }

