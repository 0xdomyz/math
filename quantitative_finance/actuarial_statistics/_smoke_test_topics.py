import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TIMEOUT_SECONDS = 45


def topic_py_files(root: Path):
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        topic = d.name
        py = d / f"{topic}.py"
        if py.exists():
            yield py


def run_file(py_file: Path):
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    cmd = [sys.executable, str(py_file)]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(py_file.parent),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env=env,
        )
        return {
            "file": str(py_file),
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout_tail": "\n".join(result.stdout.splitlines()[-20:]),
            "stderr_tail": "\n".join(result.stderr.splitlines()[-40:]),
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "file": str(py_file),
            "ok": False,
            "returncode": None,
            "stdout_tail": (exc.stdout or "")[-4000:],
            "stderr_tail": (exc.stderr or "")[-4000:],
            "timeout": True,
        }


def main():
    files = sorted(topic_py_files(ROOT))
    results = [run_file(py) for py in files]
    failures = [r for r in results if not r["ok"]]

    summary = {
        "total": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "failures": failures,
    }

    out = ROOT / "_smoke_test_report.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Total: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
