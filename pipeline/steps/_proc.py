"""步骤公用工具:定位 scripts/ 下的脚本、用 subprocess 调用并实时回显日志。

各 step 的 run() 只负责把 ctx(输入/输出目录、参数)翻译成命令行,
真正的计算逻辑都在 pipeline/scripts/ 里那些已验证的脚本中。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# pipeline/scripts/ 的绝对路径(本文件在 pipeline/steps/ 下)
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def script_path(name: str) -> Path:
    """返回 scripts/ 下某脚本的绝对路径,不存在则报错。"""
    p = SCRIPTS_DIR / name
    if not p.is_file():
        raise FileNotFoundError(f"找不到计算脚本: {p}")
    return p


def run_script(name: str, args: list, log_path: Path | None = None,
               python: str | None = None, cwd: Path | None = None,
               env: dict | None = None) -> int:
    """用指定 python 运行 scripts/<name>,把 stdout+stderr 同时写日志并回显。

    参数:
      name    —— scripts/ 下的脚本文件名
      args    —— 传给脚本的命令行参数列表(已是 str)
      log_path—— 日志文件;为 None 则只回显不落盘
      python  —— 解释器路径;默认用当前 sys.executable
      cwd/env —— 透传给 subprocess

    返回脚本退出码(不抛异常,由调用方判断)。
    """
    py = python or sys.executable
    cmd = [py, str(script_path(name)), *map(str, args)]
    print(f"[step] 运行: {' '.join(cmd)}", flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[step] 日志: {log_path}", flush=True)
        with open(log_path, "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    cwd=str(cwd) if cwd else None, env=env)
            for line in proc.stdout:
                lf.write(line)
                sys.stdout.write(line)
            proc.wait()
        return proc.returncode
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env).returncode
