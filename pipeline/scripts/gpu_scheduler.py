#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU 自动调度:检测显卡、采样显存占用、判定每张卡能否再放一个本流程任务。

设计目标
--------
流程化计算时自动利用服务器上所有显卡。策略(与需求一致):

  - 本流程的任务【一张卡只放一个】, 不自己跟自己在同卡拼多路。
  - 但允许和【别人已经在跑的任务】共享同一张卡:开跑前先检查这张卡,

        别人的显存占用峰值 × (1 + reserve) + 一个本流程任务显存 ≤ 该卡总显存

    满足(即留足别人峰值 + 冗余后还塞得下我一个任务)就用这张卡, 和别人一起跑;
    否则跳过这张卡, 不去和别人抢显存。

冗余默认 10%, 采样窗口默认 5s(取窗口内别人占用的峰值, 而非瞬时值, 更安全)。

只依赖标准库 + nvidia-smi, 因此在 pipeline 主环境和 RFDiffusion conda 环境里都能直接用。

作为库
------
    from gpu_scheduler import plan_slots
    gpus = plan_slots(task_vram_mib=8000)   # -> 例如 ["2", "3"](每卡最多出现一次)

返回的是一个「可用 GPU 编号」列表, 直接作为 redesign_pipeline 的 --gpu 传入:
列表里每个卡号最多出现一次 —— 一张卡就跑一路本流程 worker。

作为命令行(便于单独调试)
------------------------
    python gpu_scheduler.py --task-vram 8000 --reserve 0.1 --sample 5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def _run_smi(query: str) -> list[list[str]]:
    """跑一次 nvidia-smi --query-gpu, 返回按行、按逗号切好的字段矩阵。"""
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}",
         "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        rows.append([c.strip() for c in line.split(",")])
    return rows


def detect_gpus() -> list[dict]:
    """列出所有可见 GPU 的静态信息:[{index, name, total_mib}, ...]。

    检测不到(无 nvidia-smi / 无卡)返回空列表, 由调用方决定退回 CPU 或报错。
    """
    try:
        rows = _run_smi("index,name,memory.total")
    except (OSError, subprocess.CalledProcessError):
        return []
    gpus = []
    for idx, name, total in rows:
        try:
            gpus.append({"index": int(idx), "name": name,
                         "total_mib": float(total)})
        except ValueError:
            continue
    return gpus


def sample_peak_used(sample_seconds: float = 5.0,
                     interval: float = 0.5) -> dict[int, float]:
    """在采样窗口内多次读取各卡「已用显存」, 返回每张卡的峰值占用(MiB)。

    这是「别人正在占用」的量(含本进程还没起的任务), 用它加冗余作为安全边界。
    """
    peak: dict[int, float] = {}
    deadline = time.time() + max(0.0, sample_seconds)
    first = True
    while first or time.time() < deadline:
        first = False
        try:
            rows = _run_smi("index,memory.used")
        except (OSError, subprocess.CalledProcessError):
            break
        for idx, used in rows:
            try:
                i, u = int(idx), float(used)
            except ValueError:
                continue
            if u > peak.get(i, -1.0):
                peak[i] = u
        if time.time() >= deadline:
            break
        time.sleep(interval)
    return peak


def compute_slots(gpus: list[dict], peak_used: dict[int, float],
                  task_vram_mib: float, reserve: float = 0.10,
                  max_per_gpu: int = 1) -> dict[int, int]:
    """判定每张卡能放几个本流程任务(默认封顶 1:一卡一任务)。

    可放数 k = floor( (总显存 − 别人占用峰值 × (1 + reserve)) / 任务显存 ), 且 ≥ 0,
    再对 max_per_gpu 封顶。max_per_gpu=1(默认)即:别人峰值+冗余后还塞得下我一个
    任务就用(k=1, 与别人共享), 否则不用(k=0)。
    """
    slots: dict[int, int] = {}
    for g in gpus:
        idx = g["index"]
        total = g["total_mib"]
        used = peak_used.get(idx, 0.0)
        avail = total - used * (1.0 + reserve)
        k = int(avail // task_vram_mib) if task_vram_mib > 0 else 0
        if k < 0:
            k = 0
        if max_per_gpu is not None:
            k = min(k, max_per_gpu)
        slots[idx] = k
    return slots


def slots_to_gpu_list(slots: dict[int, int]) -> list[str]:
    """把 {gpu: 可放数} 展开成 round-robin 的 GPU 编号列表(字符串)。

    默认每卡 0/1, 展开后每个卡号最多出现一次。例: {0:0, 2:1, 3:1} -> ['2','3']。
    (max_per_gpu>1 时才会出现同卡多次, 一般不用。)
    """
    remaining = {k: v for k, v in slots.items() if v > 0}
    order = sorted(remaining)
    out: list[str] = []
    while remaining:
        for idx in order:
            if remaining.get(idx, 0) > 0:
                out.append(str(idx))
                remaining[idx] -= 1
                if remaining[idx] == 0:
                    del remaining[idx]
    return out


def plan_slots(task_vram_mib: float, reserve: float = 0.10,
               sample_seconds: float = 5.0, max_per_gpu: int = 1,
               only_gpus: list[int] | None = None,
               verbose: bool = False) -> list[str]:
    """一站式:检测 → 采样 → 判定 → 展开成 --gpu 列表。

    默认 max_per_gpu=1:本流程一张卡只放一个任务, 但允许与别人已有任务共享该卡
    (只要别人峰值 + 冗余后仍塞得下)。only_gpus 非空时只在这些物理卡里调度。
    检测不到显卡返回空列表。
    """
    gpus = detect_gpus()
    if only_gpus is not None:
        want = set(only_gpus)
        gpus = [g for g in gpus if g["index"] in want]
    if not gpus:
        if verbose:
            print("[gpu-scheduler] 未检测到可用 GPU", file=sys.stderr)
        return []

    peak = sample_peak_used(sample_seconds)
    slots = compute_slots(gpus, peak, task_vram_mib, reserve, max_per_gpu)
    gpu_list = slots_to_gpu_list(slots)

    if verbose:
        print("[gpu-scheduler] 任务显存估算 = %.0f MiB, 冗余 = %.0f%%, 采样 = %.1fs"
              % (task_vram_mib, reserve * 100, sample_seconds), file=sys.stderr)
        for g in gpus:
            idx = g["index"]
            ok = "可用" if slots.get(idx, 0) > 0 else "跳过(放不下)"
            print("  GPU %d %-22s 总 %.0f  别人峰值占用 %.0f  -> %s"
                  % (idx, g["name"], g["total_mib"], peak.get(idx, 0.0), ok),
                  file=sys.stderr)
        print("[gpu-scheduler] 可用显卡(%d 张): %s"
              % (len(gpu_list), ",".join(gpu_list) or "(空)"), file=sys.stderr)
    return gpu_list


def main():
    ap = argparse.ArgumentParser(description="GPU 自动调度:按空闲显存打包任务")
    ap.add_argument("--task-vram", type=float, required=True,
                    dest="task_vram", help="单个任务估算显存占用(MiB)")
    ap.add_argument("--reserve", type=float, default=0.10,
                    help="给已有占用峰值留的冗余比例(默认 0.10)")
    ap.add_argument("--sample", type=float, default=5.0,
                    help="显存采样窗口秒数(默认 5)")
    ap.add_argument("--max-per-gpu", type=int, default=1, dest="max_per_gpu",
                    help="单卡最多放几个本流程任务(默认 1:一卡一任务, 只与别人共享)")
    ap.add_argument("--only-gpus", default=None, dest="only_gpus",
                    help="只用这些物理卡, 逗号分隔, 如 2,3")
    args = ap.parse_args()

    only = None
    if args.only_gpus:
        only = [int(t) for t in args.only_gpus.replace(",", " ").split()]

    gpu_list = plan_slots(args.task_vram, args.reserve, args.sample,
                          args.max_per_gpu, only, verbose=True)
    # stdout 只打印可直接喂给 --gpu 的列表, 方便脚本 $(...) 捕获
    print(",".join(gpu_list))


if __name__ == "__main__":
    main()
