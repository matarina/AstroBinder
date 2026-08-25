#!/usr/bin/env python3
from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

from make_length_normalized_rosetta_report import SETS, load_dataset, top_rows

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "pipeline_runs/rosetta_length_normalized_comparison.md"
GROUPS = ["round1", "round2", "BindCraft"]
METRICS = [
    ("dG_per_binder", "dG/binder-res"),
    ("dG_cross_per_binder", "dGcross/binder-res"),
    ("dSASA_per_binder", "dSASA/binder-res"),
    ("nres_per_binder", "nres_int/binder-res"),
    ("iface_fraction", "binder interface fraction"),
    ("packstat", "packstat"),
    ("sc_value", "sc_value"),
    ("hbonds_per_binder", "H-bonds/binder-res"),
    ("unsat_per_binder", "unsatHbonds/binder-res"),
]


def ff(x: float) -> str:
    return f"{x:.3f}"


def avg(rows, key):
    return statistics.mean(float(r[key]) for r in rows)


def wide(data):
    lines = ["| metric | " + " | ".join(GROUPS) + " |", "|---|---:|---:|---:|"]
    for key, label in METRICS:
        lines.append("| %s | %s |" % (label, " | ".join(ff(avg(data[n], key)) for n in GROUPS)))
    return "\n".join(lines)


def length_detail(data):
    lengths = sorted({int(r["binder_len"]) for rows in data.values() for r in rows})
    headers = ["binder length"]
    for n in GROUPS:
        headers += [f"{n} N/top10 N", f"{n} dG/binder-res", f"{n} dSASA/binder-res", f"{n} interface fraction"]
    lines = ["| " + " | ".join(headers) + " |", "|---:|" + "---:|" * (len(headers)-1)]
    for length in lengths:
        cells = [str(length)]
        for n in GROUPS:
            group = [r for r in data[n] if int(r["binder_len"]) == length]
            elite = top_rows(group, .10) if group else []
            cells += [f"{len(group)}/{len(elite)}"]
            if elite:
                cells += [ff(avg(elite, "dG_per_binder")), ff(avg(elite, "dSASA_per_binder")), ff(avg(elite, "iface_fraction"))]
            else:
                cells += ["—", "—", "—"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    data = {n: load_dataset(n, SETS[n]) for n in GROUPS}
    lines = [
        "# Rosetta 残基总数归一化三组比较", "",
        "round1、round2、BindCraft 使用同一张横向表比较；所有汇总值为均值。",
        "所有归一化指标均除以 binder chain A 的总残基数，不使用界面残基数作分母。",
        "排名按 `dG_separated / binder总残基数`，越负越好；只保留前 10%。",
        "", "## 数据规模", "", "| dataset | N | binder length mean | range |", "|---|---:|---:|---:|",
    ]
    for n in GROUPS:
        lengths = [int(r["binder_len"]) for r in data[n]]
        lines.append(f"| {n} | {len(lengths)} | {ff(statistics.mean(lengths))} | {min(lengths)}–{max(lengths)} |")
    lines += ["", "## 全部结构：归一化结合相关指标", "", wide(data), "", "## 前 10%：归一化结合相关指标", ""]
    elite = {n: top_rows(data[n], .10) for n in GROUPS}
    lines += [wide(elite), "", "前 10% 结构数：" + ", ".join(f"{n}={len(elite[n])}" for n in GROUPS) + "。", "", "## 不同短肽长度：各长度层前 10%", "", "按每个数据集、每个精确 binder 长度单独取前 10%，不展示 PDB 名称。", "", length_detail(data), "", "## 指标说明", "", "- `dG/binder-res`、`dGcross/binder-res`：结合能除以 binder 总残基数，越负越好。", "- `dSASA/binder-res`：界面面积除以 binder 总残基数。", "- `nres_int/binder-res`：Rosetta 界面残基数除以 binder 总残基数。", "- `binder interface fraction`：binder 中参与界面的残基比例。", "- `packstat`、`sc_value`、`H-bonds/binder-res` 通常越高越好；`unsatHbonds/binder-res` 越低越好。", "", "原始评分文件：", "- `pipeline_runs/run_20260706_IL-4Ra/round1/3_rosetta/results_relax/binding_summary.csv`", "- `pipeline_runs/run_20260706_IL-4Ra/round2/3_rosetta/results_relax/binding_summary.csv`", "- `pipeline_runs/bindcraft_pdb/rosetta_2025.37_iface14/results_relax/binding_summary.csv"]
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
