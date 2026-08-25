#!/usr/bin/env python3
"""Generate one-table-per-comparison residue-normalized Rosetta report."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from pathlib import Path

from make_length_normalized_rosetta_report import SETS, load_dataset, top_rows

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "pipeline_runs/rosetta_length_normalized_comparison.md"

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


def f(x: float) -> str:
    return f"{x:.3f}"


def avg(rows: list[dict[str, object]], key: str) -> float:
    return statistics.mean(float(r[key]) for r in rows)


def table(data: dict[str, list[dict[str, object]]]) -> str:
    names = list(data)
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for key, label in METRICS:
        lines.append("| " + label + " | " + " | ".join(f(avg(data[n], key)) for n in names) + " |")
    return "\n".join(lines)


def length_table(data: dict[str, list[dict[str, object]]]) -> str:
    names = list(data)
    headers = (["binder length"] + [f"{n} N / top20 N" for n in names]
               + [f"{n} dG/binder-res" for n in names]
               + [f"{n} dSASA/binder-res" for n in names]
               + [f"{n} interface fraction" for n in names])
    lines = ["| " + " | ".join(headers) + " |", "|---:|" + "---:|" * (len(headers) - 1)]
    lengths = sorted({int(r["binder_len"]) for rows in data.values() for r in rows})
    for length in lengths:
        cells = [str(length)]
        groups: dict[str, list[dict[str, object]]] = {}
        for name in names:
            group = [r for r in data[name] if int(r["binder_len"]) == length]
            groups[name] = top_rows(group, 0.10) if group else []
            cells.append(f"{len(group)} / {len(groups[name])}")
        for name in names:
            group = groups[name]
            cells.append(f(avg(group, "dG_per_binder")) if group else "—")
        for name in names:
            group = groups[name]
            cells.append(f(avg(group, "dSASA_per_binder") if group else 0) if group else "—")
        for name in names:
            group = groups[name]
            cells.append(f(avg(group, "iface_fraction") if group else 0) if group else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def length_distribution(data: dict[str, list[dict[str, object]]]) -> str:
    names = list(data)
    lines = ["| binder length | " + " | ".join(names) + " |", "|---:|" + "---:|" * len(names)]
    for length in sorted({int(r["binder_len"]) for rows in data.values() for r in rows}):
        cells = [str(length)]
        for name in names:
            cells.append(str(sum(int(r["binder_len"]) == length for r in data[name])))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    data = {name: load_dataset(name, root) for name, root in SETS.items()}
    lines = [
        "# Rosetta 残基数归一化三组比较",
        "",
        "本报告将 round1、round2、BindCraft 放在同一张横向表中比较。所有汇总值均为均值。",
        "排序依据为 `dG_separated / binder 总残基数`，数值越负越优；前 10% 使用 `ceil(N×0.1)`。",
        "所有归一化指标的分母都只使用 binder A 链的总残基数，不使用界面残基数。",
        "",
        "## 数据规模与 binder 长度分布",
        "",
        "| dataset | N | binder length mean | range |",
        "|---|---:|---:|---:|",
    ]
    for name, rows in data.items():
        lengths = [int(r["binder_len"]) for r in rows]
        lines.append(f"| {name} | {len(rows)} | {f(statistics.mean(lengths))} | {min(lengths)}–{max(lengths)} |")
    lines += ["", length_distribution(data), "", "## 全部结构：归一化核心界面指标", "", table(data), "", "## 前 10%：归一化核心界面指标", ""]
    lines.append(table({name: top_rows(rows, 0.10) for name, rows in data.items()}))
    lines += ["", "前 10% 结构数：" + ", ".join(f"{n}={len(top_rows(r, .10))}" for n, r in data.items()) + "。", ""]
    lines += ["", "前 20% 结构数：" + ", ".join(f"{n}={len(top_rows(r, .20))}" for n, r in data.items()) + "。", "", "## 不同短肽长度：前 20% 细节", "", "每个长度层内单独取该组该长度的前 20%；不列出 PDB 名称。", "", length_table(data), "", "## 指标解释", "", "- `dG/binder-res`、`dGcross/binder-res`、`dG/interface-res`：越负越好。", "- `dSASA/binder-res`：每个 binder 残基贡献的界面面积，越高表示界面更展开。", "- `binder interface fraction`：binder 中参与界面的残基比例。", "- `packstat`、`sc_value`、`H-bonds/binder-res`：通常越高越好。", "- `unsatHbonds/binder-res`：越低越好。", "", "原始评分文件：", "- `pipeline_runs/run_20260706_IL-4Ra/round1/3_rosetta/results_relax/binding_summary.csv`", "- `pipeline_runs/run_20260706_IL-4Ra/round2/3_rosetta/results_relax/binding_summary.csv`", "- `pipeline_runs/bindcraft_pdb/rosetta_2025.37_iface14/results_relax/binding_summary.csv`"]
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
