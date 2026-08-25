#!/usr/bin/env python3
"""Write a residue-count-normalized Rosetta comparison report."""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SETS = {
    "round1": ROOT / "pipeline_runs/run_20260706_IL-4Ra/round1/3_rosetta",
    "round2": ROOT / "pipeline_runs/run_20260706_IL-4Ra/round2/3_rosetta",
    "BindCraft": ROOT / "pipeline_runs/bindcraft_pdb/rosetta_2025.37_iface14",
}
OUT = ROOT / "pipeline_runs/rosetta_length_normalized_top10_top20_report_full80.md"


def chain_lengths(path: Path) -> dict[str, int]:
    residues: dict[str, set[int]] = defaultdict(set)
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 26:
            continue
        try:
            residues[line[21]].add(int(line[22:26]))
        except ValueError:
            continue
    return {chain: len(nums) for chain, nums in residues.items()}


def fmt(x: float) -> str:
    return f"{x:.3f}"


def load_dataset(name: str, root: Path) -> list[dict[str, object]]:
    summary = {
        row["pdb"]: row
        for row in csv.DictReader((root / "results_relax/binding_summary.csv").open())
    }
    interface: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in csv.DictReader((root / "results_relax/interface_residues.csv").open()):
        # The normal pipeline records binder A; BindCraft includes both chains.
        if name == "BindCraft" and row["chain"] != "A":
            continue
        interface[row["pdb"]].add((row["chain"], row["resnum"]))

    rows: list[dict[str, object]] = []
    for pdb, row in summary.items():
        lengths = chain_lengths(root / "pdb_inputs" / pdb)
        binder_len = lengths.get("A", 0)
        if not binder_len:
            raise RuntimeError(f"missing binder chain A in {root / 'pdb_inputs' / pdb}")
        iface_len = len(interface.get(pdb, set()))
        if not iface_len:
            raise RuntimeError(f"missing interface residues for {name}:{pdb}")
        values = {k: float(row[k]) for k in (
            "dG_separated", "dG_cross", "dSASA_int", "nres_int",
            "packstat", "sc_value", "delta_unsatHbonds", "hbonds_int",
        )}
        values.update({
            "pdb": pdb, "binder_len": binder_len,
            "iface_binder": iface_len,
            "dG_per_binder": values["dG_separated"] / binder_len,
            "dG_cross_per_binder": values["dG_cross"] / binder_len,
            "dG_per_iface": values["dG_separated"] / iface_len,
            "dSASA_per_binder": values["dSASA_int"] / binder_len,
            "nres_per_binder": values["nres_int"] / binder_len,
            "iface_fraction": iface_len / binder_len,
            "hbonds_per_binder": values["hbonds_int"] / binder_len,
            "unsat_per_binder": values["delta_unsatHbonds"] / binder_len,
        })
        rows.append(values)
    return rows


def mean(rows: list[dict[str, object]], key: str) -> float:
    return statistics.mean(float(r[key]) for r in rows)


def median(rows: list[dict[str, object]], key: str) -> float:
    return statistics.median(float(r[key]) for r in rows)


def optimum(rows: list[dict[str, object]], key: str, direction: str) -> float:
    values = [float(r[key]) for r in rows]
    return min(values) if direction == "low" else max(values)


def top_rows(rows: list[dict[str, object]], fraction: float) -> list[dict[str, object]]:
    n = max(1, math.ceil(len(rows) * fraction))
    return sorted(rows, key=lambda r: float(r["dG_per_binder"]))[:n]


def metric_table(rows: list[dict[str, object]]) -> str:
    keys = [
        ("dG_per_binder", "dG/binder-res", "low"),
        ("dG_cross_per_binder", "dGcross/binder-res", "low"),
        ("dG_per_iface", "dG/interface-res", "low"),
        ("dSASA_per_binder", "dSASA/binder-res", "high"),
        ("nres_per_binder", "nres_int/binder-res", "high"),
        ("iface_fraction", "binder interface fraction", "high"),
        ("packstat", "packstat", "high"),
        ("sc_value", "sc_value", "high"),
        ("hbonds_per_binder", "H-bonds/binder-res", "high"),
        ("unsat_per_binder", "unsatHbonds/binder-res", "low"),
    ]
    lines = ["| metric | mean | median | optimum（最优值） |", "|---|---:|---:|---:|"]
    for key, label, direction in keys:
        lines.append(
            f"| {label} | {fmt(mean(rows, key))} | {fmt(median(rows, key))} | {fmt(optimum(rows, key, direction))} |"
        )
    return "\n".join(lines)


def main() -> None:
    data = {name: load_dataset(name, root) for name, root in SETS.items()}
    bindcraft_n = len(data["BindCraft"])
    bindcraft_top10_n = len(top_rows(data["BindCraft"], 0.10))
    bindcraft_top20_n = len(top_rows(data["BindCraft"], 0.20))
    lines: list[str] = []
    lines += [
        "# Rosetta 界面指标：按 binder 残基数归一化比较",
        "",
        f"生成时间：{date.today().isoformat()}。比较对象为 round1、round2 和补算完成后的 BindCraft。",
        "",
        "## 口径与排名规则",
        "",
        "- binder 固定为 chain A；`binder_len` 是输入 PDB 中 chain A 的实际残基编号数。",
        "- 所有能量、界面面积、界面残基数和氢键指标均额外给出按 binder 总残基数归一化的值。",
        "- `dG_per_binder` = `dG_separated / binder_len`，数值越负越优；前 10%/20% 均按此指标从负到正排序。",
        "- `dG_per_iface` 进一步按 binder 侧实际界面残基数归一化。",
        f"- 前 10% 和前 20% 使用 `ceil(N×比例)`，所以 BindCraft (N={bindcraft_n}) 分别取 {bindcraft_top10_n} 和 {bindcraft_top20_n} 个结构。",
        "- 汇总表同时给出平均数（mean）、中位数（median）和按指标方向确定的最优值（optimum）。",
        "- `dG`、`dGcross` 和 unsatHbonds 等越低越好的指标取最小值作为 optimum；其余越高越好的指标取最大值。",
        "",
        "## 数据规模与 binder 长度",
        "",
        "| dataset | N | binder length mean | range | length distribution |",
        "|---|---:|---:|---:|---|",
    ]
    for name, rows in data.items():
        lengths = [int(r["binder_len"]) for r in rows]
        dist = ", ".join(f"{k}:{v}" for k, v in sorted(Counter(lengths).items()))
        lines.append(f"| {name} | {len(rows)} | {fmt(statistics.mean(lengths))} | {min(lengths)}–{max(lengths)} | {dist} |")

    lines += ["", "## 全数据集与前 10%/20% 的归一化统计汇总", ""]
    for name, rows in data.items():
        lines += [f"### {name} (N={len(rows)})", "", "### 全部结构", "", metric_table(rows), ""]
        for fraction, label in ((0.10, "前 10%"), (0.20, "前 20%")):
            selected = top_rows(rows, fraction)
            lines += [f"### {label}（{len(selected)} 个，按 dG/binder-res）", "", metric_table(selected), ""]
            lengths = Counter(int(r["binder_len"]) for r in selected)
            lines.append("长度构成：" + ", ".join(f"{k} aa: {v}" for k, v in sorted(lengths.items())) + "。")
            lines.append("")

    lines += ["## 不同 binder 长度的前 20% 细分", "", "这里按每个数据集、每个精确 binder 长度分别取该长度层内最优的前 20%；本节长度分层表中的指标仍为均值。", ""]
    for name, rows in data.items():
        lines += [f"### {name}", "", "| length | N | top20 N | dG/binder-res | dGcross/binder-res | dSASA/binder-res | interface fraction | packstat | H-bonds/binder-res | top representative |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        by_len: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            by_len[int(row["binder_len"])].append(row)
        for length, group in sorted(by_len.items()):
            selected = top_rows(group, 0.20)
            reps = sorted(selected, key=lambda r: float(r["dG_per_binder"]))[:3]
            rep_text = "; ".join(f"{r['pdb']} ({fmt(float(r['dG_per_binder']))})" for r in reps)
            lines.append("| %d | %d | %d | %s | %s | %s | %s | %s | %s | %s |" % (
                length, len(group), len(selected), fmt(mean(selected, "dG_per_binder")),
                fmt(mean(selected, "dG_cross_per_binder")), fmt(mean(selected, "dSASA_per_binder")),
                fmt(mean(selected, "iface_fraction")), fmt(mean(selected, "packstat")),
                fmt(mean(selected, "hbonds_per_binder")), rep_text,
            ))
        lines.append("")

    lines += [
        "## 解释重点",
        "",
        "- 归一化后，短 binder 不会因为总残基较少而在原始总能量上被直接惩罚或获益。",
        "- `dG/binder-res` 衡量每个设计残基带来的平均结合能；`dG/interface-res` 衡量真正参与 binder 界面的残基效率。",
        f"- 补算后的 BindCraft N={bindcraft_n}，前 10% 为 {bindcraft_top10_n} 个、前 20% 为 {bindcraft_top20_n} 个；仍建议以前 20% 作为主要参考，前 10% 作为严格精英集。",
        "- round1 存在少量极端正的原始 dG 值，因此本报告排名始终使用残基数归一化后的 dG，而不是原始总 dG。",
        "",
        "原始评分文件：",
        "- `pipeline_runs/run_20260706_IL-4Ra/round1/3_rosetta/results_relax/binding_summary.csv`",
        "- `pipeline_runs/run_20260706_IL-4Ra/round2/3_rosetta/results_relax/binding_summary.csv`",
        "- `pipeline_runs/bindcraft_pdb/rosetta_2025.37_iface14/results_relax/binding_summary.csv`",
    ]
    # Correct an accidental curly quote if edited in a non-ASCII environment.
    OUT.write_text("\n".join(lines).replace("|”", "|") + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
