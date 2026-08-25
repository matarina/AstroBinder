#!/usr/bin/env python3
"""Summarize BindCraft success using unique launched trajectories as denominator."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT.parent / "pipeline_bindcraft/IL-4Ra/results_80_l10_15"
OUT = ROOT / "pipeline_runs/bindcraft_trajectory_success_report.md"

START_RE = re.compile(r"^Starting trajectory: (\S+)", re.MULTILINE)
SUCCESS_RE = re.compile(r"^Trajectory successful", re.MULTILINE)
ACCEPTED_RE = re.compile(r"(.+)_mpnn\d+_model\d+\.pdb$")


@dataclass
class GpuStats:
    name: str
    launched: int
    passed_trajectory: int
    low_confidence: int
    clashes: int
    unclassified: int
    accepted_trajectories: int
    accepted_designs: int

    @property
    def success_rate(self) -> float:
        return self.accepted_trajectories / self.launched

    @property
    def trajectory_pass_rate(self) -> float:
        return self.passed_trajectory / self.launched

    @property
    def conditional_accept_rate(self) -> float:
        return self.accepted_trajectories / self.passed_trajectory

    @property
    def design_yield(self) -> float:
        return self.accepted_designs / self.launched


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def trajectory_base(filename: str) -> str:
    match = ACCEPTED_RE.fullmatch(filename)
    if not match:
        raise RuntimeError(f"unexpected Accepted filename: {filename}")
    return match.group(1)


def load_gpu(name: str) -> tuple[GpuStats, set[str], set[str]]:
    root = RUN_ROOT / name
    log = (root / "run.log").read_text(errors="replace")
    launched_ids = set(START_RE.findall(log))
    if len(launched_ids) != len(START_RE.findall(log)):
        raise RuntimeError(f"{name}: duplicate trajectory IDs in run.log")

    trajectory_rows = read_csv_rows(root / "trajectory_stats.csv")
    trajectory_ids = {row["Design"] for row in trajectory_rows}
    passed_from_log = len(SUCCESS_RE.findall(log))
    if len(trajectory_rows) != len(trajectory_ids) or len(trajectory_rows) != passed_from_log:
        raise RuntimeError(f"{name}: trajectory_stats.csv and run.log disagree")

    failure_rows = read_csv_rows(root / "failure_csv.csv")
    if len(failure_rows) != 1:
        raise RuntimeError(f"{name}: failure_csv.csv must contain one aggregate row")
    failures = failure_rows[0]
    low_confidence = int(float(failures.get("Trajectory_final_pLDDT", 0) or 0))
    clashes = int(float(failures.get("Trajectory_Clashes", 0) or 0))

    accepted_files = sorted((root / "Accepted").glob("*.pdb"))
    accepted_ids = {trajectory_base(path.name) for path in accepted_files}
    if not accepted_ids <= launched_ids:
        raise RuntimeError(f"{name}: Accepted trajectory missing from run.log")

    final_rows = read_csv_rows(root / "final_design_stats.csv")
    if len(final_rows) != len(accepted_files):
        raise RuntimeError(f"{name}: final_design_stats.csv and Accepted PDB count disagree")

    unclassified = len(launched_ids) - len(trajectory_rows) - low_confidence - clashes
    if unclassified < 0:
        raise RuntimeError(f"{name}: trajectory categories exceed launched count")

    stats = GpuStats(
        name=name,
        launched=len(launched_ids),
        passed_trajectory=len(trajectory_rows),
        low_confidence=low_confidence,
        clashes=clashes,
        unclassified=unclassified,
        accepted_trajectories=len(accepted_ids),
        accepted_designs=len(accepted_files),
    )
    return stats, launched_ids, accepted_ids


def combine(rows: list[GpuStats]) -> GpuStats:
    return GpuStats(
        name="合计",
        launched=sum(row.launched for row in rows),
        passed_trajectory=sum(row.passed_trajectory for row in rows),
        low_confidence=sum(row.low_confidence for row in rows),
        clashes=sum(row.clashes for row in rows),
        unclassified=sum(row.unclassified for row in rows),
        accepted_trajectories=sum(row.accepted_trajectories for row in rows),
        accepted_designs=sum(row.accepted_designs for row in rows),
    )


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def main() -> None:
    loaded = [load_gpu(name) for name in ("gpu2", "gpu3")]
    stats = [item[0] for item in loaded]
    if loaded[0][1] & loaded[1][1]:
        raise RuntimeError("trajectory IDs overlap between gpu2 and gpu3")
    if loaded[0][2] & loaded[1][2]:
        raise RuntimeError("Accepted trajectory IDs overlap between gpu2 and gpu3")
    total = combine(stats)
    lower, upper = wilson(total.accepted_trajectories, total.launched)

    lines = [
        "# BindCraft 轨迹成功率统计",
        "",
        f"> 生成时间：{date.today().isoformat()}  ",
        "> 主要定义：一条唯一启动轨迹只要最终产生至少一个 Accepted PDB，即计为 1 条成功轨迹。",
        "",
        "## 核心结论",
        "",
        f"- 总启动轨迹：`{total.launched}` 条。",
        f"- 最终成功轨迹：`{total.accepted_trajectories}` 条。",
        f"- **轨迹成功率：`{total.accepted_trajectories}/{total.launched} = {pct(total.success_rate)}`**。",
        f"- Wilson 95% 置信区间：`{pct(lower)}–{pct(upper)}`。",
        f"- 80 个 Accepted PDB 来自 50 条轨迹，平均每条成功轨迹产生 `{total.accepted_designs / total.accepted_trajectories:.2f}` 个 Accepted。",
        f"- `80/880 = {pct(total.design_yield)}` 表示设计产出率，即每 100 条轨迹约得到 `{total.design_yield * 100:.2f}` 个 Accepted PDB；它不是轨迹成功概率。",
        "",
        "## 分卡统计",
        "",
        "| 运行 | 启动轨迹 | 通过轨迹级检查 | 低置信度淘汰 | 严重 clash 淘汰 | 未分类/中断 | 成功轨迹 | Accepted PDB | 轨迹成功率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [*stats, total]:
        lines.append(
            f"| {row.name} | {row.launched} | {row.passed_trajectory} | "
            f"{row.low_confidence} | {row.clashes} | {row.unclassified} | "
            f"{row.accepted_trajectories} | {row.accepted_designs} | {pct(row.success_rate)} |"
        )

    lines += [
        "",
        "## 转化漏斗",
        "",
        "```text",
        f"启动轨迹              {total.launched:>4}",
        f"  -> 通过轨迹级检查    {total.passed_trajectory:>4}  ({pct(total.trajectory_pass_rate)})",
        f"  -> 产生 Accepted 的轨迹 {total.accepted_trajectories:>3}  ({pct(total.success_rate)} of all; {pct(total.conditional_accept_rate)} of passed)",
        f"  -> Accepted PDB        {total.accepted_designs:>3}  ({total.accepted_designs / total.accepted_trajectories:.2f} per successful trajectory)",
        "```",
        "",
        "轨迹级未通过的主要原因是低置信度与严重 clash：",
        "",
        f"- 低置信度：`{total.low_confidence}/{total.launched} = {pct(total.low_confidence / total.launched)}`；",
        f"- 严重 clash：`{total.clashes}/{total.launched} = {pct(total.clashes / total.launched)}`；",
        f"- 两次早期运行各留下 1 条未完成分类的启动轨迹，共 `{total.unclassified}` 条；即使只用完成分类的 878 条作分母，成功率也是 `50/878 = {pct(50 / 878)}`，结论不变。",
        "",
        "## 分母说明",
        "",
        "两路 `run.log` 都包含一次早期运行和一次续跑。日志末尾的 `107` 与 `263` 只对应各自最后一次续跑，不能代表整个目录的总轨迹预算。本统计遍历完整日志中的所有 `Starting trajectory:`，并按轨迹 ID 去重；gpu2=360、gpu3=520，且两路之间没有重复 ID，因此总分母为 880。",
        "",
        "数据来源：",
        "",
        "- `pipeline_bindcraft/IL-4Ra/results_80_l10_15/gpu2/run.log`",
        "- `pipeline_bindcraft/IL-4Ra/results_80_l10_15/gpu3/run.log`",
        "- 两路 `trajectory_stats.csv`、`failure_csv.csv`、`final_design_stats.csv`",
        "- 两路 `Accepted/*.pdb`",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
