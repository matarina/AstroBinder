#!/usr/bin/env python3
"""Pipeline 命令行入口。

用法:
    python run_pipeline.py \
        --rifdock-input <rifdock 输入文件夹> \
        --config config.yaml

只负责解析参数、加载配置、启动编排器。真实的各步骤逻辑在 pipeline/steps/ 下填充。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.config_loader import PipelineConfig
from pipeline.orchestrator import Orchestrator


def main():
    ap = argparse.ArgumentParser(description="Rifdock→RFdiffusion→AF3→Rosetta 多轮循环编排框架")
    ap.add_argument("--rifdock-input", required=True,
                    help="外部 rifdock 输入文件夹(含启动所需全部内容)")
    ap.add_argument("--config", default=str(Path(__file__).parent / "pipeline" / "config.yaml"),
                    help="统一参数文件 config.yaml")
    ap.add_argument("--max-next-round-pdbs", type=int, default=80,
                    help=("下一轮最多接收的 PDB 数(默认 80)。Rosetta 筛选"
                          "超过此数时会自动收紧保留比例"))
    ap.add_argument("--filter-shrink-step", type=float, default=0.05,
                    help=("自适应筛选每次减少的保留比例(默认 0.05，"
                          "即每次收紧 5 个百分点)"))
    args = ap.parse_args()

    if args.max_next_round_pdbs < 1:
        ap.error("--max-next-round-pdbs 必须 >= 1")
    if not 0 < args.filter_shrink_step < 1:
        ap.error("--filter-shrink-step 必须在 (0, 1) 区间")

    config = PipelineConfig.load(args.config)

    # output_root 若为相对路径,锚定到项目根(本文件所在目录)的绝对路径。
    # 否则它会相对「当前工作目录」解析:一旦从别处启动、或子进程/后台任务的
    # cwd 不同,产物会落错地方、续跑的完成标记也会因路径判定失效而重复计算。
    if not Path(config.output_root).is_absolute():
        project_root = Path(__file__).resolve().parent
        config.experiment["output_root"] = str(project_root / config.output_root)
        print(f"[output_root] 相对路径已锚定为绝对: {config.experiment['output_root']}")

    orch = Orchestrator(
        config,
        rifdock_input=args.rifdock_input,
        max_next_round_pdbs=args.max_next_round_pdbs,
        filter_shrink_step=args.filter_shrink_step,
    )
    run_dir = orch.run()
    print(f"\n输出目录: {run_dir}")


if __name__ == "__main__":
    main()
