"""round0 种子步骤:RIFdock。

输入:外部传入的 rifdock 输入文件夹 ctx.input_dir,内部应含:
        - 靶点 PDB(默认 PD-1.pdb,可由 params['target'] 指定相对名)
        - 热点残基列表(默认 target_res.list,params['target_res'])
        - scaffold 子目录(默认 scaffolds,params['scaffold_subdir']);
          本次测试用 scaffolds_test。
输出:产物主目录写到 ctx.work_dir(round0_rifdock/output/);
      其中 dock_results/ 是每个 scaffold 的对接结果(PDB + all.dok),
      作为 round1 rfdiffusion 的输入向下传。

真实计算逻辑在 scripts/run_rifdock_pipeline.py(从 test_rif 原样复制)。
本 step 只把 ctx 翻译成它的命令行参数。
"""

from __future__ import annotations

import sys

from .base import Step, StepContext, StepResult
from ._proc import run_script


class RifdockStep(Step):
    name = "rifdock"

    def run(self, ctx: StepContext) -> StepResult:
        p = ctx.params or {}
        input_dir = ctx.input_dir

        # 从输入文件夹内按相对名定位三类输入(键名/默认值与脚本默认一致)
        target = input_dir / p.get("target", "PD-1.pdb")
        target_res = input_dir / p.get("target_res", "target_res.list")
        scaffold_dir = input_dir / p.get("scaffold_subdir", "scaffolds")

        for path, desc in ((target, "靶点 PDB"),
                           (target_res, "热点残基列表"),
                           (scaffold_dir, "scaffold 目录")):
            if not path.exists():
                raise FileNotFoundError(f"rifdock 输入缺少{desc}: {path}")

        # 组装 run_rifdock_pipeline.py 的命令行
        args = [
            "--target", str(target),
            "--target-res", str(target_res),
            "--scaffold-dir", str(scaffold_dir),
            "--out", str(ctx.work_dir),
        ]
        if p.get("threads"):
            args += ["--threads", str(p["threads"])]
        if p.get("database"):
            args += ["--database", str(p["database"])]
        if p.get("rifgen_bin"):
            args += ["--rifgen-bin", str(p["rifgen_bin"])]
        if p.get("rifdock_bin"):
            args += ["--rifdock-bin", str(p["rifdock_bin"])]
        if p.get("rifgen_flag"):
            args += ["--rifgen-flag", str(p["rifgen_flag"])]
        if p.get("rifdock_flag"):
            args += ["--rifdock-flag", str(p["rifdock_flag"])]
        if p.get("docking_flag"):
            # 复用已算好的 rifgen 结果:相对名相对输入文件夹解析为绝对路径
            val = str(p["docking_flag"])
            df = val if val.startswith("/") else str(input_dir / val)
            args += ["--docking-flag", df]
        if p.get("force_rifgen"):
            args += ["--force-rifgen"]
        if p.get("skip_rifgen"):
            args += ["--skip-rifgen"]
        if p.get("rifgen_extra"):
            args += ["--rifgen-extra", str(p["rifgen_extra"])]
        if p.get("rifdock_extra"):
            args += ["--rifdock-extra", str(p["rifdock_extra"])]

        log_path = ctx.work_dir / "rifdock_step.log"
        rc = run_script("run_rifdock_pipeline.py", args,
                        log_path=log_path, python=p.get("python") or sys.executable)
        if rc != 0:
            raise RuntimeError(f"rifdock 退出码 {rc},详见 {log_path}")

        # 对接结果在 work_dir/dock_results,作为下游(rfdiffusion)输入
        dock_results = ctx.work_dir / "dock_results"
        return StepResult(output_dir=dock_results,
                          info={"dock_results": str(dock_results)})
