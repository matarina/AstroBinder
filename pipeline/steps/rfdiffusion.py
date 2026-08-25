"""循环步骤 1:RFdiffusion 重扩散 + ProteinMPNN 序列重设计。

输入:上一步产物目录 ctx.input_dir。
      - round1 时为 round0_rifdock/output/dock_results:每个 scaffold 一个子文件夹,
        里面是对接结果压缩包 *.pdb.gz(已带 rif_residues 头 + A/B 链)与 all.dok。
      - 之后为上一轮 rosetta 的筛选结果(已是 *.pdb)。
      本 step 先把这些输入统一「解压 / 复制」到 work_dir/inputs/(扁平化,文件名带
      来源子文件夹前缀防重名),再把 inputs/ 作为 redesign 的输入根。
输出:写到 ctx.work_dir(roundN/1_rfdiffusion/)。其中 summary.csv 汇总全部设计序列
      及理化指标,各 per_pdb/<stem>/ 子目录含 input.pdb 与 designs/。整个 work_dir 向下传给 af3。

真实计算逻辑在 scripts/redesign_pipeline.py。
本 step 负责:①解压暂存输入 ②把 ctx 翻译成 redesign 的命令行(含自动 GPU 调度)。
"""

from __future__ import annotations

import gzip
import shutil

from .base import Step, StepContext, StepResult
from ._proc import run_script


def stage_inputs(input_dir, inputs_dir) -> int:
    """把 ctx.input_dir 下(递归)的所有 *.pdb.gz 解压、*.pdb 复制到 inputs_dir。

    扁平化输出,文件名带来源相对目录前缀(下划线连接)防同名冲突。
    返回暂存的 PDB 个数。
    """
    inputs_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(input_dir.rglob("*")):
        if not src.is_file():
            continue
        name = src.name
        if name.endswith(".pdb.gz"):
            stem = name[:-len(".pdb.gz")]
        elif name.endswith(".pdb"):
            stem = name[:-len(".pdb")]
        else:
            continue
        # 以来源相对目录做前缀,避免不同子文件夹里的同名 PDB 覆盖
        rel_parent = src.parent.relative_to(input_dir)
        prefix = "" if str(rel_parent) == "." else str(rel_parent).replace("/", "_") + "_"
        dst = inputs_dir / f"{prefix}{stem}.pdb"
        if name.endswith(".pdb.gz"):
            with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo)
        else:
            shutil.copy(src, dst)
        count += 1
    return count


class RFdiffusionStep(Step):
    name = "rfdiffusion"

    def run(self, ctx: StepContext) -> StepResult:
        p = ctx.params or {}

        # ---- ① 解压 / 复制上游产物到本步 inputs/ ----
        inputs_dir = ctx.work_dir / "inputs"
        n = stage_inputs(ctx.input_dir, inputs_dir)
        print(f"    [rfdiffusion] 暂存输入 PDB {n} 个 -> {inputs_dir}")
        if n == 0:
            raise FileNotFoundError(
                f"rfdiffusion 上游未找到任何 *.pdb(.gz): {ctx.input_dir}")

        # ---- ② 组装 redesign_pipeline 命令行 ----
        args = [
            str(inputs_dir),               # 输入根文件夹(位置参数)
            "-o", str(ctx.work_dir),
        ]
        if p.get("num_designs") is not None:
            args += ["--num-designs", str(p["num_designs"])]
        if p.get("num_seqs") is not None:
            args += ["--num-seqs", str(p["num_seqs"])]
        if p.get("temp") is not None:
            args += ["--temp", str(p["temp"])]
        if p.get("noise_scale") is not None:
            args += ["--noise-scale", str(p["noise_scale"])]
        if p.get("seed") is not None:
            args += ["--seed", str(p["seed"])]
        if p.get("limit") is not None:
            args += ["--limit", str(p["limit"])]
        if p.get("force"):
            args += ["--force"]
        if p.get("no_trajectory"):
            args += ["--no-trajectory"]
        if p.get("no_thread"):
            args += ["--no-thread"]

        # GPU:默认自动检测(按空闲显存挑卡,每卡一路,可与别人共享);
        # config 里 auto_gpu 显式设 false 时才退回手动 gpu 列表。
        if p.get("auto_gpu", True):
            args += ["--auto-gpu"]
            if p.get("task_vram") is not None:
                args += ["--task-vram", str(p["task_vram"])]
            if p.get("gpu_reserve") is not None:
                args += ["--gpu-reserve", str(p["gpu_reserve"])]
            if p.get("gpu_sample") is not None:
                args += ["--gpu-sample", str(p["gpu_sample"])]
        elif p.get("gpu") is not None:
            args += ["--gpu", str(p["gpu"])]

        log_path = ctx.work_dir / "rfdiffusion_step.log"
        rc = run_script("redesign_pipeline.py", args,
                        log_path=log_path, python=p.get("python"))
        # 退出码语义(见 redesign_pipeline.py):0=全成功 1=部分 PDB 失败 2=全部失败
        if rc >= 2:
            raise RuntimeError(
                f"rfdiffusion(redesign)严重失败(退出码 {rc},疑似全部 PDB 失败),详见 {log_path}")
        if rc == 1:
            print(f"    [rfdiffusion] 警告:部分 PDB 失败(退出码 1),用已产出的设计继续。详见 {log_path}")

        # 再校验确有产出:summary.csv 必须存在且至少 1 条设计。
        # 防止 redesign 万一以 0 退出却没产出任何序列时,写出「假的」完成标记,
        # 导致后续续跑错误地跳过本步。
        summary_csv = ctx.work_dir / "summary.csv"
        n_designs = 0
        if summary_csv.is_file():
            import csv
            with open(summary_csv, newline="", encoding="utf-8") as fh:
                n_designs = sum(1 for _ in csv.DictReader(fh))
        if n_designs == 0:
            raise RuntimeError(
                f"rfdiffusion 未产出任何设计序列(summary.csv 为空或缺失): "
                f"{summary_csv},详见 {log_path}")
        print(f"    [rfdiffusion] 产出 {n_designs} 条设计序列 -> {summary_csv}")

        # summary.csv 在 work_dir 根,向下传整个 work_dir 供 af3 读取
        return StepResult(output_dir=ctx.work_dir,
                          info={"summary_csv": str(summary_csv)})
