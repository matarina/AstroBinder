"""循环步骤 2:AlphaFold3(graft 方案,替代原「全复合物自主 docking」)。

输入:上一步 rfdiffusion 的产物目录 ctx.input_dir(roundN/1_rfdiffusion/),
      内含 summary.csv(每条 MPNN 设计一行:designed_seq / threaded_pdb /
      rif_residues_out 等)。

── 为什么改成 graft ──────────────────────────────────────────────
AF3 做全复合物预测时会自主 docking, 无法约束 binder 相对 target 的姿态, 常把
rifdock 设计里本该固定的 RIF 残基推离靶标(见 memory: af3-template-binder-docking)。
graft 方案换一条路(见 memory: peptide-graft-approach, 已实测有效):
  ① 只让 AF3 折 binder 单链, 并用「该设计序列对应的短肽结构」(threaded_pdb 的 A
     链骨架)作为模板软约束, 防止 AF3 把短肽折得偏离设计;
  ② AF3 折完后, 用 RIF 残基骨架(N/CA/C/O)做 Kabsch 刚体叠合, 把短肽整体搬回
     参考结构(threaded_pdb, RIF = rifdock 原始锚定)坐标系 —— RIF 必然精确落回
     靶标界面, 再拼上 target 成复合物;
  ③ RIF align RMSD 是天然筛选量:AF3 没折出设计的界面几何 → RMSD 大 → 淘汰。
     要求 RIF RMSD < rmsd_max(默认 2.0Å) 且 binder↔target clash 不超阈值才通过。

── 三段(都复用仓库已验证脚本)──────────────────────────────────
  1) summary.csv → AF3 单链 job:scripts/make_af3_monomer_jobs.py
     复用 filter_for_af3 的硬过滤 + 组内保留, 只放 binder A 链, 按 template_mode
     给短肽上模板(full=全长骨架防大幅变化 / rif=仅RIF残基 / none=无),
     写到 work_dir/batch_input/, 另出 _monomer_manifest.csv(定位参考结构+RIF)。
     该脚本 import alphafold3, 需用 af3 环境的 python 跑。
  2) json → 结构:scripts/run_af3_batch.py
     扫描 batch_input/ 多 GPU 并行跑 AF3(不用 MSA, 保留注入的短肽模板),
     每设计多采样(num_seeds × 扩散样本), 结构落到 batch_output/structures/。
  3) 结构 → 复合物:scripts/graft_batch.py
     遍历各设计的多采样, RIF 骨架刚体叠合回靶标 + 查 clash, 每设计按 RIF_RMSD
     升序写出通过筛选(RMSD≤rmsd_max 且 clash≤clash_max)的前 top 个复合物 PDB
     到 work_dir/complexes/。该脚本用 numpy, 需 af3 环境的 python。
输出:complexes/(通过验证的复合物 PDB, 扁平)作为下游 rosetta 输入向下传。

── 兼容旧路线 ──────────────────────────────────────────────────
config 里 af3.mode 设为 "complex" 可切回原「全复合物自主 docking」路线
(filter_for_af3 → run_af3_batch, 输出 cif structures 目录)。默认 "graft"。
"""

from __future__ import annotations

import shutil

from .base import Step, StepContext, StepResult
from ._proc import run_script

# 需要 import alphafold3 / numpy 的脚本(make_af3_monomer_jobs、graft_batch)所用解释器。
# 与 run_af3_batch.py 内部固定的 AF3_PYTHON 保持一致;config 里 af3.af3_python 可覆盖。
DEFAULT_AF3_PYTHON = "/home/opt/anaconda3/envs/af3/bin/python"


def _gpu_args(p: dict) -> list[str]:
    """把 config 里的 GPU 相关参数翻译成 run_af3_batch.py 的命令行。

    默认走 auto_gpu(与 rfdiffusion 一致:自动检测所有显卡, 按空闲显存挑可用卡,
    每卡一路 AF3 worker, 可与别人已有任务共享)。config 里 auto_gpu 显式设 false
    时才退回手动 gpus 列表。auto_gpu 下仍带上 gpus 作为「没挑到空闲卡」的兜底。
    """
    args: list[str] = []
    if p.get("auto_gpu", True):
        args += ["--auto-gpu"]
        if p.get("task_vram") is not None:
            args += ["--task-vram", str(p["task_vram"])]
        if p.get("gpu_reserve") is not None:
            args += ["--gpu-reserve", str(p["gpu_reserve"])]
        if p.get("gpu_sample") is not None:
            args += ["--gpu-sample", str(p["gpu_sample"])]
        # 兜底:auto_gpu 没挑到空闲卡时,run_af3_batch 退回这个手动列表
        if p.get("gpus") is not None:
            args += ["-g", str(p["gpus"])]
    elif p.get("gpus") is not None:
        args += ["-g", str(p["gpus"])]
    return args


class AF3Step(Step):
    name = "af3"

    def run(self, ctx: StepContext) -> StepResult:
        p = ctx.params or {}
        mode = (p.get("mode") or "graft").strip().lower()
        summary_csv = ctx.input_dir / "summary.csv"
        if not summary_csv.is_file():
            raise FileNotFoundError(f"af3 缺少上游 summary.csv: {summary_csv}")

        if mode == "complex":
            return self._run_complex(ctx, p, summary_csv)
        if mode != "graft":
            raise ValueError(f"af3.mode 只支持 'graft' 或 'complex', 收到: {mode!r}")
        return self._run_graft(ctx, p, summary_csv)

    # ============================ graft 路线(默认)============================

    def _run_graft(self, ctx: StepContext, p: dict, summary_csv) -> StepResult:
        batch_input = ctx.work_dir / "batch_input"
        batch_output = ctx.work_dir / "batch_output"
        complexes = ctx.work_dir / "complexes"

        # import alphafold3/numpy 的脚本用 af3 环境;run_af3_batch 是纯启动器,用默认 python 即可
        af3_python = p.get("af3_python") or DEFAULT_AF3_PYTHON

        # ---- 第①段:summary.csv → AF3 单链 job(短肽上模板)----
        # make_af3_monomer_jobs 要求 -o 不存在, 先删干净
        if batch_input.exists():
            shutil.rmtree(batch_input)
        j_args = ["-f", str(summary_csv), "-o", str(batch_input)]
        if p.get("seed") is not None:
            j_args += ["--seed", str(p["seed"])]
        if p.get("keep_fraction") is not None:
            j_args += ["--keep-fraction", str(p["keep_fraction"])]
        # 模板覆盖:full=全长骨架(防短肽大幅变化) / rif=仅RIF残基 / none=无
        template_mode = (p.get("template_mode") or "full").strip().lower()
        j_args += ["--template-mode", template_mode]
        rc = run_script("make_af3_monomer_jobs.py", j_args,
                        log_path=ctx.work_dir / "af3_make_jobs.log", python=af3_python)
        if rc != 0:
            raise RuntimeError(f"af3 第①段(make_af3_monomer_jobs)退出码 {rc}")

        manifest = batch_input / "_monomer_manifest.csv"
        if not manifest.is_file():
            raise RuntimeError(
                f"af3 第①段未产出 _monomer_manifest.csv(无入选设计?): {manifest}。"
                f"检查 summary.csv 是否有通过硬过滤的设计, 以及 keep_fraction 是否过小")

        # ---- 第②段:json → 结构(多采样, 保留短肽模板)----
        b_args = ["-i", str(batch_input), "-o", str(batch_output)]
        b_args += _gpu_args(p)
        if p.get("num_seeds") is not None:
            b_args += ["-s", str(p["num_seeds"])]
        if p.get("force"):
            b_args += ["-f"]
        if p.get("keep_temp"):
            b_args += ["-k"]
        rc = run_script("run_af3_batch.py", b_args,
                        log_path=ctx.work_dir / "af3_batch.log", python=p.get("python"))
        if rc != 0:
            raise RuntimeError(f"af3 第②段(run_af3_batch)退出码 {rc}")

        structures = batch_output / "structures"
        if not structures.is_dir():
            raise RuntimeError(f"af3 第②段未产出 structures 目录: {structures}")

        # ---- 第③段:RIF 骨架叠合回靶标 + RMSD/clash 筛选 → 复合物 ----
        # 每设计只写出 RIF_RMSD ≤ rmsd_max 且 clash ≤ clash_max 的前 top 个复合物。
        if complexes.exists():
            shutil.rmtree(complexes)
        g_args = ["--manifest", str(manifest),
                  "--structures", str(structures),
                  "--out-dir", str(complexes)]
        # RIF align RMSD 上限:用户要求 <2Å 才通过验证
        g_args += ["--rmsd-max", str(p.get("rmsd_max", 2.0))]
        if p.get("clash_dist") is not None:
            g_args += ["--clash-dist", str(p["clash_dist"])]
        if p.get("clash_max") is not None:
            g_args += ["--clash-max", str(p["clash_max"])]
        if p.get("top") is not None:
            g_args += ["--top", str(p["top"])]
        rc = run_script("graft_batch.py", g_args,
                        log_path=ctx.work_dir / "af3_graft.log", python=af3_python)
        if rc != 0:
            raise RuntimeError(f"af3 第③段(graft_batch)退出码 {rc}")

        # 单条设计 0 通过很正常(设计有好有坏),graft_batch 已对逐条做容错(某条 0
        # 通过只是不写出、继续下一条);但整批全灭说明本轮 AF3 都没折出设计界面几何,
        # 下游 rosetta 无输入, 属真·无米下锅, 仍需报错中断。
        n_pass = len(list(complexes.glob("*.pdb"))) if complexes.is_dir() else 0
        if n_pass == 0:
            raise RuntimeError(
                f"graft 后整批无任何样本通过 RIF_RMSD≤{p.get('rmsd_max', 2.0)}Å + clash 筛选: "
                f"{complexes}。本轮所有设计 AF3 都没折出设计的界面几何, 应回看设计质量"
                f"(或放宽 rmsd_max / 提高 num_seeds / 改 template_mode)")
        print(f"      [graft] 通过 RIF_RMSD≤{p.get('rmsd_max', 2.0)}Å 的复合物: "
              f"{n_pass} 个 → {complexes}")

        # complexes/ 是「已叠合好的复合物 PDB」, 下游 rosetta 会自动识别为 PDB 输入
        # (跳过 cif→pdb), 直接进 FastRelax。
        return StepResult(output_dir=complexes, info={
            "complexes_dir": str(complexes),
            "structures_dir": str(structures),
            "manifest": str(manifest),
            "n_passed": n_pass,
            "input_kind": "pdb",       # 提示下游:输入已是复合物 PDB
            "mode": "graft",
        })

    # ============================ 旧路线(可选)============================

    def _run_complex(self, ctx: StepContext, p: dict, summary_csv) -> StepResult:
        """原「全复合物自主 docking」路线:filter_for_af3 → run_af3_batch, 输出 cif structures。"""
        batch_input = ctx.work_dir / "batch_input"
        batch_output = ctx.work_dir / "batch_output"

        if batch_input.exists():
            shutil.rmtree(batch_input)
        f_args = ["-f", str(summary_csv), "-o", str(batch_input)]
        if p.get("seed") is not None:
            f_args += ["--seed", str(p["seed"])]
        if p.get("keep_fraction") is not None:
            f_args += ["--keep-fraction", str(p["keep_fraction"])]
        rc = run_script("filter_for_af3.py", f_args,
                        log_path=ctx.work_dir / "af3_filter.log", python=p.get("python"))
        if rc != 0:
            raise RuntimeError(f"af3 第①段(filter_for_af3)退出码 {rc}")

        b_args = ["-i", str(batch_input), "-o", str(batch_output)]
        b_args += _gpu_args(p)
        if p.get("num_seeds") is not None:
            b_args += ["-s", str(p["num_seeds"])]
        if p.get("force"):
            b_args += ["-f"]
        if p.get("keep_temp"):
            b_args += ["-k"]
        rc = run_script("run_af3_batch.py", b_args,
                        log_path=ctx.work_dir / "af3_batch.log", python=p.get("python"))
        if rc != 0:
            raise RuntimeError(f"af3 第②段(run_af3_batch)退出码 {rc}")

        structures = batch_output / "structures"
        return StepResult(output_dir=structures,
                          info={"structures_dir": str(structures),
                                "input_kind": "cif", "mode": "complex"})
