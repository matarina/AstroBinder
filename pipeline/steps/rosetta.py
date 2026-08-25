"""循环步骤 3:Rosetta 2025.37 界面结合能打分 + 筛选。

输入:上一步 af3 的产物目录 ctx.input_dir。两种可能(自动识别):
  - graft 路线(af3.mode=graft, 默认):ctx.input_dir = roundN/2_af3/complexes/,
    里面已是叠合好的复合物 PDB(A=binder, 其余=target), 首行带 rif_residues 注释。
    此时跳过 cif→pdb, 只把 PDB 清洗(去掉 rif_residues 等非标准行)后作为输入。
  - complex 路线(af3.mode=complex):ctx.input_dir = .../batch_output/structures/,
    是 AF3 输出的 mmCIF。Rosetta 需要 PDB, 故先做 cif→pdb 转换。

三段(都复用仓库已验证脚本):
  0) 准备 PDB 输入:
     - PDB 输入(graft):清洗后扁平复制到 work_dir/pdb_inputs/<name>.pdb;
     - cif 输入(complex):scripts/af3_cif_to_pdb.py 遍历 structures/<design>/,
       把每个设计顶层 best-ranked <design>_model.cif 转成 Rosetta 可读 PDB,
       扁平写到 work_dir/pdb_inputs/<design>.pdb。which=all 时连同各 seed-sample 一起转。
     两种情况都: A 链=设计序列, 其余=靶标, 链间加 TER。
  1) 算结合能:scripts/relax_binding.py
     调用已安装 Rosetta 迁移包的固定协议,对 pdb_inputs/ 里每个 PDB 做
     binder A 链 + 14 Å 邻域 FastRelax,再算界面结合能 dG_separated
     与逐残基界面贡献(残基能量拆分),产物落到 work_dir/results_relax/
     (含 binding_summary.csv 全局打分 / interface_residues.csv 残基拆分)。
  2) 筛选:scripts/filter_binding_top70.py
     读 binding_summary.csv 多维度排序保留前 keep 比例,入选 PDB 写头部
     rif_residues 标注,产物落到 results_relax/filtered_top70/。
输出:filtered_top70/ 作为下一轮 rfdiffusion 的输入向下传。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .base import Step, StepContext, StepResult
from ._proc import run_script

# 清洗 PDB 时保留的标准记录类型(graft 复合物首行的 rif_residues 等非标准行会被丢弃,
# 否则 Rosetta 读到无法解析的行可能告警/出错)。TER/END 保留以维持链边界。
_PDB_KEEP_PREFIXES = ("ATOM", "HETATM", "TER", "END", "MODEL", "ENDMDL", "ANISOU")


def _looks_like_pdb_inputs(input_dir: Path) -> bool:
    """判断上游产物是否已是「扁平复合物 PDB」(graft 路线),而非 AF3 cif structures 目录。

    graft 输出目录下直接是 *.pdb 文件;complex 路线是 structures/<design>/ 子目录、
    里面才是 *_model.cif。据此区分两条路线。
    """
    if not input_dir.is_dir():
        return False
    if any(input_dir.glob("*.pdb")):
        return True
    return False


def _sanitize_pdb(src: Path, dst: Path) -> None:
    """把 graft 复合物 PDB 复制到 dst, 只保留标准坐标记录, 丢掉 rif_residues 等非标准首行。

    rif_residues 信息本轮 Rosetta 不需要(下游 filter_binding_top70 会按结合能重新采样
    写回),这里去掉以免 Rosetta 解析异常。
    """
    kept = []
    for line in src.read_text().splitlines(keepends=True):
        if line.startswith(_PDB_KEEP_PREFIXES):
            kept.append(line)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(kept))


class RosettaStep(Step):
    name = "rosetta"

    def run(self, ctx: StepContext) -> StepResult:
        p = ctx.params or {}
        pdb_inputs = ctx.work_dir / "pdb_inputs"
        results_relax = ctx.work_dir / "results_relax"

        # ---- 第①段:准备 Rosetta 可读的复合物 PDB ----
        # 自动识别上游是 graft 复合物 PDB(直接清洗复用)还是 AF3 cif structures(需转换)。
        if not ctx.input_dir.is_dir():
            raise FileNotFoundError(f"rosetta 缺少上游 AF3 结构目录: {ctx.input_dir}")
        if p.get("force") and pdb_inputs.exists():
            shutil.rmtree(pdb_inputs)     # force 时清掉旧结果,保证与上游一致

        if _looks_like_pdb_inputs(ctx.input_dir):
            # graft 路线:上游已是复合物 PDB(A=binder + target), 清洗后扁平复制。
            pdb_inputs.mkdir(parents=True, exist_ok=True)
            src_pdbs = sorted(ctx.input_dir.glob("*.pdb"))
            for src in src_pdbs:
                _sanitize_pdb(src, pdb_inputs / src.name)
            n_pdb = len(list(pdb_inputs.glob("*.pdb")))
            print(f"      [graft PDB] 复用上游复合物 {n_pdb} 个 → {pdb_inputs}")
        else:
            # complex 路线:AF3 mmCIF → Rosetta 可读 PDB(扁平输出)。
            which = p.get("which", "best")    # best=每设计取最优模型;all=连样本一起转
            c_args = [str(ctx.input_dir), str(pdb_inputs), "--which", str(which)]
            rc = run_script("af3_cif_to_pdb.py", c_args,
                            log_path=ctx.work_dir / "rosetta_cif2pdb.log", python=p.get("python"))
            if rc != 0:
                raise RuntimeError(f"rosetta 第①段(af3_cif_to_pdb)退出码 {rc}")
            n_pdb = len(list(pdb_inputs.glob("*.pdb")))
            print(f"      [cif→pdb] 转换出 {n_pdb} 个 PDB → {pdb_inputs}")

        if n_pdb == 0:
            raise RuntimeError(f"未准备出任何 PDB 输入,检查上游结构目录: {ctx.input_dir}")

        # ---- 第②段:FastRelax + 结合能 + 残基能量拆分 ----
        # relax_binding.py 递归遍历 pdb_inputs/ 的 .pdb,对每个:
        #   界面 14 Å FastRelax → InterfaceAnalyzer(dG_separated 等全局指标)
        #   → residue_energy_breakdown(逐残基界面贡献)
        # 产出 binding_summary.csv(全局) + interface_residues.csv(残基拆分)。
        r_args = [str(pdb_inputs), str(results_relax)]
        if p.get("procs") is not None:
            r_args += ["--nproc", str(p["procs"])]
        if p.get("limit") is not None:
            r_args += ["--limit", str(p["limit"])]
        if p.get("force"):
            r_args += ["--force"]
        if p.get("rosetta"):
            r_args += ["--rosetta", str(p["rosetta"])]
        rc = run_script("relax_binding.py", r_args,
                        log_path=ctx.work_dir / "rosetta_relax.log", python=p.get("python"))
        if rc != 0:
            raise RuntimeError(f"rosetta 第②段(relax_binding)退出码 {rc}")

        # ---- 第③段:多维度筛选 ----
        filtered = results_relax / "filtered_top70"
        f_args = [str(results_relax), str(filtered)]
        if p.get("keep") is not None:
            f_args += ["--keep", str(p["keep"])]
        if p.get("e_full") is not None:
            f_args += ["--e-full", str(p["e_full"])]
        if p.get("kT") is not None:
            f_args += ["--kT", str(p["kT"])]
        if p.get("seed") is not None:
            f_args += ["--seed", str(p["seed"])]
        if p.get("min_rif") is not None:
            f_args += ["--min-rif", str(p["min_rif"])]
        if p.get("max_pdbs") is not None:
            f_args += ["--max-pdbs", str(p["max_pdbs"])]
        if p.get("keep_step") is not None:
            f_args += ["--keep-step", str(p["keep_step"])]
        rc = run_script("filter_binding_top70.py", f_args,
                        log_path=ctx.work_dir / "rosetta_filter.log", python=p.get("python"))
        if rc != 0:
            raise RuntimeError(f"rosetta 第③段(filter_binding_top70)退出码 {rc}")

        # 筛选结果作为下一轮 rfdiffusion 输入
        n_filtered_pdbs = len(list(filtered.glob("*.pdb")))
        if p.get("max_pdbs") is not None and n_filtered_pdbs > int(p["max_pdbs"]):
            raise RuntimeError(
                f"rosetta 自适应筛选后仍有 {n_filtered_pdbs} 个 PDB,"
                f"超过上限 {p['max_pdbs']}: {filtered}"
            )
        return StepResult(output_dir=filtered, info={
            "filtered_dir": str(filtered),
            "n_filtered_pdbs": n_filtered_pdbs,
            "pdb_inputs_dir": str(pdb_inputs),
            "n_pdb_inputs": n_pdb,
            "binding_summary": str(results_relax / "binding_summary.csv"),
            "interface_residues": str(results_relax / "interface_residues.csv"),
        })
