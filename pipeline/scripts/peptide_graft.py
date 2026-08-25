#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 AF3 折叠出的短肽(binder)按 RIF 残基刚体叠合回靶标坐标系, 合并成复合物。

背景 / 动机
-----------
AF3 做全复合物预测时会自主 docking, 无法约束 binder 相对 target 的姿态, 常把
rifdock 设计里本该固定的 RIF 残基推离靶标(见 memory: af3-template-binder-docking)。
本脚本换一条路:
  1) 让 AF3 只折叠短肽自身(单链, 见 make_af3_monomer_jobs.py);
  2) 用 RIF 残基的骨架原子(N/CA/C/O)做 Kabsch 刚体叠合, 把 AF3 短肽整体搬到参考
     结构(RFdiffusion 输出的 threaded_pdb, 其 RIF 残基骨架 = rifdock 原始锚定位置)
     的坐标系里 —— 于是 RIF 残基必然精确落回靶标界面;
  3) 把叠合后的短肽 + 参考里的 target 链拼成复合物, 顺带检查 binder↔target 重原子
     clash, 严重冲突的丢弃。

RIF align RMSD 本身就是一个天然筛选量: AF3 短肽若没折出设计的界面几何, 几个 RIF
残基无法同时叠合, RMSD 就大 —— 直接淘汰, 比界面 iptm 更直接。

参考结构约定(threaded_pdb): binder 链(默认 A) = 设计序列骨架, 其余链 = target,
且 RIF 残基号(rif_residues_out)与 AF3 短肽残基号一一对应。

用法
----
  python peptide_graft.py \
      --peptide <AF3 短肽 .cif 或 .pdb> \
      --ref     <threaded_pdb 参考复合物> \
      --rif     20,30,34,42,43,46 \
      --out     <输出复合物 .pdb>

依赖: numpy(af3 环境自带)。.cif 输入复用 af3_cif_to_pdb 转 PDB。
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

# 复用同目录 af3_cif_to_pdb 的 cif→pdb 转换(统一走 PDB 解析路径)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from af3_cif_to_pdb import cif_to_pdb  # noqa: E402

BACKBONE = ("N", "CA", "C", "O")


# ----------------------------- PDB 解析 -----------------------------

class Atom:
    """保留原始 PDB 行文本, 只在写出时替换坐标字段, 从而不破坏原子名/元素等格式。"""
    __slots__ = ("line", "chain", "resid", "name", "element", "xyz")

    def __init__(self, line: str):
        self.line = line.rstrip("\n")
        self.chain = line[21]
        self.resid = int(line[22:26])
        self.name = line[12:16].strip()
        self.element = line[76:78].strip() or self.name.lstrip("0123456789")[:1]
        self.xyz = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float64,
        )

    def with_xyz(self, xyz: np.ndarray) -> str:
        """返回替换坐标后的 PDB 行(其余列原样)。"""
        x, y, z = xyz
        return f"{self.line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{self.line[54:]}"

    @property
    def is_heavy(self) -> bool:
        return self.element.upper() != "H"


def read_pdb_atoms(pdb_path: Path) -> list[Atom]:
    atoms = []
    for line in Path(pdb_path).read_text().splitlines(keepends=True):
        if line.startswith(("ATOM", "HETATM")):
            atoms.append(Atom(line))
    if not atoms:
        raise ValueError(f"{pdb_path} 未解析到任何 ATOM 记录")
    return atoms


def load_atoms_any(path: Path) -> list[Atom]:
    """输入既可能是 .cif(AF3 输出)也可能是 .pdb, 统一返回 Atom 列表。"""
    path = Path(path)
    if path.suffix.lower() == ".cif":
        tmp = Path(tempfile.mkstemp(suffix=".pdb")[1])
        try:
            cif_to_pdb(path, tmp)
            return read_pdb_atoms(tmp)
        finally:
            tmp.unlink(missing_ok=True)
    return read_pdb_atoms(path)


# ----------------------------- 几何 -----------------------------

def kabsch(P: np.ndarray, Q: np.ndarray):
    """求把 P 叠到 Q 的最优刚体变换 (R, t): R@p + t ≈ q。返回 (R, t, rmsd)。"""
    cP, cQ = P.mean(0), Q.mean(0)
    P0, Q0 = P - cP, Q - cQ
    H = P0.T @ Q0
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cQ - R @ cP
    diff = (P @ R.T + t) - Q
    rmsd = float(np.sqrt((diff * diff).sum() / len(P)))
    return R, t, rmsd


def backbone_map(atoms: list[Atom], chain: str, resids: set[int]) -> dict:
    """取指定链、指定残基号的骨架原子, 键=(resid, atomname), 值=坐标。"""
    out = {}
    for a in atoms:
        if a.chain == chain and a.resid in resids and a.name in BACKBONE:
            out[(a.resid, a.name)] = a.xyz
    return out


def count_clashes(moved: list[Atom], target: list[Atom], dist: float):
    """统计 binder 重原子与 target 重原子距离 < dist 的原子对数。返回 (对数, 最近距离)。"""
    B = np.array([a.xyz for a in moved if a.is_heavy])
    T = np.array([a.xyz for a in target if a.is_heavy])
    if len(B) == 0 or len(T) == 0:
        return 0, float("inf")
    # 分块算距离矩阵, 控内存
    n_clash, min_d = 0, float("inf")
    for i in range(0, len(B), 512):
        blk = B[i:i + 512]
        d = np.sqrt(((blk[:, None, :] - T[None, :, :]) ** 2).sum(-1))
        n_clash += int((d < dist).sum())
        min_d = min(min_d, float(d.min()))
    return n_clash, min_d


# ----------------------------- 主流程 -----------------------------

def graft(peptide_path, ref_path, rif_resids, out_path,
          pep_chain="A", ref_binder_chain="A",
          clash_dist=2.0, clash_max=0, rmsd_max=None, write=True):
    """把短肽按 RIF 骨架叠到参考、拼 target、查 clash。返回结果 dict。"""
    rif = set(rif_resids)
    pep_atoms = load_atoms_any(peptide_path)
    ref_atoms = read_pdb_atoms(Path(ref_path))

    # 参考里 binder 链之外的都算 target
    target_atoms = [a for a in ref_atoms if a.chain != ref_binder_chain]
    if not target_atoms:
        raise ValueError(f"参考 {ref_path} 里除 {ref_binder_chain} 链外没有 target 链")

    # 配对 RIF 骨架原子
    pep_bb = backbone_map(pep_atoms, pep_chain, rif)
    ref_bb = backbone_map(ref_atoms, ref_binder_chain, rif)
    keys = sorted(set(pep_bb) & set(ref_bb))
    if len(keys) < 3:
        raise ValueError(
            f"可配对的 RIF 骨架原子太少({len(keys)}): 检查 --rif 残基号是否与两侧链一致。"
            f"\n  短肽侧命中残基: {sorted({k[0] for k in pep_bb})}"
            f"\n  参考侧命中残基: {sorted({k[0] for k in ref_bb})}")
    P = np.array([pep_bb[k] for k in keys])   # 短肽 RIF 骨架
    Q = np.array([ref_bb[k] for k in keys])   # 参考 RIF 骨架(目标位置)
    R, t, rmsd = kabsch(P, Q)

    # 变换短肽全部原子
    moved = []
    for a in pep_atoms:
        if a.chain != pep_chain:
            continue
        a.xyz = R @ a.xyz + t
        moved.append(a)

    n_clash, min_d = count_clashes(moved, target_atoms, clash_dist)

    passed = True
    reasons = []
    if rmsd_max is not None and rmsd > rmsd_max:
        passed = False
        reasons.append(f"RIF_RMSD={rmsd:.2f}>{rmsd_max}")
    if n_clash > clash_max:
        passed = False
        reasons.append(f"clash={n_clash}>{clash_max}")

    result = {
        "rif_rmsd": rmsd, "n_rif_atoms": len(keys),
        "n_clash": n_clash, "min_dist": min_d,
        "passed": passed, "reasons": reasons,
    }

    if write and passed:
        _write_complex(moved, target_atoms, rif_resids, out_path)
        result["out"] = str(out_path)
    return result


def _write_complex(binder: list[Atom], target: list[Atom], rif_resids, out_path):
    """写复合物 PDB: binder(A 链原样) + target(原样), 首行 rif_residues 头。"""
    lines = ["rif_residues " + ",".join(map(str, rif_resids)) + "\n"]
    serial = 0
    for a in binder:
        serial += 1
        lines.append(_renumber(a.with_xyz(a.xyz), serial) + "\n")
    lines.append(f"TER   {serial + 1:>5d}\n")
    for a in target:
        serial += 1
        lines.append(_renumber(a.with_xyz(a.xyz), serial) + "\n")
    lines.append("END\n")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("".join(lines))


def _renumber(line: str, serial: int) -> str:
    """替换原子序号(第 7-11 列), 其余原样。"""
    return f"{line[:6]}{serial:>5d}{line[11:]}"


def _parse_rif(s: str) -> list[int]:
    return sorted({int(x) for x in s.replace(",", " ").split()})


def main():
    ap = argparse.ArgumentParser(description="AF3 短肽按 RIF 残基叠合回靶标, 拼复合物 + 查 clash")
    ap.add_argument("--peptide", required=True, help="AF3 折叠出的短肽结构 (.cif 或 .pdb)")
    ap.add_argument("--ref", required=True, help="参考复合物 (threaded_pdb): binder 链 + target 链")
    ap.add_argument("--rif", required=True, help="RIF 残基号 (逗号分隔, 1-based, binder 链编号)")
    ap.add_argument("--out", required=True, help="输出复合物 .pdb")
    ap.add_argument("--pep-chain", default="A", help="短肽里 binder 链 id (默认 A)")
    ap.add_argument("--ref-binder-chain", default="A", help="参考里 binder 链 id (默认 A)")
    ap.add_argument("--clash-dist", type=float, default=2.0, help="重原子 clash 距离阈值 Å (默认 2.0)")
    ap.add_argument("--clash-max", type=int, default=5,
                    help="允许的 clash 原子对数上限 (默认 5; 单链预测界面难免轻微接触, "
                         "少量 2Å 级接触下游 Rosetta FastRelax 可消除)")
    ap.add_argument("--rmsd-max", type=float, default=None, help="RIF align RMSD 上限, 超过则判失败 (默认不限)")
    args = ap.parse_args()

    res = graft(args.peptide, args.ref, _parse_rif(args.rif), args.out,
                pep_chain=args.pep_chain, ref_binder_chain=args.ref_binder_chain,
                clash_dist=args.clash_dist, clash_max=args.clash_max,
                rmsd_max=args.rmsd_max)
    flag = "✅ PASS" if res["passed"] else "❌ FAIL"
    print(f"{flag}  RIF_RMSD={res['rif_rmsd']:.3f}Å ({res['n_rif_atoms']} 原子)  "
          f"clash={res['n_clash']} (最近 {res['min_dist']:.2f}Å)")
    if res["reasons"]:
        print("  原因:", "; ".join(res["reasons"]))
    if res.get("out"):
        print("  写出:", res["out"])
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
