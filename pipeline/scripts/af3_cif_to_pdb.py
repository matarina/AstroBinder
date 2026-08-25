#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 AlphaFold3 的 mmCIF 结构转成 Rosetta 能直接读的 PDB(纯标准库,无第三方依赖)。

背景:
  AF3 只输出 mmCIF(ModelCIF 方言),而本仓库的 Rosetta 打分脚本
  relax_binding.py 递归找 .pdb。为固定 PDB 头、原子命名和链约定,并让计算
  输入不依赖 Rosetta 对 ModelCIF 方言的兼容细节,在进 Rosetta 前先把
  cif 规规矩矩转成经典 PDB ATOM 记录最稳妥。

AF3 输出目录结构(每个设计一个子目录):
  structures/<design>/
      <design>_model.cif              ← 顶层 = 排名最高的模型(best)
      <design>_ranking_scores.csv
      seed-<S>_sample-<K>/<design>_seed-<S>_sample-<K>_model.cif  ← 全部 25 个样本

链约定:A 链 = 设计序列(scaffold),B 链 = 靶标。转换保留 auth_asym_id / auth_seq_id
(即作者链号/残基号),与下游 InterfaceAnalyzer「A vs B」的界面判定一致。

用法:
  # 单文件转换
  python af3_cif_to_pdb.py <input.cif> <output.pdb>

  # 批量:遍历 AF3 structures 目录,每个设计取一个/全部模型转成 PDB 存到输出目录
  python af3_cif_to_pdb.py <structures_dir> <out_pdb_dir> [--which best|all]

    --which best  每个设计只转顶层 best-ranked 模型 → <design>.pdb (默认)
    --which all   连同各 seed-sample 一起转 → <design>__<seed-sample>.pdb
"""
import argparse
import sys
from pathlib import Path


# mmCIF _atom_site 里需要用到的列名(按列名取值,不依赖列顺序)
_NEEDED = ("group_PDB", "label_atom_id", "label_comp_id", "auth_asym_id",
           "auth_seq_id", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
           "B_iso_or_equiv", "type_symbol", "label_alt_id",
           "pdbx_PDB_ins_code", "pdbx_PDB_model_num")


def _atom_name_field(name: str, element: str) -> str:
    """按 PDB 约定把原子名放进第 13–16 列(4 字符宽)。

    单字符元素且原子名 ≤3 位时,名字前留一格(如 ' CA ', ' N  ', ' OXT');
    四字符名或双字符元素则从第 13 列顶格填。"""
    if len(name) >= 4 or len(element) >= 2:
        return f"{name:<4.4s}"
    return f" {name:<3s}"


def _parse_atom_site(cif_path: Path):
    """解析 mmCIF 的 _atom_site loop,产出每条原子记录的 dict(仅取第一个模型)。

    mmCIF 的 loop_ 里字段名逐行列在 `_atom_site.<col>`,随后是数据行(空白分隔)。
    AF3 输出的坐标行字段无内嵌空格,可安全按空白切分。"""
    cols = []           # _atom_site 列名(按出现顺序)
    in_loop = False     # 是否处在 _atom_site 的字段声明区
    reading = False     # 是否处在数据区
    first_model = None
    with open(cif_path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("_atom_site."):
                cols.append(s.split(".", 1)[1])
                in_loop = True
                continue
            if in_loop and not reading:
                # 字段声明结束、进入数据区的第一行
                if not s or s.startswith("_") or s.startswith("#"):
                    continue
                reading = True
            if reading:
                if not s or s.startswith("#") or s.startswith("loop_") or s.startswith("_"):
                    break               # 数据区结束
                toks = s.split()
                if len(toks) != len(cols):
                    continue            # 不合规行,跳过
                row = dict(zip(cols, toks))
                mdl = row.get("pdbx_PDB_model_num")
                if first_model is None:
                    first_model = mdl
                if mdl != first_model:  # 只保留第一个模型
                    break
                yield row
    if not cols:
        raise ValueError(f"{cif_path} 中未找到 _atom_site 记录")


def cif_to_pdb(cif_path, pdb_path) -> int:
    """把单个 mmCIF 转为 PDB,返回写出的原子数。链切换处插 TER,末尾写 END。"""
    cif_path, pdb_path = Path(cif_path), Path(pdb_path)
    lines = []
    serial = 0
    ter = 0
    prev_chain = None
    last = None                          # 记录上一条,用于写 TER
    for row in _parse_atom_site(cif_path):
        chain = row["auth_asym_id"]
        if prev_chain is not None and chain != prev_chain:
            ter += 1
            lines.append(
                f"TER   {serial + 1:>5d}      "
                f"{last['comp']:>3s} {last['chain']:1.1s}{last['resseq']:>4s}\n")
            serial += 1
        serial += 1

        group = row["group_PDB"]         # ATOM / HETATM
        atom = row["label_atom_id"].strip('"')
        comp = row["label_comp_id"]
        element = row["type_symbol"]
        alt = row.get("label_alt_id", ".")
        alt = " " if alt in (".", "?") else alt
        ins = row.get("pdbx_PDB_ins_code", "?")
        ins = " " if ins in (".", "?") else ins
        resseq = row["auth_seq_id"]
        x = float(row["Cartn_x"]); y = float(row["Cartn_y"]); z = float(row["Cartn_z"])
        occ = float(row["occupancy"]); bfac = float(row["B_iso_or_equiv"])
        name_field = _atom_name_field(atom, element)

        lines.append(
            f"{group:<6.6s}{serial:>5d} {name_field}{alt:1.1s}{comp:>3.3s} "
            f"{chain:1.1s}{resseq:>4s}{ins:1.1s}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}"
            f"          {element:>2.2s}\n")
        prev_chain = chain
        last = {"comp": comp, "chain": chain, "resseq": resseq}

    if serial == 0:
        raise ValueError(f"{cif_path} 未解析到任何原子")
    # 末尾补 TER + END
    lines.append(
        f"TER   {serial + 1:>5d}      "
        f"{last['comp']:>3s} {last['chain']:1.1s}{last['resseq']:>4s}\n")
    lines.append("END\n")

    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    pdb_path.write_text("".join(lines))
    return serial - ter                  # 真实原子数(扣除 TER 占的 serial)


# ---------- 批量:遍历 AF3 structures 目录 ----------

def _best_cif(design_dir: Path):
    """返回该设计目录顶层的 best-ranked 模型 cif(<design>_model.cif)。"""
    hits = sorted(design_dir.glob("*_model.cif"))
    return hits[0] if hits else None


def _all_cifs(design_dir: Path):
    """返回该设计所有 seed-sample 子目录里的 model.cif,键为 seed-sample 标签。"""
    out = []
    for sub in sorted(design_dir.iterdir()):
        if sub.is_dir():
            for cif in sorted(sub.glob("*_model.cif")):
                out.append((sub.name, cif))
    return out


def batch(struct_dir: Path, out_dir: Path, which: str) -> list:
    """遍历 AF3 structures 目录,把每个设计的模型转成 PDB 写到 out_dir(扁平)。

    which=best: 每个设计一个 <design>.pdb(顶层 best-ranked)
    which=all : 额外把每个 seed-sample 转成 <design>__<seed-sample>.pdb
    返回 [(pdb_path, natoms), ...]。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    designs = sorted(d for d in struct_dir.iterdir() if d.is_dir())
    if not designs:
        raise SystemExit(f"在 {struct_dir} 下没找到任何设计子目录")
    for d in designs:
        best = _best_cif(d)
        if best is None:
            print(f"[warn] {d.name} 顶层无 *_model.cif,跳过")
            continue
        pdb = out_dir / f"{d.name}.pdb"
        n = cif_to_pdb(best, pdb)
        done.append((pdb, n))
        print(f"[ok] {d.name}.pdb  ({n} atoms)")
        if which == "all":
            for tag, cif in _all_cifs(d):
                pdb2 = out_dir / f"{d.name}__{tag}.pdb"
                n2 = cif_to_pdb(cif, pdb2)
                done.append((pdb2, n2))
    return done


def main():
    ap = argparse.ArgumentParser(description="AF3 mmCIF → Rosetta 可读 PDB")
    ap.add_argument("inp", help="单个 .cif 文件,或 AF3 structures 目录")
    ap.add_argument("out", help="输出 .pdb 文件,或输出目录(批量时)")
    ap.add_argument("--which", choices=["best", "all"], default="best",
                    help="批量时每个设计转 best-ranked 模型还是全部样本(默认 best)")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    if inp.is_file():
        n = cif_to_pdb(inp, out)
        print(f"{inp}  →  {out}  ({n} atoms)")
    elif inp.is_dir():
        done = batch(inp, out, args.which)
        print(f"\n共转换 {len(done)} 个 PDB → {out}")
    else:
        sys.exit(f"输入不存在: {inp}")


if __name__ == "__main__":
    main()
