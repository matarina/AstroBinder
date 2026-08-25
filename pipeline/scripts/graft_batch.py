#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量把 AF3 单链短肽的多采样结果叠合回靶标, 筛选 + 统计。

配合 make_af3_monomer_jobs.py(出 _monomer_manifest.csv)与 run_af3_batch.py 的
输出目录使用。对每条设计的每个 seed-sample:
  peptide_graft 叠合 -> RIF align RMSD + binder↔target clash
然后:
  - 每条设计按 RIF_RMSD 升序, 写出通过筛选的前 --top 个复合物 PDB(供 rosetta);
  - 打印 RMSD 分布 / 各 clash 阈值通过率 / 非 RIF 区跨样本多样性(以 RIF 叠合后 RMSF)。

用法:
  <af3 env python> graft_batch.py \
      --manifest <batch_input_mono/_monomer_manifest.csv> \
      --structures <batch_output_mono/structures> \
      --out-dir <复合物输出目录> \
      [--rmsd-max 2.0] [--clash-dist 2.0] [--clash-max 0] [--top 3]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import peptide_graft as pg


def sample_cifs(struct_dir: Path, seq_id: str):
    """返回该设计所有 seed-sample 的 model.cif(标签, 路径)。"""
    base = struct_dir / seq_id
    out = []
    for cf in sorted(glob.glob(str(base / "seed-*/*_model.cif"))):
        tag = Path(cf).parent.name        # seed-<S>_sample-<K>
        out.append((tag, cf))
    return out


def nonrif_diversity(cifs, rif, pep_chain="A"):
    """以 RIF CA 为基准把各样本叠到首样本, 算非 RIF 区 CA 跨样本 RMSF 均值。"""
    rif = set(rif)
    per_res, base = {}, None
    for _, cf in cifs:
        d = {a.resid: a.xyz for a in pg.load_atoms_any(Path(cf))
             if a.chain == pep_chain and a.name == "CA"}
        if base is None:
            base = d
        rk = sorted(rif & set(d) & set(base))
        if len(rk) < 3:
            continue
        R, t, _ = pg.kabsch(np.array([d[k] for k in rk]),
                            np.array([base[k] for k in rk]))
        for k, v in d.items():
            per_res.setdefault(k, []).append(R @ v + t)
    if not per_res:
        return None, None
    rmsf = {k: float(np.sqrt(((np.array(v) - np.array(v).mean(0)) ** 2).sum(1).mean()))
            for k, v in per_res.items() if len(v) > 1}
    rif_r = [rmsf[k] for k in rmsf if k in rif]
    non_r = [rmsf[k] for k in rmsf if k not in rif]
    return (np.mean(rif_r) if rif_r else None,
            np.mean(non_r) if non_r else None)


def main():
    ap = argparse.ArgumentParser(description="批量 graft AF3 单链多采样 + 筛选统计")
    ap.add_argument("--manifest", required=True, help="_monomer_manifest.csv")
    ap.add_argument("--structures", required=True, help="AF3 输出 structures 目录")
    ap.add_argument("--out-dir", required=True, help="通过筛选的复合物 PDB 输出目录")
    ap.add_argument("--rmsd-max", type=float, default=2.0, help="RIF align RMSD 上限 Å")
    ap.add_argument("--clash-dist", type=float, default=2.0, help="重原子 clash 距离阈值 Å")
    ap.add_argument("--clash-max", type=int, default=5,
                    help="允许 clash 原子对数上限 (默认 5; 轻微接触交下游 Rosetta relax)")
    ap.add_argument("--top", type=int, default=3, help="每设计写出通过筛选的前 N 个")
    args = ap.parse_args()

    struct_dir = Path(args.structures)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.manifest, newline="")))

    grand_pass = 0
    grand_total = 0
    for r in rows:
        seq_id = r["seq_id"]
        ref = r["threaded_pdb"]
        rif = [int(x) for x in r["rif_residues_out"].replace(",", " ").split()]
        cifs = sample_cifs(struct_dir, seq_id)
        if not cifs:
            print(f"[{seq_id}] 无样本, 跳过"); continue

        recs = []
        for tag, cf in cifs:
            try:
                res = pg.graft(cf, ref, rif, None, clash_dist=args.clash_dist,
                               clash_max=args.clash_max, rmsd_max=args.rmsd_max,
                               write=False)
            except Exception as e:
                print(f"  [{seq_id}/{tag}] graft 失败: {e}"); continue
            res["tag"] = tag; res["cif"] = cf
            recs.append(res)
        if not recs:
            continue

        rmsds = np.array([x["rif_rmsd"] for x in recs])
        clashes = np.array([x["n_clash"] for x in recs])
        passed = [x for x in recs if x["passed"]]
        grand_pass += len(passed); grand_total += len(recs)
        rif_div, non_div = nonrif_diversity(cifs, rif)

        print(f"\n===== {seq_id}  ({len(recs)} 样本, RIF={rif}) =====")
        print(f"  RIF_RMSD: 最小 {rmsds.min():.2f} / 中位 {np.median(rmsds):.2f} / "
              f"最大 {rmsds.max():.2f} Å")
        print(f"  clash(<{args.clash_dist}Å): 最小 {clashes.min()} / 中位 "
              f"{int(np.median(clashes))} / 最大 {clashes.max()}")
        print(f"  通过(RMSD≤{args.rmsd_max} 且 clash≤{args.clash_max}): "
              f"{len(passed)}/{len(recs)} = {100*len(passed)/len(recs):.0f}%")
        if non_div is not None:
            print(f"  跨样本多样性(以RIF叠合): RIF区 {rif_div:.2f}Å | 非RIF区 {non_div:.2f}Å")

        # 写通过筛选的前 top 个复合物
        passed.sort(key=lambda x: x["rif_rmsd"])
        for i, x in enumerate(passed[:args.top]):
            out = out_dir / f"{seq_id}__{x['tag']}.pdb"
            pg.graft(x["cif"], ref, rif, out, clash_dist=args.clash_dist,
                     clash_max=args.clash_max, rmsd_max=args.rmsd_max, write=True)
            print(f"    写出 #{i+1}: {out.name}  (RMSD {x['rif_rmsd']:.2f}Å, clash {x['n_clash']})")

    print(f"\n总通过率: {grand_pass}/{grand_total} = "
          f"{100*grand_pass/max(grand_total,1):.0f}%  -> {out_dir}")


if __name__ == "__main__":
    main()
