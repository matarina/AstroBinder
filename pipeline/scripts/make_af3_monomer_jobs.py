#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 graft 方案生成 AF3「单链短肽」推理 job(只折 binder, 不放 target)。

与 filter_for_af3.py 的区别: 那个生成的是 A+target 全复合物 job(让 AF3 自主
docking); 本脚本只放 binder A 链, 并给 RIF 残基上部分模板, 让 AF3 尽量把短肽内部
(尤其界面 RIF 残基间)的几何折成设计的样子。后续 peptide_graft.py 再用 RIF 残基把
短肽刚体叠回靶标坐标系 —— 从而绕开 AF3 管不住的链间对接(见 memory:
af3-template-binder-docking)。

复用 filter_for_af3 的硬过滤 + 组内保留逻辑, 保证入选设计集与全复合物路线一致。
RIF 模板坐标取自 threaded_pdb 的 binder 链(RFdiffusion 输出骨架, RIF 残基为固定
motif, 骨架位置 = rifdock 原始锚定)。

产物:
  <out>/<seq_id>.json        AF3 单链 job(带 RIF 部分模板), 直接喂 run_af3_batch.py
  <out>/_monomer_manifest.csv  每条设计的 threaded_pdb / rif_residues_out,
                               供 peptide_graft.py 批量叠合时定位参考结构与锚点

用法:
  <af3 env python> make_af3_monomer_jobs.py -f <summary.csv> -o <输出文件夹> \
      [--keep-fraction 0.3] [--seed 1] [--no-template]

需在 af3 环境跑(import alphafold3)。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import filter_for_af3 as ff          # 复用硬过滤/分组/target 定位
from af3_template import build_template


def parse_rif(s: str) -> list[int]:
    return sorted({int(x) for x in s.replace(",", " ").split() if x.strip()})


def select_rows(summary_csv: str, keep_fraction: float):
    """复用 filter_for_af3 的规则: 硬过滤 -> 同(scaffold,长度)组内按 mpnn_score 留前比例。"""
    with open(summary_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    kept = [r for r in rows if ff.passes_hard_filters(r, [])]
    groups: dict[tuple, list] = {}
    for r in kept:
        groups.setdefault((r.get("pdb_stem", ""), r.get("seq_length", "")), []).append(r)
    final = []
    for members in groups.values():
        members.sort(key=lambda x: ff.to_float(x.get("mpnn_score")) or float("inf"))
        final.extend(members[:math.floor(len(members) * keep_fraction)])
    return len(rows), len(kept), final


def main():
    ap = argparse.ArgumentParser(description="生成 AF3 单链短肽 job(RIF 上模板, 不放 target)")
    ap.add_argument("-f", "--summary", required=True, help="summary.csv 路径")
    ap.add_argument("-o", "--out", required=True, help="输出文件夹(需不存在)")
    ap.add_argument("--seed", type=int, default=1, help="AF3 modelSeeds(默认 1)")
    ap.add_argument("--keep-fraction", type=float, default=ff.KEEP_FRACTION,
                    help=f"组内保留比例(默认 {ff.KEEP_FRACTION})")
    ap.add_argument("--template-mode", choices=["rif", "full", "none"], default="rif",
                    help="binder 模板覆盖: rif=只RIF残基(默认) / full=全长骨架 / none=无模板。"
                         "覆盖越少非RIF区越自由(多样性↑但graft通过率↓)")
    args = ap.parse_args()

    if not os.path.isfile(args.summary):
        sys.exit(f"错误: 找不到 summary.csv: {args.summary}")
    if os.path.exists(args.out):
        sys.exit(f"错误: 输出文件夹已存在, 请换路径或先删除: {args.out}")
    os.makedirs(args.out)

    total, n_kept, final = select_rows(args.summary, args.keep_fraction)

    written, skipped = 0, []
    manifest = []
    for r in final:
        seq_id = (r.get("seq_id", "") or f"design_{written}").strip()
        a_seq = r.get("designed_seq", "").strip()
        threaded = r.get("threaded_pdb", "").strip()
        rif = parse_rif(r.get("rif_residues_out", "") or r.get("rif_residues", ""))
        if not a_seq or not threaded or not os.path.isfile(threaded) or not rif:
            skipped.append(seq_id)
            continue

        a_entry = {"protein": {"id": "A", "sequence": a_seq,
                               "unpairedMsa": "", "pairedMsa": "", "templates": []}}
        if args.template_mode != "none":
            # 模板坐标取自 threaded_pdb 的 A 链(设计骨架); full=全长, rif=只RIF残基
            q = None if args.template_mode == "full" else rif
            tmpl, tmpl_seq = build_template(threaded, "A", name=f"{seq_id}_A",
                                            query_indices_1based=q)
            if tmpl_seq != a_seq:
                # 序列应一致(threaded 即 designed_seq thread 回骨架); 不一致则告警但仍用坐标
                print(f"[警告] {seq_id}: threaded A 链序列与 designed_seq 不一致, "
                      f"仅用坐标做模板", flush=True)
            a_entry["protein"]["templates"] = [tmpl]

        job = {"name": seq_id, "modelSeeds": [args.seed],
               "sequences": [a_entry], "dialect": "alphafold3", "version": 1}
        with open(os.path.join(args.out, f"{seq_id}.json"), "w") as fh:
            import json
            json.dump(job, fh, indent=2)
        written += 1
        manifest.append({"seq_id": seq_id, "threaded_pdb": threaded,
                         "rif_residues_out": ",".join(map(str, rif)),
                         "a_length": r.get("seq_length", "")})

    if manifest:
        with open(os.path.join(args.out, "_monomer_manifest.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
            w.writeheader(); w.writerows(manifest)

    print("=" * 60)
    print(f"总设计数          : {total}")
    print(f"硬过滤通过        : {n_kept}")
    print(f"组内留前 {int(args.keep_fraction*100)}% 后  : {len(final)}")
    print(f"写出单链 job      : {written}  -> {args.out}  "
          f"(模板: {args.template_mode})")
    if skipped:
        print(f"⚠️ 缺序列/threaded/rif 跳过: {len(skipped)} ({', '.join(skipped[:10])})")
    print("=" * 60)


if __name__ == "__main__":
    main()
