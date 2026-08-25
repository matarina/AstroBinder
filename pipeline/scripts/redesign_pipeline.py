#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量「A 链重扩散 + 序列重设计」流水线
================================================

功能概述
--------
给定一个根文件夹, 递归遍历其下所有子文件夹, 找出全部 *.pdb, 对每个 PDB:

  1. 读取头部 `rif_residues 33,42,44,46` 标注 (A 链中要固定空间位置的关键残基)。
  2. 用 RFdiffusion 做 motif scaffolding (重扩散):
       - RIF 残基作为坐标固定的 motif, 空间位置不动;
       - A 链其余残基忽略原氨基酸类型, 重新生成主链骨架;
       - 其余所有非 A 链 (target 靶标, B/C/D...) 整链按原坐标固定。
  3. 用 ProteinMPNN 给新骨架设计 A 链序列:
       - RIF 位点固定为原氨基酸;
       - 其余所有非 A 链固定不设计。
  4. 每个 PDB 的产物放在各自独立的文件夹里, 输出一个总 CSV 汇总所有设计序列及其属性。

输出布局 (OUTROOT 下):
  per_pdb/                         # 所有输入 PDB 的任务目录统一归拢在此
    <stem>/                        # 每个输入 PDB 一个文件夹
      input.pdb                    # 原始输入副本
      rfdiffusion/                 # 重扩散骨架 (design_0.pdb / .trb / 日志)
      mpnn/                        # ProteinMPNN 原始输出
      designs/                     # thread 好的完整复合物 (新序列贴回骨架)
      sequences.csv                # 该 PDB 的逐序列明细
  summary.csv                      # 全部 PDB 的总汇总表

CSV 字段见 CSV_HEADER。
"""

import os
import sys
import csv
import glob
import json
import shutil
import pickle
import argparse
import subprocess
import multiprocessing as mp
import queue
from collections import OrderedDict

# ===== 路径常量 (与本机环境绑定) =====
PY = "/mnt/data2/wtk/software/miniconda3/envs/RFDiffusion/bin/python"
RFDIFF_DIR = "/mnt/data2/wtk/software/RFdiffusion/RFdiffusion-main"
RFDIFF_RUN = os.path.join(RFDIFF_DIR, "scripts/run_inference.py")
MPNN_DIR = "/mnt/data2/wtk/software/ProteinMPNN/ProteinMPNN-main"
MPNN_HELPER = os.path.join(MPNN_DIR, "helper_scripts")

# 各输入 PDB 的任务目录统一归拢到 out_root 下的这个子文件夹,
# 避免骨架多时一堆 <stem>/ 目录直接散落在 out_root 根部。
PER_PDB_SUBDIR = "per_pdb"

# 三字母 -> 单字母
THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U",
}

# Kyte-Doolittle 疏水性 (用于 GRAVY)
KD_HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}

# 单残基平均分子量 (Da), 计算肽链时减去 (n-1) 个水
MONO_MW = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "Q": 146.15, "E": 147.13, "G": 75.07, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}
WATER_MW = 18.015

# pKa (Bjellqvist), 用于净电荷 / pI
PKA_POS = {"K": 10.54, "R": 12.48, "H": 6.04}     # 带正电
PKA_NEG = {"D": 3.90, "E": 4.07, "C": 8.18, "Y": 10.46}  # 带负电
PKA_NTERM = 9.69
PKA_CTERM = 2.34

AROMATIC = set("FWY")


# ============================================================
#  序列属性计算 (纯 Python, 不依赖 biopython)
# ============================================================
def seq_mw(seq):
    """肽链分子量 (Da)。"""
    if not seq:
        return 0.0
    total = sum(MONO_MW.get(a, 0.0) for a in seq)
    return round(total - WATER_MW * (len(seq) - 1), 2)


def net_charge(seq, ph=7.0):
    """给定 pH 下的净电荷 (Henderson-Hasselbalch)。"""
    pos = 1.0 / (1.0 + 10 ** (ph - PKA_NTERM))
    for a, pk in PKA_POS.items():
        pos += seq.count(a) * (1.0 / (1.0 + 10 ** (ph - pk)))
    neg = 1.0 / (1.0 + 10 ** (PKA_CTERM - ph))
    for a, pk in PKA_NEG.items():
        neg += seq.count(a) * (1.0 / (1.0 + 10 ** (pk - ph)))
    return pos - neg


def isoelectric_point(seq):
    """二分法求 pI。"""
    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if net_charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2.0, 2)


def gravy(seq):
    """平均疏水性 (GRAVY)。"""
    if not seq:
        return 0.0
    return round(sum(KD_HYDRO.get(a, 0.0) for a in seq) / len(seq), 3)


def aromatic_frac(seq):
    """芳香族残基比例 (F/W/Y)。"""
    if not seq:
        return 0.0
    return round(sum(1 for a in seq if a in AROMATIC) / len(seq), 3)


def seq_identity(a, b):
    """两条等长序列的一致性 (%)。"""
    if not a or len(a) != len(b):
        return None
    same = sum(1 for x, y in zip(a, b) if x == y)
    return round(100.0 * same / len(a), 2)


def mutation_list(orig, design, offset=1):
    """逐位比较, 返回突变列表 (如 ['S1A', 'G5V']) 与突变数。offset = A 链起始残基号。"""
    muts = []
    if not orig or len(orig) != len(design):
        return muts, None
    for i, (o, d) in enumerate(zip(orig, design)):
        if o != d:
            muts.append("%s%d%s" % (o, i + offset, d))
    return muts, len(muts)


# ============================================================
#  PDB / 头部解析
# ============================================================
def parse_rif_residues(pdb_path):
    """从头部读取 `rif_residues 33,42,44,46` -> [33,42,44,46] (升序去重)。"""
    res = []
    with open(pdb_path) as fh:
        for line in fh:
            low = line.strip().lower()
            if low.startswith("rif_residues"):
                body = line.split(None, 1)[1] if len(line.split(None, 1)) > 1 else ""
                for tok in body.replace(",", " ").split():
                    try:
                        res.append(int(tok))
                    except ValueError:
                        pass
                break
    return sorted(set(res))


def parse_chain_seq(pdb_path, chain):
    """读取指定链的序列与残基号列表 (按 CA 顺序)。返回 (seq_str, [resi,...])。"""
    seq, resis = [], []
    seen = set()
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[21] != chain:
                continue
            atom = line[12:16].strip()
            if atom != "CA":
                continue
            resi = int(line[22:26])
            if resi in seen:
                continue
            seen.add(resi)
            resn = line[17:20].strip()
            seq.append(THREE2ONE.get(resn, "X"))
            resis.append(resi)
    return "".join(seq), resis


def chain_bounds(pdb_path):
    """返回 {chain: (min_resi, max_resi, n_residues)}。"""
    info = OrderedDict()
    seen = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            ch = line[21]
            atom = line[12:16].strip()
            if atom != "CA":
                continue
            resi = int(line[22:26])
            seen.setdefault(ch, set()).add(resi)
    for ch, rs in seen.items():
        info[ch] = (min(rs), max(rs), len(rs))
    return info


# ============================================================
#  contig 构造
# ============================================================
def build_contig(rif_positions, a_start, a_len, targets):
    """
    构造 motif scaffolding 的 contig 字符串。
    - A 链共 a_len 个残基 (从 a_start 开始, 连续编号);
    - rif_positions 为 1-based 的 PDB 残基号, 作为固定 motif;
    - 各生成段长度严格等于原长 -> 输出残基号与输入保持一致;
    - targets 为所有非 A 链 [(chain, start, end), ...], 每条整链按原坐标固定。
    返回 (contig_str, design_chain_id)。design_chain_id 恒为 'A'。
    """
    a_end = a_start + a_len - 1
    rif = [p for p in rif_positions if a_start <= p <= a_end]
    segs = []
    cursor = a_start
    for p in rif:
        gap = p - cursor          # p 之前的可生成段长度
        if gap > 0:
            segs.append("%d-%d" % (gap, gap))
        segs.append("A%d-%d" % (p, p))   # 固定 motif 单残基
        cursor = p + 1
    tail = a_end - cursor + 1
    if tail > 0:
        segs.append("%d-%d" % (tail, tail))
    a_contig = "/".join(segs) + "/0"           # /0 表示链断点
    # 每条非 A 链各拼一段, 整链固定; 按链出现顺序排列
    parts = [a_contig]
    for ch, t_start, t_end in targets:
        parts.append("%s%d-%d/0" % (ch, t_start, t_end))
    return " ".join(parts), "A"


def motif_out_positions(trb_path, design_chain="A"):
    """从 .trb 读取输出侧 motif 在 design_chain 上的残基号 (即 RIF 落点)。"""
    with open(trb_path, "rb") as fh:
        d = pickle.load(fh)
    hal = d.get("complex_con_hal_pdb_idx", [])
    return sorted(r for (c, r) in hal if c == design_chain)


# ============================================================
#  外部程序调用
# ============================================================
def run_rfdiffusion(input_pdb, out_prefix, contig, num_designs, gpu,
                    noise_scale=1.0, log_path=None,
                    write_trajectory=True):
    """跑 RFdiffusion motif scaffolding。输出 <out_prefix>_{0..n-1}.pdb / .trb。

    write_trajectory=False 时不写 traj/ 下的逐步轨迹 PDB (只省磁盘/IO, 不省算力;
    .trb 仍会照常写, 后续 MPNN 固定位点要用)。
    """
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        PY, RFDIFF_RUN,
        "inference.input_pdb=%s" % input_pdb,
        "inference.output_prefix=%s" % out_prefix,
        "contigmap.contigs=[%s]" % contig,
        "inference.num_designs=%d" % num_designs,
        "denoiser.noise_scale_ca=%g" % noise_scale,
        "denoiser.noise_scale_frame=%g" % noise_scale,
        "inference.write_trajectory=%s" % ("True" if write_trajectory else "False"),
    ]
    logf = open(log_path, "w") if log_path else subprocess.DEVNULL
    try:
        subprocess.run(cmd, cwd=RFDIFF_DIR, env=env, stdout=logf,
                       stderr=subprocess.STDOUT, check=True)
    finally:
        if log_path:
            logf.close()


def run_mpnn(pdb_dir, out_dir, fixed_positions, num_seqs, temp, gpu, seed=37,
             log_path=None):
    """
    对 pdb_dir 里的 PDB 跑 ProteinMPNN: 只设计 A 链, 其余所有非 A 链整链固定,
    A 链内 fixed_positions (1-based A 链残基号) 固定不设计。
    """
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.makedirs(out_dir, exist_ok=True)
    parsed = os.path.join(out_dir, "parsed.jsonl")
    assigned = os.path.join(out_dir, "assigned.jsonl")
    fixed = os.path.join(out_dir, "fixed.jsonl")
    logf = open(log_path, "w") if log_path else subprocess.DEVNULL

    def call(args):
        subprocess.run(args, env=env, stdout=logf,
                       stderr=subprocess.STDOUT, check=True)

    try:
        call([PY, os.path.join(MPNN_HELPER, "parse_multiple_chains.py"),
              "--input_path", pdb_dir, "--output_path", parsed])
        call([PY, os.path.join(MPNN_HELPER, "assign_fixed_chains.py"),
              "--input_path", parsed, "--output_path", assigned,
              "--chain_list", "A"])
        if fixed_positions:
            pos_str = " ".join(str(p) for p in fixed_positions)
            call([PY, os.path.join(MPNN_HELPER, "make_fixed_positions_dict.py"),
                  "--input_path", parsed, "--output_path", fixed,
                  "--chain_list", "A", "--position_list", pos_str])
        run_args = [
            PY, os.path.join(MPNN_DIR, "protein_mpnn_run.py"),
            "--jsonl_path", parsed,
            "--chain_id_jsonl", assigned,
            "--out_folder", out_dir,
            "--num_seq_per_target", str(num_seqs),
            "--sampling_temp", str(temp),
            "--seed", str(seed),
            "--batch_size", "1",
        ]
        if fixed_positions:
            run_args += ["--fixed_positions_jsonl", fixed]
        call(run_args)
    finally:
        if log_path:
            logf.close()


def parse_mpnn_fasta(fa_path):
    """
    解析 MPNN 输出 .fa。返回 list[dict]:
      {sample, score, global_score, seq_recovery, seq, is_original}
    第 1 条是输入骨架序列 (这里是 RIF+poly-G, 仅作参考), 其余为设计。
    """
    out = []
    header, seqbuf = None, []

    def flush():
        if header is None:
            return
        seq = "".join(seqbuf)
        # A 链是第一段 (按 '/' 切分多链)
        a_seq = seq.split("/")[0]
        rec = {"seq": a_seq, "header": header}
        for field in header.split(","):
            field = field.strip()
            if "=" in field:
                k, v = field.split("=", 1)
                rec[k.strip()] = v.strip()
        out.append(rec)

    with open(fa_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                header, seqbuf = line[1:], []
            else:
                seqbuf.append(line.strip())
        flush()
    return out


def thread_seq_to_backbone(backbone_pdb, a_seq, out_pdb):
    """
    把设计序列贴回骨架: A 链残基名改为新序列, 主链原子保留;
    B 链原样保留。原子序号重新连续编号。
    """
    one2three = {v: k for k, v in THREE2ONE.items() if len(k) == 3}
    # 优先用标准 20 种
    std = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
           "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
           "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
           "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}
    # 建 A 链残基号 -> 新氨基酸 (按出现顺序)
    _, a_resis = parse_chain_seq(backbone_pdb, "A")
    resi2aa = {}
    for i, resi in enumerate(a_resis):
        if i < len(a_seq):
            resi2aa[resi] = std.get(a_seq[i], "GLY")

    serial = 0
    out_lines = []
    with open(backbone_pdb) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            ch = line[21]
            if ch == "A":
                resi = int(line[22:26])
                atom = line[12:16].strip()
                if atom not in ("N", "CA", "C", "O"):
                    continue  # 骨架模型只留主链
                new_resn = resi2aa.get(resi, line[17:20].strip())
                serial += 1
                nl = (line[:6] + "%5d" % serial + line[11:17]
                      + "%3s" % new_resn + line[20:])
                out_lines.append(nl)
            else:
                serial += 1
                nl = line[:6] + "%5d" % serial + line[11:]
                out_lines.append(nl)
    with open(out_pdb, "w") as fh:
        fh.writelines(out_lines)
        fh.write("TER\nEND\n")


# ============================================================
#  主流程
# ============================================================
CSV_HEADER = [
    "source_pdb",          # 原始输入 PDB 的绝对路径
    "pdb_stem",            # PDB 文件名(去扩展名)
    "design_index",        # RFdiffusion 第几个骨架 (0-based)
    "mpnn_sample",         # MPNN 第几条采样序列
    "seq_id",              # 全局唯一标识: <stem>_d<design>_s<sample>
    "design_chain",        # 设计链 (A)
    "a_chain_range",       # A 链残基号范围 (输入)
    "a_length",            # A 链长度
    "rif_residues",        # RIF 关键残基号 (输入侧, 逗号分隔)
    "rif_residues_out",    # RIF 关键残基号 (输出/MPNN 侧)
    "fixed_positions",     # MPNN 实际固定的 A 链位点
    "original_seq",        # 原始 A 链序列 (输入 PDB)
    "designed_seq",        # MPNN 设计的 A 链序列
    "seq_length",          # 设计序列长度
    "num_mutations",       # 相对原始序列的突变数
    "pct_identity",        # 与原始序列一致性 (%)
    "mutations",           # 突变列表 (如 S1A;G5V)
    "mpnn_score",          # MPNN score (越低越好, neg log prob)
    "mpnn_global_score",   # MPNN global_score
    "mpnn_seq_recovery",   # MPNN seq_recovery (相对骨架参考序列)
    "mol_weight_Da",       # 分子量
    "net_charge_pH7",      # pH7 净电荷
    "isoelectric_point",   # 等电点 pI
    "gravy",               # 平均疏水性
    "aromatic_fraction",   # 芳香族比例
    "rfdiff_backbone_pdb", # 重扩散骨架 PDB 路径
    "threaded_pdb",        # thread 好的完整复合物 PDB 路径
]


def load_records_from_csv(csv_path):
    """从已完成 PDB 的 sequences.csv 回收记录 list[OrderedDict], 用于续跑时重建总表。"""
    recs = []
    if not os.path.exists(csv_path):
        return recs
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            recs.append(OrderedDict((k, row.get(k, "")) for k in CSV_HEADER))
    return recs


def process_one_pdb(src_pdb, out_root, args, gpu):
    """处理单个 PDB, 返回该 PDB 的所有序列记录 list[dict]。

    断点续跑: 每个 PDB 处理完在其输出目录写 .done 标记。若标记已存在且未开 --force,
    直接从该 PDB 的 sequences.csv 回收记录返回, 跳过重算。
    """
    stem = os.path.splitext(os.path.basename(src_pdb))[0]
    # 用相对路径的目录前缀避免同名 stem 冲突
    rel = os.path.relpath(os.path.dirname(src_pdb), args.input_dir)
    sub = stem if rel in (".", "") else os.path.join(rel.replace("/", "_"), stem)
    # 各 PDB 的任务目录统一归拢到 out_root/per_pdb/ 下,
    # 避免骨架多时一堆 <stem>/ 目录与 inputs/、summary.csv 散落在 out_root 根部。
    pdb_out = os.path.join(out_root, PER_PDB_SUBDIR, sub)
    done_marker = os.path.join(pdb_out, ".done")
    seq_csv = os.path.join(pdb_out, "sequences.csv")

    # --- 续跑: 已完成则跳过, 回收已有记录 ---
    if os.path.exists(done_marker) and not args.force:
        recs = load_records_from_csv(seq_csv)
        print("  [续跑] 已完成, 跳过: %s (%d 条记录)" % (stem, len(recs)))
        return recs
    # --force 或未完成: 清掉旧的半成品目录重算, 保证干净
    if os.path.isdir(pdb_out):
        shutil.rmtree(pdb_out)

    rf_dir = os.path.join(pdb_out, "rfdiffusion")
    mpnn_dir = os.path.join(pdb_out, "mpnn")
    designs_dir = os.path.join(pdb_out, "designs")
    for d in (rf_dir, mpnn_dir):
        os.makedirs(d, exist_ok=True)
    if args.thread_pdb:
        os.makedirs(designs_dir, exist_ok=True)
    shutil.copy(src_pdb, os.path.join(pdb_out, "input.pdb"))

    # --- 解析输入 ---
    rif = parse_rif_residues(src_pdb)
    bounds = chain_bounds(src_pdb)
    if "A" not in bounds:
        print("  [跳过] 无 A 链: %s" % src_pdb)
        return []
    a_start, a_end, a_len = bounds["A"]
    orig_a_seq, _ = parse_chain_seq(src_pdb, "A")
    # target 链: 除 A 外的所有链, 全部按原坐标固定
    targets = [(c, bounds[c][0], bounds[c][1]) for c in bounds if c != "A"]
    if not targets:
        print("  [跳过] 无 target 链: %s" % src_pdb)
        return []

    contig, dchain = build_contig(rif, a_start, a_len, targets)
    target_desc = " ".join("%s%d-%d" % (c, s, e) for c, s, e in targets)
    print("  RIF=%s  A=%d-%d  targets=%s" %
          (rif, a_start, a_end, target_desc))
    print("  contig: %s" % contig)

    # --- 1. RFdiffusion 重扩散 ---
    rf_prefix = os.path.join(rf_dir, "design")
    run_rfdiffusion(
        os.path.abspath(os.path.join(pdb_out, "input.pdb")), rf_prefix, contig,
        args.num_designs, gpu,
        noise_scale=args.noise_scale,
        log_path=os.path.join(rf_dir, "rfdiffusion.log"),
        write_trajectory=args.write_trajectory)

    records = []
    for di in range(args.num_designs):
        bb_pdb = "%s_%d.pdb" % (rf_prefix, di)
        trb = "%s_%d.trb" % (rf_prefix, di)
        if not os.path.exists(bb_pdb):
            print("  [警告] 缺骨架: %s" % bb_pdb)
            continue
        rif_out = motif_out_positions(trb, dchain) if os.path.exists(trb) else rif

        # --- 2. ProteinMPNN 设计该骨架 ---
        d_mpnn = os.path.join(mpnn_dir, "design_%d" % di)
        in_for_mpnn = os.path.join(d_mpnn, "in")
        os.makedirs(in_for_mpnn, exist_ok=True)
        shutil.copy(bb_pdb, os.path.join(in_for_mpnn, "bb.pdb"))
        run_mpnn(in_for_mpnn, d_mpnn, rif_out, args.num_seqs, args.temp, gpu,
                 seed=args.seed, log_path=os.path.join(d_mpnn, "mpnn.log"))

        fa = os.path.join(d_mpnn, "seqs", "bb.fa")
        if not os.path.exists(fa):
            print("  [警告] 缺 MPNN 输出: %s" % fa)
            continue
        entries = parse_mpnn_fasta(fa)
        # 第 1 条为骨架参考序列, 跳过; 其余为设计
        designs = entries[1:] if len(entries) > 1 else entries
        for si, rec in enumerate(designs):
            dseq = rec["seq"]
            seq_id = "%s_d%d_s%d" % (stem, di, si + 1)
            muts, nmut = mutation_list(orig_a_seq, dseq, offset=a_start)
            if args.thread_pdb:
                threaded = os.path.join(designs_dir, "%s.pdb" % seq_id)
                thread_seq_to_backbone(bb_pdb, dseq, threaded)
                threaded_path = os.path.abspath(threaded)
            else:
                threaded_path = ""   # 不回贴复合物时留空
            records.append(OrderedDict([
                ("source_pdb", os.path.abspath(src_pdb)),
                ("pdb_stem", stem),
                ("design_index", di),
                ("mpnn_sample", si + 1),
                ("seq_id", seq_id),
                ("design_chain", dchain),
                ("a_chain_range", "%d-%d" % (a_start, a_end)),
                ("a_length", a_len),
                ("rif_residues", ",".join(map(str, rif))),
                ("rif_residues_out", ",".join(map(str, rif_out))),
                ("fixed_positions", ",".join(map(str, rif_out))),
                ("original_seq", orig_a_seq),
                ("designed_seq", dseq),
                ("seq_length", len(dseq)),
                ("num_mutations", nmut),
                ("pct_identity", seq_identity(orig_a_seq, dseq)),
                ("mutations", ";".join(muts)),
                ("mpnn_score", rec.get("score", "")),
                ("mpnn_global_score", rec.get("global_score", "")),
                ("mpnn_seq_recovery", rec.get("seq_recovery", "")),
                ("mol_weight_Da", seq_mw(dseq)),
                ("net_charge_pH7", round(net_charge(dseq, 7.0), 2)),
                ("isoelectric_point", isoelectric_point(dseq)),
                ("gravy", gravy(dseq)),
                ("aromatic_fraction", aromatic_frac(dseq)),
                ("rfdiff_backbone_pdb", os.path.abspath(bb_pdb)),
                ("threaded_pdb", threaded_path),
            ]))

    # 该 PDB 的逐序列 CSV
    if records:
        with open(seq_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_HEADER)
            w.writeheader()
            w.writerows(records)
    # 写完成标记(续跑时据此跳过)。记 0 条也标完成, 避免下次重算空结果的 PDB。
    with open(done_marker, "w") as fh:
        fh.write("%d\n" % len(records))
    print("  -> %d 条设计序列" % len(records))
    return records


# ============================================================
#  多 GPU 并行调度
# ============================================================
def _gpu_worker(gpu, task_q, result_q, out_root, args):
    """单卡 worker: 从任务队列里不断领取 PDB, 在指定 gpu 上处理, 结果回传。

    收到 None 结束哨兵才退出 -> 自动负载均衡
    (谁先跑完谁领下一个, 不按固定均分)。

    不能使用 get_nowait(): multiprocessing.Queue 由后台 feeder 线程异步
    投递数据，worker 可能在任务尚未全部 flush 时短暂看到空队列，
    误以为全部完成而提前退出。
    回传 (idx, recs, ok, pdb): ok=False 表示该 PDB 处理时抛异常(失败)。
    """
    while True:
        task = task_q.get()
        if task is None:
            break
        idx, pdb, total = task
        print("\n[GPU %s][%d/%d] %s" % (gpu, idx, total, pdb))
        ok = True
        try:
            recs = process_one_pdb(pdb, out_root, args, gpu)
        except subprocess.CalledProcessError as e:
            print("  [错误] 外部程序失败 (GPU %s): %s" % (gpu, e))
            recs = []
            ok = False
        except Exception as e:
            print("  [错误] (GPU %s) %s" % (gpu, e))
            recs = []
            ok = False
        result_q.put((idx, recs, ok, pdb))


def run_parallel(pdbs, gpus, out_root, args, summary_csv):
    """把 pdbs 分发到多张卡上并行处理, 增量写 summary.csv。
    返回 (全部记录, 失败的 PDB 列表)。"""
    ctx = mp.get_context("spawn")  # 避免 CUDA 在 fork 子进程里的状态问题
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    total = len(pdbs)
    for i, pdb in enumerate(pdbs, 1):
        task_q.put((i, pdb, total))
    # 同一个 producer 依次写入时 Queue 保证 FIFO：所有真实任务
    # 都位于哨兵之前。每个 worker 最终取到一个 None 后安全退出。
    for _ in gpus:
        task_q.put(None)

    workers = [
        ctx.Process(target=_gpu_worker,
                    args=(g, task_q, result_q, out_root, args))
        for g in gpus
    ]
    for w in workers:
        w.start()

    # 边收边写: 按 PDB 完成顺序增量落盘, 防中途崩溃丢结果
    collected = {}
    failed = []
    done = 0
    all_records = []
    while done < total:
        try:
            idx, recs, ok, pdb = result_q.get(timeout=5)
        except queue.Empty:
            # 正常计算期间 5 秒内没有新结果很常见。但如果 worker
            # 被 OOM/SIGKILL/段错误等异常终止，必须立即报错，否则
            # 主进程会永久卡在等待一个永远不会回传的任务。
            crashed = [
                (w.pid, w.exitcode)
                for w in workers
                if w.exitcode not in (None, 0)
            ]
            if crashed:
                for w in workers:
                    if w.is_alive():
                        w.terminate()
                for w in workers:
                    w.join()
                raise RuntimeError(
                    "GPU worker 异常退出: %s; 已停止其余 worker, "
                    "保留已完成 PDB 的 .done 标记供续跑" % crashed
                )
            continue
        collected[idx] = recs
        if not ok:
            failed.append(pdb)
        done += 1
        # 按原始 PDB 顺序拼接已完成的部分 (保证 summary 行序稳定)
        all_records = []
        for i in range(1, total + 1):
            if i in collected:
                all_records.extend(collected[i])
        with open(summary_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_HEADER)
            w.writeheader()
            w.writerows(all_records)

    for w in workers:
        w.join()
    return all_records, failed


def main():
    ap = argparse.ArgumentParser(
        description="批量 A 链重扩散(RFdiffusion motif scaffolding)+ 序列重设计(ProteinMPNN)")
    ap.add_argument("input_dir", help="输入根文件夹 (递归找 *.pdb)")
    ap.add_argument("-o", "--out", required=True, help="输出根文件夹")
    ap.add_argument("--num-designs", type=int, default=2,
                    dest="num_designs", help="每个 PDB 的重扩散骨架数 (默认 2)")
    ap.add_argument("--num-seqs", type=int, default=2,
                    dest="num_seqs", help="每个骨架的 MPNN 序列数 (默认 2)")
    ap.add_argument("--temp", default="0.2", help="MPNN 采样温度 (默认 0.2)")
    ap.add_argument("--noise-scale", type=float, default=2.0, dest="noise_scale",
                    help="RFdiffusion 噪声尺度 (默认 2)")
    ap.add_argument("--gpu", default="0",
                    help="CUDA 设备号; 逗号分隔可指定多卡并行, 如 0,1,2,3 (默认 0)。"
                         "开 --auto-gpu 时本项被忽略")
    ap.add_argument("--auto-gpu", action="store_true", dest="auto_gpu",
                    help="自动检测服务器显卡, 按空闲显存挑选可用卡(每卡一路本流程任务, "
                         "可与别人任务共享), 覆盖 --gpu")
    ap.add_argument("--task-vram", type=float, default=8000.0, dest="task_vram",
                    help="单任务估算显存(MiB), 供 --auto-gpu 判定能否塞进某卡(默认 8000)")
    ap.add_argument("--gpu-reserve", type=float, default=0.10, dest="gpu_reserve",
                    help="--auto-gpu 给别人占用峰值留的冗余比例(默认 0.10)")
    ap.add_argument("--gpu-sample", type=float, default=5.0, dest="gpu_sample",
                    help="--auto-gpu 显存采样窗口秒数(默认 5)")
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理前 N 个 PDB (0=全部, 用于冒烟测试)")
    ap.add_argument("--force", action="store_true",
                    help="忽略 .done 标记, 强制重算所有 PDB")
    ap.add_argument("--seed", type=int, default=37, help="MPNN 随机种子")
    ap.add_argument("--no-trajectory", dest="write_trajectory",
                    action="store_false",
                    help="不写 RFdiffusion 逐步轨迹 PDB (省磁盘/IO; .trb 仍保留)")
    ap.add_argument("--no-thread", dest="thread_pdb", action="store_false",
                    help="不回贴生成完整复合物 PDB (designs/), 只留 MPNN 序列与属性")
    ap.set_defaults(write_trajectory=True, thread_pdb=True)
    args = ap.parse_args()

    args.input_dir = os.path.abspath(args.input_dir)
    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)

    pdbs = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.pdb"),
                            recursive=True))
    if not pdbs:
        print("未找到任何 PDB: %s" % args.input_dir)
        sys.exit(1)
    if args.limit and args.limit > 0:
        pdbs = pdbs[:args.limit]

    # ---- 确定 GPU 列表 ----
    if args.auto_gpu:
        # 自动检测: 按空闲显存挑可用卡(每卡一路, 可与别人共享)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from gpu_scheduler import plan_slots
        gpus = plan_slots(task_vram_mib=args.task_vram, reserve=args.gpu_reserve,
                          sample_seconds=args.gpu_sample, max_per_gpu=1,
                          verbose=True)
        if not gpus:
            print("[auto-gpu] 没有显存放得下的空闲卡, 退回 GPU 0 串行(可能与别人抢显存)")
            gpus = ["0"]
    else:
        # 手动 GPU 列表 (逗号分隔, 去空白去重保序)
        gpus = []
        for tok in str(args.gpu).replace(",", " ").split():
            if tok not in gpus:
                gpus.append(tok)
        if not gpus:
            gpus = ["0"]

    n_par = min(len(gpus), len(pdbs))   # 卡比 PDB 多时, 多余的卡用不上
    print("=" * 60)
    print("输入根目录 : %s" % args.input_dir)
    print("输出根目录 : %s" % out_root)
    print("PDB 数量   : %d%s" %
          (len(pdbs), "" if not args.limit else " (--limit %d)" % args.limit))
    print("每PDB骨架数: %d   每骨架序列数: %d" %
          (args.num_designs, args.num_seqs))
    if args.force:
        print("模式       : --force 强制重算(忽略 .done)")
    if n_par > 1:
        print("GPU        : %s  (多卡并行, %d 路)" % (",".join(gpus), n_par))
    else:
        print("GPU        : %s  (单卡串行)" % gpus[0])
    print("=" * 60)

    summary_csv = os.path.join(out_root, "summary.csv")

    if n_par > 1:
        # ---- 多卡并行 ----
        all_records, failed = run_parallel(pdbs, gpus[:n_par], out_root, args, summary_csv)
    else:
        # ---- 单卡串行 (原逻辑) ----
        all_records = []
        failed = []
        gpu = gpus[0]
        for i, pdb in enumerate(pdbs, 1):
            print("\n[%d/%d] %s" % (i, len(pdbs), pdb))
            try:
                recs = process_one_pdb(pdb, out_root, args, gpu)
                all_records.extend(recs)
            except subprocess.CalledProcessError as e:
                print("  [错误] 外部程序失败: %s" % e)
                failed.append(pdb)
            except Exception as e:
                print("  [错误] %s" % e)
                failed.append(pdb)
            # 每个 PDB 处理完即增量写总表, 防中途崩溃丢结果
            with open(summary_csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=CSV_HEADER)
                w.writeheader()
                w.writerows(all_records)

    print("\n" + "=" * 60)
    print("完成。共 %d 条设计序列。" % len(all_records))
    print("总汇总表: %s" % summary_csv)
    if failed:
        print("⚠️  失败 %d/%d 个 PDB:" % (len(failed), len(pdbs)))
        for p in failed:
            print("   - %s" % p)
    print("=" * 60)

    # 退出码语义(供上层框架判定成败):
    #   0 = 全部成功;1 = 部分 PDB 失败(仍有产出);2 = 全部 PDB 失败(无产出)
    # 之前无论成败都返回 0,导致「全失败」被上层误判为成功并写完成标记。
    if failed and len(failed) == len(pdbs):
        print("‼️  所有 PDB 均失败,以退出码 2 退出。")
        sys.exit(2)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
