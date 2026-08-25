#!/usr/bin/env python3
# =============================================================================
# filter_for_af3.py
#   读取 redesign pipeline 的 summary.csv, 按理化/MPNN 指标初筛, 再在
#   同一 (scaffold, A链长度) 组内按 mpnn_score 升序保留前 50%,
#   为每条入选设计生成一个可直接用于 AlphaFold3 的复合物 JSON。
#
#   复合物 = A 链 (MPNN 设计序列 designed_seq) + 其余所有链 (target, 从 input.pdb
#           提取的 A 链之外的全部链, 如 B/C/D...)。
#
# 用法:
#   python filter_for_af3.py -f <summary.csv> -o <输出文件夹> [选项]
#
# 硬过滤规则 (不满足即剔除):
#   mpnn_score        <= 2.0
#   (gravy / aromatic_fraction / pI / net_charge 四项已停用, 不再参与过滤)
# 之后: 每个 (scaffold, len) 组内按 mpnn_score 升序, 保留前 floor(30%) (向下取整, 组可清零)。
# 每条蛋白链关闭 MSA (unpairedMsa/pairedMsa 空串, templates 空) -> AF3 单序列模式。
# =============================================================================
import argparse
import csv
import json
import math
import os
import sys

# 硬过滤阈值 (集中放此处, 方便调整)
# 现仅按 mpnn_score 单项过滤; gravy/aromatic/pI/net_charge 四项已按需求停用
# (对 graft 短肽路线不适用)。相关常量保留仅为兼容 import, 不再参与判定。
MPNN_SCORE_MAX = 2.0      # 要求 mpnn_score <= 2.0
GRAVY_MAX = 0.0           # (停用) 曾要求 gravy < 0
AROMATIC_MAX = 0.15       # (停用) 曾要求 aromatic_fraction <= 0.15
PI_LOW, PI_HIGH = 6.5, 7.5  # (停用) 曾剔除 pI 落在该闭区间内的
NET_CHARGE_ABS_MIN = 1.0  # (停用) 曾要求 |net_charge| >= 1
KEEP_FRACTION = 0.3       # 组内保留比例

THREE2ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


def chain_seq(pdb_path, chain):
    """从 PDB 提取指定链的单字母序列 (按 CA 原子去重, 保持出现顺序)。"""
    seq, seen = [], set()
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            if line[12:16].strip() != 'CA':
                continue
            if line[21] != chain:
                continue
            resnum = line[22:27]  # resSeq + iCode
            if resnum in seen:
                continue
            seen.add(resnum)
            seq.append(THREE2ONE.get(line[17:20].strip(), 'X'))
    return ''.join(seq)


def all_chains_seq(pdb_path, exclude=('A',)):
    """提取 PDB 中所有链的单字母序列 (按 CA 去重), 跳过 exclude 中的链。
    返回 [(chain_id, seq), ...], 链顺序按其在文件中首次出现的次序。"""
    order, seqs, seen = [], {}, {}
    exclude = set(exclude)
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            if line[12:16].strip() != 'CA':
                continue
            ch = line[21]
            if ch in exclude:
                continue
            resnum = line[22:27]  # resSeq + iCode
            if ch not in seqs:
                order.append(ch)
                seqs[ch] = []
                seen[ch] = set()
            if resnum in seen[ch]:
                continue
            seen[ch].add(resnum)
            seqs[ch].append(THREE2ONE.get(line[17:20].strip(), 'X'))
    return [(ch, ''.join(seqs[ch])) for ch in order]


def target_seq_for_row(row):
    """取该设计对应的所有 target 链 (A 链之外的全部链)。
    优先用 threaded_pdb 所在 scaffold 的 input.pdb, 回退到 source_pdb。
    返回 ([(chain_id, seq), ...], pdb_path)。"""
    candidates = []
    threaded = row.get('threaded_pdb', '').strip()
    if threaded:
        scaffold_dir = os.path.dirname(os.path.dirname(threaded))
        candidates.append(os.path.join(scaffold_dir, 'input.pdb'))
    src = row.get('source_pdb', '').strip()
    if src:
        candidates.append(src)
    for pdb in candidates:
        if pdb and os.path.isfile(pdb):
            chains = all_chains_seq(pdb, exclude=('A',))
            if chains:
                return chains, pdb
    return None, None


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def passes_hard_filters(row, reasons):
    """硬过滤: 仅按 mpnn_score 单项判定 (其余理化指标已按需求停用)。
    失败原因写入 reasons 列表。返回 bool。"""
    ms = to_float(row.get('mpnn_score'))
    if ms is None or ms > MPNN_SCORE_MAX:
        reasons.append(f'mpnn_score={ms}>{MPNN_SCORE_MAX}')
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description='按指标过滤并生成 AF3 复合物 JSON')
    ap.add_argument('-f', '--summary', required=True, help='summary.csv 路径')
    ap.add_argument('-o', '--out', required=True, help='输出文件夹 (需不存在)')
    ap.add_argument('--seed', type=int, default=1, help='AF3 modelSeeds (默认 1)')
    ap.add_argument('--keep-fraction', type=float, default=KEEP_FRACTION,
                    help='组内保留比例 (默认 0.3)')
    args = ap.parse_args()

    if not os.path.isfile(args.summary):
        sys.exit(f'错误: 找不到 summary.csv: {args.summary}')
    if os.path.exists(args.out):
        sys.exit(f'错误: 输出文件夹已存在, 请换路径或先删除: {args.out}')

    with open(args.summary, newline='') as fh:
        rows = list(csv.DictReader(fh))
    total = len(rows)

    # --- 第 1 步: 硬过滤 ---
    kept, dropped = [], []
    for r in rows:
        reasons = []
        if passes_hard_filters(r, reasons):
            kept.append(r)
        else:
            dropped.append((r, reasons))

    # --- 第 2 步: 同 (scaffold, A链长度) 组内按 mpnn_score 升序留前 50% ---
    groups = {}
    for r in kept:
        key = (r.get('pdb_stem', ''), r.get('seq_length', ''))
        groups.setdefault(key, []).append(r)

    final = []
    for key, members in groups.items():
        members.sort(key=lambda x: to_float(x.get('mpnn_score')) or float('inf'))
        # 向下取整, 允许组清零 (floor(0.3*1)=0 时该组不保留)
        n_keep = math.floor(len(members) * args.keep_fraction)
        final.extend(members[:n_keep])

    # --- 第 3 步: 写 AF3 JSON ---
    os.makedirs(args.out, exist_ok=True)
    written, no_target = 0, []
    manifest = []
    for r in final:
        a_seq = r.get('designed_seq', '').strip()
        seq_id = r.get('seq_id', '').strip() or f'design_{written}'
        b_chains, b_src = target_seq_for_row(r)
        if not a_seq or not b_chains:
            no_target.append(seq_id)
            continue
        # A 链 = MPNN 设计序列; 其余链 = input.pdb 原样提取的 target 链
        sequences = [
            # 关闭 MSA: unpairedMsa/pairedMsa 置空串, templates 置空 -> 单序列模式
            {'protein': {'id': 'A', 'sequence': a_seq,
                         'unpairedMsa': '', 'pairedMsa': '', 'templates': []}},
        ]
        for ch_id, ch_seq in b_chains:
            sequences.append(
                {'protein': {'id': ch_id, 'sequence': ch_seq,
                             'unpairedMsa': '', 'pairedMsa': '', 'templates': []}})
        job = {
            'name': seq_id,
            'modelSeeds': [args.seed],
            'sequences': sequences,
            'dialect': 'alphafold3',
            'version': 1,
        }
        with open(os.path.join(args.out, f'{seq_id}.json'), 'w') as fh:
            json.dump(job, fh, indent=2)
        written += 1
        manifest.append({
            'seq_id': seq_id,
            'scaffold': r.get('pdb_stem', ''),
            'a_length': r.get('seq_length', ''),
            'target_chains': ''.join(ch for ch, _ in b_chains),
            'mpnn_score': r.get('mpnn_score', ''),
            'mpnn_global_score': r.get('mpnn_global_score', ''),
            'gravy': r.get('gravy', ''),
            'net_charge_pH7': r.get('net_charge_pH7', ''),
            'isoelectric_point': r.get('isoelectric_point', ''),
            'aromatic_fraction': r.get('aromatic_fraction', ''),
            'target_src': os.path.basename(b_src) if b_src else '',
        })

    # 写一个清单, 方便核对
    if manifest:
        with open(os.path.join(args.out, '_selected.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
            w.writeheader()
            w.writerows(manifest)

    # --- 报告 ---
    print('=' * 60)
    print(f'总设计数            : {total}')
    print(f'硬过滤通过          : {len(kept)}  (剔除 {len(dropped)})')
    print(f'组内留前 {int(args.keep_fraction*100)}% 后    : {len(final)}')
    print(f'成功写出 AF3 JSON   : {written}  -> {args.out}')
    if no_target:
        print(f'⚠️ 缺 target/序列跳过: {len(no_target)} ({", ".join(no_target)})')
    print('=' * 60)
    print('被剔除明细 (前 15 条):')
    for r, reasons in dropped[:15]:
        print(f'  {r.get("seq_id","?"):40s} | {"; ".join(reasons)}')


if __name__ == '__main__':
    main()
