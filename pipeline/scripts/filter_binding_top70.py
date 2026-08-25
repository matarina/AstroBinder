#!/usr/bin/env python3
"""
筛选 relax_binding.py 的结果。

用法:
    python filter_binding_top70.py <结果文件夹> [输出文件夹] [--keep 0.70]

逻辑:
    1. 进入指定文件夹, 读取 binding_summary.csv
    2. 对四个维度分别计算“好坏分数”(已统一为越大越好):
         - dG_separated                : 结合自由能 (越负越好)
         - dG_separated/dSASAx100       : 单位埋藏面积结合能 (越负越好)
         - 几何/堆积质量                : packstat + sc_value (越大越好, z-score 合成)
         - 氢键/极性满足                : hbonds_int - delta_unsatHbonds (z-score 合成)
    3. 每个维度保留“前 70%”(剔除最差 30%), 四个维度同时通过才算合格。
       如果设置 --max-pdbs 且合格数超过上限，则每次将保留比例降低
       --keep-step(默认 5 个百分点)，直到合格数不超过上限。
    4. 对每个合格 pdb, 从候选池(界面上能量落在 [e_full, 0) 内、即有结合贡献的 A 链
       残基)选出 rif_residues: 累积式随机采样(每次命中并入集合, 达到 min_rif 即停,
       最多 max_resample 次), 采满仍不足 min_rif(默认 3)则淘汰该结构。
       候选池本身不足 min_rif 时同样淘汰(不复制、不进下一轮)。
    5. 把保留下来的 pdb 复制到输出文件夹, 首行写 rif_residues 标注。
"""

import sys
import os
import csv
import math
import random
import shutil
import argparse

# ---- 玻尔兹曼边界采样参数 ----
INTERFACE_CSV = "interface_residues.csv"  # 界面残基能量表
SAMPLE_CHAIN = "A"                        # 只对 A 链残基采样
# 直接用残基 ddG_contrib 作为能量: 越负结合贡献越强
# E_ZERO 固定为 0, 不对外暴露: 归一化公式依赖 e_zero=0 才严格成立(E=0 时 w=0)
E_ZERO = 0.0    # 能量上界(固定), ddG_contrib >= 0 的残基直接舍弃 (w=0)
E_FULL = -5.0   # 能量下界, ddG_contrib <= 此值的残基必定入选 (w=1)
KT = 1.0        # 玻尔兹曼温度参数, 越小越偏好低能量(强结合)残基

# ---- 维度定义: (维度名, [(列名, 是否越大越好), ...]) ----
# 对“越小越好”的列, 取负后再参与排序/合成, 统一成“越大越好”
DIMENSIONS = [
    ("dG_separated",            [("dG_separated", False)]),
    ("dG_separated/dSASAx100",  [("dG_separated/dSASAx100", False)]),
    ("几何/堆积质量",            [("packstat", True), ("sc_value", True)]),
    ("氢键/极性满足",            [("hbonds_int", True), ("delta_unsatHbonds", False)]),
]


def percentile(sorted_vals, q):
    """numpy 线性插值法计算分位数, sorted_vals 必须已升序排列, q 在 [0,1]。"""
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("空数据无法计算分位数")
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def zscores(values):
    """返回每个值的 z-score; 若标准差为 0, 全部返回 0(该列不产生区分度)。"""
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = var ** 0.5
    if std == 0:
        return [0.0] * n
    return [(v - mean) / std for v in values]


def select_by_keep(rows, dim_scores, keep):
    """按 keep 计算四维度阈值并返回筛选结果。

    返回 (thresholds, pass_count_per_dim, selected_rows)。抽成独立函数后，
    自适应逻辑可以只重算廉价的分位数，不会重复读 CSV 或复制 PDB。
    """
    n = len(rows)
    drop_q = 1.0 - keep
    thresholds = []
    for d_idx in range(len(DIMENSIONS)):
        col_scores = sorted(dim_scores[i][d_idx] for i in range(n))
        thresholds.append(percentile(col_scores, drop_q))

    pass_count_per_dim = [0] * len(DIMENSIONS)
    selected = []
    for i, row in enumerate(rows):
        ok_flags = [
            dim_scores[i][d] >= thresholds[d]
            for d in range(len(DIMENSIONS))
        ]
        for d, ok in enumerate(ok_flags):
            if ok:
                pass_count_per_dim[d] += 1
        if all(ok_flags):
            selected.append(row)
    return thresholds, pass_count_per_dim, selected


def clear_previous_outputs(output_dir):
    """删除本脚本上次写入的 PDB/清单，避免收紧筛选后旧文件残留。"""
    os.makedirs(output_dir, exist_ok=True)
    removed = 0
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isfile(path) and (name.endswith(".pdb") or name == "selected_summary.csv"):
            os.remove(path)
            removed += 1
    return removed


def load_interface_residues(csv_path):
    """读取 interface_residues.csv, 返回:
       - by_pdb: {pdb相对路径: [(resnum:int, ddG_contrib:float), ...]} 仅含 SAMPLE_CHAIN 链

    ddG_contrib 直接作为残基能量参与边界采样(越负结合贡献越强), 不再计算全局阈值。
    """
    by_pdb = {}
    n_res = 0
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["chain"] != SAMPLE_CHAIN:
                continue
            try:
                resnum = int(r["resnum"])
                ddg = float(r["ddG_contrib"])
            except (ValueError, KeyError):
                continue
            by_pdb.setdefault(r["pdb"], []).append((resnum, ddg))
            n_res += 1

    if n_res == 0:
        raise ValueError(f"{csv_path} 中没有链 {SAMPLE_CHAIN} 的残基记录")

    return by_pdb


def residue_weight(energy, e_full=E_FULL, kT=KT):
    """玻尔兹曼边界归一化采样权重 w in [0,1]。

    直接以残基 ddG_contrib 作为能量 (越负结合越强), 能量上界固定为 E_ZERO=0:
      - energy >= 0      -> 0  (舍弃, 无结合贡献)
      - energy <= e_full -> 1  (必定入选, 强结合)
      - e_full < energy < 0 -> (exp(-E/kT) - 1) / (exp(-e_full/kT) - 1)
        玻尔兹曼因子在 [e_full, 0] 区间内归一化(上界固定 0 时严格成立),
        E=0 时 w=0, E=e_full 时 w=1, 中间为指数(凸)曲线; kT 越小越保守。
    """
    if energy >= E_ZERO:
        return 0.0
    if energy <= e_full:
        return 1.0
    return (math.exp(-energy / kT) - 1.0) / (math.exp(-e_full / kT) - 1.0)


def sample_residues(residues, rng, e_full=E_FULL, kT=KT):
    """对一个 pdb 的 A 链残基做逐残基独立伯努利采样, 返回命中的 resnum 升序列表。

    residues: [(resnum, ddG_contrib), ...]
    """
    hit = []
    for resnum, ddg in residues:
        w = residue_weight(ddg, e_full=e_full, kT=kT)
        if w <= 0.0:
            continue
        if w >= 1.0 or rng.random() < w:
            hit.append(resnum)
    return sorted(set(hit))


def count_positive_weight(residues, e_full=E_FULL, kT=KT):
    """界面上权重 w>0(即 ddG_contrib<0, 有结合贡献)的残基数——理论上能采到的 rif 上限。"""
    return sum(1 for _, ddg in residues
               if residue_weight(ddg, e_full=e_full, kT=kT) > 0.0)


def select_rif_residues(residues, rng, min_rif, max_resample, e_full=E_FULL, kT=KT):
    """选出该结构的 rif 残基, 保证 >= min_rif 个(凑不够则淘汰)。

    候选池 = 能量落在 [e_full, e_zero=0) 区间内(即 ddG_contrib < 0, 有结合贡献)
    的残基。参考 rifgen/rifdock: 只要残基能量在该区间内即为合格候选, 不必是最强的。

    步骤(累积式随机采样, 达到即停):
      循环最多 max_resample 次, 每次按玻尔兹曼权重做一轮伯努利采样, 命中的残基
      并入集合; 集合一旦达到 min_rif 立即停止并返回。

    返回:
      - resnum 升序列表(len >= min_rif) —— 成功;
      - None —— 候选池本身 < min_rif, 或采满 max_resample 次仍不足 min_rif,
        直接淘汰该结构(不按 ddG_contrib 最负补齐)。
    """
    # 候选池: 能量在 [e_full, e_zero) 内的残基 (ddG_contrib < 0 即有结合贡献)
    candidates = [(rn, ddg) for rn, ddg in residues if ddg < E_ZERO]
    if len(candidates) < min_rif:
        return None
    # 累积式随机采样: 逐轮并入命中残基, 达到 min_rif 即停
    hit = set()
    for _ in range(max_resample):
        hit.update(sample_residues(residues, rng, e_full=e_full, kT=kT))
        if len(hit) >= min_rif:
            return sorted(hit)
    # 采满最大次数仍不足 min_rif -> 淘汰, 不补齐
    return None


def prepend_rif_line(pdb_path, resnums):
    """在 pdb 文件首行写入 `rif_residues 33,42,44,46` 形式的注释行。"""
    header = "rif_residues " + ",".join(str(n) for n in resnums) + "\n"
    with open(pdb_path, "r") as f:
        body = f.read()
    with open(pdb_path, "w") as f:
        f.write(header)
        f.write(body)


def main():
    parser = argparse.ArgumentParser(description="按四个维度的前70%筛选 binding 结果并复制 pdb")
    parser.add_argument("result_dir", help="relax_binding.py 的结果文件夹")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="输出文件夹 (默认: <结果文件夹>/filtered_top70)")
    parser.add_argument("--keep", type=float, default=0.70,
                        help="每个维度保留的比例 (默认 0.70, 即前70%%)")
    parser.add_argument("--e-full", type=float, default=E_FULL,
                        help=f"能量下界, ddG_contrib <= 此值的残基必定入选 (默认 {E_FULL}; "
                             f"能量上界固定为 {E_ZERO}, 不可改)")
    parser.add_argument("--kT", type=float, default=KT,
                        help=f"玻尔兹曼温度参数, 越小越偏好强结合残基 (默认 {KT})")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子, 用于复现采样结果 (默认随机)")
    parser.add_argument("--min-rif", type=int, default=3,
                        help="每个入选结构界面 rif 残基的最少个数 (默认 3); 累积随机采样凑够为止, "
                             "候选池(能量在[e_full,0)内、有结合贡献的残基)不足、或采满 "
                             "max-resample 次仍不足则淘汰该结构")
    parser.add_argument("--max-resample", type=int, default=500,
                        help="累积随机采样的最大次数 (默认 500); 采满仍不足 min-rif 则淘汰该结构, 不补齐")
    parser.add_argument("--max-pdbs", type=int, default=None,
                        help=("筛选结果 PDB 数上限。超过时按 --keep-step 逐步"
                              "收紧 keep；默认不限制"))
    parser.add_argument("--keep-step", type=float, default=0.05,
                        help="超过 --max-pdbs 时每次减少的 keep 比例 (默认 0.05)")
    args = parser.parse_args()

    result_dir = os.path.abspath(args.result_dir)
    if not os.path.isdir(result_dir):
        sys.exit(f"错误: 文件夹不存在: {result_dir}")

    csv_path = os.path.join(result_dir, "binding_summary.csv")
    if not os.path.isfile(csv_path):
        sys.exit(f"错误: 未找到 binding_summary.csv: {csv_path}")

    output_dir = args.output_dir or os.path.join(result_dir, "filtered_top70")
    output_dir = os.path.abspath(output_dir)

    keep = args.keep
    if not 0 < keep <= 1:
        sys.exit("错误: --keep 必须在 (0, 1] 区间")
    if args.max_pdbs is not None and args.max_pdbs < 1:
        sys.exit("错误: --max-pdbs 必须 >= 1")
    if not 0 < args.keep_step < 1:
        sys.exit("错误: --keep-step 必须在 (0, 1) 区间")

    if args.kT <= 0:
        sys.exit("错误: --kT 必须 > 0")
    if args.e_full >= E_ZERO:
        sys.exit(f"错误: --e-full ({args.e_full}) 必须小于能量上界 {E_ZERO}")

    # ---- 读取界面残基能量表 (ddG_contrib 直接作为边界采样能量) ----
    interface_csv = os.path.join(result_dir, INTERFACE_CSV)
    by_pdb_res = None
    if os.path.isfile(interface_csv):
        try:
            by_pdb_res = load_interface_residues(interface_csv)
            print(f"读取界面残基表: {interface_csv}")
            print(f"链 {SAMPLE_CHAIN} 边界采样: e_zero={E_ZERO} (>= 舍弃, 固定), "
                  f"e_full={args.e_full} (<= 必选), kT={args.kT}")
        except ValueError as e:
            print(f"警告: 解析 {INTERFACE_CSV} 失败, 跳过残基采样: {e}")
    else:
        print(f"警告: 未找到 {INTERFACE_CSV}, 跳过残基采样: {interface_csv}")

    rng = random.Random(args.seed)

    # ---- 读取 CSV ----
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("错误: binding_summary.csv 没有数据行")

    n = len(rows)
    print(f"读取 {n} 条记录: {csv_path}")

    # ---- 校验所需列是否存在 ----
    needed_cols = {col for _, subs in DIMENSIONS for col, _ in subs}
    missing = needed_cols - set(rows[0].keys())
    if missing:
        sys.exit(f"错误: CSV 缺少列: {', '.join(sorted(missing))}")

    # ---- 计算每个维度的好坏分数 (越大越好) ----
    # dim_scores[i][dim_index] = 第 i 行在该维度的合成分数
    dim_scores = [[0.0] * len(DIMENSIONS) for _ in range(n)]
    for d_idx, (dim_name, subs) in enumerate(DIMENSIONS):
        # 每个子指标: 取列值 -> 方向归一(越小越好则取负) -> z-score, 再相加
        normalized_cols = []
        for col, higher_better in subs:
            try:
                vals = [float(r[col]) for r in rows]
            except ValueError as e:
                sys.exit(f"错误: 列 {col} 含非数值: {e}")
            if not higher_better:
                vals = [-v for v in vals]
            normalized_cols.append(zscores(vals))
        for i in range(n):
            dim_scores[i][d_idx] = sum(col[i] for col in normalized_cols)

    # ---- 按默认 keep 筛选；过多时每次收紧 keep_step ----
    requested_keep = keep
    while True:
        _, pass_count_per_dim, selected = select_by_keep(rows, dim_scores, keep)
        print(f"[自适应筛选] keep={keep:.0%} -> {len(selected)} 个四维度合格 PDB")
        if args.max_pdbs is None or len(selected) <= args.max_pdbs:
            break

        next_keep = round(keep - args.keep_step, 10)
        if next_keep <= 0:
            # 极端并列分数可能让分位数阈值无法继续减少数量。
            # 为保证下一轮不会超过硬上限，用“最弱维度分数、总分”做稳定排序截断。
            row_index = {id(row): i for i, row in enumerate(rows)}
            selected.sort(
                key=lambda row: (
                    min(dim_scores[row_index[id(row)]]),
                    sum(dim_scores[row_index[id(row)]]),
                    row.get("pdb", ""),
                ),
                reverse=True,
            )
            selected = selected[:args.max_pdbs]
            print(f"[自适应筛选] keep 已无法继续收紧，按综合分稳定截断为 "
                  f"{len(selected)} 个")
            break
        print(f"[自适应筛选] {len(selected)} > {args.max_pdbs}，"
              f"keep 从 {keep:.0%} 收紧到 {next_keep:.0%}")
        keep = next_keep

    if keep != requested_keep:
        print(f"[自适应筛选] 最终 keep={keep:.0%} "
              f"(初始 {requested_keep:.0%})")

    print(f"\n每个维度通过 (前{int(keep*100)}%) 的数量:")
    for (dim_name, _), c in zip(DIMENSIONS, pass_count_per_dim):
        print(f"  {dim_name:<22}: {c}/{n}")
    print(f"\n四个维度同时满足: {len(selected)}/{n}")

    removed = clear_previous_outputs(output_dir)
    if removed:
        print(f"清理旧筛选产物: {removed} 个文件")

    if not selected:
        print("没有同时满足全部维度的条目, 不复制任何文件。")
        return

    # ---- 复制 pdb + 玻尔兹曼采样 A 链残基(要求 >= min_rif 个, 否则淘汰), 写 rif_residues 首行 ----
    copied, miss, dropped_rif = 0, 0, 0
    do_sampling = by_pdb_res is not None
    kept_rows = []  # 实际保留(复制)的行 -> selected_summary.csv
    if args.min_rif < 1:
        sys.exit("错误: --min-rif 必须 >= 1")
    if not do_sampling:
        print(f"警告: 无 {INTERFACE_CSV}, min_rif>={args.min_rif} 门槛无法生效, 按四维度结果全部保留")
    print(f"\n复制到: {output_dir}")
    for r in selected:
        rel = r["pdb"]
        src = os.path.join(result_dir, rel)
        if not os.path.isfile(src):
            print(f"  [缺失] {rel}")
            miss += 1
            continue

        if do_sampling:
            # 累积随机采样凑够 min_rif(达到即停); 候选池不足、或采满 max_resample
            # 次仍不足 min_rif, 均淘汰该结构(不复制、不补齐、不进下一轮)
            residues = by_pdb_res.get(rel, [])
            hit = select_rif_residues(residues, rng, args.min_rif, args.max_resample,
                                      e_full=args.e_full, kT=args.kT)
            if hit is None:
                n_pos = count_positive_weight(residues, e_full=args.e_full, kT=args.kT)
                if n_pos < args.min_rif:
                    reason = (f"界面有结合贡献残基 {n_pos}/{len(residues)} < "
                              f"min_rif={args.min_rif}, 候选池不足")
                else:
                    reason = (f"采样 {args.max_resample} 次仍不足 min_rif={args.min_rif} "
                              f"(候选池 {n_pos}/{len(residues)})")
                print(f"  [淘汰] {rel}  (dG={r['dG_separated']})  {reason}")
                dropped_rif += 1
                continue
            # 平铺到输出目录 (文件名本身已含 design 名, 不会重名)
            dst = os.path.join(output_dir, os.path.basename(rel))
            shutil.copy2(src, dst)
            prepend_rif_line(dst, hit)
            copied += 1
            kept_rows.append(r)
            print(f"  [复制] {rel}  (dG={r['dG_separated']})  "
                  f"rif_residues({len(hit)}/{len(residues)}): {','.join(map(str, hit))}")
        else:
            dst = os.path.join(output_dir, os.path.basename(rel))
            shutil.copy2(src, dst)
            copied += 1
            kept_rows.append(r)
            print(f"  [复制] {rel}  (dG={r['dG_separated']})")

    tail = f", 因 rif<{args.min_rif} 淘汰 {dropped_rif} 个" if do_sampling else ""
    print(f"\n完成: 复制 {copied} 个, 缺失 {miss} 个{tail}。")

    if not kept_rows:
        print(f"警告: 全部 {len(selected)} 个四维度合格结构都因界面 rif<{args.min_rif} 被淘汰, "
              f"输出目录为空(下一轮将无输入)。")
        return

    # ---- 同时导出被保留条目的清单, 方便核对 ----
    out_csv = os.path.join(output_dir, "selected_summary.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(kept_rows)
    print(f"已写出筛选清单: {out_csv}")


if __name__ == "__main__":
    main()
