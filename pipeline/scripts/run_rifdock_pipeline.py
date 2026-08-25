#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RIFdock 批量对接流程脚本(一条命令跑完)

做的事:
  1. 复制 default_flags 里的默认 rifgen/rifdock flag,只把 target / target_res
     (以及输出/缓存目录)按命令行参数替换,其余保持默认。
  2. 跑一次 rifgen 生成靶点的 RIF 场。rifgen 结束时会在日志里打印
     "what you need for docking" 区块 —— 里边的 -rif_dock:target_rf_cache
     带有 rifgen 自动生成的哈希串(__RF_..._trhashXXXX_...),手写必错,
     所以本脚本直接从该日志区块解析,不做任何手工拼接。
  3. 对 scaffold 文件夹里的每个 PDB 各跑一次 rif_dock,
     每个 scaffold 独立 log、独立结果目录。

所有产物都收进当前目录下的【一个】主文件夹(默认 rifdock_pipeline/):
    rifdock_pipeline/
      flags/            生成的 rifgen.flag / rifdock.base.flag
      rifgen_out/       rifgen 产物(RIF 场 + 哈希命名的 rosetta_field 缓存)
      cache/            流程本地缓存:scaffdata(scaffold 二体表)+ rifgen 兜底。
                        与靶点无关的共享缓存(rifgen HBOND_GEOMS / rifdock rotrf)
                        优先走默认 flag 里配置的公共库,本地只作兜底。
      logs/             rifgen.log + 每个 scaffold 一个 dock_<名字>.log
      dock_results/     每个 scaffold 一个结果子目录(含 PDB 与 all.dok)
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# ---- 该环境下的默认路径 ----
RIF_ROOT = "/mnt/data2/wtk/software/rif/rifdock-master"
DEF_RIFGEN_BIN = f"{RIF_ROOT}/build/apps/rosetta/rifgen"
DEF_RIFDOCK_BIN = f"{RIF_ROOT}/build/apps/rosetta/rif_dock_test"
DEF_RIFGEN_FLAG = f"{RIF_ROOT}/default_flags/rifgen.default.flag"
# 默认用 benchmark 调过的速度/质量平衡快速预设(rifdock.fast.flag);要换回原始默认,
# 命令行加 --rifdock-flag {RIF_ROOT}/default_flags/rifdock.default.flag 即可。
DEF_RIFDOCK_FLAG = f"{RIF_ROOT}/default_flags/rifdock.fast.flag"


def log(msg):
    print(f"[pipeline] {msg}", flush=True)


def _cache_values(line):
    """取出一行 flag 中 key 之后的所有路径值,忽略行内 # 注释。"""
    code = line.split("#", 1)[0]
    toks = code.split()
    return toks[1:] if len(toks) > 1 else []


def _abs_paths(vals):
    """只保留绝对路径(即用户配置的公共库目录),丢掉 ./cache 这类相对兜底。"""
    return [v for v in vals if os.path.isabs(v)]


def build_rifgen_flag(default_text, target, target_res, outdir, outfile,
                      cache_dir, database):
    """以默认 rifgen flag 为底,只改 target / target_res / outdir / outfile /
    缓存目录(以及可选 database),其余原样保留。
    其中 -rifgen:data_cache_dir 保留默认里的公共库(绝对路径)放最前,再把
    流程本地 cache 追加为兜底——不能用单一本地目录覆盖,否则会丢掉公共库共享。"""
    out_lines = []
    for line in default_text.splitlines():
        s = line.strip()
        if s.startswith("-rifgen:target_res"):
            out_lines.append(f"-rifgen:target_res       {target_res}")
        elif s.startswith("-rifgen:target"):
            out_lines.append(f"-rifgen:target           {target}")
        elif s.startswith("-rifgen:outdir"):
            out_lines.append(f"-rifgen:outdir           {outdir}")
        elif s.startswith("-rifgen:outfile"):
            out_lines.append(f"-rifgen:outfile          {outfile}")
        elif s.startswith("-rifgen:data_cache_dir"):
            # 公共库(默认里的绝对路径)优先放前,流程本地 cache 追加为兜底
            shared = _abs_paths(_cache_values(line))
            dirs = shared + [cache_dir]
            out_lines.append("-rifgen:data_cache_dir    " + "    ".join(dirs))
        elif s.startswith("-database") and database:
            out_lines.append(f"-database {database}")
        else:
            out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def parse_docking_block(log_text):
    """从 rifgen 日志里解析 'what you need for docking' 区块,
    返回其中所有 -rif_dock: 开头的行(含哈希命名的 target_rf_cache)。"""
    lines = log_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "what you need for docking" in line:
            start = i + 1
            break
    if start is None:
        return None
    block = []
    for line in lines[start:]:
        s = line.strip()
        if s.startswith("####") or s.startswith("==="):
            break  # 区块结束(分隔线或脚本自加的 EXIT 标记)
        if s.startswith("-rif_dock:"):
            block.append(s)
    return block if block else None


def build_rifdock_base_flag(default_text, docking_block, cache_dir, database):
    """以默认 rifdock flag 为底:
      - 用 rifgen 解析出的 docking_block 替换原占位的 docking 区块;
      - 删掉 scaffolds / outdir / dokfile(这三项每个 scaffold 在命令行单独给);
      - 重定向缓存目录到主文件夹;可选改 database。
    """
    # docking 区块涉及的 key,先把默认模板里的同名占位行全删掉,再统一插入解析结果
    block_keys = {ln.split()[0] for ln in docking_block}
    out_lines = []
    inserted = False
    for line in default_text.splitlines():
        s = line.strip()
        key = s.split()[0] if s else ""
        if key in block_keys:
            if not inserted:  # 在第一处占位位置插入权威区块
                out_lines.append(
                    "########### what you need for docking (来自 rifgen 日志,自动解析) ###########")
                out_lines.extend(docking_block)
                out_lines.append(
                    "#############################################################################")
                inserted = True
            continue  # 跳过模板里的旧占位行
        if key in ("-rif_dock:scaffolds", "-rif_dock:outdir", "-rif_dock:dokfile"):
            continue  # 这三项命令行逐个 scaffold 指定
        if key == "-rif_dock:data_cache_dir":
            # scaffdata 与靶点/scaffold 相关,保持流程本地(与默认 ./cache/scaffdata 同义)
            out_lines.append(f"-rif_dock:data_cache_dir  {cache_dir}/scaffdata")
        elif key == "-rif_dock:rotrf_cache_dir":
            # rotrf 与靶点无关、可跨项目复用:保留默认里的公共库(绝对路径),
            # 默认没配公共库时才退回流程本地 cache/rotrf
            shared = _abs_paths(_cache_values(line))
            out_lines.append("-rif_dock:rotrf_cache_dir " +
                             (shared[0] if shared else f"{cache_dir}/rotrf"))
        elif key == "-database" and database:
            out_lines.append(f"-database {database}")
        else:
            out_lines.append(line)
    if not inserted:  # 模板里没有占位行时,补在文件头
        head = ["########### what you need for docking (来自 rifgen 日志,自动解析) ###########"]
        head += docking_block
        head += ["#############################################################################"]
        out_lines = head + out_lines
    return "\n".join(out_lines) + "\n"


def _is_opt_key(tok):
    """判断一个 token 是不是选项名(key)。
    选项名形如 -rifgen:xxx / -hash_cart_resl;负数值(如 -1.5、-.3)不算 key。"""
    return tok.startswith("-") and not re.match(r"^-[\d.]", tok)


def parse_extra_opts(tokens):
    """把命令行风格的额外参数 token 列表解析成有序的 [(key, [values...]), ...]。
    key 以 '-' 开头,其后所有非 key 的 token 都是它的值(支持多值向量选项)。
    同名 key 多次出现时,后者覆盖前者(与 Rosetta 命令行语义一致)。"""
    opts = []
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        if _is_opt_key(tok):
            key, vals = tok, []
            i += 1
            while i < n and not _is_opt_key(tokens[i]):
                vals.append(tokens[i])
                i += 1
            opts.append((key, vals))
        else:
            i += 1  # 跳过没有归属 key 的游离值(正常不该出现)
    return opts


def merge_extra_into_flag(flag_text, extra_tokens):
    """把额外参数【直接写进】flag 文本,而不是挂在命令行末尾覆盖:
      - flag 里已有同名项 -> 就地整行替换(保留行内 # 注释);
      - flag 里没有的项   -> 追加到文件末尾,并加一段来源说明注释。
    Rosetta 中 '-a::b' 与 '-a:b' 是同一选项,这里归一化后按 key 精确匹配
    (因此 '-hash_cart_resl' 不会误改 '-hash_cart_resls')。
    返回 (新文本, 改动说明列表)。"""
    def norm(key):
        return key.replace("::", ":")

    opts = parse_extra_opts(extra_tokens)
    if not opts:
        return flag_text, []

    pending, order = {}, []
    for k, v in opts:
        nk = norm(k)
        pending[nk] = (k, v)
        if nk not in order:
            order.append(nk)

    used, notes, out_lines = set(), [], []
    for line in flag_text.splitlines():
        # 拆出行内注释原样保留
        if "#" in line:
            code_part, hash_rest = line.split("#", 1)
            comment = "  #" + hash_rest.rstrip()
        else:
            code_part, comment = line, ""
        cs = code_part.strip()
        if cs.startswith("-"):
            toks = cs.split()
            line_key = toks[0]
            nk = norm(line_key)
            if nk in pending and nk not in used:
                _, vals = pending[nk]
                old_val = " ".join(toks[1:])
                new_val = " ".join(vals)
                out_lines.append(f"{line_key} {new_val}{comment}".rstrip())
                used.add(nk)
                notes.append(f"改 {line_key}: {old_val!r} -> {new_val!r}")
                continue
        out_lines.append(line)

    appended = [pending[nk] for nk in order if nk not in used]
    if appended:
        out_lines.append("")
        out_lines.append("# ==== 以下为命令行 extra 参数自动写入(flag 模板中原无此项) ====")
        for k, v in appended:
            new_val = " ".join(v)
            out_lines.append(f"{k} {new_val}".rstrip())
            notes.append(f"加 {k}: {new_val!r}")

    return "\n".join(out_lines) + "\n", notes


def run(cmd, log_path, env):
    """运行命令,stdout+stderr 同时写日志文件并回显到终端,返回退出码。"""
    log(f"运行: {' '.join(cmd)}")
    log(f"日志: {log_path}")
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=env, text=True)
        for line in proc.stdout:
            lf.write(line)
            sys.stdout.write(line)
        proc.wait()
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(
        description="RIFdock 批量对接流程(rifgen 一次 + 每个 scaffold 各对接一次)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--target", default="input/PD-1.pdb",
                    help="靶点蛋白 PDB(写入 -rifgen:target)")
    ap.add_argument("--target-res", default="input/target_res.list",
                    help="热点残基列表文件(写入 -rifgen:target_res)")
    ap.add_argument("--scaffold-dir", default="input/scaffolds",
                    help="存放多个 scaffold PDB 的文件夹,里面每个 .pdb 各对接一次")
    ap.add_argument("--out", default="rifdock_pipeline",
                    help="主输出文件夹(所有产物都收在这里)")
    ap.add_argument("--threads", type=int, default=None,
                    help="OMP 线程数(设置 OMP_NUM_THREADS)")
    ap.add_argument("--database", default=None,
                    help="覆盖 Rosetta database 路径(默认用 flag 里的)")
    ap.add_argument("--rifgen-bin", default=DEF_RIFGEN_BIN)
    ap.add_argument("--rifdock-bin", default=DEF_RIFDOCK_BIN)
    ap.add_argument("--rifgen-flag", default=DEF_RIFGEN_FLAG,
                    help="默认 rifgen flag 模板")
    ap.add_argument("--rifdock-flag", default=DEF_RIFDOCK_FLAG,
                    help="默认 rifdock flag 模板")
    ap.add_argument("--force-rifgen", action="store_true",
                    help="即使已有可用的 rifgen 结果也强制重跑")
    ap.add_argument("--skip-rifgen", action="store_true",
                    help="跳过 rifgen,直接用已有日志里的 docking 区块做对接")
    ap.add_argument("--docking-flag", default=None,
                    help="复用已算好的 rifgen 结果:直接从该 flag 文件顶部的 "
                         "'what you need for docking' 区块取 docking 输入,"
                         "跳过 rifgen 且不读取 rifgen.log。区块里的路径原样使用,"
                         "故须为绝对路径或相对运行 cwd 可达。")
    ap.add_argument("--rifgen-extra", default="",
                    help="额外的 rifgen 参数,会【直接写进 rifgen.flag】(同名项就地替换、"
                         "新项追加到文件末尾),而非挂命令行末尾覆盖,便于事后审计。"
                         "如 \"-rifgen:beam_size_M 1 -rifgen:rot_samp_resl 12.0\"")
    ap.add_argument("--rifdock-extra", default="",
                    help="额外的 rif_dock 参数,会【直接写进 rifdock.base.flag】(同名项就地替换、"
                         "新项追加到文件末尾),而非挂命令行末尾覆盖,便于事后审计。")
    args = ap.parse_args()

    import shlex
    rifgen_extra = shlex.split(args.rifgen_extra)
    rifdock_extra = shlex.split(args.rifdock_extra)

    # ---- 路径全部转绝对,避免 rif_dock 在不同 cwd 下找不到缓存 ----
    target = Path(args.target).resolve()
    target_res = Path(args.target_res).resolve()
    scaffold_dir = Path(args.scaffold_dir).resolve()
    master = Path(args.out).resolve()
    flags_dir = master / "flags"
    rifgen_out = master / "rifgen_out"
    cache_dir = master / "cache"
    logs_dir = master / "logs"
    results_dir = master / "dock_results"

    for p in (flags_dir, rifgen_out, cache_dir, logs_dir, results_dir):
        p.mkdir(parents=True, exist_ok=True)

    # ---- 基本校验 ----
    if not target.is_file():
        sys.exit(f"找不到靶点 PDB: {target}")
    if not target_res.is_file():
        sys.exit(f"找不到 target_res 列表: {target_res}")
    if not scaffold_dir.is_dir():
        sys.exit(f"找不到 scaffold 文件夹: {scaffold_dir}")
    scaffolds = sorted([p for p in scaffold_dir.iterdir()
                        if p.suffix == ".pdb" or p.name.endswith(".pdb.gz")])
    if not scaffolds:
        sys.exit(f"scaffold 文件夹里没有 .pdb / .pdb.gz: {scaffold_dir}")
    log(f"发现 {len(scaffolds)} 个 scaffold 待对接")

    env = os.environ.copy()
    if args.threads:
        env["OMP_NUM_THREADS"] = str(args.threads)
        log(f"OMP_NUM_THREADS = {args.threads}")

    rifgen_log = logs_dir / "rifgen.log"
    outfile = "rif.rif.gz"  # 保持默认 RIF 主文件名

    # ---- 复用已算好的 rifgen 结果(--docking-flag):跳过 rifgen,不读 rifgen.log ----
    docking_block = None
    if args.docking_flag:
        df = Path(args.docking_flag)
        if not df.is_file():
            sys.exit(f"--docking-flag 指定的文件不存在: {df}")
        docking_block = parse_docking_block(df.read_text())
        if not docking_block:
            sys.exit("--docking-flag 文件里找不到 'what you need for docking' 区块"
                     "(需含 -rif_dock: 开头的行): " + str(df))
        log(f"复用现成 docking 区块(来自 {df},跳过 rifgen,不读 rifgen.log),"
            f"共 {len(docking_block)} 行:")
        for ln in docking_block:
            if "target_rf_cache" in ln or "target_rif " in ln:
                log("  " + ln)

    # ---- 阶段一:rifgen(未提供 --docking-flag 时才走)----
    if docking_block is None:
        need_rifgen = not args.skip_rifgen
        if need_rifgen and not args.force_rifgen and rifgen_log.is_file():
            blk = parse_docking_block(rifgen_log.read_text())
            if blk:  # 已有可用结果,跳过重跑
                log("检测到已有可用的 rifgen 日志(含 docking 区块),跳过 rifgen。"
                    "如需重跑请加 --force-rifgen")
                need_rifgen = False

        if need_rifgen:
            rifgen_default = Path(args.rifgen_flag).read_text()
            rifgen_flag_text = build_rifgen_flag(
                rifgen_default, str(target), str(target_res),
                str(rifgen_out), outfile, str(cache_dir), args.database)
            # 把 --rifgen-extra 直接写进 flag(就地替换/末尾追加),不再挂命令行覆盖
            rifgen_flag_text, notes = merge_extra_into_flag(rifgen_flag_text, rifgen_extra)
            rifgen_flag_path = flags_dir / "rifgen.flag"
            rifgen_flag_path.write_text(rifgen_flag_text)
            log(f"已写入 rifgen flag: {rifgen_flag_path}")
            if notes:
                log(f"--rifgen-extra 已写入 flag 文件({len(notes)} 处改动):")
                for nt in notes:
                    log("  " + nt)
            rc = run([args.rifgen_bin, f"@{rifgen_flag_path}"], rifgen_log, env)
            if rc != 0:
                log(f"警告: rifgen 退出码 {rc}(143=被 SIGTERM 终止)。"
                    "仍尝试解析 docking 区块,可能产物不完整。")

        # ---- 解析 rifgen 打印的权威 docking 区块(含哈希缓存名) ----
        if not rifgen_log.is_file():
            sys.exit(f"没有 rifgen 日志可解析: {rifgen_log}")
        docking_block = parse_docking_block(rifgen_log.read_text())
        if not docking_block:
            sys.exit("无法从 rifgen 日志解析出 'what you need for docking' 区块,"
                     "说明 rifgen 没正常跑到打印该区块。请检查 " + str(rifgen_log))
        log(f"已解析 docking 区块({len(docking_block)} 行),"
            "其中 target_rf_cache 为 rifgen 哈希命名:")
        for ln in docking_block:
            if "target_rf_cache" in ln:
                log("  " + ln)

    # ---- 写 rifdock 基础 flag(替换 docking 区块 + 重定向缓存) ----
    rifdock_default = Path(args.rifdock_flag).read_text()
    rifdock_base_text = build_rifdock_base_flag(
        rifdock_default, docking_block, str(cache_dir), args.database)
    # 把 --rifdock-extra 直接写进 base flag(就地替换/末尾追加),不再挂命令行覆盖
    rifdock_base_text, notes = merge_extra_into_flag(rifdock_base_text, rifdock_extra)
    rifdock_base_path = flags_dir / "rifdock.base.flag"
    rifdock_base_path.write_text(rifdock_base_text)
    log(f"已写入 rifdock 基础 flag: {rifdock_base_path}")
    if notes:
        log(f"--rifdock-extra 已写入 base flag 文件({len(notes)} 处改动):")
        for nt in notes:
            log("  " + nt)

    # ---- 阶段二:逐个 scaffold 对接 ----
    # scaffdata 在流程本地;rotrf 已重定向到公共库(见 build_rifdock_base_flag),
    # 故这里只建本地 scaffdata,不再建本地 cache/rotrf 以免留下误导性空目录。
    (cache_dir / "scaffdata").mkdir(exist_ok=True)
    ok, fail = 0, 0
    for sc in scaffolds:
        name = sc.name[:-7] if sc.name.endswith(".pdb.gz") else sc.stem
        sc_outdir = results_dir / name
        sc_outdir.mkdir(parents=True, exist_ok=True)
        sc_log = logs_dir / f"dock_{name}.log"
        cmd = [
            args.rifdock_bin, f"@{rifdock_base_path}",
            "-rif_dock:scaffolds", str(sc),
            "-rif_dock:outdir", str(sc_outdir),
            # rif_dock 内部用 outdir + dokfile 拼出最终路径,所以 dokfile 只能给
            # 文件名(basename),不能给绝对路径——否则会拼成 outdir/绝对路径 的
            # 畸形双重路径导致写盘失败(对接 PDB 照常产出,但 all.dok 丢失)。
            "-rif_dock:dokfile", "all.dok",
        ]
        log(f"[{ok + fail + 1}/{len(scaffolds)}] 对接 scaffold: {name}")
        rc = run(cmd, sc_log, env)
        if rc == 0:
            ok += 1
        else:
            fail += 1
            log(f"  scaffold {name} 对接退出码 {rc},详见 {sc_log}")

    log(f"全部完成: 成功 {ok} 个, 失败 {fail} 个。结果在 {results_dir}")


if __name__ == "__main__":
    main()


