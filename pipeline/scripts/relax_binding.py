#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""遍历输入文件夹中的所有 PDB,调用已安装的 Rosetta 迁移包完成打分。

对每个 PDB 依次执行:
  1) binder A 链及其 14 Å 邻域的界面受限 FastRelax
  2) InterfaceAnalyzer  —— 全局结合能 dG_separated、界面面积、packstat 等
  3) residue_energy_breakdown —— 残基对相互作用能,折算成每残基界面贡献

输出(<output_dir> 下):
  按输入目录结构镜像存放的 relax 后 PDB
  binding_summary.csv      每个 PDB 的全局结合能/界面指标
  interface_residues.csv   每个界面残基对结合的贡献(ΔΔG,越负贡献越大)
中间产物(ia/*.sc / reb/*.out / 日志)放在 <output_dir>/_work/ 下的镜像目录,
保持镜像目录本身只含 relax 后的 PDB。

用法:
  python relax_binding.py <input_dir> <output_dir> [--nproc N] [--limit K] [--force]
"""
import argparse, csv, os, queue, re, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_ROSETTA = os.environ.get(
    "PIPELINE_ROSETTA_ROOT",
    "/mnt/data2/wtk/software/rosetta/rosetta-2025.37-df75a9c",
)
SAFE_TASK_NAME = re.compile(r"[^A-Za-z0-9_.-]+")

# 每个 Rosetta 进程强制单线程:钉死可能触发多线程的环境变量,避免单个 relax/IA/reb
# 因外部 OMP/BLAS 设置而偷偷多开线程,和线程池的并发数打架、抢占 CPU。
# 这样"每个 PDB 只占一个线程",并发规模完全由 --nproc(进程数)控制。
_SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def default_nproc():
    """本机可用物理核心数,作为默认并行任务数。

    从 /proc/cpuinfo 读取 socket/core 拓扑,并尊重 cgroup/taskset 的 CPU
    亲和性限制。取不到物理核拓扑时退回可用逻辑 CPU 数,最后兜底 1。
    """
    try:
        available = set(os.sched_getaffinity(0))
    except AttributeError:            # 非 Linux 无此接口
        available = set(range(os.cpu_count() or 1))

    try:
        physical = set()
        for block in Path("/proc/cpuinfo").read_text().strip().split("\n\n"):
            fields = {}
            for line in block.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    fields[key.strip()] = value.strip()
            cpu = int(fields["processor"])
            if cpu in available and "physical id" in fields and "core id" in fields:
                physical.add((fields["physical id"], fields["core id"]))
        if physical:
            return len(physical)
    except (OSError, KeyError, ValueError):
        pass
    return max(1, len(available))


def physical_core_cpu_ids():
    """返回当前亲和性范围内每个物理核的一个逻辑 CPU ID。"""
    try:
        available = set(os.sched_getaffinity(0))
    except AttributeError:
        available = set(range(os.cpu_count() or 1))

    representatives = {}
    try:
        for block in Path("/proc/cpuinfo").read_text().strip().split("\n\n"):
            fields = {}
            for line in block.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    fields[key.strip()] = value.strip()
            cpu = int(fields["processor"])
            if cpu not in available:
                continue
            key = (fields.get("physical id", "0"), fields.get("core id", str(cpu)))
            representatives[key] = min(cpu, representatives.get(key, cpu))
    except (OSError, KeyError, ValueError):
        return sorted(available)
    return sorted(representatives.values()) or sorted(available)

# score.sc 里关心的全局字段
KEEP = ["dG_separated", "dG_separated/dSASAx100", "dG_cross",
        "dSASA_int", "dSASA_hphobic", "dSASA_polar",
        "nres_int", "packstat", "sc_value", "delta_unsatHbonds",
        "hbonds_int", "total_score"]


def safe_task_name(name):
    """生成迁移包单任务脚本接受的任务名。"""
    value = SAFE_TASK_NAME.sub("_", name.replace(" ", "_")).strip("_")
    return value or "pdb_task"


def resolve_runner(rosetta_root):
    """定位自包含 Rosetta 安装及其固定协议 runner。"""
    root = Path(rosetta_root).expanduser().resolve()
    runner = root / "bin/run_rosetta_one.sh"
    runtime = root / "runtime/rosetta"
    required = [
        runner,
        root / "config/protocol.env",
        runtime / "database",
        runtime / "bin/rosetta_scripts.linuxgccrelease",
        runtime / "bin/InterfaceAnalyzer.linuxgccrelease",
        runtime / "bin/residue_energy_breakdown.linuxgccrelease",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        sys.exit("找不到已安装的 Rosetta 迁移包组件:\n  " + "\n  ".join(missing))
    return runner


def run(cmd, logfile, cpu_id=None):
    """执行命令,stdout/stderr 重定向到日志文件,返回退出码。

    子进程环境里钉死单线程相关变量,保证每个 Rosetta 进程只占一个线程。
    """
    env = {**os.environ, **_SINGLE_THREAD_ENV}
    if cpu_id is not None:
        cmd = ["taskset", "-c", str(cpu_id), *cmd]
    with open(logfile, "w") as lf:
        return subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode


def task_output_paths(wdir, task_name):
    """返回迁移包任务目录内供汇总使用的 IA/REB 文件。"""
    return wdir / "ia" / f"{task_name}.sc", wdir / "reb" / f"{task_name}.out"


def find_relaxed_structure(wdir):
    """定位单任务 runner 产生的 relaxed 结构。"""
    structures = sorted((wdir / "relaxed").glob("relaxed_*_0001.pdb"))
    return structures[-1] if structures else None


def process_one(pdb, rel, out_dir, work_dir, runner, force, cpu_queue=None):
    """处理单个 PDB。
    pdb      输入 PDB 绝对路径
    rel      相对输入根目录的相对路径(用于镜像输出与标识)
    返回 (rel, 是否成功, 提示信息)。relax 后 PDB 写到 out_dir/rel,
    中间产物写到 work_dir/<rel 去掉扩展名>/。
    """
    cpu_id = cpu_queue.get() if cpu_queue is not None else None
    try:
        name = safe_task_name(pdb.stem)
        relaxed = out_dir / rel                      # 镜像位置的 relax 后 PDB
        wdir = work_dir / rel.with_suffix("")        # 该 PDB 的中间产物目录
        relaxed.parent.mkdir(parents=True, exist_ok=True)
        if force and wdir.exists():
            shutil.rmtree(wdir)
        wdir.mkdir(parents=True, exist_ok=True)
        score_file, breakdown_file = task_output_paths(wdir, name)
        produced_structure = find_relaxed_structure(wdir)

        # 只承认新迁移包写出的完整标记,避免把旧 Rosetta 结果误当成新版本结果复用。
        if (wdir / ".pipeline_done").exists() and produced_structure and \
           score_file.exists() and breakdown_file.exists() and not force:
            if not relaxed.exists():
                shutil.copy2(produced_structure, relaxed)
            return rel, True, "skip(已存在)"

        # 固定协议由安装包维护:界面 14 Å FastRelax → IA → REB。这里不复制协议参数,
        # 以免流水线与安装包产生两套逐渐漂移的 Rosetta 命令。
        rc = run(
            ["bash", str(runner), str(pdb), str(wdir), name],
            wdir / "driver.log",
            cpu_id=cpu_id,
        )
        produced_structure = find_relaxed_structure(wdir)
        if rc != 0 or not (wdir / ".pipeline_done").exists():
            return rel, False, "Rosetta 三阶段失败(见 driver.log/relax.log/ia.log/reb.log)"
        if not produced_structure or not score_file.exists() or not breakdown_file.exists():
            return rel, False, "Rosetta 输出不完整(见任务目录)"

        # 下游筛选沿用原流水线布局:results_relax/<输入相对路径> 是 relax 后结构。
        shutil.copy2(produced_structure, relaxed)

        suffix = f"(cpu={cpu_id})" if cpu_id is not None else ""
        return rel, True, f"ok{suffix}"
    finally:
        if cpu_queue is not None:
            cpu_queue.put(cpu_id)


# ---------- 解析 ----------
def parse_score(scfile):
    """读取 InterfaceAnalyzer 的 score.sc,返回 {列名: 值}。"""
    header = data = None
    with open(scfile) as f:
        for line in f:
            if line.startswith("SCORE:"):
                toks = line.split()[1:]
                if toks[0] == "total_score":      # 表头行
                    header = toks
                elif header:                       # 数据行
                    data = toks
    return dict(zip(header, data)) if header and data else None


def parse_breakdown(bdfile):
    """读 residue_energy_breakdown 的 silent 输出,折算每个残基的跨界面贡献。
    规则:对所有跨链 two-body 残基对,把 total 的一半分别记到两个残基上
    (这样所有残基贡献之和 ≈ 跨界面总能量 dG_cross)。
    返回 {(chain, resnum): [restype, energy]}。"""
    contrib = {}
    header = None
    with open(bdfile) as f:
        for line in f:
            if not line.startswith("SCORE:"):
                continue
            toks = line.split()[1:]
            if toks[0] == "pose_id":               # 表头
                header = toks
                continue
            if not header:
                continue
            row = dict(zip(header, toks))
            if row.get("restype2") == "onebody" or row.get("resi2") == "--":
                continue                            # 跳过单体项
            p1, p2 = row["pdbid1"], row["pdbid2"]   # 形如 4A / 102B
            if p1[-1] == p2[-1]:                    # 同链,非跨界面
                continue
            try:
                half = float(row["total"]) / 2.0
            except ValueError:
                continue
            for pid, ctype in ((p1, row["restype1"]), (p2, row["restype2"])):
                key = (pid[-1], int(pid[:-1]))
                contrib.setdefault(key, [ctype, 0.0])
                contrib[key][1] += half
    return contrib


def _fnum(x):
    """安全转 float,失败返回正无穷(排序时排到最后)。"""
    try:
        return float(x)
    except (ValueError, TypeError):
        return float("inf")


def collect(results, out_dir, work_dir):
    """汇总成功项的 score.sc / breakdown.out,写出两张 CSV。"""
    summary, residues = [], []
    for rel, ok, _ in results:
        if not ok:
            continue
        wdir = work_dir / rel.with_suffix("")
        task_name = safe_task_name(rel.stem)
        design = rel.parent.as_posix() if rel.parent != Path(".") else ""
        sc, bd = task_output_paths(wdir, task_name)
        if sc.exists() and (row := parse_score(sc)):
            rec = {"pdb": rel.as_posix(), "design": design}
            rec.update({k: row.get(k, "") for k in KEEP})
            summary.append(rec)
        if bd.exists():
            for (ch, num), (rtype, e) in parse_breakdown(bd).items():
                residues.append({"pdb": rel.as_posix(), "design": design,
                                 "chain": ch, "resnum": num,
                                 "restype": rtype, "ddG_contrib": round(e, 3)})

    summary.sort(key=lambda r: _fnum(r["dG_separated"]))          # 结合能越负越靠前
    residues.sort(key=lambda r: (r["pdb"], r["ddG_contrib"]))     # 同 PDB 内贡献从负到正

    bs = out_dir / "binding_summary.csv"
    with open(bs, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb", "design"] + KEEP)
        w.writeheader(); w.writerows(summary)

    ir = out_dir / "interface_residues.csv"
    with open(ir, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb", "design", "chain",
                                          "resnum", "restype", "ddG_contrib"])
        w.writeheader(); w.writerows(residues)
    return bs, len(summary), ir, len(residues)


def main():
    ap = argparse.ArgumentParser(
        description="遍历文件夹做 FastRelax 并计算界面结合能/残基贡献")
    ap.add_argument("input_dir", help="输入文件夹(递归查找其中所有 .pdb)")
    ap.add_argument("output_dir", help="输出文件夹(镜像输入结构存放 relax 后 PDB)")
    ap.add_argument("--nproc", type=int, default=0,
                    help="并行任务数(每任务单进程/单线程);<=0 表示自动使用可用物理核")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 K 个 PDB(0=全部,用于测试)")
    ap.add_argument("--force", action="store_true", help="强制重算已有结果")
    ap.add_argument(
        "--pin-cores",
        action="store_true",
        help="把并行任务分别绑定到不同物理核，避免单线程 Rosetta 都挤到同一核",
    )
    ap.add_argument(
        "--rosetta",
        default=DEFAULT_ROSETTA,
        help="已安装的 Rosetta 2025.37 自包含迁移包根目录",
    )
    args = ap.parse_args()

    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    if not in_dir.is_dir():
        sys.exit(f"输入文件夹不存在: {in_dir}")
    work_dir = out_dir / "_work"
    runner = resolve_runner(args.rosetta)

    pdbs = sorted(p for p in in_dir.rglob("*.pdb") if p.is_file())
    if args.limit:
        pdbs = pdbs[:args.limit]
    if not pdbs:
        sys.exit(f"在 {in_dir} 下没找到 .pdb 文件")
    out_dir.mkdir(parents=True, exist_ok=True)
    # nproc <=0 时自动取本机可用物理核心数;每个 PDB 由一个进程单线程处理。
    if args.nproc <= 0:
        args.nproc = default_nproc()
    nproc = min(args.nproc, len(pdbs))    # PDB 少于进程数时没必要开那么多
    cpu_queue = None
    if args.pin_cores:
        if shutil.which("taskset") is None:
            sys.exit("--pin-cores 需要系统提供 taskset")
        cpu_ids = physical_core_cpu_ids()
        nproc = min(nproc, len(cpu_ids))
        cpu_queue = queue.Queue()
        for cpu_id in cpu_ids[:nproc]:
            cpu_queue.put(cpu_id)
    print(
        f"共 {len(pdbs)} 个 PDB,{nproc} 路并行(每任务单进程/单线程),"
        f"Rosetta=2025.37,relax=iface14/ref2015,绑核={args.pin_cores}"
    )

    # relax 是单线程外部进程,用线程池并发分发即可
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=nproc) as ex:
        futs = {ex.submit(process_one, p, p.relative_to(in_dir),
                          out_dir, work_dir, runner, args.force, cpu_queue): p for p in pdbs}
        for fut in as_completed(futs):
            rel, ok, msg = fut.result()
            results.append((rel, ok, msg))
            done += 1
            flag = "OK " if ok else "FAIL"
            print(f"[{done}/{len(pdbs)}] {flag} {rel.as_posix()}  {msg}")

    nfail = sum(1 for _, ok, _ in results if not ok)
    bs, ns, ir, nr = collect(results, out_dir, work_dir)
    print(f"\n=== 完成: {len(results)-nfail} 成功 / {nfail} 失败 ===")
    print(f"{bs}: {ns} 行")
    print(f"{ir}: {nr} 行(界面残基)")
    if nfail:
        sys.exit(1)


if __name__ == "__main__":
    main()
