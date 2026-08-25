#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AF3 批量并行推理调度器(不使用 MSA)。

用法:
  python run_af3_batch.py -i <存放 json 的文件夹> \
                          -o <输出文件夹> \
                          -g 4,5
  (长选项 --input_dir / --output_dir / --gpus 仍可用)

特点:
  - 扫描 input_dir 下所有 *.json
  - 预处理:为每条 protein/RNA 链注入"单序列 MSA"(不做数据库比对),
    并清空 templates,从而配合 --norun_data_pipeline 实现"不用 MSA"
  - 多 GPU 并行:把待跑任务轮转分到每块 GPU 一个子目录,每块 GPU 只起
    一个 run_alphafold.py 进程用 --input_dir 跑完整组。模型只构建一次、
    每个 bucket 只做一次 XLA 编译,后续任务复用已编译可执行体(大幅提速:
    原本每任务 ~140s 里约 130s 是初始化/编译而非真正前向)
  - 断点续跑:输出里已存在 *_summary_confidences.json 的任务自动跳过
  - 每块 GPU 一份批处理日志写到 <output_dir>/logs/gpu<id>.log

输出目录布局:
  <output_dir>/
    structures/   <- 所有结构结果(每个任务一个子目录),这是你要的东西
    logs/         <- 每 GPU 批处理日志(保留,失败排查靠它)
    运行期间还会临时生成 _jax_cache(XLA 编译缓存)、_prepared_inputs
    (注入单序列 MSA 后的中间 json)和 _groups(按 GPU 分组的输入副本),
    三者都不是结果,全部跑完后默认自动删除。
    若想复用编译缓存以加速下次运行,加 --keep_temp 保留它们。
"""

import argparse
import json
import os
import random
import shutil
import string
import subprocess
import sys
import threading
import time
from pathlib import Path

# ===================== 环境固定配置(本服务器已验证可用) =====================
AF3_PYTHON = "/home/opt/anaconda3/envs/af3/bin/python"          # jax 0.4.34 正确版本
AF3_RUNNER = "/home/opt/alphafold3/run_alphafold.py"
AF3_PKG_PYTHONPATH = "/mnt/data2/wtk/software/af3_pkg"          # 组装好的完整 alphafold3 包
MODEL_DIR = "/home/opt/alphafold3_database/alphafold3_weights"
# ===========================================================================

_ALLOWED = set(string.ascii_letters + string.digits + "_-.")

# 种子相关:默认每个 sample 生成 5 个随机种子,取值范围 [1, 1_000_000)。
DEFAULT_NUM_SEEDS = 5
SEED_MAX = 1_000_000  # 1M 以内(不含)


def gen_random_seeds(n: int) -> list[int]:
    """生成 n 个互不重复的随机种子,取值范围 [1, SEED_MAX)。"""
    return random.sample(range(1, SEED_MAX), n)


def sanitised_name(name: str) -> str:
    """与 AF3 内部 sanitised_name 一致,决定输出子目录名。"""
    return "".join(c for c in name.replace(" ", "_") if c in _ALLOWED)


def inject_single_seq_msa(fold_input: dict) -> dict:
    """为每条 protein / rna 链注入单序列 MSA(no-MSA 模式必需)。

    templates 处理: 若 JSON 里该链已带非空 templates(即上游 filter_for_af3 已
    注入的骨架模板), 原样保留; 否则置空列表 [] 表示不用模板。这样"无模板"和
    "有模板"两种输入都能走 --norun_data_pipeline(该模式要求每条蛋白链
    unpairedMsa/pairedMsa/templates 三者都存在, templates 可为 [])。
    """
    for entity in fold_input.get("sequences", []):
        for mol_type in ("protein", "rna"):
            if mol_type not in entity:
                continue
            chain = entity[mol_type]
            ids = chain.get("id", "A")
            first_id = ids[0] if isinstance(ids, list) else ids
            seq = chain["sequence"]
            # 单序列 A3M:仅含自身,等价于"无同源信息"
            chain["unpairedMsa"] = f">{first_id}\n{seq}\n"
            if mol_type == "protein":
                chain["pairedMsa"] = ""
                # 保留上游注入的骨架模板;没有则置空列表
                if not chain.get("templates"):
                    chain["templates"] = []
    # 强制 dialect/version,确保支持内置 MSA 字段
    fold_input["dialect"] = "alphafold3"
    if fold_input.get("version", 0) < 2:
        fold_input["version"] = 2
    return fold_input


def prepare_inputs(input_dir: Path, prepared_dir: Path, num_seeds: int) -> list[dict]:
    """读取所有 json,注入单序列 MSA 与随机种子,写到 prepared_dir。返回任务列表。

    每个 json 的 modelSeeds 会被覆盖为 num_seeds 个互不重复的随机种子
    (取值范围 [1, 1_000_000)),从而每个 sample 产出 num_seeds × num_diffusion_samples
    个重复结构。
    """
    prepared_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for jp in sorted(input_dir.glob("*.json")):
        try:
            data = json.loads(jp.read_text())
        except Exception as e:
            print(f"[警告] 跳过无法解析的 json: {jp.name} ({e})", flush=True)
            continue
        # 若未指定 name,用文件名兜底
        if not data.get("name", "").strip():
            data["name"] = jp.stem
        data = inject_single_seq_msa(data)
        # 覆盖为随机种子:每个任务各自独立采样,互不重复
        data["modelSeeds"] = gen_random_seeds(num_seeds)
        out_name = sanitised_name(data["name"])
        prepared_path = prepared_dir / f"{out_name}.json"
        prepared_path.write_text(json.dumps(data, indent=2))
        jobs.append({"name": out_name, "json": str(prepared_path), "src": jp.name})
    return jobs


def is_done(output_dir: Path, name: str) -> bool:
    """已有顶层 summary_confidences.json 视为完成。"""
    return (output_dir / name / f"{name}_summary_confidences.json").exists()


def run_group(gpu_id: str, group_dir: Path, output_dir: Path, log_dir: Path,
              jax_cache: Path) -> tuple[int, float]:
    """在指定 GPU 上用单进程批处理一整组任务,返回 (退出码, 耗时秒)。

    用 --input_dir 让 run_alphafold.py 在一个进程内顺序跑完该目录下所有 json:
    模型只构建一次、每个 bucket 只做一次 XLA 编译,后续任务复用内存里已编译的
    可执行体,从而省掉"每个 json 都重新初始化 + 重新编译"的巨大开销
    (实测每任务原本 ~140s,其中 ~130s 是初始化/编译而非真正前向)。

    jax_cache 为"该 GPU 专属"的编译缓存目录,跨进程/跨次运行时可被复用。
    """
    log_path = log_dir / f"gpu{gpu_id}.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    # 关闭显存预分配:共享 GPU 上避免一次性吞掉大块显存导致冲突
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["PYTHONPATH"] = AF3_PKG_PYTHONPATH + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    cmd = [
        AF3_PYTHON, AF3_RUNNER,
        f"--input_dir={group_dir}",       # 整组目录,一个进程顺序跑完
        f"--model_dir={MODEL_DIR}",
        f"--output_dir={output_dir}",
        "--norun_data_pipeline",          # 不做 MSA 搜索
        "--gpu_device=0",                 # CUDA_VISIBLE_DEVICES 已映射,这里恒为 0
        "--force_output_dir",             # 强制用固定输出目录名(否则非空时会新建时间戳目录)
        f"--jax_compilation_cache_dir={jax_cache}",
    ]
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write("CMD: " + " ".join(cmd) + f"\nGPU(物理): {gpu_id}\n\n")
        lf.flush()
        rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    return rc, time.time() - t0


def worker(gpu_id: str, group_dir: Path, jobs_in_group: list, output_dir: Path,
           log_dir: Path, jax_cache_root: Path, results: list,
           lock: threading.Lock):
    """单 GPU worker:用一个进程批处理分配给本 GPU 的整组任务。

    每块 GPU 使用独立的编译缓存子目录 _jax_cache/gpu<id>,
    避免多进程并发写同一缓存目录导致损坏。

    批处理模式下拿不到单任务退出码,只能在进程结束后按"输出目录里是否生成了
    summary_confidences.json"逐个判定任务成败。
    """
    gpu_cache = jax_cache_root / f"gpu{gpu_id}"
    gpu_cache.mkdir(parents=True, exist_ok=True)
    n = len(jobs_in_group)
    print(f"🚀 [GPU {gpu_id}] 启动批处理:{n} 个任务(单进程,模型只编译一次)",
          flush=True)
    rc, dt = run_group(gpu_id, group_dir, output_dir, log_dir, gpu_cache)
    done = [j for j in jobs_in_group if is_done(output_dir, j["name"])]
    failed = [j for j in jobs_in_group if not is_done(output_dir, j["name"])]
    flag = "✅" if not failed else "⚠️"
    print(f"{flag} [GPU {gpu_id}] 完成 {len(done)}/{n},失败 {len(failed)}"
          f"(进程 rc={rc}),组用时 {dt:.1f}s,均摊 {dt / max(n, 1):.1f}s/任务",
          flush=True)
    with lock:
        for j in done:
            results.append({"name": j["name"], "rc": 0, "gpu": gpu_id})
        for j in failed:
            results.append({"name": j["name"], "rc": rc or 1, "gpu": gpu_id})


def _cleanup(path: Path, label: str) -> None:
    """删除临时目录,失败只告警不中断。"""
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
        print(f"已清理临时目录 {label}", flush=True)
    except Exception as e:
        print(f"[警告] 清理 {label} 失败,可手动删除 {path} ({e})", flush=True)


def main():
    ap = argparse.ArgumentParser(description="AF3 批量并行推理(不使用 MSA)")
    ap.add_argument("-i", "--input_dir", required=True, help="存放输入 json 的文件夹")
    ap.add_argument("-o", "--output_dir", required=True, help="结果输出文件夹")
    ap.add_argument("-g", "--gpus", default="4,5",
                    help="逗号分隔的物理 GPU 编号,如 4,5;开 --auto-gpu 时本项被忽略")
    ap.add_argument("--auto-gpu", action="store_true", dest="auto_gpu",
                    help="自动检测所有显卡, 按空闲显存挑可用卡(每卡一路 AF3 worker, "
                         "可与别人已有任务共享), 覆盖 --gpus")
    ap.add_argument("--task-vram", type=float, default=20000.0, dest="task_vram",
                    help="单个 AF3 worker 估算显存(MiB), 供 --auto-gpu 判定能否塞进某卡"
                         "(默认 20000;AF3 权重+推理较重, 比 rfdiffusion 高)")
    ap.add_argument("--gpu-reserve", type=float, default=0.10, dest="gpu_reserve",
                    help="--auto-gpu 给别人占用峰值留的冗余比例(默认 0.10)")
    ap.add_argument("--gpu-sample", type=float, default=5.0, dest="gpu_sample",
                    help="--auto-gpu 显存采样窗口秒数(默认 5)")
    ap.add_argument("-f", "--force", action="store_true", help="忽略已完成,全部重跑")
    ap.add_argument("-s", "--num_seeds", type=int, default=DEFAULT_NUM_SEEDS,
                    help=f"每个 sample 的随机种子数量(默认 {DEFAULT_NUM_SEEDS}),"
                         "种子为 [1, 1000000) 内互不重复的随机数。"
                         f"总重复数 = num_seeds × 扩散样本数(后者默认 5)")
    ap.add_argument("-k", "--keep_temp", action="store_true",
                    help="保留临时目录 _jax_cache / _prepared_inputs(默认跑完自动删除)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if args.num_seeds < 1:
        print("[错误] --num_seeds 必须 >= 1", flush=True)
        sys.exit(1)

    # ---- 确定 GPU 列表 ----
    # --auto-gpu: 自动检测, 按空闲显存挑可用卡(每卡一路 AF3 worker, 可与别人共享);
    #             检测不到空闲卡时退回 --gpus 手动列表(保持原行为)。
    # 否则: 用 --gpus 手动列表。
    manual_gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if args.auto_gpu:
        # gpu_scheduler 只依赖标准库 + nvidia-smi, 与本脚本同目录
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from gpu_scheduler import plan_slots
        gpus = plan_slots(task_vram_mib=args.task_vram, reserve=args.gpu_reserve,
                          sample_seconds=args.gpu_sample, max_per_gpu=1,
                          verbose=True)
        if not gpus:
            print(f"[auto-gpu] 没有显存放得下的空闲卡, 退回手动 --gpus 列表: "
                  f"{manual_gpus or ['(空)']}", flush=True)
            gpus = manual_gpus
    else:
        gpus = manual_gpus
    if not gpus:
        print("[错误] 没有可用 GPU(--auto-gpu 未挑到卡且 --gpus 为空)", flush=True)
        sys.exit(1)
    # 所有结构结果统一收进 structures/,与日志、临时目录分开
    results_dir = output_dir / "structures"
    log_dir = output_dir / "logs"
    prepared_dir = output_dir / "_prepared_inputs"
    jax_cache = output_dir / "_jax_cache"
    for d in (output_dir, results_dir, log_dir, jax_cache):
        d.mkdir(parents=True, exist_ok=True)

    print(f"输入: {input_dir}\n输出: {output_dir}\nGPU: {gpus}\n"
          f"每 sample 随机种子数: {args.num_seeds}(× 扩散样本 5 = "
          f"{args.num_seeds * 5} 个重复/sample)\n", flush=True)

    jobs = prepare_inputs(input_dir, prepared_dir, args.num_seeds)
    if not jobs:
        print("未找到任何 json,退出。", flush=True)
        sys.exit(1)

    # 过滤已完成
    pending, skipped = [], []
    for j in jobs:
        if not args.force and is_done(results_dir, j["name"]):
            skipped.append(j["name"])
        else:
            pending.append(j)
    print(f"共 {len(jobs)} 个任务,跳过已完成 {len(skipped)},待跑 {len(pending)}", flush=True)
    if not pending:
        print("没有需要执行的任务。", flush=True)
        return

    # 按 GPU 数量把待跑任务轮转分组,每块 GPU 一个独立子目录。
    # 每块 GPU 只起一个 run_alphafold.py 进程、用 --input_dir 跑完整组,
    # 模型只构建/编译一次,后续任务复用已编译可执行体。
    group_root = output_dir / "_groups"
    _cleanup(group_root, "_groups")  # 清掉上次残留的分组目录,避免混入旧 json
    groups: dict[str, list] = {g: [] for g in gpus}
    for idx, j in enumerate(pending):
        g = gpus[idx % len(gpus)]
        gdir = group_root / f"gpu{g}"
        gdir.mkdir(parents=True, exist_ok=True)
        # 把预处理好的 json 复制进该 GPU 的组目录
        shutil.copy2(j["json"], gdir / Path(j["json"]).name)
        groups[g].append(j)
    for g in gpus:
        print(f"  GPU {g}: 分到 {len(groups[g])} 个任务", flush=True)

    results, lock = [], threading.Lock()
    t0 = time.time()
    threads = [
        threading.Thread(
            target=worker,
            args=(g, group_root / f"gpu{g}", groups[g], results_dir, log_dir,
                  jax_cache, results, lock))
        for g in gpus if groups[g]
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok = sum(1 for r in results if r["rc"] == 0)
    fail = [r for r in results if r["rc"] != 0]
    print(f"\n===== 全部完成,总用时 {time.time()-t0:.1f}s =====", flush=True)
    print(f"成功 {ok} / {len(results)}", flush=True)
    print(f"结构结果在: {results_dir}", flush=True)
    if fail:
        print("失败任务(批处理日志在 logs/gpu<id>.log):", flush=True)
        for r in fail:
            print(f"  - {r['name']} (GPU {r['gpu']} → logs/gpu{r['gpu']}.log)", flush=True)

    # 清理临时目录:_jax_cache / _prepared_inputs / _groups 都不是结果,可重新生成。
    # 有失败任务时保留 _prepared_inputs,方便直接定位/重跑那条输入。
    if not args.keep_temp:
        _cleanup(jax_cache, "_jax_cache")
        _cleanup(group_root, "_groups")
        if fail:
            print(f"检测到失败任务,保留 {prepared_dir.name} 以便排查/重跑。", flush=True)
        else:
            _cleanup(prepared_dir, "_prepared_inputs")


if __name__ == "__main__":
    main()
