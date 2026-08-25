"""编排器:串接种子轮 + 多轮循环,管理步骤间数据流。

数据流:

    外部 rifdock 输入文件夹
        │
        ▼
    round0_rifdock/output/              (RifdockStep)
        │
        ▼
    round1/1_rfdiffusion ─► 2_af3 ─► 3_rosetta/filtered_structures
        │                                              │
        │   (rosetta 筛选结果作为下一轮 rfdiffusion 输入)│
        ▼                                              ▼
    round2/1_rfdiffusion ─► 2_af3 ─► 3_rosetta/filtered_structures
        ...

编排器只负责「建目录、定输入、按顺序调 run()、把产物接到下一步」,
不掺杂任何算法。每步的真实逻辑由对应 Step 子类填充。
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from .config_loader import PipelineConfig
from .layout import RunLayout, ROUND_STEP_DIRS
from .steps.base import StepContext, StepResult
from .steps.rifdock import RifdockStep
from .steps.rfdiffusion import RFdiffusionStep
from .steps.af3 import AF3Step
from .steps.rosetta import RosettaStep


# 每个步骤完成后在其 work_dir 写的标记文件名
STEP_DONE_FILE = ".step_complete.json"


# 循环内步骤的执行顺序(与 layout.ROUND_STEP_DIRS 对应)
ROUND_STEP_ORDER = ["rfdiffusion", "af3", "rosetta"]

STEP_CLASSES = {
    "rifdock": RifdockStep,
    "rfdiffusion": RFdiffusionStep,
    "af3": AF3Step,
    "rosetta": RosettaStep,
}


def detect_target_pdb(input_dir: Path, scaffold_subdir: str = "scaffolds") -> Path:
    """自动定位输入文件夹里的靶点 PDB。

    只看输入文件夹顶层的 *.pdb(scaffolds 在子目录里,不会被误选)。
    恰好一个则用它;零个或多个都无法自动确定,报错并提示手动指定。
    """
    input_dir = Path(input_dir)
    scaffold_path = input_dir / scaffold_subdir
    pdbs = sorted(
        p for p in input_dir.glob("*.pdb")
        if p.is_file() and p.parent != scaffold_path
    )
    if not pdbs:
        raise FileNotFoundError(
            f"输入文件夹顶层未找到靶点 PDB(*.pdb): {input_dir}"
        )
    if len(pdbs) > 1:
        names = [p.name for p in pdbs]
        raise ValueError(
            f"输入文件夹顶层存在多个 *.pdb,无法自动确定靶点: {names};"
            f"请在 config 的 seed.rifdock.target 手动指定"
        )
    return pdbs[0]


class Orchestrator:
    def __init__(self, config: PipelineConfig, rifdock_input: Path | None,
                 max_next_round_pdbs: int = 80,
                 filter_shrink_step: float = 0.05,
                 precomputed_seed: bool = False):
        self.config = config
        self.rifdock_input = Path(rifdock_input) if rifdock_input is not None else None
        self.max_next_round_pdbs = max_next_round_pdbs
        self.filter_shrink_step = filter_shrink_step
        self.precomputed_seed = precomputed_seed

        # ---- 自动确定靶点 PDB 与靶点名 ----
        if precomputed_seed:
            if not config.target:
                raise ValueError("precomputed_seed 模式要求 experiment.target 非空")
            self.target_pdb_name = None
            self.target_name = config.target
            print(f"[target] 复用预计算 RIFdock seed  靶点名 = {self.target_name}")
        else:
            if self.rifdock_input is None:
                raise ValueError("非 precomputed_seed 模式必须提供 rifdock_input")
            rifdock_params = config.seed_params("rifdock")
            explicit_target = rifdock_params.get("target")
            if explicit_target:
                # config 里显式写了靶点 PDB(相对名):以它为准,向后兼容
                target_pdb = self.rifdock_input / explicit_target
                if not target_pdb.exists():
                    raise FileNotFoundError(
                        f"config 指定的靶点 PDB 不存在: {target_pdb}"
                    )
            else:
                # 未指定:根据输入文件夹内容自动推断
                scaffold_subdir = rifdock_params.get("scaffold_subdir", "scaffolds")
                target_pdb = detect_target_pdb(self.rifdock_input, scaffold_subdir)

            self.target_pdb_name = target_pdb.name  # 传给 rifdock step 的相对名
            # 靶点名:experiment.target 显式指定则优先,否则用 PDB 文件名(去扩展名)
            self.target_name = config.target or target_pdb.stem
            print(f"[target] 靶点 PDB = {target_pdb.name}  靶点名 = {self.target_name}")

        self.layout = RunLayout.create(
            output_root=config.output_root,
            target=self.target_name,
            run_name=config.run_name,
        )

    def run(self):
        self.layout.make_run_dir()
        print(f"[run] {self.layout.run_dir}")

        # ---- round0:种子 rifdock(只跑一次) ----
        seed_out = self._run_seed()

        # ---- round1..N:循环 ----
        prev_output = seed_out
        for r in range(1, self.config.rounds + 1):
            prev_output = self._run_round(r, prev_output)

        print("[done]")
        return self.layout.run_dir

    # ---- 步骤级完成标记 / 续跑 ----

    def _run_step(self, step_name: str, ctx: StepContext, force: bool) -> StepResult:
        """跑一个步骤,带完成标记与续跑。

        - 完成后在 ctx.work_dir 写 .step_complete.json(output_dir / info / 时间)。
        - 未开 force 且标记存在:直接读标记重建 StepResult 跳过重算。
        - 校验标记里的 output_dir 仍存在,否则视为失效重算(防止产物被清理)。
        """
        marker = ctx.work_dir / STEP_DONE_FILE
        if not force and marker.is_file():
            try:
                data = json.loads(marker.read_text(encoding="utf-8"))
                out = Path(data["output_dir"])
                if out.exists():
                    print(f"      [跳过] 已完成(标记存在): {marker}")
                    return StepResult(output_dir=out, info=data.get("info", {}))
                print(f"      [重算] 标记存在但产物已丢失: {out}")
            except (ValueError, KeyError) as e:
                print(f"      [重算] 完成标记损坏({e}),重新计算")

        result = STEP_CLASSES[step_name](ctx.params).run(ctx)

        marker.write_text(json.dumps({
            "step": step_name,
            "round": ctx.round_index,
            "output_dir": str(result.output_dir),
            "info": result.info,
            "completed_at": _dt.datetime.now().isoformat(timespec="seconds"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    # ---- 种子 ----

    def _run_seed(self) -> Path:
        step_name = "rifdock"
        work_dir = self.layout.make_seed_dirs()
        params = self.config.seed_params(step_name)
        if self.target_pdb_name is not None:
            # 注入自动确定的靶点 PDB(相对名),覆盖/补齐 config 里的 target
            params["target"] = self.target_pdb_name
        if self.precomputed_seed:
            if params.get("force"):
                raise ValueError("precomputed_seed 模式禁止 seed.rifdock.force=true")
            marker = work_dir / STEP_DONE_FILE
            if not marker.is_file():
                raise FileNotFoundError(f"预计算 RIFdock seed 缺少完成标记: {marker}")
            try:
                marker_data = json.loads(marker.read_text(encoding="utf-8"))
                marked_output = Path(marker_data["output_dir"])
            except (ValueError, KeyError) as exc:
                raise RuntimeError(f"预计算 RIFdock seed 标记损坏: {marker}") from exc
            if not marked_output.exists():
                raise FileNotFoundError(
                    f"预计算 RIFdock seed 标记指向的产物不存在: {marked_output}"
                )
        ctx = StepContext(
            round_index=0,
            step_name=step_name,
            work_dir=work_dir,
            input_dir=(self.rifdock_input or Path("__precomputed_seed_no_input__")),
            params=params,
            target=self.target_name,
        )
        print(f"  [round0] {step_name} -> {work_dir}")
        result = self._run_step(step_name, ctx, force=bool(params.get("force")))
        return result.output_dir

    # ---- 单轮循环 ----

    def _run_round(self, round_index: int, round_input: Path) -> Path:
        print(f"  [round{round_index}]")
        prev_output = round_input
        for step_name in ROUND_STEP_ORDER:
            work_dir = self.layout.make_step_dir(round_index, step_name)
            params = self.config.step_params(round_index, step_name)
            # Rosetta 产物会直接成为下一轮的 PDB 输入。为避免默认
            # keep 比例在大批量数据上留下过多结构，非最后一轮启用
            # 自适应上限；最后一轮不需要再向后传递，保留原筛选设置。
            if step_name == "rosetta" and round_index < self.config.rounds:
                params["max_pdbs"] = self.max_next_round_pdbs
                params["keep_step"] = self.filter_shrink_step
            ctx = StepContext(
                round_index=round_index,
                step_name=step_name,
                work_dir=work_dir,
                input_dir=prev_output,
                params=params,
                target=self.target_name,
            )
            print(f"    {ROUND_STEP_DIRS[step_name]} <- {prev_output}")
            result = self._run_step(step_name, ctx, force=bool(params.get("force")))
            prev_output = result.output_dir
        # 本轮最后一步(rosetta)的产物 = 下一轮的输入
        return prev_output
