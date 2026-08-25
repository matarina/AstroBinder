"""步骤基类与上下文。

每个步骤(rifdock / rfdiffusion / af3 / rosetta)都继承 Step,
实现 run()。框架只规定接口和数据流,真实命令由你后续在 run() 里填充。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class StepContext:
    """传给每个步骤 run() 的上下文。"""

    round_index: int            # 0 表示种子轮(rifdock),>=1 为循环轮
    step_name: str              # 'rifdock' / 'rfdiffusion' / ...
    work_dir: Path              # 本步骤的输出目录(已创建)
    input_dir: Optional[Path]   # 上游产物目录;round0 时为外部 rifdock 输入文件夹
    params: dict                # 本步骤的特异性参数(已合并 defaults+override)
    target: str                 # 靶点名
    extras: dict = field(default_factory=dict)  # 预留:跨步骤传递的额外信息


@dataclass
class StepResult:
    """每个步骤 run() 的返回值,告诉编排器产物在哪、是否成功。"""

    output_dir: Path            # 下游步骤要读取的产物目录
    ok: bool = True
    info: dict = field(default_factory=dict)


class Step:
    """所有步骤的基类。"""

    name: str = "base"

    def __init__(self, params: dict):
        self.params = params or {}

    def run(self, ctx: StepContext) -> StepResult:
        """执行本步骤。子类必须重写。

        约定:
          - 读取 ctx.input_dir 作为输入
          - 把产物写到 ctx.work_dir(或其子目录)
          - 返回 StepResult,output_dir 指向供下游读取的目录
        """
        raise NotImplementedError(f"步骤 {self.name!r} 尚未实现 run()")
