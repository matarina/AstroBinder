"""配置加载与合并。

统一参数文件 config.yaml 的结构(详见该文件内注释):

    experiment:        # 全局实验设置
      target: ...
      output_root: ...
      rounds: 2
    seed:              # round0_rifdock 的参数
      rifdock: {...}
    defaults:          # 每个循环步骤的默认参数
      rfdiffusion: {...}
      af3: {...}
      rosetta: {...}
    rounds:            # 按轮号覆盖默认参数(可选,缺省则用 defaults)
      1:
        rfdiffusion: {...}   # 只写要改的键,深合并进 defaults
      2:
        rosetta: {...}

取参数时:某轮某步的参数 = defaults[step] 深合并 rounds[round][step]。
"""

from __future__ import annotations

import copy
from pathlib import Path


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并 override 进 base 的副本;override 的标量/列表直接覆盖。"""
    result = copy.deepcopy(base) if base else {}
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


class PipelineConfig:
    """加载后的配置访问器。"""

    def __init__(self, raw: dict):
        self.raw = raw or {}
        self.experiment = self.raw.get("experiment", {})
        self._seed = self.raw.get("seed", {})
        self._defaults = self.raw.get("defaults", {})
        self._rounds = self.raw.get("rounds", {})

    @classmethod
    def load(cls, path) -> "PipelineConfig":
        import yaml  # 延迟导入,框架不强依赖

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(raw)

    # ---- 全局 ----

    @property
    def target(self):
        """靶点名。留空则返回 None,由编排器根据输入 PDB 自动推断。"""
        return self.experiment.get("target")

    @property
    def output_root(self) -> str:
        return self.experiment.get("output_root", "pipeline_runs")

    @property
    def rounds(self) -> int:
        return int(self.experiment.get("rounds", 1))

    @property
    def run_name(self):
        return self.experiment.get("run_name")  # 可为 None,布局会自动生成

    # ---- 取某步参数 ----

    def seed_params(self, step: str) -> dict:
        """种子步骤(round0)参数。"""
        return copy.deepcopy(self._seed.get(step, {}))

    def step_params(self, round_index: int, step: str) -> dict:
        """某轮某步参数 = defaults[step] 深合并 rounds[round][step]。"""
        base = self._defaults.get(step, {})
        # yaml 的轮号键可能是 int 或 str,两种都兜住
        override = {}
        for k in (round_index, str(round_index)):
            if k in self._rounds:
                override = self._rounds[k].get(step, {})
                break
        return _deep_merge(base, override)
