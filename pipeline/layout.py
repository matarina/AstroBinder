"""目录布局管理。

负责把一次实验的输出目录结构计算出来并按需创建。结构形如:

    <output_root>/
    └── run_<日期>_<靶点>/              # 一次完整实验
        ├── round0_rifdock/            # 种子,只跑一次
        │   └── output/
        ├── round1/
        │   ├── 1_rfdiffusion/
        │   ├── 2_af3/
        │   └── 3_rosetta/
        └── round2/
            └── ...

只管「路径在哪、怎么建」,不掺杂任何算法逻辑。
"""

from __future__ import annotations

import datetime as _dt
import unicodedata
from dataclasses import dataclass
from pathlib import Path


# 常见希腊字母 -> 拉丁转写(靶点名如 IL-4Rα / TNF-α 很常见)。
# RFdiffusion 依赖的 Hydra 命令行解析器不接受路径中的非 ASCII 字符,
# 故凡进入文件路径的靶点名都必须先 ASCII 化,否则 run_inference.py 直接报
# LexerNoViableAltException。
_GREEK_MAP = {
    "α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "e",
    "θ": "th", "ι": "i", "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x",
    "ο": "o", "π": "p", "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "u",
    "φ": "ph", "χ": "ch", "ψ": "ps", "ω": "o",
    "Α": "A", "Β": "B", "Γ": "G", "Δ": "D", "Ε": "E", "Ζ": "Z", "Η": "E",
    "Θ": "Th", "Ι": "I", "Κ": "K", "Λ": "L", "Μ": "M", "Ν": "N", "Ξ": "X",
    "Ο": "O", "Π": "P", "Ρ": "R", "Σ": "S", "Τ": "T", "Υ": "U", "Φ": "Ph",
    "Χ": "Ch", "Ψ": "Ps", "Ω": "O",
}


def slugify_target(name: str) -> str:
    """把靶点名转成仅含 ASCII 字母数字与 -_. 的 slug,供 run 目录命名。

    - 希腊字母按 _GREEK_MAP 转写(α→a);
    - 其余非 ASCII 先 NFKD 去重音再取 ASCII 字母数字,取不到则丢弃;
    - 空格等非法字符丢弃。
    结果为空时回退 'target'。
    """
    if not name:
        return "target"
    out = []
    for ch in name:
        if ch in _GREEK_MAP:
            out.append(_GREEK_MAP[ch])
        elif ch.isascii() and (ch.isalnum() or ch in "-_."):
            out.append(ch)
        else:
            conv = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()
            out.append("".join(c for c in conv if c.isalnum()))
    return "".join(out) or "target"


# 循环内的步骤目录名(序号前缀决定顺序,后续要加步骤就往这里加)
ROUND_STEP_DIRS = {
    "rfdiffusion": "1_rfdiffusion",
    "af3": "2_af3",
    "rosetta": "3_rosetta",
}

# 种子步骤(round0,只跑一次)
SEED_STEP = "rifdock"
SEED_DIR = "round0_rifdock"


@dataclass
class RunLayout:
    """一次实验的所有路径。"""

    output_root: Path
    target: str
    run_name: str

    @classmethod
    def create(cls, output_root, target, run_name=None, date=None):
        """构造布局。run_name 为空时自动生成 run_<日期>_<靶点slug>。

        无论自动生成还是显式传入,run_name 都经 slugify_target 处理成 ASCII
        安全形式(路径里的希腊字母/非 ASCII 会让 RFdiffusion 的 Hydra 解析失败)。
        target 字段保留原名(仅供显示/传给各步骤当标签)。
        """
        slug = slugify_target(target)
        if run_name is None:
            day = (date or _dt.date.today()).strftime("%Y%m%d")
            run_name = f"run_{day}_{slug}"
        else:
            run_name = slugify_target(run_name)
        return cls(output_root=Path(output_root), target=target, run_name=run_name)

    # ---- 路径计算(不创建) ----

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_name

    def seed_dir(self) -> Path:
        """round0_rifdock/ 根目录。"""
        return self.run_dir / SEED_DIR

    def seed_output_dir(self) -> Path:
        """round0_rifdock/output/ —— 种子产物,作为 round1 的输入。"""
        return self.seed_dir() / "output"

    def round_dir(self, round_index: int) -> Path:
        """roundN/ 根目录。"""
        return self.run_dir / f"round{round_index}"

    def step_dir(self, round_index: int, step: str) -> Path:
        """roundN/<序号>_<步骤>/ 目录。"""
        if step not in ROUND_STEP_DIRS:
            raise KeyError(f"未知的循环步骤: {step!r},可选: {list(ROUND_STEP_DIRS)}")
        return self.round_dir(round_index) / ROUND_STEP_DIRS[step]

    # ---- 创建 ----

    def make_run_dir(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def make_seed_dirs(self) -> Path:
        out = self.seed_output_dir()
        out.mkdir(parents=True, exist_ok=True)
        return out

    def make_step_dir(self, round_index: int, step: str) -> Path:
        d = self.step_dir(round_index, step)
        d.mkdir(parents=True, exist_ok=True)
        return d
