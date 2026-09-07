# 🚀 AstroBinder

<div align="center">

**Multi-round protein binder design pipeline** — dock once, then design–score–iterate.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-2ea44f?style=for-the-badge)]()
[![Pipeline](https://img.shields.io/badge/Status-Active-8A2BE2?style=for-the-badge)]()

</div>

---

```
        🌌  AstroBinder — de novo protein binder design
        ═══════════════════════════════════════════════════

 🧬  RIFdock ─▶ RFdiffusion ─▶ ProteinMPNN ─▶ AlphaFold 3 ─▶ Rosetta
        ────────────────────────────────────────────────
        "dock once, then design–score–iterate"

        ┌─────────────────────────────────────────────────────┐
        │  external rifdock input folder                       │
        └──────────────────────┬──────────────────────────────┘
                               │
                               ▼
        ╭──────────────────────────────────────────────╮
        │            · round0_rifdock ·                │  seed docking
        │            (runs once — never recomputed)    │  (RIFgen/RIFdock)
        ╰──────────────────────┬───────────────────────╯
                               │
            ┌──────────────────┴──────────────────┐
            │                                     │
            ▼                                     ▼
   ┌────────────────┐                   ┌────────────────┐
   │  1_rfdiffusion │                   │  1_rfdiffusion │
   │  motif re-     │                   │  motif re-     │
   │  scaffold +    │                   │  scaffold +    │
   │  ProteinMPNN   │                   │  ProteinMPNN   │
   │  seq design    │                   │  seq design    │
   └───────┬────────┘                   └───────┬────────┘
           ▼                                     ▼
   ┌────────────────┐                   ┌────────────────┐
   │    2_af3       │                   │    2_af3       │
   │  AF3 monomer   │                   │  AF3 monomer   │
   │  folding +     │                   │  folding +     │
   │  RIF graft     │                   │  RIF graft     │
   └───────┬────────┘                   └───────┬────────┘
           ▼                                     ▼
   ┌────────────────┐                   ┌────────────────┐
   │   3_rosetta    │                   │   3_rosetta    │
   │  interface     │                   │  interface     │
   │  score &       │                   │  score &       │
   │  filter        │                   │  filter        │
   └───────┬────────┘                   └───────┬────────┘
           │                                     │
           └─────── filtered structures ─────────┘
               become the next round's input 🔄
                               │
                               ▼
                       round3, round4, … ♾️
```

---

## ✨ Features

| | |
|---|---|
| 🔁 **Multi-round loop** | Regenerates scaffolds from the previous round's best binders |
| 🧠 **Resume / step-skip** | `.step_complete.json` markers skip finished steps automatically |
| 🎛️ **Unified config** | One YAML file via deep-merge of per-round overrides |
| 🖥️ **Smart GPU scheduling** | `auto_gpu` places tasks on cards with enough free VRAM |
| 📉 **Adaptive filtering** | Shrinks the Rosetta keep-ratio until the batch fits `--max-next-round-pdbs` |
| 🧪 **Smoke-tested** | `config.smoke.yaml` verifies the whole loop end-to-end |

---

## 🗂️ Repository Layout

```
📦 AstroBinder
├── 📄 run_pipeline.py            CLI entry point
└── 📁 pipeline/
    ├── orchestrator.py           round orchestration, data flow, step-level resume
    ├── config_loader.py          config loading + deep merge
    ├── layout.py                 output directory layout
    ├── config*.yaml              configs (see below)
    ├── 📁 steps/                 one class per step: rifdock / rfdiffusion / af3 / rosetta
    ├── 📁 scripts/               real model calls, structure conversion, graft, scoring
    └── 📄 README.md              detailed Chinese docs (main flow + conventions)

pipeline_runs/                    experiment outputs (created at runtime, not committed)
```

---

## ⚙️ Requirements

- **🐍 Python** ≥ 3.10 with: `PyYAML`, `torch`, `biopython`, `scipy`
- **🧩 External model stacks** (invoked as subprocesses / imports by `pipeline/scripts/`):

| Tool | Role |
|---|---|
| RFdiffusion + ProteinMPNN | scaffold / sequence design |
| AlphaFold 3 (`alphafold3`) | monomer folding + RIF graft |
| Rosetta | interface scoring |
| RIFgen / RIFdock | seed docking |

- **🎮 GPU**: steps auto-schedule across free GPUs by default (see [GPU scheduling](#gpu-scheduling)).

> Binaries/conda paths are overridable per step via config keys such as `rifgen_bin`, `rosetta` flags, etc. — see `pipeline/config.yaml` for the full key list with inline comments.

---

## 📥 Input Folder

`--rifdock-input` points to a folder containing everything the seed step needs:

```
📁 input/
├── 📄 <target>.pdb            target structure (top-level, exactly one *.pdb — auto-detected)
├── 📄 target_res.list         hot-spot residue list
├── 📁 scaffolds/              HHH_bc scaffold PDBs for RIFdock
└── 📄 rifdock.flag            (optional) pre-computed rifgen results → skips re-running rifgen
```

> ⚠️ If the top level has **zero or multiple** `*.pdb`, auto-detection fails — set `seed.rifdock.target` to the explicit relative PDB name instead.

---

## 🚀 Running

```bash
# 🏭 Production run (2 rounds)
python run_pipeline.py \
    --rifdock-input input \
    --config pipeline/config.yaml

# 🧪 End-to-end smoke test (minimal generation, both rounds, ~verifies the loop only)
python run_pipeline.py \
    --rifdock-input input \
    --config pipeline/config.smoke.yaml

# 🔧 Finish Rosetta of round1 only, reusing an existing run
python run_pipeline.py \
    --rifdock-input input \
    --config pipeline/config.rosetta_only.yaml
```

### 🎛️ CLI Options

| Option | Default | Description |
|---|---|---|
| `--rifdock-input` | *(required)* | External rifdock input folder |
| `--config` | `pipeline/config.yaml` | Unified parameter file |
| `--max-next-round-pdbs` | `80` | Max PDBs passed to the next round; when Rosetta output exceeds this, the keep ratio is tightened adaptively |
| `--filter-shrink-step` | `0.05` | How much the keep ratio shrinks per tightening (5 percentage points) |

> 📍 A relative `output_root` in the config is anchored to the **project root**, so outputs land in the same place no matter where you launch from (this keeps resume markers working).

---

## 🧾 Configuration

Available configs in `pipeline/`:

| File | Purpose |
|---|---|
| `config.yaml` | Production parameters |
| `config.smoke.yaml` | Minimal end-to-end smoke test (1 design, 1 seed, keep all) |
| `config.rosetta_only.yaml` | Reuse an existing run, only finish Rosetta |
| `config.IL-13Ra.yaml` | IL-13Ra target run |
| `config.test.yaml`, `config.paramtest.yaml` | Ad-hoc / parameter tests |

Structure (full key list with comments in `pipeline/config.yaml`):

```yaml
experiment:            # 🌍 global: target, output_root, rounds, run_name
seed:                  # 🌱 round0_rifdock parameters (paths relative to --rifdock-input)
  rifdock: {...}
defaults:              # 🧬 per-step defaults used by every round
  rfdiffusion: {...}
  af3: {...}
  rosetta: {...}
rounds:                # 🎯 optional per-round overrides, deep-merged over defaults
  2:
    rosetta: {keep: 0.5}
```

📐 Step parameters for a given round = `defaults[step]` **deep-merged** with `rounds[round][step]`.

---

## 📂 Output Layout

```
📁 pipeline_runs/
└── 📁 run_<YYYYMMDD>_<target-slug>/
    ├── 📁 round0_rifdock/output/        seed docking results → round1 input
    ├── 📁 round1/
    │   ├── 📁 1_rfdiffusion/
    │   ├── 📁 2_af3/
    │   └── 📁 3_rosetta/filtered_structures/
    └── 📁 round2/...
```

> ℹ️ `run_name` defaults to `run_<date>_<target>`. Non-ASCII (e.g. Greek letters) is **slugified** — RFdiffusion's Hydra parser rejects non-ASCII paths.

---

## 💾 Resume / Step-Skipping

- ✅ Each step writes `.step_complete.json` in its work dir on success.
- 🔁 Re-running with the **same `run_name`** skips already-completed steps (e.g. rifdock runs once and is never recomputed).
- 🔄 After changing models, code, or a step's parameters: use a **new `run_name`**, or set that step's `force: true` to recompute it.

---

## 🖥️ GPU Scheduling

GPU-heavy steps (`rfdiffusion`, `af3`) support `auto_gpu: true`: all GPUs are scanned and tasks are placed on cards with enough free VRAM (`task_vram` MiB estimate, minus `gpu_reserve` fraction), so they can share cards with other users' jobs.

```
🏴‍☠️  auto_gpu: true   →  scan & auto-place across free VRAM
🔒  auto_gpu: false  +  gpu: "0,1"  →  pin cards manually
```

---

## 📉 Adaptive Filtering

If a Rosetta step produces more than `--max-next-round-pdbs` structures, its keep ratio is reduced by `--filter-shrink-step` and re-scored until the output fits, so the next round always gets a **manageable batch**.

```
rosetta output > max-next-round-pdbs ?
        │ yes
        ▼
  reduce keep ratio by --filter-shrink-step 🔽
        │
        ▼
  re-score until it fits ✅  →  pass to next round
```

---

<div align="center">

Made with 🧬 by the AstroBinder team — design better binders, faster.

</div>
