# AstroBinder

Multi-round protein binder design pipeline. It chains **RIFdock → RFdiffusion → ProteinMPNN → AlphaFold 3 → Rosetta** into a "dock once, then design–score–iterate" loop: each round regenerates scaffolds from the previous round's best binders, and Rosetta filtering decides which structures advance.

```text
external rifdock input folder
        │
        ▼
round0_rifdock            (seed docking, runs once)
        │
        ▼
round1:  1_rfdiffusion ─► 2_af3 ─► 3_rosetta ─┐
        │   (RFdiffusion motif       (AF3 monomer     (interface
        │    re-scaffolding +          folding +        score & filter)
        │    ProteinMPNN seq          RIF graft)
        │    design)
        │
        │  filtered structures become next round's input
        ▼
round2:  1_rfdiffusion ─► 2_af3 ─► 3_rosetta ─► ...
```

## Repository layout

```text
run_pipeline.py            CLI entry point
pipeline/
├── orchestrator.py        round orchestration, data flow, step-level resume
├── config_loader.py       config loading + deep merge
├── layout.py              output directory layout
├── config*.yaml           configs (see below)
├── steps/                 one class per step: rifdock / rfdiffusion / af3 / rosetta
├── scripts/               real model calls, structure conversion, graft, scoring
└── README.md              detailed Chinese docs (main flow + conventions)
pipeline_runs/             experiment outputs (created at runtime, not committed)
```

## Requirements

- **Python** ≥ 3.10 with: `PyYAML`, `torch`, `biopython`, `scipy`
- **External model stacks** (invoked as subprocesses / imports by `pipeline/scripts/`):
  - RFdiffusion + ProteinMPNN
  - AlphaFold 3 (`alphafold3` package)
  - Rosetta (interface scoring binary)
  - RIFgen / RIFdock (docking binaries)
- **GPU**: steps auto-schedule across free GPUs by default (see [GPU scheduling](#gpu-scheduling)).

Binaries/conda paths are overridable per step via config keys such as `rifgen_bin`, `rosetta` flags, etc. — see `pipeline/config.yaml` for the full key list with inline comments.

## Input folder

`--rifdock-input` points to a folder containing everything the seed step needs:

```text
input/
├── <target>.pdb            target structure (top-level, exactly one *.pdb — auto-detected)
├── target_res.list         hot-spot residue list
├── scaffolds/              HHH_bc scaffold PDBs for RIFdock
└── rifdock.flag            (optional) pre-computed rifgen results → skips re-running rifgen
```

If the top level has zero or multiple `*.pdb`, auto-detection fails — set `seed.rifdock.target` to the explicit relative PDB name instead.

## Running

```bash
# Production run (2 rounds)
python run_pipeline.py \
    --rifdock-input input \
    --config pipeline/config.yaml

# End-to-end smoke test (minimal generation, both rounds, ~verifies the loop only)
python run_pipeline.py \
    --rifdock-input input \
    --config pipeline/config.smoke.yaml

# Finish Rosetta of round1 only, reusing an existing run
python run_pipeline.py \
    --rifdock-input input \
    --config pipeline/config.rosetta_only.yaml
```

### CLI options

| Option | Default | Description |
|---|---|---|
| `--rifdock-input` | (required) | External rifdock input folder |
| `--config` | `pipeline/config.yaml` | Unified parameter file |
| `--max-next-round-pdbs` | `80` | Max PDBs passed to the next round; when Rosetta output exceeds this, the keep ratio is tightened adaptively |
| `--filter-shrink-step` | `0.05` | How much the keep ratio shrinks per tightening (5 percentage points) |

A relative `output_root` in the config is anchored to the project root, so outputs land in the same place no matter where you launch from (this keeps resume markers working).

## Configuration

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
experiment:            # global: target, output_root, rounds, run_name
seed:                  # round0_rifdock parameters (paths relative to --rifdock-input)
  rifdock: {...}
defaults:              # per-step defaults used by every round
  rfdiffusion: {...}
  af3: {...}
  rosetta: {...}
rounds:                # optional per-round overrides, deep-merged over defaults
  2:
    rosetta: {keep: 0.5}
```

Step parameters for a given round = `defaults[step]` deep-merged with `rounds[round][step]`.

## Output layout

```text
pipeline_runs/
└── run_<YYYYMMDD>_<target-slug>/
    ├── round0_rifdock/output/        seed docking results → round1 input
    ├── round1/
    │   ├── 1_rfdiffusion/
    │   ├── 2_af3/
    │   └── 3_rosetta/filtered_structures/
    └── round2/...
```

`run_name` defaults to `run_<date>_<target>` (non-ASCII, e.g. Greek letters, is slugified — RFdiffusion's Hydra parser rejects non-ASCII paths).

## Resume / step skipping

- Each step writes `.step_complete.json` in its work dir on success.
- Re-running with the **same `run_name`** skips already-completed steps (e.g. rifdock runs once and is never recomputed).
- After changing models, code, or a step's parameters: use a **new `run_name`**, or set that step's `force: true` to recompute it.

## GPU scheduling

GPU-heavy steps (`rfdiffusion`, `af3`) support `auto_gpu: true`: all GPUs are scanned and tasks are placed on cards with enough free VRAM (`task_vram` MiB estimate, minus `gpu_reserve` fraction), so they can share cards with other users' jobs. Set `auto_gpu: false` and `gpu: "0,1"` to pin cards manually.

## Adaptive filtering

If a Rosetta step produces more than `--max-next-round-pdbs` structures, its keep ratio is reduced by `--filter-shrink-step` and re-scored until the output fits, so the next round always gets a manageable batch.
