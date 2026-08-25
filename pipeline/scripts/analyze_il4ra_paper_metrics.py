#!/usr/bin/env python3
"""Paper-inspired extended analysis for the IL-4Ra binder benchmark.

The script analyzes the exact PDB cohorts already used by
``IL-4Ra_benchmark_report.md``:

* pipeline round1 (Rosetta-scored full cohort)
* pipeline round2 (Rosetta-scored full cohort)
* BindCraft accepted designs (Rosetta-scored N=80 cohort)

It adds metrics that can be reproduced from the existing PDB/CSV files:

* DSSP secondary-structure composition and an explicitly defined fold class
* sequence uniqueness, pairwise identity, and 80%-identity clusters
* atomic/residue contacts, PRODIGY-style CC/CP/CA/PP/PA/AA contacts
* salt bridges, hydrophobic contacts, pi-pi and cation-pi geometry proxies
* IL-4Ra common/configured hotspot coverage
* inter-chain clashes (<2.2 A), pre/post-relaxation RMSD, contact retention
* input PDB CA B-factor confidence, sequence charge/hydrophobicity proxies
* per-residue Rosetta cross-interface energy concentration

iPSAE/ipTM/pTM require predictor PAE/confidence outputs and DeltaForge is
proprietary; this script deliberately does not fabricate those quantities.

Run with the BindCraft environment, which supplies pandas/scipy/Biopython and
matplotlib::

    /mnt/data2/wtk/software/miniconda3/envs/BindCraft/bin/python \
      /mnt/data2/wtk/pipeline/pipeline/scripts/analyze_il4ra_paper_metrics.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.spatial import cKDTree


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

# PRODIGY-style residue chemistry classes.
RESIDUE_CLASS = {
    **{aa: "C" for aa in ("ARG", "LYS", "ASP", "GLU")},
    **{aa: "P" for aa in ("CYS", "HIS", "HID", "HIE", "HIP", "ASN", "GLN", "SER", "THR", "TYR")},
    **{aa: "A" for aa in ("ALA", "PHE", "GLY", "ILE", "LEU", "MET", "MSE", "PRO", "TRP", "VAL")},
}
CONTACT_CLASS_ORDER = ("CC", "CP", "CA", "PP", "PA", "AA")

HYDROPHOBIC_AA1 = frozenset("AVILMFWYC")
CHARGED_AA1 = frozenset("DEKR")

ACID_ATOMS = {
    "ASP": frozenset(("OD1", "OD2")),
    "GLU": frozenset(("OE1", "OE2")),
}
BASE_ATOMS = {
    "LYS": frozenset(("NZ",)),
    "ARG": frozenset(("NE", "NH1", "NH2")),
}

RING_ATOMS = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TRP": ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "HID": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "HIE": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "HIP": ("CG", "ND1", "CD2", "CE1", "NE2"),
}

CATION_ATOMS = {
    "LYS": ("NZ",),
    "ARG": ("CZ",),
}

POLAR_ELEMENTS = frozenset(("N", "O", "S"))
WATER_NAMES = frozenset(("HOH", "WAT", "H2O"))

PIPELINE_HOTSPOTS = frozenset((13, 39, 41, 43, 67, 69, 70, 71, 72, 121, 123, 127, 173, 174))
BINDCRAFT_HOTSPOTS = frozenset((13, 39, 41, 43, 67, 69, 70, 71, 72, 125, 127, 131, 182, 183))
COMMON_HOTSPOTS = PIPELINE_HOTSPOTS & BINDCRAFT_HOTSPOTS


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    label: str
    score_dir: Path
    configured_hotspots: frozenset[int]

    @property
    def input_dir(self) -> Path:
        return self.score_dir / "pdb_inputs"

    @property
    def result_dir(self) -> Path:
        return self.score_dir / "results_relax"

    @property
    def binding_csv(self) -> Path:
        return self.result_dir / "binding_summary.csv"

    @property
    def residues_csv(self) -> Path:
        return self.result_dir / "interface_residues.csv"


@dataclass
class AtomTable:
    coords: np.ndarray
    chains: np.ndarray
    res_indices: np.ndarray
    resseqs: np.ndarray
    resnames: np.ndarray
    atomnames: np.ndarray
    elements: np.ndarray
    bfactors: np.ndarray
    residue_keys: list[tuple[str, int, str]]
    residue_names: list[str]
    residue_atoms: dict[tuple[str, int, str], dict[str, np.ndarray]]
    residue_order: dict[str, list[tuple[str, int, str]]]
    ca_coords: dict[tuple[str, int, str], np.ndarray]
    ca_bfactors: dict[tuple[str, int, str], float]
    water_oxygens: np.ndarray


def _infer_element(atom_name: str) -> str:
    stripped = atom_name.strip()
    while stripped and stripped[0].isdigit():
        stripped = stripped[1:]
    if not stripped:
        return ""
    if stripped[:2].upper() in {"CL", "BR"}:
        return stripped[:2].upper()
    return stripped[0].upper()


def parse_pdb(path: Path) -> AtomTable:
    coords: list[tuple[float, float, float]] = []
    chains: list[str] = []
    res_indices: list[int] = []
    resseqs: list[int] = []
    resnames: list[str] = []
    atomnames: list[str] = []
    elements: list[str] = []
    bfactors: list[float] = []
    residue_keys: list[tuple[str, int, str]] = []
    residue_names: list[str] = []
    residue_atoms: dict[tuple[str, int, str], dict[str, np.ndarray]] = defaultdict(dict)
    residue_order: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    residue_index_by_key: dict[tuple[str, int, str], int] = {}
    ca_coords: dict[tuple[str, int, str], np.ndarray] = {}
    ca_bfactors: dict[tuple[str, int, str], float] = {}
    water_oxygens: list[tuple[float, float, float]] = []
    seen_atoms: set[tuple[str, int, str, str]] = set()

    with path.open(errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()
            if record not in {"ATOM", "HETATM"}:
                continue
            altloc = line[16:17]
            if altloc not in {" ", "A", "1"}:
                continue
            atom_name = line[12:16].strip()
            resname = line[17:20].strip().upper()
            chain = line[21:22].strip() or "_"
            try:
                resseq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                bfactor = float(line[60:66]) if line[60:66].strip() else math.nan
            except ValueError:
                continue
            icode = line[26:27].strip()
            element = line[76:78].strip().upper() or _infer_element(atom_name)
            coord = np.asarray((x, y, z), dtype=np.float64)

            if resname in WATER_NAMES:
                if element == "O" or atom_name in {"O", "OW"}:
                    water_oxygens.append((x, y, z))
                continue
            if resname not in AA3_TO_1:
                continue

            atom_key = (chain, resseq, icode, atom_name)
            if atom_key in seen_atoms:
                continue
            seen_atoms.add(atom_key)
            residue_key = (chain, resseq, icode)
            if residue_key not in residue_index_by_key:
                residue_index_by_key[residue_key] = len(residue_keys)
                residue_keys.append(residue_key)
                residue_names.append(resname)
                residue_order[chain].append(residue_key)
            residue_index = residue_index_by_key[residue_key]

            coords.append((x, y, z))
            chains.append(chain)
            res_indices.append(residue_index)
            resseqs.append(resseq)
            resnames.append(resname)
            atomnames.append(atom_name)
            elements.append(element)
            bfactors.append(bfactor)
            residue_atoms[residue_key][atom_name] = coord
            if atom_name == "CA":
                ca_coords[residue_key] = coord
                ca_bfactors[residue_key] = bfactor

    if not coords:
        raise ValueError(f"No standard protein atoms parsed from {path}")

    return AtomTable(
        coords=np.asarray(coords, dtype=np.float64),
        chains=np.asarray(chains, dtype="U2"),
        res_indices=np.asarray(res_indices, dtype=np.int32),
        resseqs=np.asarray(resseqs, dtype=np.int32),
        resnames=np.asarray(resnames, dtype="U3"),
        atomnames=np.asarray(atomnames, dtype="U4"),
        elements=np.asarray(elements, dtype="U2"),
        bfactors=np.asarray(bfactors, dtype=np.float64),
        residue_keys=residue_keys,
        residue_names=residue_names,
        residue_atoms=dict(residue_atoms),
        residue_order=dict(residue_order),
        ca_coords=ca_coords,
        ca_bfactors=ca_bfactors,
        water_oxygens=np.asarray(water_oxygens, dtype=np.float64).reshape((-1, 3)),
    )


def chain_sequence(table: AtomTable, chain: str = "A") -> str:
    seq = []
    for key in table.residue_order.get(chain, []):
        residue_index = table.residue_keys.index(key)
        seq.append(AA3_TO_1.get(table.residue_names[residue_index], "X"))
    return "".join(seq)


def _resname_for_key(table: AtomTable, key: tuple[str, int, str]) -> str:
    idx = table.residue_keys.index(key)
    return table.residue_names[idx]


def _heavy_chain_indices(table: AtomTable, chain: str) -> np.ndarray:
    return np.flatnonzero((table.chains == chain) & (table.elements != "H"))


def _pair_class(class_a: str, class_b: str) -> str:
    if class_a == class_b:
        return class_a + class_b
    order = {"C": 0, "P": 1, "A": 2}
    return class_a + class_b if order[class_a] < order[class_b] else class_b + class_a


def _ring_geometries(table: AtomTable, chain: str) -> list[tuple[tuple[str, int, str], np.ndarray, np.ndarray]]:
    rings = []
    for key in table.residue_order.get(chain, []):
        resname = _resname_for_key(table, key)
        wanted = RING_ATOMS.get(resname)
        if not wanted:
            continue
        atoms = table.residue_atoms.get(key, {})
        points = np.asarray([atoms[name] for name in wanted if name in atoms], dtype=np.float64)
        if len(points) < 3:
            continue
        centroid = points.mean(axis=0)
        _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
        normal = vh[-1]
        norm = np.linalg.norm(normal)
        if norm == 0:
            continue
        rings.append((key, centroid, normal / norm))
    return rings


def _cation_centers(table: AtomTable, chain: str) -> list[tuple[tuple[str, int, str], np.ndarray]]:
    centers = []
    for key in table.residue_order.get(chain, []):
        resname = _resname_for_key(table, key)
        wanted = CATION_ATOMS.get(resname)
        if not wanted:
            continue
        atoms = table.residue_atoms.get(key, {})
        points = [atoms[name] for name in wanted if name in atoms]
        if points:
            centers.append((key, np.asarray(points, dtype=np.float64).mean(axis=0)))
    return centers


def _pi_interactions(table: AtomTable) -> tuple[int, int]:
    rings_a = _ring_geometries(table, "A")
    rings_b = _ring_geometries(table, "B")
    pi_pairs: set[tuple[tuple[str, int, str], tuple[str, int, str]]] = set()
    for key_a, center_a, normal_a in rings_a:
        for key_b, center_b, normal_b in rings_b:
            distance = float(np.linalg.norm(center_a - center_b))
            if distance > 6.0:
                continue
            cos_angle = float(np.clip(abs(np.dot(normal_a, normal_b)), 0.0, 1.0))
            angle = math.degrees(math.acos(cos_angle))
            # Paper-inspired geometry proxy: face-to-face or edge-to-face stacking.
            if angle <= 35.0 or (angle >= 55.0 and distance <= 5.5):
                pi_pairs.add((key_a, key_b))

    cations_a = _cation_centers(table, "A")
    cations_b = _cation_centers(table, "B")
    cation_pi_pairs: set[tuple[tuple[str, int, str], tuple[str, int, str]]] = set()
    for cation_key, cation in cations_a:
        for ring_key, center, _ in rings_b:
            if np.linalg.norm(cation - center) <= 6.0:
                cation_pi_pairs.add((cation_key, ring_key))
    for cation_key, cation in cations_b:
        for ring_key, center, _ in rings_a:
            if np.linalg.norm(cation - center) <= 6.0:
                cation_pi_pairs.add((ring_key, cation_key))
    return len(pi_pairs), len(cation_pi_pairs)


def _water_bridges(table: AtomTable) -> tuple[int, int]:
    if table.water_oxygens.size == 0:
        return 0, 0
    idx_a = np.flatnonzero((table.chains == "A") & np.isin(table.elements, list(POLAR_ELEMENTS)))
    idx_b = np.flatnonzero((table.chains == "B") & np.isin(table.elements, list(POLAR_ELEMENTS)))
    if len(idx_a) == 0 or len(idx_b) == 0:
        return int(len(table.water_oxygens)), 0
    tree_a = cKDTree(table.coords[idx_a])
    tree_b = cKDTree(table.coords[idx_b])
    bridges = 0
    for oxygen in table.water_oxygens:
        if tree_a.query_ball_point(oxygen, 3.5) and tree_b.query_ball_point(oxygen, 3.5):
            bridges += 1
    return int(len(table.water_oxygens)), bridges


def interface_features(
    table: AtomTable,
    configured_hotspots: frozenset[int],
    *,
    calculate_pi: bool = True,
) -> tuple[dict[str, float], set[tuple[tuple[str, int, str], tuple[str, int, str]]]]:
    idx_a = _heavy_chain_indices(table, "A")
    idx_b = _heavy_chain_indices(table, "B")
    if len(idx_a) == 0 or len(idx_b) == 0:
        raise ValueError("Expected binder chain A and receptor chain B")

    coords_a = table.coords[idx_a]
    coords_b = table.coords[idx_b]
    sparse = cKDTree(coords_a).sparse_distance_matrix(
        cKDTree(coords_b), max_distance=5.5, output_type="coo_matrix"
    )
    rows = np.asarray(sparse.row, dtype=np.int32)
    cols = np.asarray(sparse.col, dtype=np.int32)
    distances = np.asarray(sparse.data, dtype=np.float64)
    global_a = idx_a[rows]
    global_b = idx_b[cols]

    atom_contacts_45 = int(np.count_nonzero(distances < 4.5))
    clashes_22 = int(np.count_nonzero(distances < 2.2))
    severe_clashes_15 = int(np.count_nonzero(distances < 1.5))
    min_distance = float(distances.min()) if len(distances) else math.nan

    residue_pairs: set[tuple[tuple[str, int, str], tuple[str, int, str]]] = set()
    for ia, ib in zip(global_a.tolist(), global_b.tolist()):
        key_a = table.residue_keys[int(table.res_indices[ia])]
        key_b = table.residue_keys[int(table.res_indices[ib])]
        residue_pairs.add((key_a, key_b))

    class_counts = Counter({name: 0 for name in CONTACT_CLASS_ORDER})
    for key_a, key_b in residue_pairs:
        cls_a = RESIDUE_CLASS.get(_resname_for_key(table, key_a), "P")
        cls_b = RESIDUE_CLASS.get(_resname_for_key(table, key_b), "P")
        class_counts[_pair_class(cls_a, cls_b)] += 1

    salt_pairs: set[tuple[tuple[str, int, str], tuple[str, int, str]]] = set()
    close_mask = distances < 4.0
    for ia, ib in zip(global_a[close_mask].tolist(), global_b[close_mask].tolist()):
        key_a = table.residue_keys[int(table.res_indices[ia])]
        key_b = table.residue_keys[int(table.res_indices[ib])]
        res_a = str(table.resnames[ia])
        res_b = str(table.resnames[ib])
        atom_a = str(table.atomnames[ia])
        atom_b = str(table.atomnames[ib])
        favorable = (
            (atom_a in ACID_ATOMS.get(res_a, ()) and atom_b in BASE_ATOMS.get(res_b, ()))
            or (atom_b in ACID_ATOMS.get(res_b, ()) and atom_a in BASE_ATOMS.get(res_a, ()))
        )
        if favorable:
            salt_pairs.add((key_a, key_b))

    binder_contact_keys = {pair[0] for pair in residue_pairs}
    receptor_contact_keys = {pair[1] for pair in residue_pairs}
    binder_len = len(table.residue_order.get("A", []))
    receptor_order = table.residue_order.get("B", [])
    # Pipeline graft PDBs use Rosetta pose-style numbering for chain B, i.e.
    # target residue 1 becomes binder_length+1. BindCraft keeps target-local
    # numbering. Normalize both to the original IL-4Ra numbering (first B
    # residue is 1) before hotspot accounting.
    receptor_numbering_offset = receptor_order[0][1] - 1 if receptor_order else 0
    receptor_contact_resseqs = {key[1] - receptor_numbering_offset for key in receptor_contact_keys}
    common_contacts = receptor_contact_resseqs & COMMON_HOTSPOTS
    configured_contacts = receptor_contact_resseqs & configured_hotspots
    pi_pairs, cation_pi_pairs = _pi_interactions(table) if calculate_pi else (0, 0)
    water_count, water_bridges = _water_bridges(table)

    features: dict[str, float] = {
        "atomic_contacts_4p5": atom_contacts_45,
        "atomic_contacts_4p5_per_binder": atom_contacts_45 / binder_len,
        "residue_contact_pairs_5p5": len(residue_pairs),
        "residue_contact_pairs_5p5_per_binder": len(residue_pairs) / binder_len,
        "binder_contact_residues_5p5": len(binder_contact_keys),
        "binder_contact_fraction_5p5": len(binder_contact_keys) / binder_len,
        "receptor_contact_residues_5p5": len(receptor_contact_keys),
        "receptor_residue_number_offset": receptor_numbering_offset,
        "salt_bridge_pairs_4p0": len(salt_pairs),
        "salt_bridge_pairs_4p0_per_binder": len(salt_pairs) / binder_len,
        "hydrophobic_AA_pairs_5p5": class_counts["AA"],
        "hydrophobic_AA_pairs_5p5_per_binder": class_counts["AA"] / binder_len,
        "pi_pi_pairs_proxy": pi_pairs,
        "cation_pi_pairs_proxy": cation_pi_pairs,
        "interchain_clashes_2p2": clashes_22,
        "interchain_severe_clashes_1p5": severe_clashes_15,
        "minimum_interchain_heavy_atom_distance": min_distance,
        "common_hotspot_contacts": len(common_contacts),
        "common_hotspot_coverage": len(common_contacts) / len(COMMON_HOTSPOTS),
        "configured_hotspot_contacts": len(configured_contacts),
        "configured_hotspot_coverage": len(configured_contacts) / len(configured_hotspots),
        "water_oxygen_count": water_count,
        "water_bridge_count_3p5": water_bridges,
    }
    for name in CONTACT_CLASS_ORDER:
        features[f"contact_{name}_5p5"] = class_counts[name]
        features[f"contact_{name}_5p5_per_binder"] = class_counts[name] / binder_len
    return features, residue_pairs


def _kabsch_transform(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mobile_centroid = mobile.mean(axis=0)
    target_centroid = target.mean(axis=0)
    mob0 = mobile - mobile_centroid
    tar0 = target - target_centroid
    u, _, vt = np.linalg.svd(mob0.T @ tar0)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_centroid - mobile_centroid @ rotation.T
    return rotation, translation


def _apply_transform(coords: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return coords @ rotation.T + translation


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def _matched_ca(
    before: AtomTable, after: AtomTable, chain: str
) -> tuple[list[tuple[str, int, str]], np.ndarray, np.ndarray]:
    keys = [
        key
        for key in before.residue_order.get(chain, [])
        if key in before.ca_coords and key in after.ca_coords
    ]
    if not keys:
        return [], np.empty((0, 3)), np.empty((0, 3))
    return (
        keys,
        np.asarray([before.ca_coords[key] for key in keys]),
        np.asarray([after.ca_coords[key] for key in keys]),
    )


def relaxation_features(
    before: AtomTable,
    after: AtomTable,
    before_pairs: set[tuple[tuple[str, int, str], tuple[str, int, str]]],
    after_pairs: set[tuple[tuple[str, int, str], tuple[str, int, str]]],
) -> dict[str, float]:
    _, receptor_before, receptor_after = _matched_ca(before, after, "B")
    _, binder_before, binder_after = _matched_ca(before, after, "A")
    if len(receptor_before) < 3 or len(binder_before) < 2:
        return {
            "receptor_ca_rmsd_after_relax": math.nan,
            "binder_pose_ca_rmsd_after_receptor_alignment": math.nan,
            "binder_internal_ca_rmsd_after_relax": math.nan,
            "interface_contact_retention": math.nan,
            "interface_contact_jaccard": math.nan,
        }

    rotation, translation = _kabsch_transform(receptor_before.copy(), receptor_after)
    receptor_aligned = _apply_transform(receptor_before, rotation, translation)
    binder_pose_aligned = _apply_transform(binder_before, rotation, translation)
    receptor_rmsd = _rmsd(receptor_aligned, receptor_after)
    binder_pose_rmsd = _rmsd(binder_pose_aligned, binder_after)

    binder_rotation, binder_translation = _kabsch_transform(binder_before.copy(), binder_after)
    binder_internal_rmsd = _rmsd(
        _apply_transform(binder_before, binder_rotation, binder_translation), binder_after
    )

    intersection = len(before_pairs & after_pairs)
    union = len(before_pairs | after_pairs)
    retention = intersection / len(before_pairs) if before_pairs else math.nan
    jaccard = intersection / union if union else math.nan
    return {
        "receptor_ca_rmsd_after_relax": receptor_rmsd,
        "binder_pose_ca_rmsd_after_receptor_alignment": binder_pose_rmsd,
        "binder_internal_ca_rmsd_after_relax": binder_internal_rmsd,
        "interface_contact_retention": retention,
        "interface_contact_jaccard": jaccard,
    }


def parse_dssp(path: Path, dssp_executable: Path) -> dict[tuple[str, int, str], str]:
    proc = subprocess.run(
        # DSSP 2.0.4 treats the literal string "stdout" as a filename despite
        # its help text. /dev/stdout is safe here and remains capturable in
        # parallel worker processes.
        [str(dssp_executable), "-i", str(path), "-o", "/dev/stdout"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        message = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown DSSP error"
        raise RuntimeError(message)
    in_records = False
    result: dict[tuple[str, int, str], str] = {}
    for line in proc.stdout.splitlines():
        if line.startswith("  #  RESIDUE"):
            in_records = True
            continue
        if not in_records or len(line) < 17:
            continue
        if line[13:14] == "!":
            continue
        try:
            resseq = int(line[5:10])
        except ValueError:
            continue
        chain = line[11:12].strip() or "_"
        icode = line[10:11].strip()
        result[(chain, resseq, icode)] = line[16:17].strip() or "C"
    return result


def _count_structured_segments(collapsed: Sequence[str]) -> int:
    count = 0
    i = 0
    while i < len(collapsed):
        code = collapsed[i]
        if code not in {"H", "E"}:
            i += 1
            continue
        j = i + 1
        while j < len(collapsed) and collapsed[j] == code:
            j += 1
        minimum = 3 if code == "H" else 2
        if j - i >= minimum:
            count += 1
        i = j
    return count


def dssp_features(table: AtomTable, assignments: Mapping[tuple[str, int, str], str]) -> dict[str, object]:
    raw_codes = [assignments.get(key, "C") for key in table.residue_order.get("A", [])]
    collapsed = []
    for code in raw_codes:
        if code in {"H", "G", "I"}:
            collapsed.append("H")
        elif code in {"E", "B"}:
            collapsed.append("E")
        elif code in {"T", "S"}:
            collapsed.append("T")
        else:
            collapsed.append("C")
    n = len(collapsed)
    helix = collapsed.count("H") / n
    sheet = collapsed.count("E") / n
    turn = collapsed.count("T") / n
    coil = collapsed.count("C") / n
    segments = _count_structured_segments(collapsed)

    # The paper does not publish its exact topology classifier. These explicit
    # rules preserve its broad categories without claiming an exact replica.
    if helix >= 0.15 and sheet >= 0.15:
        fold_class = "mixed_alpha_beta"
    elif segments >= 2 and (helix + sheet) >= 0.40:
        fold_class = "multi_segment_proxy"
    elif helix >= 0.40:
        fold_class = "alpha_helical"
    elif sheet >= 0.30:
        fold_class = "beta_sheet"
    else:
        fold_class = "coil_dominated"
    return {
        "dssp_helix_fraction": helix,
        "dssp_sheet_fraction": sheet,
        "dssp_turn_fraction": turn,
        "dssp_coil_fraction": coil,
        "dssp_structured_segment_count": segments,
        "dssp_fold_class": fold_class,
        "dssp_string": "".join(collapsed),
    }


def sequence_features(sequence: str) -> dict[str, float]:
    analysis = ProteinAnalysis(sequence)
    counts = Counter(sequence)
    entropy = -sum((count / len(sequence)) * math.log(count / len(sequence)) for count in counts.values())
    hydro_values = [1 if aa in HYDROPHOBIC_AA1 else 0 for aa in sequence]
    longest_run = 0
    current_run = 0
    for value in hydro_values:
        current_run = current_run + 1 if value else 0
        longest_run = max(longest_run, current_run)
    window = min(5, len(sequence))
    max_window_hydrophobic = max(
        sum(hydro_values[i : i + window]) / window for i in range(len(sequence) - window + 1)
    )
    return {
        "sequence_charge_pH7": float(analysis.charge_at_pH(7.0)),
        "sequence_isoelectric_point": float(analysis.isoelectric_point()),
        "sequence_gravy": float(analysis.gravy()),
        "sequence_hydrophobic_fraction": sum(hydro_values) / len(sequence),
        "sequence_charged_fraction": sum(aa in CHARGED_AA1 for aa in sequence) / len(sequence),
        "sequence_aromatic_fraction": sum(aa in "FYW" for aa in sequence) / len(sequence),
        "sequence_cysteine_fraction": sequence.count("C") / len(sequence),
        "sequence_max_hydrophobic_run": longest_run,
        "sequence_max_5aa_hydrophobic_fraction": max_window_hydrophobic,
        "sequence_composition_entropy": entropy,
        "sequence_effective_alphabet": math.exp(entropy),
    }


def energetic_features(
    rows: Sequence[tuple[str, int, float]],
    configured_hotspots: frozenset[int],
    receptor_numbering_offset: int,
) -> dict[str, float]:
    binder_values = [value for chain, _, value in rows if chain == "A" and math.isfinite(value)]
    receptor_values = [
        (resnum - receptor_numbering_offset, value)
        for chain, resnum, value in rows
        if chain == "B" and math.isfinite(value)
    ]
    favorable_binder = sorted(value for value in binder_values if value < 0)
    favorable_target = [(resnum, value) for resnum, value in receptor_values if value < 0]
    total_favorable_binder = sum(-value for value in favorable_binder)
    total_favorable_target = sum(-value for _, value in favorable_target)
    top3 = sum(-value for value in favorable_binder[:3])
    common_hotspot = sum(-value for resnum, value in favorable_target if resnum in COMMON_HOTSPOTS)
    configured_hotspot = sum(-value for resnum, value in favorable_target if resnum in configured_hotspots)
    return {
        "reb_binder_residue_count": len(binder_values),
        "reb_binder_favorable_residue_count": len(favorable_binder),
        "reb_binder_ddg_sum": sum(binder_values),
        "reb_binder_strongest_ddg": min(binder_values) if binder_values else math.nan,
        "reb_binder_top3_favorable_share": top3 / total_favorable_binder if total_favorable_binder else math.nan,
        "reb_common_hotspot_favorable_energy_share": common_hotspot / total_favorable_target if total_favorable_target else math.nan,
        "reb_configured_hotspot_favorable_energy_share": configured_hotspot / total_favorable_target if total_favorable_target else math.nan,
    }


def analyze_one(task: tuple[str, str, str, str, tuple[int, ...], str]) -> dict[str, object]:
    dataset_slug, dataset_label, input_path_s, relaxed_path_s, hotspot_tuple, dssp_s = task
    input_path = Path(input_path_s)
    relaxed_path = Path(relaxed_path_s)
    hotspots = frozenset(hotspot_tuple)
    input_table = parse_pdb(input_path)
    relaxed_table = parse_pdb(relaxed_path)
    sequence = chain_sequence(input_table, "A")
    if not sequence:
        raise ValueError(f"No binder chain A sequence in {input_path}")

    before_features, before_pairs = interface_features(input_table, hotspots, calculate_pi=False)
    relaxed_features, relaxed_pairs = interface_features(relaxed_table, hotspots, calculate_pi=True)
    stability = relaxation_features(input_table, relaxed_table, before_pairs, relaxed_pairs)
    assignments = parse_dssp(input_path, Path(dssp_s))
    secondary = dssp_features(input_table, assignments)
    confidence = [
        input_table.ca_bfactors[key]
        for key in input_table.residue_order.get("A", [])
        if key in input_table.ca_bfactors and math.isfinite(input_table.ca_bfactors[key])
    ]

    result: dict[str, object] = {
        "dataset": dataset_slug,
        "dataset_label": dataset_label,
        "pdb": input_path.name,
        "input_pdb": str(input_path),
        "relaxed_pdb": str(relaxed_path),
        "binder_sequence": sequence,
        "binder_length": len(sequence),
        "input_binder_ca_bfactor_mean": float(np.mean(confidence)) if confidence else math.nan,
        "input_binder_ca_bfactor_median": float(np.median(confidence)) if confidence else math.nan,
        "input_binder_ca_bfactor_ge80_fraction": float(np.mean(np.asarray(confidence) >= 80.0)) if confidence else math.nan,
    }
    result.update({f"pre_relax_{key}": value for key, value in before_features.items()})
    result.update(relaxed_features)
    result.update(stability)
    result["interchain_clash_change"] = (
        relaxed_features["interchain_clashes_2p2"] - before_features["interchain_clashes_2p2"]
    )
    result["post_relax_clash_free"] = int(relaxed_features["interchain_clashes_2p2"] == 0)
    result.update(secondary)
    result.update(sequence_features(sequence))
    return result


def load_energy_rows(spec: DatasetSpec) -> dict[str, list[tuple[str, int, float]]]:
    df = pd.read_csv(spec.residues_csv)
    df["ddG_contrib"] = pd.to_numeric(df["ddG_contrib"], errors="coerce")
    result: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    for row in df.itertuples(index=False):
        result[Path(str(row.pdb)).name].append((str(row.chain), int(row.resnum), float(row.ddG_contrib)))
    return result


def load_score_table(spec: DatasetSpec) -> pd.DataFrame:
    scores = pd.read_csv(spec.binding_csv)
    scores["pdb"] = scores["pdb"].map(lambda value: Path(str(value)).name)
    for column in scores.columns:
        if column not in {"pdb", "design"}:
            scores[column] = pd.to_numeric(scores[column], errors="coerce")
    return scores


def _identity_statistics(sequences: Sequence[str]) -> tuple[dict[str, float], np.ndarray]:
    n = len(sequences)
    if n < 2:
        return {
            "pairwise_comparisons": 0,
            "mean_pairwise_identity": math.nan,
            "median_pairwise_identity": math.nan,
            "sd_pairwise_identity": math.nan,
            "p90_pairwise_identity": math.nan,
            "max_pairwise_identity": math.nan,
            "fraction_pairwise_identity_lt_10pct": math.nan,
            "fraction_pairwise_identity_ge_80pct": math.nan,
            "identity80_cluster_count": n,
        }, np.empty(0, dtype=np.float32)

    max_len = max(map(len, sequences))
    encoded = np.full((n, max_len), -1, dtype=np.int16)
    alphabet = {aa: idx for idx, aa in enumerate("ACDEFGHIKLMNPQRSTVWYX")}
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int16)
    for i, sequence in enumerate(sequences):
        encoded[i, : len(sequence)] = [alphabet.get(aa, alphabet["X"]) for aa in sequence]

    identities = np.empty(n * (n - 1) // 2, dtype=np.float32)
    offset = 0
    parent = np.arange(n, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(n - 1):
        other = encoded[i + 1 :]
        both_valid = (other >= 0) & (encoded[i] >= 0)
        matches = np.sum((other == encoded[i]) & both_valid, axis=1)
        denominator = np.maximum(lengths[i], lengths[i + 1 :])
        values = matches / denominator
        identities[offset : offset + len(values)] = values.astype(np.float32)
        for rel_j in np.flatnonzero(values >= 0.80).tolist():
            union(i, i + 1 + rel_j)
        offset += len(values)

    cluster_count = len({find(i) for i in range(n)})
    return {
        "pairwise_comparisons": len(identities),
        "mean_pairwise_identity": float(np.mean(identities)),
        "median_pairwise_identity": float(np.median(identities)),
        "sd_pairwise_identity": float(np.std(identities)),
        "p90_pairwise_identity": float(np.quantile(identities, 0.90)),
        "max_pairwise_identity": float(np.max(identities)),
        "fraction_pairwise_identity_lt_10pct": float(np.mean(identities < 0.10)),
        "fraction_pairwise_identity_ge_80pct": float(np.mean(identities >= 0.80)),
        "identity80_cluster_count": cluster_count,
    }, identities


def pairwise_identity_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    summaries = []
    arrays: dict[str, np.ndarray] = {}
    for dataset, group in df.groupby("dataset", sort=False):
        structure_sequences = group["binder_sequence"].astype(str).tolist()
        unique_sequences = list(dict.fromkeys(structure_sequences))
        structure_stats, _ = _identity_statistics(structure_sequences)
        unique_stats, unique_identities = _identity_statistics(unique_sequences)
        sequence_counts = Counter(structure_sequences)
        row = {
            "dataset": dataset,
            "dataset_label": group["dataset_label"].iloc[0],
            "n_structures": len(structure_sequences),
            "n_unique_sequences": len(unique_sequences),
            "unique_sequence_fraction": len(unique_sequences) / len(structure_sequences),
            "largest_exact_sequence_multiplicity": max(sequence_counts.values()),
        }
        row.update(structure_stats)
        row.update({f"unique_sequence_{key}": value for key, value in unique_stats.items()})
        row["identity80_effective_fraction_of_structures"] = (
            structure_stats["identity80_cluster_count"] / len(structure_sequences)
        )
        row["identity80_effective_fraction_of_unique_sequences"] = (
            unique_stats["identity80_cluster_count"] / len(unique_sequences)
        )
        summaries.append(row)
        # Plot the deduplicated distribution; structure-weighted identities are
        # retained in the CSV for auditing seed/sample multiplicity.
        arrays[dataset] = unique_identities
    return pd.DataFrame(summaries), arrays


def scalar_summary(df: pd.DataFrame, columns: Sequence[str], subset_name: str) -> pd.DataFrame:
    rows = []
    for (dataset, label), group in df.groupby(["dataset", "dataset_label"], sort=False):
        for column in columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            rows.append(
                {
                    "subset": subset_name,
                    "dataset": dataset,
                    "dataset_label": label,
                    "metric": column,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def fold_summary(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    rows = []
    classes = (
        "alpha_helical",
        "beta_sheet",
        "mixed_alpha_beta",
        "multi_segment_proxy",
        "coil_dominated",
    )
    for (dataset, label), group in df.groupby(["dataset", "dataset_label"], sort=False):
        counts = group["dssp_fold_class"].value_counts()
        for fold_class in classes:
            rows.append(
                {
                    "subset": subset_name,
                    "dataset": dataset,
                    "dataset_label": label,
                    "fold_class": fold_class,
                    "count": int(counts.get(fold_class, 0)),
                    "fraction": float(counts.get(fold_class, 0) / len(group)),
                }
            )
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, fold_df: pd.DataFrame, identity_arrays: Mapping[str, np.ndarray], output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    palette = {
        "round1": "#4C78A8",
        "round2": "#F58518",
        "bindcraft": "#54A24B",
    }
    label_order = ["pipeline round1", "pipeline round2", "BindCraft"]

    fold_order = [
        "alpha_helical",
        "beta_sheet",
        "mixed_alpha_beta",
        "multi_segment_proxy",
        "coil_dominated",
    ]
    fold_colors = ["#E45756", "#4C78A8", "#B279A2", "#72B7B2", "#BAB0AC"]
    pivot = (
        fold_df[fold_df["subset"] == "all"]
        .pivot(index="dataset_label", columns="fold_class", values="fraction")
        .reindex(label_order)
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(10, 6.5))
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for fold_class, color in zip(fold_order, fold_colors):
        values = pivot.get(fold_class, pd.Series(0, index=pivot.index)).to_numpy()
        ax.bar(x, values, bottom=bottom, label=fold_class.replace("_", " "), color=color)
        bottom += values
    ax.set_xticks(x, pivot.index, rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of structures")
    ax.set_title("DSSP-derived binder fold classes (input PDBs)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "secondary_structure_classes.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "secondary_structure_classes.pdf", bbox_inches="tight")
    plt.close(fig)

    composition_columns = [
        ("dssp_helix_fraction", "helix", "#E45756"),
        ("dssp_sheet_fraction", "sheet", "#4C78A8"),
        ("dssp_turn_fraction", "turn", "#F2CF5B"),
        ("dssp_coil_fraction", "coil", "#BAB0AC"),
    ]
    composition = (
        df.groupby("dataset_label")[[column for column, _, _ in composition_columns]]
        .mean()
        .reindex(label_order)
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    bottom = np.zeros(len(pivot))
    for fold_class, color in zip(fold_order, fold_colors):
        values = pivot.get(fold_class, pd.Series(0, index=pivot.index)).to_numpy()
        axes[0].bar(x, values, bottom=bottom, label=fold_class.replace("_", " "), color=color)
        bottom += values
    axes[0].set_xticks(x, pivot.index)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fraction")
    axes[0].set_title("Structure-level fold class")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=11)

    bottom = np.zeros(len(composition))
    for column, label, color in composition_columns:
        values = composition[column].to_numpy()
        axes[1].bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[1].set_xticks(x, composition.index)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Mean residue fraction")
    axes[1].set_title("Per-residue DSSP composition")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=11)
    fig.suptitle("IL-4Ra binder secondary-structure bias", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "secondary_structure_summary.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "secondary_structure_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    metrics = [
        ("residue_contact_pairs_5p5_per_binder", "Residue contacts / binder residue"),
        ("salt_bridge_pairs_4p0_per_binder", "Salt bridges / binder residue"),
        ("common_hotspot_coverage", "Common hotspot coverage"),
        ("pre_relax_interchain_clashes_2p2", "Pre-relax inter-chain clashes"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (column, title) in zip(axes.flat, metrics):
        sns.boxplot(
            data=df,
            x="dataset_label",
            y=column,
            order=label_order,
            hue="dataset",
            palette=palette,
            showfliers=False,
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=12)
    fig.suptitle("Paper-inspired interface geometry metrics", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "interface_geometry_metrics.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "interface_geometry_metrics.pdf", bbox_inches="tight")
    plt.close(fig)

    stability_metrics = [
        ("binder_pose_ca_rmsd_after_receptor_alignment", "Binder pose Cα RMSD (Å)"),
        ("binder_internal_ca_rmsd_after_relax", "Binder internal Cα RMSD (Å)"),
        ("interface_contact_retention", "Interface contact retention"),
        ("input_binder_ca_bfactor_mean", "Input binder CA confidence/B-factor"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (column, title) in zip(axes.flat, stability_metrics):
        sns.violinplot(
            data=df,
            x="dataset_label",
            y=column,
            order=label_order,
            hue="dataset",
            palette=palette,
            inner="quartile",
            cut=0,
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=12)
    fig.suptitle("Prediction confidence and Rosetta-relax stability", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "relaxation_stability_metrics.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "relaxation_stability_metrics.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for dataset, values in identity_arrays.items():
        if not len(values):
            continue
        sample = values
        if len(sample) > 500_000:
            rng = np.random.default_rng(20260806)
            sample = rng.choice(sample, size=500_000, replace=False)
        label = df.loc[df["dataset"] == dataset, "dataset_label"].iloc[0]
        sns.histplot(
            sample,
            bins=np.linspace(0, 1, 51),
            stat="density",
            element="step",
            fill=False,
            linewidth=2,
            color=palette[dataset],
            label=label,
            ax=ax,
        )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pairwise identity (N-to-C ungapped; matches / longer length)")
    ax.set_ylabel("Density")
    ax.set_title("Within-method sequence diversity (exact sequences deduplicated)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "sequence_identity_distribution.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "sequence_identity_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def write_readme(output_dir: Path, n_by_dataset: Mapping[str, int], dssp_executable: Path) -> None:
    text = f"""# IL-4Ra paper-inspired extended metrics

Generated by `pipeline/pipeline/scripts/analyze_il4ra_paper_metrics.py`.

## Cohorts

{os.linesep.join(f'- `{key}`: {value} structures' for key, value in n_by_dataset.items())}

## Main files

- `per_structure_metrics.csv`: one row per Rosetta-scored complex.
- `scalar_summary.csv`: mean/median/IQR/range for full, energy-top20%, and common-length cohorts.
- `fold_class_summary.csv`: DSSP-derived fold-class counts and fractions.
- `sequence_diversity.csv`: exact uniqueness plus structure-weighted and
  deduplicated pairwise identity/80%-identity clusters.
- `secondary_structure_classes.*`, `secondary_structure_summary.*`, `interface_geometry_metrics.*`,
  `relaxation_stability_metrics.*`, `sequence_identity_distribution.*`: figures.

## Reproducible definitions

- Binder is chain A; receptor is chain B.
- Geometry/contact metrics use Rosetta-relaxed PDBs unless prefixed `pre_relax_`.
- Residue contact: any inter-chain heavy-atom distance < 5.5 A.
- Atomic contact: inter-chain heavy-atom distance < 4.5 A.
- Salt bridge: ASP/GLU carboxylate O to LYS/ARG basic N < 4.0 A.
- Clash: inter-chain heavy-atom distance < 2.2 A (paper threshold).
- Common hotspot coverage uses residues {sorted(COMMON_HOTSPOTS)}.
- Chain-B residue numbers are normalized back to target-local IL-4Ra numbering:
  pipeline graft PDBs carry a binder-length offset, whereas BindCraft PDBs do not.
- DSSP uses input prediction PDBs and `{dssp_executable}`. H/G/I -> helix,
  E/B -> sheet, T/S -> turn, all remaining states -> coil.
- Fold classes are an explicit paper-inspired proxy because the paper does not
  publish its exact topology classifier: mixed if helix and sheet are each >=15%;
  otherwise multi-segment if >=2 structured elements and >=40% helix+sheet;
  otherwise alpha-helical if helix >=40%, beta-sheet if sheet >=30%, else coil-dominated.
- Pairwise identity is an exhaustive N-to-C ungapped comparison normalized by
  the longer sequence length. The paper does not state its alignment algorithm,
  so this result is comparable in intent, not an exact reimplementation.
- The paper also does not disclose the similarity kernel used for its Vendi
  score, so no nominally exact Vendi value is reported; 80%-identity clusters
  provide the auditable effective-diversity summary instead.
- Pi-pi and cation-pi fields are geometry proxies; DeltaForge thresholds and
  regression weights are proprietary and unavailable.
- Pipeline `interface_residues.csv` contains binder-chain rows only, so target
  hotspot energy-share fields are unavailable there; geometric hotspot coverage
  remains available for all three cohorts.
- Input CA B factors are confidence-like values from AF3 (pipeline) or AF2
  (BindCraft). They measure local backbone confidence, not interface confidence.
- iPSAE/ipTM/pTM cannot be recovered from PDB coordinates alone because PAE and
  predictor score files are required. No DeltaForge Kd is inferred.
"""
    (output_dir / "README.md").write_text(text)


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--dssp",
        type=Path,
        default=workspace_default / "software/BindCraft-main/functions/dssp",
    )
    parser.add_argument("--workers", type=int, default=min(16, max(1, os.cpu_count() or 1)))
    parser.add_argument("--limit", type=int, default=0, help="Per-dataset debug limit; 0 means full cohort")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else workspace / "pipeline/pipeline_runs/il4ra_extended_metrics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.dssp.is_file():
        raise FileNotFoundError(f"DSSP executable not found: {args.dssp}")

    run_dir = workspace / "pipeline/pipeline_runs/run_20260706_IL-4Ra"
    bindcraft_dir = workspace / "pipeline/pipeline_runs/bindcraft_pdb/rosetta_2025.37_iface14"
    specs = [
        DatasetSpec("round1", "pipeline round1", run_dir / "round1/3_rosetta", PIPELINE_HOTSPOTS),
        DatasetSpec("round2", "pipeline round2", run_dir / "round2/3_rosetta", PIPELINE_HOTSPOTS),
        DatasetSpec("bindcraft", "BindCraft", bindcraft_dir, BINDCRAFT_HOTSPOTS),
    ]

    tasks = []
    score_tables: dict[str, pd.DataFrame] = {}
    energy_rows: dict[str, dict[str, list[tuple[str, int, float]]]] = {}
    for spec in specs:
        scores = load_score_table(spec)
        if args.limit:
            scores = scores.head(args.limit).copy()
        score_tables[spec.slug] = scores
        energy_rows[spec.slug] = load_energy_rows(spec)
        for pdb_name in scores["pdb"].tolist():
            input_path = spec.input_dir / pdb_name
            relaxed_path = spec.result_dir / pdb_name
            if not input_path.is_file() or not relaxed_path.is_file():
                raise FileNotFoundError(f"Missing input/result PDB pair for {spec.slug}: {pdb_name}")
            tasks.append(
                (
                    spec.slug,
                    spec.label,
                    str(input_path),
                    str(relaxed_path),
                    tuple(sorted(spec.configured_hotspots)),
                    str(args.dssp.resolve()),
                )
            )

    print(f"Analyzing {len(tasks)} PDB pairs with {args.workers} workers", flush=True)
    records = []
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(analyze_one, task): task for task in tasks}
        completed = 0
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                records.append(future.result())
            except Exception as exc:  # pragma: no cover - surfaced in manifest
                errors.append({"dataset": task[0], "pdb": Path(task[2]).name, "error": repr(exc)})
            completed += 1
            if completed % 100 == 0 or completed == len(tasks):
                print(f"  completed {completed}/{len(tasks)}; errors={len(errors)}", flush=True)
    if errors:
        (output_dir / "analysis_errors.json").write_text(json.dumps(errors, indent=2))
        raise RuntimeError(f"{len(errors)} structures failed; see {output_dir / 'analysis_errors.json'}")

    metrics = pd.DataFrame(records)
    combined = []
    spec_by_slug = {spec.slug: spec for spec in specs}
    for dataset, group in metrics.groupby("dataset", sort=False):
        spec = spec_by_slug[dataset]
        merged = group.merge(score_tables[dataset], on="pdb", how="left", validate="one_to_one")
        energetic = []
        for pdb_name, offset_value in zip(
            merged["pdb"], merged["receptor_residue_number_offset"]
        ):
            offset = int(offset_value)
            energetic.append(
                energetic_features(
                    energy_rows[dataset].get(pdb_name, []), spec.configured_hotspots, offset
                )
            )
        energetic_df = pd.DataFrame(energetic, index=merged.index)
        merged = pd.concat([merged, energetic_df], axis=1)
        combined.append(merged)
    df = pd.concat(combined, ignore_index=True)

    numeric_rosetta = [
        "dG_separated",
        "dG_cross",
        "dSASA_int",
        "nres_int",
        "packstat",
        "sc_value",
        "delta_unsatHbonds",
        "hbonds_int",
    ]
    for column in numeric_rosetta:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["dG_per_binder"] = df["dG_separated"] / df["binder_length"]
    df["dG_cross_per_binder"] = df["dG_cross"] / df["binder_length"]
    df["dSASA_per_binder"] = df["dSASA_int"] / df["binder_length"]
    df["hbonds_per_binder"] = df["hbonds_int"] / df["binder_length"]
    df["unsat_hbonds_per_binder"] = df["delta_unsatHbonds"] / df["binder_length"]
    df["reb_binder_favorable_fraction"] = (
        df["reb_binder_favorable_residue_count"] / df["binder_length"]
    )
    duplicate_counts = df.groupby(["dataset", "binder_sequence"])["pdb"].transform("count")
    df["exact_sequence_multiplicity"] = duplicate_counts.astype(int)

    df["energy_rank"] = df.groupby("dataset")["dG_per_binder"].rank(method="first", ascending=True)
    cohort_sizes = df.groupby("dataset")["pdb"].transform("size")
    df["energy_top20"] = df["energy_rank"] <= np.ceil(cohort_sizes * 0.20)
    common_lengths = sorted(set.intersection(*(set(group["binder_length"]) for _, group in df.groupby("dataset"))))
    df["common_length_cohort"] = df["binder_length"].isin(common_lengths)

    column_order = [
        "dataset",
        "dataset_label",
        "pdb",
        "binder_sequence",
        "binder_length",
        "exact_sequence_multiplicity",
        "energy_rank",
        "energy_top20",
        "common_length_cohort",
        "dG_separated",
        "dG_per_binder",
        "dG_cross",
        "dG_cross_per_binder",
        "dSASA_int",
        "dSASA_per_binder",
        "packstat",
        "sc_value",
        "hbonds_int",
        "hbonds_per_binder",
        "delta_unsatHbonds",
        "unsat_hbonds_per_binder",
    ]
    remaining = [column for column in df.columns if column not in column_order]
    df = df[column_order + sorted(remaining)]
    df.sort_values(["dataset", "dG_per_binder", "pdb"], inplace=True)
    df.to_csv(output_dir / "per_structure_metrics.csv", index=False, float_format="%.6f")

    sequence_all, identity_arrays = pairwise_identity_summary(df)
    sequence_all.insert(0, "subset", "all")
    sequence_top20, _ = pairwise_identity_summary(df[df["energy_top20"]])
    sequence_top20.insert(0, "subset", "energy_top20")
    sequence_common, _ = pairwise_identity_summary(df[df["common_length_cohort"]])
    sequence_common.insert(0, "subset", "common_lengths")
    sequence_diversity = pd.concat(
        [sequence_all, sequence_top20, sequence_common], ignore_index=True
    )
    sequence_diversity.to_csv(output_dir / "sequence_diversity.csv", index=False, float_format="%.6f")

    summary_columns = [
        "dssp_helix_fraction",
        "dssp_sheet_fraction",
        "dssp_turn_fraction",
        "dssp_coil_fraction",
        "input_binder_ca_bfactor_mean",
        "atomic_contacts_4p5_per_binder",
        "residue_contact_pairs_5p5_per_binder",
        "binder_contact_fraction_5p5",
        "receptor_contact_residues_5p5",
        "hydrophobic_AA_pairs_5p5_per_binder",
        "salt_bridge_pairs_4p0",
        "salt_bridge_pairs_4p0_per_binder",
        "contact_CC_5p5_per_binder",
        "contact_CP_5p5_per_binder",
        "contact_CA_5p5_per_binder",
        "contact_PP_5p5_per_binder",
        "contact_PA_5p5_per_binder",
        "contact_AA_5p5_per_binder",
        "pi_pi_pairs_proxy",
        "cation_pi_pairs_proxy",
        "common_hotspot_coverage",
        "configured_hotspot_coverage",
        "interchain_clashes_2p2",
        "pre_relax_interchain_clashes_2p2",
        "binder_pose_ca_rmsd_after_receptor_alignment",
        "binder_internal_ca_rmsd_after_relax",
        "receptor_ca_rmsd_after_relax",
        "interface_contact_retention",
        "interface_contact_jaccard",
        "post_relax_clash_free",
        "sequence_charge_pH7",
        "sequence_isoelectric_point",
        "sequence_gravy",
        "sequence_hydrophobic_fraction",
        "sequence_charged_fraction",
        "sequence_max_5aa_hydrophobic_fraction",
        "reb_binder_favorable_residue_count",
        "reb_binder_favorable_fraction",
        "reb_binder_strongest_ddg",
        "reb_binder_top3_favorable_share",
        "reb_common_hotspot_favorable_energy_share",
        "reb_configured_hotspot_favorable_energy_share",
    ]
    scalar = pd.concat(
        [
            scalar_summary(df, summary_columns, "all"),
            scalar_summary(df[df["energy_top20"]], summary_columns, "energy_top20"),
            scalar_summary(df[df["common_length_cohort"]], summary_columns, "common_lengths"),
        ],
        ignore_index=True,
    )
    scalar.to_csv(output_dir / "scalar_summary.csv", index=False, float_format="%.6f")

    folds = pd.concat(
        [
            fold_summary(df, "all"),
            fold_summary(df[df["energy_top20"]], "energy_top20"),
            fold_summary(df[df["common_length_cohort"]], "common_lengths"),
        ],
        ignore_index=True,
    )
    folds.to_csv(output_dir / "fold_class_summary.csv", index=False, float_format="%.6f")

    length_summary = (
        df.groupby(["dataset", "dataset_label", "binder_length"])
        .agg(
            n=("pdb", "size"),
            unique_sequences=("binder_sequence", "nunique"),
            dG_per_binder_mean=("dG_per_binder", "mean"),
            helix_fraction_mean=("dssp_helix_fraction", "mean"),
            contacts_per_binder_mean=("residue_contact_pairs_5p5_per_binder", "mean"),
            common_hotspot_coverage_mean=("common_hotspot_coverage", "mean"),
            binder_pose_rmsd_median=("binder_pose_ca_rmsd_after_receptor_alignment", "median"),
        )
        .reset_index()
    )
    length_summary.to_csv(output_dir / "length_stratified_summary.csv", index=False, float_format="%.6f")

    make_plots(df, folds, identity_arrays, output_dir)
    n_by_dataset = df.groupby("dataset")["pdb"].size().to_dict()
    write_readme(output_dir, n_by_dataset, args.dssp.resolve())
    manifest = {
        "workspace": str(workspace),
        "output_dir": str(output_dir),
        "dssp": str(args.dssp.resolve()),
        "workers": args.workers,
        "n_structures": n_by_dataset,
        "common_hotspots": sorted(COMMON_HOTSPOTS),
        "common_lengths": common_lengths,
        "errors": 0,
        "limitations": [
            "iPSAE/ipTM/pTM require PAE/predictor outputs and are not recoverable from PDB alone",
            "DeltaForge feature thresholds and regression weights are proprietary",
            "DSSP topology classes and pi interaction fields are explicit paper-inspired proxies",
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote extended analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
