#!/usr/bin/env python3
"""Create Rosetta-ready BindCraft complexes with the binder on chain A.

BindCraft exports IL-4Ra as chain A and its designed binder as chain B.  The
pipeline's fixed iface14 protocol defines chain A as the binder, so this
utility swaps only those PDB chain IDs; coordinates, residue numbers, atom
records, TER records, and all other fields are preserved verbatim.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def swap_chain_ids(line: str) -> str:
    if not line.startswith(("ATOM  ", "HETATM", "ANISOU")) or len(line) < 22:
        return line
    chain = line[21]
    if chain == "A":
        return line[:21] + "B" + line[22:]
    if chain == "B":
        return line[:21] + "A" + line[22:]
    return line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    inputs = sorted(args.input_dir.glob("*.pdb"))
    if not inputs:
        parser.error(f"no PDB files in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in inputs:
        text = source.read_text()
        chains = {line[21] for line in text.splitlines() if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 22}
        if chains != {"A", "B"}:
            parser.error(f"{source} must contain exactly chains A and B; found {sorted(chains)}")
        (args.output_dir / source.name).write_text("".join(swap_chain_ids(line) for line in text.splitlines(keepends=True)))
    print(f"Prepared {len(inputs)} PDBs in {args.output_dir}")


if __name__ == "__main__":
    main()
