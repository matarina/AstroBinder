#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 PDB 单链转成 AF3 模板 mmCIF, 并构造带模板的 AF3 复合物 JSON。

用途: 让 AF3 在预测时以 RFdiffusion 生成的骨架为模板软约束——
  - target 链: 全长模板 -> 构象基本不动;
  - binder(A)链: 只对 RIF 固定残基提供部分模板 -> 界面锚定, 其余自由折叠。

模板 mmCIF 的硬要求(经 af3_pkg 源码核实):
  单条 polymer 链 + release date + entry name + _atom_site 坐标。
  这里用 AF3 自带的 alphafold3.structure(即 AF3 解析模板用的同一套代码)生成,
  从而保证 from_mmcif 一定能读回。

依赖: 需能 import alphafold3(即 PYTHONPATH 含 af3_pkg), 用 af3 环境的 python 跑。
"""

from __future__ import annotations

import datetime
import numpy as np

from alphafold3 import structure
from alphafold3.constants import mmcif_names

THREE2ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

# 元素猜测: PDB 第 77-78 列若无, 用原子名首字母兜底
def _element(line: str, atom_name: str) -> str:
    el = line[76:78].strip()
    if el:
        return el
    a = atom_name.lstrip('0123456789')
    return a[0] if a else 'C'


def parse_pdb_chain(pdb_path: str, chain: str):
    """解析 PDB 中指定链的全部原子, 返回按原子排列的字段字典 (numpy 数组)。

    残基顺序 = 文件中出现顺序; res_id 用「出现次序」重新连续编号(1-based),
    这样模板里的 templateIndex(0-based) 与残基出现顺序一一对应, 与 AF3 query
    序列的残基顺序对齐。同时返回该链的单字母序列, 供核对。
    """
    res_id, chain_id, chain_type = [], [], []
    res_name, atom_name, atom_element = [], [], []
    xs, ys, zs = [], [], []
    seq = []
    seen = {}          # (resSeq+iCode) -> 连续编号
    order_counter = 0
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            if line[21] != chain:
                continue
            key = line[22:27]                     # resSeq + iCode
            if key not in seen:
                order_counter += 1
                seen[key] = order_counter
                seq.append(THREE2ONE.get(line[17:20].strip(), 'X'))
            rid = seen[key]
            an = line[12:16].strip()
            res_id.append(rid)
            chain_id.append(chain)
            chain_type.append(mmcif_names.PROTEIN_CHAIN)
            res_name.append(line[17:20].strip())
            atom_name.append(an)
            atom_element.append(_element(line, an))
            xs.append(float(line[30:38]))
            ys.append(float(line[38:46]))
            zs.append(float(line[46:54]))
    if not res_id:
        raise ValueError(f'链 {chain} 在 {pdb_path} 中没有原子')
    arrays = dict(
        res_id=np.array(res_id, dtype=np.int32),
        chain_id=np.array(chain_id, dtype=object),
        chain_type=np.array(chain_type, dtype=object),
        res_name=np.array(res_name, dtype=object),
        atom_name=np.array(atom_name, dtype=object),
        atom_element=np.array(atom_element, dtype=object),
        atom_x=np.array(xs, dtype=np.float32),
        atom_y=np.array(ys, dtype=np.float32),
        atom_z=np.array(zs, dtype=np.float32),
    )
    return arrays, ''.join(seq), order_counter


def pdb_chain_to_mmcif(pdb_path: str, chain: str, name: str,
                       release_date: str = '1970-01-01') -> tuple[str, str, int]:
    """把 PDB 单链转成 mmCIF 文本字符串。返回 (mmcif_str, 单字母序列, 残基数)。"""
    arrays, seq, nres = parse_pdb_chain(pdb_path, chain)
    y, m, d = (int(x) for x in release_date.split('-'))
    struc = structure.from_atom_arrays(
        name=name,
        release_date=datetime.date(y, m, d),
        **arrays,
    )
    return struc.to_mmcif(), seq, nres


def build_template(pdb_path: str, chain: str, name: str,
                   query_indices_1based: list[int] | None = None) -> dict:
    """为一条链构造一个 AF3 template 元素。

    query_indices_1based: 该链上要提供模板坐标的残基号(1-based, 按链内出现顺序)。
      None -> 全长(该链所有残基都上模板)。
    模板残基顺序与 query 残基顺序一致(见 parse_pdb_chain 的连续重编号),
    故 templateIndices 与 queryIndices 相同, 均为 0-based。
    """
    cif, seq, nres = pdb_chain_to_mmcif(pdb_path, chain, name=name)
    if query_indices_1based is None:
        idx0 = list(range(nres))
    else:
        idx0 = sorted(i - 1 for i in query_indices_1based if 1 <= i <= nres)
    return {
        'mmcif': cif,
        'queryIndices': idx0,
        'templateIndices': idx0,
    }, seq


def build_job(name: str, a_seq: str, pdb_path: str,
              target_chains: list[str], rif_residues_1based: list[int],
              seed: int = 1, use_template: bool = True,
              binder_mode: str = 'rif') -> dict:
    """构造一条 AF3 复合物 job(A=binder 设计序列, 其余=target)。

    binder_mode(仅在 use_template=True 时生效):
      'none' -> A 链不上模板;
      'rif'  -> A 链只对 rif 残基上部分模板;
      'full' -> A 链全长上模板。
    target 链在 use_template=True 时恒为全长模板。
    use_template=False: 所有链 templates=[](无模板对照)。
    unpairedMsa/pairedMsa 留待 run_af3_batch 注入单序列(此处给空串占位)。
    """
    a_entry = {'protein': {'id': 'A', 'sequence': a_seq,
                           'unpairedMsa': '', 'pairedMsa': '', 'templates': []}}
    if use_template and binder_mode != 'none':
        q = None if binder_mode == 'full' else rif_residues_1based
        tmpl, _ = build_template(pdb_path, 'A', name=f'{name}_A',
                                 query_indices_1based=q)
        a_entry['protein']['templates'] = [tmpl]
    sequences = [a_entry]

    for ch in target_chains:
        entry = {'protein': {'id': ch, 'sequence': '',
                             'unpairedMsa': '', 'pairedMsa': '', 'templates': []}}
        tmpl, seq = build_template(pdb_path, ch, name=f'{name}_{ch}',
                                   query_indices_1based=None)
        entry['protein']['sequence'] = seq
        if use_template:
            entry['protein']['templates'] = [tmpl]
        sequences.append(entry)

    return {
        'name': name,
        'modelSeeds': [seed],
        'sequences': sequences,
        'dialect': 'alphafold3',
        'version': 1,
    }


if __name__ == '__main__':
    # 自检: 转一条链, 再用 from_mmcif 读回, 打印关键属性
    import sys
    pdb, ch = sys.argv[1], sys.argv[2]
    cif, seq, n = pdb_chain_to_mmcif(pdb, ch, name=f'tmpl_{ch}')
    print(f'链 {ch}: {n} 残基  seq={seq[:60]}{"..." if len(seq)>60 else ""}')
    back = structure.from_mmcif(cif, name='readback')
    print('读回成功: release_date =', back.release_date,
          '| chains =', list(back.chains))
    print('mmcif 前 500 字符:\n', cif[:500])
