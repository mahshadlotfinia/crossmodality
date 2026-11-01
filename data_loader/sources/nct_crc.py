"""
data_loader/sources/nct_crc.py
Created on October 1, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, Tuple

import pandas as pd

from data_loader.build_utils import make_case_id, rel_to_root, tree_signature


DATASET = "nct_crc"


def source_signature(dcfg: Dict) -> Dict:
    return {"eval_dir": tree_signature(dcfg["eval_dir"]), "dev_dir": tree_signature(dcfg["dev_dir"])}


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    rows = []
    for partition, folder in (("eval", dcfg["eval_dir"]), ("dev", dcfg["dev_dir"])):
        split = os.path.basename(os.path.normpath(folder))
        for cls in sorted(os.listdir(folder)):
            cdir = os.path.join(folder, cls)
            if not os.path.isdir(cdir):
                continue
            for f in sorted(os.listdir(cdir)):
                if f.lower().endswith(tuple(dcfg["image_extensions"])):
                    key = f"{split}{os.sep}{cls}{os.sep}{os.path.splitext(f)[0]}"
                    rows.append((make_case_id(DATASET, key), key, cls, split, partition, rel_to_root(os.path.join(cdir, f), root)))
    t = pd.DataFrame(rows, columns=["case_id", "key", "tissue_class", "split", "partition", "image_path"])
    return t.assign(dataset=DATASET, group_id=t["key"])


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cls = ev["key"].str.split(os.sep).str[1]
    pos = cls == fcfg["source_label"]
    neg = ~pos & (cls != dcfg["background_class"])
    return ev[pos].copy(), ev[neg].copy()
