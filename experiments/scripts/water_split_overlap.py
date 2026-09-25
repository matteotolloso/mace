import sys, logging
import numpy as np
from ase.io import read
sys.path.insert(0, "eval"); logging.disable(logging.CRITICAL)
from reliability import CHECKPOINT_PATTERN, select_best_checkpoints
from pathlib import Path

def key(a):
    return np.round(a.get_positions(), 4).tobytes()

print("## Geometry overlap between CCSDT and BLYP roles (exact match of positions rounded to 1e-4 A)")
print("split | CCSDT test (50) in BLYP train / val / test | CCSDT val in BLYP train | CCSDT train (130) in BLYP train / val / test")
for s in range(5):
    d = Path(f"dataset/water_{s}")
    b = {r: {key(a) for a in read(d / "blyp" / f"{r}.xyz", ":")} for r in ("train", "val", "test")}
    c = {r: [key(a) for a in read(d / "ccsdt" / f"{r}.xyz", ":")] for r in ("train", "val", "test")}
    allb = b["train"] | b["val"] | b["test"]
    assert all(k in allb for r in c for k in c[r]), "CCSDT geometry not found in BLYP"
    cnt = lambda cr, br: sum(k in b[br] for k in c[cr])
    print(f"{s} | n_test={len(c['test'])}: {cnt('test','train')} / {cnt('test','val')} / {cnt('test','test')} | "
          f"n_val={len(c['val'])}: {cnt('val','train')} | n_train={len(c['train'])}: "
          f"{cnt('train','train')} / {cnt('train','val')} / {cnt('train','test')}")

print("\n## Validation-NLL selected epoch per member (0-based, as eval/reliability.py selects it)")
for e in ("wA", "wB", "wC"):
    root = Path(f"experiment_{e}")
    ep = np.array([[int(CHECKPOINT_PATTERN.match(p.name).group("epoch"))
                    for p in select_best_checkpoints(root / f"checkpoints_{s}", root / f"results_{s}",
                                                     None, "loss", "min", None)] for s in range(5)])
    print(f"{e}: members {ep.size}, median {np.median(ep):.1f}, range {ep.min()}-{ep.max()}, "
          f"per-split medians {np.median(ep, axis=1).tolist()}")
