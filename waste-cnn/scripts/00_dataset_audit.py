"""
00_dataset_audit.py — dataset verification + canonical split generation (v2, faster).

Run once on Kaggle (attach dataset `techsash/waste-classification-data`, CPU is enough):
    python 00_dataset_audit.py --root /kaggle/input/waste-classification-data --out /kaggle/working/audit

Produces (in --out):
    manifest.csv        one row per image file
    splits.csv          canonical split for ALL experiments: path,label,split,group_id,test_leak
    audit_report.json   summary statistics (counts, formats, corrupt files, duplicates, leakage, timings)
    near_dup_pairs.csv  cross-split near-duplicate pairs for visual inspection

Notes:
  * near-duplicate search is exact and fast (pigeonhole banding) instead of comparing every pair;
  * extra copies of the dataset are compared by MD5 only (no image decoding);
  * files are inspected in parallel (one process per CPU), output order unchanged;
  * progress and per-stage timings are printed;
  * split groups merge every pair with dHash distance <= near_dist (plan section 6.1), not only distance 0.
"""
import argparse, hashlib, json, os, random, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
LABELS = {"O": 0, "R": 1}  # paper p.84106: organic = 0, recyclable = 1 (positive class)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def find_roots(root):
    """Return every directory that contains TRAIN/ and TEST/ sub-folders (case-insensitive)."""
    roots = []
    for d, subdirs, _ in os.walk(root):
        names = {s.upper(): s for s in subdirs}
        if "TRAIN" in names and "TEST" in names:
            roots.append((d, names["TRAIN"], names["TEST"]))
    return roots


def file_jobs(root_dir, train_name, test_name, decode):
    jobs = []
    for split_name, official in ((train_name, "TRAIN"), (test_name, "TEST")):
        base = os.path.join(root_dir, split_name)
        for cls in sorted(os.listdir(base)):
            cdir = os.path.join(base, cls)
            if not os.path.isdir(cdir):
                continue
            for fn in sorted(os.listdir(cdir)):
                jobs.append((os.path.join(cdir, fn), official, cls, decode))
    return jobs


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(img, size=8):
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def inspect_file(job):
    """Worker: MD5 of every file; PIL check, decode and dHash only when decode=True."""
    p, official, cls, decode = job
    r = dict(path=p, split_official=official, class_dir=cls,
             label=LABELS.get(cls.upper(), -1), ext=os.path.splitext(p)[1].lower(),
             md5="", dhash="", width=0, height=0, mode="", format="", ok=0, error="")
    try:
        r["md5"] = md5_file(p)
        if decode:
            with Image.open(p) as im:
                im.verify()  # structural check
            with Image.open(p) as im:  # verify() invalidates the object; reopen to decode
                im.load()
                r.update(width=im.width, height=im.height, mode=im.mode, format=im.format or "")
                r["dhash"] = f"{dhash(im):016x}"
            r["ok"] = 1
    except Exception as e:  # corrupted / unreadable
        r["error"] = f"{type(e).__name__}: {e}"[:200]
    return r


def md5_only(job):
    return md5_file(job[0])


def run_pool(fn, jobs, workers, label):
    """Apply fn to jobs in parallel; results keep the input order."""
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, res in enumerate(ex.map(fn, jobs, chunksize=32), 1):
            out.append(res)
            if k % 2000 == 0 or k == len(jobs):
                log(f"  {label}: {k}/{len(jobs)}")
    return out


def near_pairs_banded(hashes, max_dist):
    """Exact search for all pairs (i<j) with Hamming(dHash_i, dHash_j) <= max_dist.

    Pigeonhole: cut the 64 bits into max_dist+1 bands. Two hashes within max_dist differ in at most
    max_dist bits, so at least one band is identical. Only pairs that share an identical band are
    compared, and every candidate is checked with the exact Hamming distance.
    Returns a sorted list of (i, j, distance)."""
    n = len(hashes)
    if max_dist < 0 or n < 2:
        return []
    vals = [int(x, 16) for x in hashes]
    nb = max_dist + 1
    edges = [(64 * k) // nb for k in range(nb + 1)]
    found = {}
    for b in range(nb):
        mask = ((1 << (edges[b + 1] - edges[b])) - 1) << edges[b]
        buckets = defaultdict(list)
        for i, v in enumerate(vals):
            buckets[v & mask].append(i)
        for idxs in buckets.values():
            m = len(idxs)
            for x in range(m):
                i = idxs[x]
                vi = vals[i]
                for y in range(x + 1, m):
                    j = idxs[y]
                    if (i, j) in found:
                        continue
                    d = bin(vi ^ vals[j]).count("1")
                    if d <= max_dist:
                        found[(i, j)] = d
    return sorted((i, j, d) for (i, j), d in found.items())


class UF:
    def __init__(self, n): self.p = list(range(n))
    def f(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def u(self, a, b):
        a, b = self.f(a), self.f(b)
        if a != b: self.p[b] = a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="audit")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--near-dist", type=int, default=4,
                    help="dHash Hamming distance for near-duplicates; also used to group split units")
    ap.add_argument("--skip-near", action="store_true", help="skip the near-duplicate search (MD5 only)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rep = {"near_dist": a.near_dist, "workers": a.workers,
           "group_rule": "union of exact-MD5 groups and dHash pairs with distance <= near_dist"}
    timings = {}

    roots = find_roots(a.root)
    rep["roots_found"] = [r[0] for r in roots]
    if not roots:
        sys.exit(f"No TRAIN/TEST folders under {a.root}")
    primary = roots[0]
    rep["root_used"] = primary[0]
    log(f"roots found: {len(roots)}; using {primary[0]}; workers={a.workers}")

    # ---------- 1. inspect the primary copy (MD5 + decode + dHash) ----------
    t = time.time()
    rows = run_pool(inspect_file, file_jobs(*primary, decode=True), a.workers, "inspect")
    timings["inspect_s"] = round(time.time() - t, 1)

    # ---------- 2. extra copies: MD5 only ----------
    if len(roots) > 1:
        t = time.time()
        first = sorted(r["md5"] for r in rows)
        copies = {}
        for root in roots[1:]:
            md5s = run_pool(md5_only, file_jobs(*root, decode=False), a.workers, f"md5 {root[0]}")
            copies[root[0]] = sorted(md5s) == first
        rep["copies_identical_to_used_root"] = copies
        rep["copies_identical"] = all(copies.values())
        timings["copy_check_s"] = round(time.time() - t, 1)

    # ---------- basic statistics ----------
    rep["total_files"] = len(rows)
    rep["counts"] = {f"{s}/{c}": n for (s, c), n in Counter((r["split_official"], r["class_dir"]) for r in rows).items()}
    rep["extensions"] = dict(Counter(r["ext"] for r in rows))
    rep["non_image_ext"] = [r["path"] for r in rows if r["ext"] not in IMG_EXT][:50]
    rep["formats"] = dict(Counter(r["format"] for r in rows if r["ok"]))
    rep["modes"] = dict(Counter(r["mode"] for r in rows if r["ok"]))
    rep["unknown_label_dirs"] = sorted({r["class_dir"] for r in rows if r["label"] < 0})
    bad = [r for r in rows if not r["ok"]]
    rep["corrupt_count"] = len(bad)
    rep["corrupt_examples"] = [(r["path"], r["error"]) for r in bad[:50]]
    ok = [r for r in rows if r["ok"] and r["label"] >= 0]
    W = np.array([r["width"] for r in ok]); H = np.array([r["height"] for r in ok]); AR = W / H
    q = lambda v: {k: float(np.percentile(v, p)) for k, p in (("min", 0), ("p5", 5), ("median", 50), ("p95", 95), ("max", 100))}
    rep["width"], rep["height"], rep["aspect_ratio"] = q(W), q(H), q(AR)
    rep["smaller_than_224_either_side"] = int(((W < 224) | (H < 224)).sum())
    for s in ("TRAIN", "TEST"):
        lab = [r["label"] for r in ok if r["split_official"] == s]
        rep[f"class_balance_{s}"] = {"organic": lab.count(0), "recyclable": lab.count(1),
                                     "recyclable_share": round(lab.count(1) / max(len(lab), 1), 4)}
    log(f"usable images: {len(ok)} (corrupt: {len(bad)})")

    # ---------- 3. duplicates ----------
    t = time.time()
    uf = UF(len(ok))
    by_md5 = defaultdict(list)
    for i, r in enumerate(ok): by_md5[r["md5"]].append(i)
    exact_groups = [g for g in by_md5.values() if len(g) > 1]
    for g in exact_groups:
        for j in g[1:]: uf.u(g[0], j)
    rep["exact_dup_groups"] = len(exact_groups)
    rep["exact_dup_extra_files"] = sum(len(g) - 1 for g in exact_groups)

    pairs = [] if a.skip_near else near_pairs_banded([r["dhash"] for r in ok], a.near_dist)
    timings["near_dup_s"] = round(time.time() - t, 1)
    log(f"near-duplicate pairs (dHash distance <= {a.near_dist}): {len(pairs)}")
    for i, j, d in pairs:
        uf.u(i, j)  # group every near-duplicate pair, as the plan requires
    rep[f"near_dup_pairs_dist<={a.near_dist}"] = len(pairs)
    rep["near_dup_pairs_dist0"] = sum(1 for p in pairs if p[2] == 0)

    gid = [uf.f(i) for i in range(len(ok))]
    groups = defaultdict(list)
    for i, g in enumerate(gid): groups[g].append(i)
    rep["split_units_groups"] = len(groups)
    rep["largest_group_sizes"] = sorted((len(g) for g in groups.values()), reverse=True)[:10]
    conflict = [g for g in groups.values() if len({ok[i]["label"] for i in g}) > 1]
    rep["label_conflict_groups"] = len(conflict)
    rep["label_conflict_examples"] = [[ok[i]["path"] for i in g] for g in conflict[:20]]

    # ---------- 4. train/test leakage (official TEST vs TRAIN) ----------
    train_idx = {i for i, r in enumerate(ok) if r["split_official"] == "TRAIN"}
    leak = set()
    with open(os.path.join(a.out, "near_dup_pairs.csv"), "w") as f:
        f.write("path_a,path_b,hamming,split_a,split_b,label_a,label_b\n")
        for i, j, d in pairs:
            ri, rj = ok[i], ok[j]
            if ri["split_official"] != rj["split_official"]:
                f.write(f'"{ri["path"]}","{rj["path"]}",{d},{ri["split_official"]},{rj["split_official"]},{ri["label"]},{rj["label"]}\n')
                leak.add(j if i in train_idx else i)
        for g in exact_groups:
            sp = {ok[i]["split_official"] for i in g}
            if len(sp) > 1: leak.update(i for i in g if i not in train_idx)
    n_test = len(ok) - len(train_idx)
    rep["test_images_with_train_near_duplicate"] = len(leak)
    rep["test_leak_share"] = round(len(leak) / max(n_test, 1), 4)

    # ---------- 5. canonical split: official TEST = test; TRAIN -> train/val, stratified, group-aware ----------
    rng = random.Random(a.seed)
    split = {}
    for lab in (0, 1):
        gs = [g for g in groups.values()
              if all(ok[i]["split_official"] == "TRAIN" for i in g)
              and Counter(ok[i]["label"] for i in g).most_common(1)[0][0] == lab]
        gs.sort(key=lambda g: ok[g[0]]["path"]); rng.shuffle(gs)
        target = a.val_frac * sum(len(g) for g in gs); n = 0
        for g in gs:
            s = "val" if n < target else "train"; n += len(g) if s == "val" else 0
            for i in g: split[i] = s
    for i, r in enumerate(ok):
        if r["split_official"] == "TEST": split[i] = "test"
        split.setdefault(i, "train")  # TRAIN images grouped with a TEST image stay in train; flagged via test_leak
    with open(os.path.join(a.out, "splits.csv"), "w") as f:
        f.write("path,label,split,group_id,test_leak\n")
        for i, r in enumerate(ok):
            f.write(f'"{r["path"]}",{r["label"]},{split[i]},{gid[i]},{int(i in leak)}\n')
    rep["split_counts"] = {s: dict(Counter(ok[i]["label"] for i in range(len(ok)) if split[i] == s)) for s in ("train", "val", "test")}
    rep["seed"], rep["val_frac"] = a.seed, a.val_frac
    with open(os.path.join(a.out, "splits.csv"), "rb") as f:
        rep["splits_md5"] = hashlib.md5(f.read()).hexdigest()

    # ---------- 6. manifest and report ----------
    with open(os.path.join(a.out, "manifest.csv"), "w") as f:
        cols = ["path", "split_official", "class_dir", "label", "ext", "md5", "dhash", "width", "height", "mode", "format", "ok", "error"]
        f.write(",".join(cols) + "\n")
        for r in rows: f.write(",".join(f'"{r[c]}"' if c in ("path", "error") else str(r[c]) for c in cols) + "\n")
    rep["timings_s"] = dict(timings, total=round(time.time() - T0, 1))
    with open(os.path.join(a.out, "audit_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    log("done")
    print(json.dumps({k: v for k, v in rep.items() if "examples" not in k and k != "non_image_ext"}, indent=2))


if __name__ == "__main__":
    main()