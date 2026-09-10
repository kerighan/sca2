"""The gap gen3 - GDN along training, aggregate and by position, from
runs/cdelta.jsonl (n=3 vs n=3 by default). Source of the table in CATCHUP.md.

    python dump_gap.py [--a shape_dv56] [--b gdn4] [--seeds-b 3,4,5]
"""
import argparse, collections, json, math
import numpy as np


def curves(path, prefix):
    r = collections.defaultdict(list)
    for line in open(path):
        q = json.loads(line)
        if q["model"].startswith(prefix):
            r[q["model"]].append((q["tokens"], q["val"], q.get("pos")))
    return {k: sorted(v) for k, v in r.items()}


def ip(c, x):
    if x < c[0][0] or x > c[-1][0]:
        return None
    for i in range(1, len(c)):
        if c[i][0] >= x:
            (x0, y0, p0), (x1, y1, p1) = c[i - 1], c[i]
            w = (x - x0) / max(x1 - x0, 1)
            return y0 + (y1 - y0) * w, [u + (v - u) * w for u, v in zip(p0, p1)]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="runs/cdelta.jsonl")
    p.add_argument("--a", default="shape_dv56"); p.add_argument("--b", default="gdn4")
    a = p.parse_args(argv)
    A, B = curves(a.log, a.a), curves(a.log, a.b)
    print(f"{a.a}: {sorted(A)}\n{a.b}: {sorted(B)}")
    grid = [20e6 * 1.15 ** i for i in range(40)]
    rows = []
    for x in grid:
        va = [ip(c, x) for c in A.values()]; vb = [ip(c, x) for c in B.values()]
        if None in va or None in vb or not va or not vb:
            continue
        ma = np.mean([v for v, _ in va]); mb = np.mean([v for v, _ in vb])
        pa = np.mean([q for _, q in va], 0); pb = np.mean([q for _, q in vb], 0)
        rows.append((x, ma, mb, ma - mb, pa[0] - pb[0], pa[-1] - pb[-1]))
    print(f"\n{'tok(M)':>7} {a.a:>10} {a.b:>8} {'gap':>8} {'gap@first':>10} {'gap@last':>9}   (position buckets)")
    for x, ma, mb, g, g0, g7 in rows:
        print(f"{x/1e6:7.1f} {ma:10.4f} {mb:8.4f} {g:+8.4f} {g0:+10.4f} {g7:+9.4f}")
    L = np.log([r[0] for r in rows]); g = np.array([r[3] for r in rows]); h = len(rows) // 2
    s2 = np.polyfit(L[h:], g[h:], 1)
    print(f"\nlate-half slope d(gap)/dln(tokens) = {s2[0]:+.3f}"
          f"   -> zero crossing at ~{math.exp(-s2[1]/s2[0])/1e6:.0f}M tokens if it holds")
    for lab, col in ((a.a, 1), (a.b, 2)):
        y = np.array([r[col] for r in rows])
        print(f"{lab}: dval/dln(tokens) first half {np.polyfit(L[:h], y[:h], 1)[0]:+.3f}, "
              f"second half {np.polyfit(L[h:], y[h:], 1)[0]:+.3f}")


if __name__ == "__main__":
    main()
