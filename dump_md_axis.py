"""Read runs/md_axis.jsonl and read off the Mc -> Md trade at equal parameters.

Two contrasts, both at ~743.6k layer params, 4 layers, one epoch over the same
177.4M-token corpus:

  md4_dv32 vs deep4     the dv -> Mc trade  (dv 64/Mc 128 -> dv 32/Mc 378)
  md16_dv32 vs md4_dv32 the Mc -> Md trade  (Mc 378/Md 4 -> Mc 166/Md 16)

gdn4 (2.8379) is the target. Endpoints only -- the sweep established that fitted
slopes over a partial run carry ~0.2 nats of projection noise.

    python dump_md_axis.py
"""
import json
from collections import defaultdict
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
# state floats per layer, computed from the config each arm was launched with
STATE = {"md4_dv32": (24192, 256), "md16_dv32": (10624, 1024),
         "deep4": (16384, 512), "gdn4": (10800, 0), "decay2": (16384, 512),
         "decay_deep4": (16384, 512)}


def load(path):
    d = defaultdict(list)
    if not path.exists():
        return d
    for line in path.open():
        r = json.loads(line)
        d[r["model"]].append(r)
    for v in d.values():
        v.sort(key=lambda r: r["step"])
    return d


def main():
    d = load(RUNS / "md_axis.jsonl")
    prev = load(RUNS / "confirm_pycode.jsonl")
    for k in ("deep4", "gdn4"):
        if k in prev:
            d[k] = prev[k]
    if not d:
        print("no runs/md_axis.jsonl yet")
        return

    common = min(v[-1]["tokens"] for v in d.values())
    print(f"common token count {common/1e6:.1f}M of 177.4M\n")
    print(f"{'arm':12s} {'tok/s':>7} {'val@common':>11} {'val end':>8} "
          f"{'pos slope':>10} {'C state':>8} {'D state':>8}")
    print("-" * 70)
    rows = []
    for k, v in d.items():
        ok = [r for r in v if r["tokens"] <= common]
        cut = ok[-1] if ok else v[0]
        p = v[-1].get("pos")
        slope = (p[-1] - p[0]) if p else float("nan")
        c, dd = STATE.get(k, (0, 0))
        rows.append((cut["val"], k, v[-1], slope, c, dd))
    for val, k, last, slope, c, dd in sorted(rows):
        print(f"{k:12s} {last['tok_s']:>7} {val:>11.4f} {last['val']:>8.4f} "
              f"{slope:>+10.3f} {c:>8d} {dd:>8d}")

    def val(k):
        ok = [r for r in d[k] if r["tokens"] <= common]
        return (ok[-1] if ok else d[k][0])["val"]

    print()
    if "md4_dv32" in d and "deep4" in d:
        print(f"dv 64->32 at Md=4  (Mc 128->378): {val('md4_dv32')-val('deep4'):+.4f}")
    if "md16_dv32" in d and "md4_dv32" in d:
        print(f"Mc->Md at dv=32    (Md 4->16)   : {val('md16_dv32')-val('md4_dv32'):+.4f}")
    if "gdn4" in d:
        best = min((val(k), k) for k in d if k != "gdn4")
        print(f"best SCA2 ({best[1]}) vs gdn4      : {best[0]-val('gdn4'):+.4f}")


if __name__ == "__main__":
    main()
