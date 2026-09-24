"""Upload the code and the tokenizer, install deps, build the corpus remotely.

    python -m vast.setup                       # code + deps, then build 15B tokens
    python -m vast.setup --tokens 4_000_000_000
    python -m vast.setup --no-corpus           # code only, corpus later

Why the corpus is BUILT here rather than uploaded. Zyda-2 at 15B tokens is
~30 GB as uint16; uploading that over a rented host's uplink costs hours and
the host streams it from HuggingFace far faster than we can push it. What must
travel is the TOKENIZER -- about 1 MB -- because a corpus built with a
different BPE is a different experiment, and refitting remotely would silently
produce one.

Why --no-deps on the project install. The image ships a CUDA-enabled Torch;
letting pip resolve `torch` from a requirement pulls the CPU wheel over it and
the first training step fails with no GPU. Only the packages the image lacks
are installed, and Torch is never named.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tarfile
import tempfile

from .common import PROJECT, REMOTE, live, run_remote, ssh_target, ROOT

# Everything the training loop imports, plus the corpus builder and the BPE.
UPLOAD = [
    "lapa", "sca2",
    "pretrain.py", "bench_tinypython.py", "prep_zyda.py", "prep_longdoc.py",
    "bench_vocab.py", "plot_lm.py", "curves_t2048.sh",
    "vast",
]
# The tokenizer files, matched by prefix. Fitted locally so both sides agree.
BPE_PREFIX = "zyda_bpe32k"

# The image lacks these. torch is deliberately absent, and so is triton: torch
# ships its own pinned build (pytorch-triton) and a pip `triton` installs over
# it with a version the compiled kernels were not built against.
DEPS = "datasets tokenizers numpy pyarrow einops transformers ninja"

# flash-linear-attention provides the GDN baseline's Triton kernels, and it is
# SHIPPED FROM THIS MACHINE rather than installed from PyPI. The 0.5.2 wheel on
# PyPI unpacks to `fla/layers` and `fla/models` only -- no `fla/ops`, so
# `from fla.ops.gated_delta_rule import chunk_gated_delta_rule` fails and the
# GDN baseline silently disappears from the registry. The local install has the
# full tree (its own RECORD does not list fla/ops either, so it did not come
# from that wheel), and every GDN number in the campaign was produced with it.
# Copying it verbatim is also the stronger choice scientifically: the baseline
# is then byte-identical between the GB10 results and the rented GPU.
FLA_LOCAL = "fla"



def _ship_fla(info: dict) -> None:
    """Copy this machine's flash-linear-attention into the host's site-packages.

    See FLA_LOCAL: the PyPI wheel is missing fla/ops, which is the half the GDN
    baseline imports. Shipping the working tree keeps the baseline identical on
    both machines, which is what makes the two sets of numbers comparable.
    """
    import site
    src = None
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        cand = Path(base) / FLA_LOCAL
        if (cand / "ops").is_dir():
            src = cand
            break
    if src is None:
        raise FileNotFoundError(
            "no local flash-linear-attention with fla/ops; the GDN baseline "
            "cannot run remotely. Install one that has it, or drop the gdn arm."
        )
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
        with tarfile.open(archive.name, "w:gz") as bundle:
            bundle.add(src, arcname="fla",
                       filter=lambda i: None if "__pycache__" in i.name else i)
        print(f"shipping flash-linear-attention from {src} "
              f"({Path(archive.name).stat().st_size/1e6:.1f} MB)")
        subprocess.run(["scp", "-q", "-o", "StrictHostKeyChecking=accept-new",
                        "-P", str(info["ssh_port"]), archive.name,
                        f"root@{info['ssh_host']}:/tmp/fla.tar.gz"],
                       check=True, timeout=600)
    run_remote(info,
               "SP=$(python -c 'import site; print(site.getsitepackages()[0])') && "
               "rm -rf $SP/fla && tar xzf /tmp/fla.tar.gz -C $SP && "
               "python -c 'from fla.ops.gated_delta_rule import chunk_gated_delta_rule'",
               timeout=300)


def _exclude(item: tarfile.TarInfo):
    parts = Path(item.name).parts
    if "__pycache__" in parts or ".git" in parts:
        return None
    if "runtime" in parts:                      # never ship instance metadata
        return None
    if Path(item.name).suffix in {".pt", ".bin", ".png", ".jsonl", ".log"}:
        return None
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=15_000_000_000)
    parser.add_argument("--corpus-name", default="zyda32k", dest="corpus_name")
    parser.add_argument("--no-corpus", action="store_true", dest="no_corpus")
    args = parser.parse_args()

    bpe_files = sorted(ROOT.glob(f"{BPE_PREFIX}-*"))
    if not bpe_files:
        raise FileNotFoundError(
            f"no {BPE_PREFIX}-vocab.json / -merges.txt: fit it first with\n"
            f"  python prep_zyda.py --tokens 2000000000 --out zyda32k_2B --bpe {BPE_PREFIX}"
        )

    info = live(args.slot)
    target = ssh_target(info)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
        with tarfile.open(archive.name, "w:gz") as bundle:
            for name in UPLOAD:
                path = ROOT / name
                if not path.exists():
                    raise FileNotFoundError(f"missing upload item: {path}")
                bundle.add(path, arcname=name, filter=_exclude)
            for f in bpe_files:
                bundle.add(f, arcname=f.name)
        size = Path(archive.name).stat().st_size
        print(f"uploading {size/1e6:.1f} MB ({len(bpe_files)} tokenizer files)")
        subprocess.run(["scp", "-q", "-o", "StrictHostKeyChecking=accept-new",
                        "-P", str(info["ssh_port"]), archive.name,
                        f"root@{info['ssh_host']}:/tmp/{PROJECT}.tar.gz"],
                       check=True, timeout=900)

    _ship_fla(info)

    unpack = (
        f"mkdir -p {REMOTE} && cd {REMOTE} && "
        f"tar xzf /tmp/{PROJECT}.tar.gz && mkdir -p runs plot && "
        f"pip install -q {DEPS} && "
        f"python -c \"import torch, triton; "
        f"from fla.ops.gated_delta_rule import chunk_gated_delta_rule; "
        f"print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
        f"'triton', triton.__version__, 'fla ops ok')\""
    )
    out = run_remote(info, unpack, timeout=900)
    print(out.stdout.strip() or out.stderr.strip())

    smoke = (
        f"cd {REMOTE} && python -c \""
        "from sca2.ref import LayerCfg;"
        "from lapa.layer import LaplaceConfig, LaplaceAttention;"
        "import torch;"
        "m=LaplaceAttention(LaplaceConfig(d=256,M=64,dv=64,L=32,ff=512,max_len=256,"
        "chunk=64,conv=4,gdn_gate=True,lam_free=True,mem_range=(4.,2e4),"
        "layer_scale=True,init_v2=True,v_silu=True)).cuda();"
        "x=torch.randn(2,256,256,device='cuda',requires_grad=True);"
        "y=m(x); y.float().square().mean().backward();"
        "print('layer fwd+bwd ok', tuple(y.shape))\""
    )
    out = run_remote(info, smoke, timeout=600, check=False)
    print(out.stdout.strip() or out.stderr.strip()[-800:])
    if out.returncode != 0:
        raise RuntimeError("the layer does not run on this host; keeping the instance")

    if args.no_corpus:
        print(f"\ncode ready at {REMOTE}. Build the corpus when you want:\n"
              f"  python -m vast.submit --name corpus "
              f"'python prep_zyda.py --tokens {args.tokens} "
              f"--out {args.corpus_name} --bpe {BPE_PREFIX}'")
        return

    build = (
        f"cd {REMOTE} && nohup python prep_zyda.py --tokens {args.tokens} "
        f"--out {args.corpus_name} --bpe {BPE_PREFIX} "
        f"> runs/prep_zyda.log 2>&1 < /dev/null & echo $!"
    )
    out = run_remote(info, build, timeout=120)
    pid = out.stdout.strip()
    (ROOT / "vast" / "runtime").mkdir(parents=True, exist_ok=True)
    (ROOT / "vast" / "runtime" / "corpus.pid").write_text(pid)
    print(f"\ncorpus build started, remote pid {pid} "
          f"({args.tokens/1e9:.0f}B tokens, ~{args.tokens/5e5/3600:.1f} h)")
    print(f"  watch: python -m vast.logs --name prep_zyda")


if __name__ == "__main__":
    main()
