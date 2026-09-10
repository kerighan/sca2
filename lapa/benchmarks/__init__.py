"""Benchmarks for Laplace Attention against its baselines. Each is a script:

    python -m lapa.benchmarks.copy --help       copy capacity: per-length accuracy curves
    python -m lapa.benchmarks.copy plot ...     figures from a JSONL log

Baselines (same prefill/step/init_state protocol as LaplaceAttention): lapa.benchmarks.baselines
"""
