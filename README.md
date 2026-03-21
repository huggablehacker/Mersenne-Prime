# Mersenne-Prime
MERSENNE PRIME HUNTER — Targeting a 100,000,000-digit prime


GOAL:
  Find a Mersenne prime M_p = 2^p - 1 with at least 100,000,000 decimal digits.
  This requires p ≈ 332,192,810 or larger.

  The current world record (Oct 2024) is M_136279841 with ~41 million digits.
  A 100-million-digit prime would be a NEW WORLD RECORD.

METHOD:
  Lucas-Lehmer (LL) test — the gold-standard algorithm for Mersenne primes.
  - For M_p = 2^p - 1 (p prime):
    - 1. s_0 = 4
    - 2. s_{i+1} = (s_i^2 - 2) mod M_p   for i = 0..p-3
    - 3. M_p is prime iff s_{p-2} == 0

  Only prime values of p need to be tested (composite p → composite M_p).

PARALLELISM:
  - Multiprocessing: trial-factor sieve runs in parallel across all CPU cores.
    Multiple LL candidates can be tested simultaneously on separate cores.
  - GPU: if CuPy or Numba is detected, the inner LL squaring loop is offloaded
    to the GPU using NTT (Number Theoretic Transform) multiplication.
  - Worker pool uses a producer/consumer model: N_CPU-1 sieve workers feed
    passing candidates to LL worker processes.

STRUCTURE:
  - Checkpoint: saves progress to disk so you can resume interrupted runs.
  - Self-test: verifies correctness against known Mersenne primes.
  - Sieve: filters candidates using trial factors, eliminating ~70% cheaply.
  - Benchmark: measures LL throughput and projects time for large candidates.

USAGE:
  - python3 mersenne_prime_hunter.py                   (Auto-detect best backend)
  - python3 mersenne_prime_hunter.py --selftest        (Verify algorithm)
  - python3 mersenne_prime_hunter.py --benchmark       (Benchmark all backends)
  - python3 mersenne_prime_hunter.py --workers N       (Override CPU worker count)
  - python3 mersenne_prime_hunter.py --backend cpu     (Force CPU (no GPU))
  - python3 mersenne_prime_hunter.py --backend gpu     (Force GPU (error if absent))
  - python3 mersenne_prime_hunter.py --start P         (Start from exponent P)
  - python3 mersenne_prime_hunter.py --digits D        (Target D decimal digits)
  - python3 mersenne_prime_hunter.py --demo            (Quick demo near p=10000)

================================================================================
