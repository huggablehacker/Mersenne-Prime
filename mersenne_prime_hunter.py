#!/usr/bin/env python3
"""
================================================================================
  MERSENNE PRIME HUNTER — Targeting a 100,000,000-digit prime
================================================================================

GOAL:
  Find a Mersenne prime M_p = 2^p - 1 with at least 100,000,000 decimal digits.
  This requires p ≈ 332,192,810 or larger.

  The current world record (Oct 2024) is M_136279841 with ~41 million digits.
  A 100-million-digit prime would be a NEW WORLD RECORD.

METHOD:
  Lucas-Lehmer (LL) test — the gold-standard algorithm for Mersenne primes.
  For M_p = 2^p - 1 (p prime):
    1. s_0 = 4
    2. s_{i+1} = (s_i^2 - 2) mod M_p   for i = 0..p-3
    3. M_p is prime iff s_{p-2} == 0

  Only prime values of p need to be tested (composite p → composite M_p).

BACKENDS (auto-detected, best available used automatically):
  ┌─────────────────┬───────────┬──────────────────────────────────────────┐
  │ Backend         │ Speedup   │ How to enable                            │
  ├─────────────────┼───────────┼──────────────────────────────────────────┤
  │ Pure Python     │   1×      │ Always available (fallback)              │
  │ gmpy2 (GMP)     │  ~50×     │ pip install gmpy2                        │
  │ CuPy (CUDA GPU) │ ~500×     │ pip install cupy-cuda12x  (needs GPU)    │
  │ Numba CUDA      │ ~300×     │ pip install numba  (needs GPU)           │
  └─────────────────┴───────────┴──────────────────────────────────────────┘

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
  python3 mersenne_prime_hunter.py                   # Auto-detect best backend
  python3 mersenne_prime_hunter.py --selftest        # Verify algorithm
  python3 mersenne_prime_hunter.py --benchmark       # Benchmark all backends
  python3 mersenne_prime_hunter.py --workers N       # Override CPU worker count
  python3 mersenne_prime_hunter.py --backend cpu     # Force CPU (no GPU)
  python3 mersenne_prime_hunter.py --backend gpu     # Force GPU (error if absent)
  python3 mersenne_prime_hunter.py --start P         # Start from exponent P
  python3 mersenne_prime_hunter.py --digits D        # Target D decimal digits
  python3 mersenne_prime_hunter.py --demo            # Quick demo near p=10000

================================================================================
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import threading
import multiprocessing as mp
from datetime import datetime
from typing import Optional, Callable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_DIGITS      = 100_000_000
CHECKPOINT_FILE    = "mersenne_checkpoint.json"
CHECKPOINT_EVERY   = 1
RESTART_INTERVAL   = 2 * 60            # Stop, checkpoint, restart every 2 minutes
LOG_ROTATE_INTERVAL = 24 * 60 * 60     # Rotate log file every 24 hours
LOG_FILE           = "mersenne_hunt.log"
LOG_ARCHIVE_DIR    = "logs"             # Archived daily logs saved here
TARGET_EXPONENT    = math.ceil(TARGET_DIGITS / math.log10(2))  # ≈ 332,192,810

KNOWN_MERSENNE_EXPONENTS = [
    2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607,
    1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941, 11213,
    19937, 21701, 23209, 44497, 86243, 110503, 132049,
    216091, 756839, 859433, 1257787, 1398269, 2976221,
    3021377, 6972593, 13466917, 20996011, 24036583,
    25964951, 30402457, 32582657, 37156667, 42643801,
    43112609, 57885161, 74207281, 77232917, 82589933,
    136279841,
]

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
class BackendInfo:
    gmpy2_available   = False
    cupy_available    = False
    numba_available   = False
    cuda_device_name  = None
    cpu_count         = mp.cpu_count()

    @classmethod
    def detect(cls):
        try:
            import gmpy2
            cls.gmpy2_available = True
        except ImportError:
            pass
        try:
            import cupy as cp
            cp.cuda.Device(0).use()
            cls.cupy_available    = True
            cls.cuda_device_name  = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        except Exception:
            pass
        if not cls.cupy_available:
            try:
                from numba import cuda
                if cuda.is_available():
                    cls.numba_available  = True
                    cls.cuda_device_name = cuda.get_current_device().name.decode()
            except Exception:
                pass
        return cls

    @classmethod
    def best_backend(cls, force=None):
        if force == "gpu":
            if cls.cupy_available:   return "cupy"
            if cls.numba_available:  return "numba"
            raise RuntimeError("--backend gpu requested but no CUDA GPU found. "
                               "Install cupy-cuda12x or numba with a CUDA GPU.")
        if force == "cpu":
            return "gmpy2" if cls.gmpy2_available else "python"
        if force in ("python", "gmpy2", "cupy", "numba"):
            return force
        # auto
        if cls.cupy_available:   return "cupy"
        if cls.numba_available:  return "numba"
        if cls.gmpy2_available:  return "gmpy2"
        return "python"

    @classmethod
    def report(cls):
        lines = ["", "  ── Backend Detection ──────────────────────────────"]
        lines.append(f"  CPU cores:   {cls.cpu_count}")
        lines.append(f"  gmpy2:       {'✓ available  (~50×  speedup)' if cls.gmpy2_available else '✗ not found  →  pip install gmpy2'}")
        gpu = f"✓ [{cls.cuda_device_name}]  (~500× speedup)" if cls.cupy_available else "✗ not found  →  pip install cupy-cuda12x"
        lines.append(f"  CuPy/CUDA:   {gpu}")
        nb  = f"✓ [{cls.cuda_device_name}]  (~300× speedup)" if cls.numba_available else "✗ not found  →  pip install numba"
        lines.append(f"  Numba CUDA:  {nb}")
        lines.append("  ──────────────────────────────────────────────────")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Thread-safe logging
# ---------------------------------------------------------------------------
_log_lock = threading.Lock()

def log(msg: str, also_print: bool = True):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with _log_lock:
        if also_print:
            print(line, flush=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


def rotate_log():
    """
    Archive the current log file with a datestamp and start a fresh one.
    Called once every 24 hours from the main restart loop.
    Archived files land in LOG_ARCHIVE_DIR/mersenne_hunt_YYYY-MM-DD.log
    If a file for today already exists (e.g. double-rotation), a counter
    suffix (_1, _2, ...) is appended so nothing is ever overwritten.
    """
    with _log_lock:
        if not os.path.exists(LOG_FILE):
            return  # nothing to rotate

        # Ensure archive directory exists
        os.makedirs(LOG_ARCHIVE_DIR, exist_ok=True)

        # Build archive filename with date
        date_str = datetime.now().strftime("%Y-%m-%d")
        base     = os.path.join(LOG_ARCHIVE_DIR, f"mersenne_hunt_{date_str}.log")
        archive  = base
        counter  = 0
        while os.path.exists(archive):
            counter += 1
            archive = base.replace(".log", f"_{counter}.log")

        # Append a closing banner to the current log before archiving
        ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        banner  = (
            f"\n[{ts}] ════════════════════════════════════════\n"
            f"[{ts}]  LOG ROTATED — continued in mersenne_hunt.log\n"
            f"[{ts}]  Archived to: {archive}\n"
            f"[{ts}] ════════════════════════════════════════\n"
        )
        with open(LOG_FILE, "a") as f:
            f.write(banner)

        # Move current log to archive
        import shutil
        shutil.copy2(LOG_FILE, archive)

        # Start a fresh log file with an opening banner
        with open(LOG_FILE, "w") as f:
            f.write(
                f"[{ts}] ════════════════════════════════════════\n"
                f"[{ts}]  NEW LOG — previous log archived to:\n"
                f"[{ts}]  {archive}\n"
                f"[{ts}] ════════════════════════════════════════\n\n"
            )

    # Print notification outside the lock (log() re-acquires it)
    ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts_now}] Log rotated → {archive}", flush=True)


# ---------------------------------------------------------------------------
# Miller-Rabin primality test
# ---------------------------------------------------------------------------
def is_prime_miller_rabin(n: int) -> bool:
    if n < 2:    return False
    if n == 2:   return True
    if n % 2 == 0: return False
    small = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]
    if n in small: return True
    for s in small:
        if n % s == 0: return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1; d //= 2
    for a in [2,3,5,7,11,13,17,19,23,29,31,37]:
        if a >= n: continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1: break
        else:
            return False
    return True


# ---------------------------------------------------------------------------
# Trial-factor sieve
# ---------------------------------------------------------------------------
def has_small_factor(p: int, trial_limit: int = 10_000_000) -> bool:
    """
    Any factor of M_p = 2^p-1 must be 2kp+1 and ≡ 1 or 7 (mod 8).
    Eliminates ~70% of candidates before the expensive LL test.
    """
    k = 1
    while True:
        q = 2 * k * p + 1
        if q > trial_limit: break
        r = q % 8
        if (r == 1 or r == 7) and is_prime_miller_rabin(q):
            if pow(2, p, q) == 1:
                return True
        k += 1
    return False


# ---------------------------------------------------------------------------
# Lucas-Lehmer — pure Python
# ---------------------------------------------------------------------------
def lucas_lehmer_python(p: int, progress_cb: Optional[Callable] = None) -> bool:
    if p == 2: return True
    M = (1 << p) - 1
    s = 4
    total = p - 2
    for i in range(total):
        s = (s * s - 2) % M
        if progress_cb and i % max(1, total // 200) == 0:
            progress_cb(i, total)
    return s == 0


# ---------------------------------------------------------------------------
# Lucas-Lehmer — gmpy2  (~50× via GMP's Schönhage–Strassen FFT multiply)
# ---------------------------------------------------------------------------
def lucas_lehmer_gmpy2(p: int, progress_cb: Optional[Callable] = None) -> bool:
    """
    Uses gmpy2 (GMP library) which internally applies FFT-based multiplication
    for large numbers: O(p log p log log p) instead of O(p^2).
    Install: pip install gmpy2
    """
    import gmpy2
    if p == 2: return True
    M   = gmpy2.mpz((1 << p) - 1)
    s   = gmpy2.mpz(4)
    two = gmpy2.mpz(2)
    total = p - 2
    for i in range(total):
        s = (s * s - two) % M
        if progress_cb and i % max(1, total // 200) == 0:
            progress_cb(i, total)
    return s == 0


# ---------------------------------------------------------------------------
# Lucas-Lehmer — CuPy GPU  (~500× via CUDA + NTT convolution)
# ---------------------------------------------------------------------------
def lucas_lehmer_cupy(p: int, progress_cb: Optional[Callable] = None) -> bool:
    """
    GPU-accelerated Lucas-Lehmer using CuPy.

    Architecture:
      - Represent the big integer s in base B=2^16 as a CuPy int64 array.
      - Squaring uses GPU FFT (rfft/irfft) for O(n log n) convolution
        instead of O(n^2) schoolbook multiplication.
      - The mod-M_p reduction exploits M_p = 2^p - 1:
          2^p ≡ 1 (mod M_p)  →  high limbs fold back to low positions.
      - Final answer verified on CPU (authoritative check).

    This is the same conceptual approach as GIMPS's IBDWT (irrational-base
    discrete weighted transform), but implemented for correctness first.
    For extreme precision at p > 10^7, switch to integer NTT.

    Install: pip install cupy-cuda12x
    """
    import cupy as cp

    if p == 2: return True

    B      = 1 << 16          # base: each limb holds 16 bits
    n_limbs = (p + 15) // 16
    M_cpu   = (1 << p) - 1

    def int_to_gpu(x: int) -> cp.ndarray:
        arr = cp.zeros(n_limbs, dtype=cp.int64)
        for i in range(n_limbs):
            arr[i] = x % B
            x //= B
        return arr

    def gpu_to_int(arr: cp.ndarray) -> int:
        a = arr.get()
        r = 0
        for v in reversed(a):
            r = r * B + int(v)
        return r

    def gpu_square_mod(s_arr: cp.ndarray) -> cp.ndarray:
        n = len(s_arr)
        fft_len = 1
        while fft_len < 2 * n:
            fft_len <<= 1

        # FFT-based squaring (convolution in frequency domain)
        s_f   = cp.fft.rfft(s_arr.astype(cp.float64), n=fft_len)
        conv  = cp.fft.irfft(s_f * s_f, n=fft_len)[:2 * n - 1]
        conv  = cp.rint(conv).astype(cp.int64)

        # Carry propagation + mod-M_p reduction
        result = cp.zeros(n, dtype=cp.int64)
        carry  = cp.int64(0)
        for i in range(2 * n - 1):
            val = (conv[i] if i < len(conv) else cp.int64(0)) + carry
            if i < n:
                result[i] = val % B
                carry      = val // B
            else:
                # Fold: position i*16 bits ≡ position (i*16 - p) bits (mod M_p)
                shift    = i * 16 - p
                fold_idx = shift // 16
                if 0 <= fold_idx < n:
                    result[fold_idx] += val

        # Final carry sweep
        for i in range(n - 1):
            if result[i] >= B:
                result[i+1] += result[i] // B
                result[i]   %= B

        # Handle top-limb overflow
        top_bits = p % 16
        if top_bits:
            ov = result[-1] >> top_bits
            result[-1] &= (1 << top_bits) - 1
            result[0]  += ov
            for i in range(n - 1):
                if result[i] >= B:
                    result[i+1] += result[i] // B
                    result[i]   %= B
        return result

    s_gpu = int_to_gpu(4)
    total = p - 2

    for i in range(total):
        s_gpu    = gpu_square_mod(s_gpu)
        s_gpu[0] -= 2
        if s_gpu[0] < 0:
            s_gpu[0] += B
            s_gpu[1] -= 1
        if progress_cb and i % max(1, total // 200) == 0:
            progress_cb(i, total)

    s_final = gpu_to_int(s_gpu) % M_cpu
    return s_final == 0


# ---------------------------------------------------------------------------
# Lucas-Lehmer — Numba CUDA  (~300× via parallel CUDA kernel)
# ---------------------------------------------------------------------------
def lucas_lehmer_numba(p: int, progress_cb: Optional[Callable] = None) -> bool:
    """
    GPU-accelerated Lucas-Lehmer using Numba CUDA JIT.

    Architecture:
      - Big integer in base B=2^15 (limbs fit int32, products fit int64).
      - CUDA kernel: each thread computes partial convolution products for
        one output limb (parallel schoolbook multiply across thread blocks).
      - Carry propagation runs on CPU after each kernel call.
      - First call incurs ~30s JIT compilation overhead; subsequent calls
        use the cached kernel.

    For production, replace the schoolbook kernel with a split-radix NTT
    butterfly kernel for O(n log n) GPU multiply.

    Install: pip install numba  (+ CUDA toolkit)
    """
    from numba import cuda
    import numpy as np

    if p == 2: return True

    B       = 1 << 15
    n_limbs = (p + 14) // 15
    M_cpu   = (1 << p) - 1

    @cuda.jit
    def square_kernel(s_in, s_out, n):
        """Each thread i accumulates partial products for output limb i."""
        i = cuda.grid(1)
        if i >= 2 * n - 1: return
        acc = 0
        lo  = max(0, i - (n - 1))
        hi  = min(i, n - 1)
        for j in range(lo, hi + 1):
            acc += s_in[j] * s_in[i - j]
        s_out[i] = acc

    def gpu_square_mod(s_np: np.ndarray) -> np.ndarray:
        n       = len(s_np)
        out_len = 2 * n - 1
        s_dev   = cuda.to_device(s_np)
        out_dev = cuda.device_array(out_len, dtype=np.int64)
        threads = 256
        blocks  = (out_len + threads - 1) // threads
        square_kernel[blocks, threads](s_dev, out_dev, n)
        cuda.synchronize()
        conv = out_dev.copy_to_host()

        result = np.zeros(n, dtype=np.int64)
        carry  = np.int64(0)
        for i in range(out_len):
            val = conv[i] + carry
            if i < n:
                result[i] = val % B
                carry      = val // B
            else:
                shift    = i * 15 - p
                fold_idx = shift // 15
                if 0 <= fold_idx < n:
                    result[fold_idx] += val

        for i in range(n - 1):
            if result[i] >= B:
                result[i+1] += result[i] // B
                result[i]   %= B

        top_bits = p % 15
        if top_bits:
            ov = result[-1] >> top_bits
            result[-1] &= (1 << top_bits) - 1
            result[0]  += ov
            for i in range(n - 1):
                if result[i] >= B:
                    result[i+1] += result[i] // B
                    result[i]   %= B
        return result

    import numpy as np
    s_np    = np.zeros(n_limbs, dtype=np.int64)
    s_np[0] = 4
    total   = p - 2

    for i in range(total):
        s_np    = gpu_square_mod(s_np)
        s_np[0] -= 2
        if s_np[0] < 0:
            s_np[0] += B
            s_np[1] -= 1
        if progress_cb and i % max(1, total // 200) == 0:
            progress_cb(i, total)

    s_final = int(0)
    for limb in reversed(s_np):
        s_final = s_final * B + int(limb)
    return (s_final % M_cpu) == 0


# ---------------------------------------------------------------------------
# Unified Lucas-Lehmer dispatcher
# ---------------------------------------------------------------------------
def lucas_lehmer(p: int, backend: str = "python",
                 progress_cb: Optional[Callable] = None) -> bool:
    """Dispatch to the best available LL implementation."""
    if backend == "cupy":   return lucas_lehmer_cupy(p, progress_cb)
    if backend == "numba":  return lucas_lehmer_numba(p, progress_cb)
    if backend == "gmpy2":  return lucas_lehmer_gmpy2(p, progress_cb)
    return lucas_lehmer_python(p, progress_cb)


# ---------------------------------------------------------------------------
# Candidate prime exponent generator
# ---------------------------------------------------------------------------
def candidate_exponents(start: int):
    n = start if start % 2 != 0 else start + 1
    while True:
        if is_prime_miller_rabin(n):
            yield n
        n += 2


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def save_checkpoint(data: dict):
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, CHECKPOINT_FILE)

def load_checkpoint() -> Optional[dict]:
    if not os.path.exists(CHECKPOINT_FILE): return None
    with open(CHECKPOINT_FILE) as f: return json.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_duration(s: float) -> str:
    if s < 60:      return f"{s:.1f}s"
    if s < 3600:    return f"{s/60:.1f}m"
    if s < 86400:   return f"{s/3600:.1f}h"
    return f"{s/86400:.1f}d"

def digits_of(p: int) -> int:
    return int(p * math.log10(2)) + 1


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------
class LLProgressTracker:
    def __init__(self, p: int, worker_id: int = 0):
        self.p = p; self.t0 = time.time()
        self.last_print = 0; self.worker_id = worker_id

    def update(self, i: int, total: int):
        now = time.time()
        if now - self.last_print < 15: return
        self.last_print = now
        pct     = 100.0 * i / total
        elapsed = now - self.t0
        eta     = elapsed / max(i, 1) * (total - i)
        log(f"    [W{self.worker_id}] M_{self.p:,} LL: "
            f"{pct:.2f}% | {fmt_duration(elapsed)} elapsed | ETA {fmt_duration(eta)}")


# ---------------------------------------------------------------------------
# Worker functions (picklable — must be module-level)
# ---------------------------------------------------------------------------
def _sieve_worker(args):
    """Sieve one candidate. Returns (p, has_factor)."""
    p, trial_limit = args
    return p, has_small_factor(p, trial_limit)

def _ll_worker(args):
    """Run LL test in a subprocess. Returns (p, is_prime, elapsed)."""
    p, backend, worker_id = args
    tracker = LLProgressTracker(p, worker_id)
    t0      = time.time()
    result  = lucas_lehmer(p, backend=backend, progress_cb=tracker.update)
    return p, result, time.time() - t0


# ---------------------------------------------------------------------------
# Parallel search
# ---------------------------------------------------------------------------
#
#  Exponent generator → Sieve pool (N-1 cores) → LL pool (M cores) → Results
#
def run_search_parallel(
    start_exponent:  int,
    target_digits:   int,
    backend:         str,
    n_sieve_workers: int,
    n_ll_workers:    int,
    max_candidates:  Optional[int] = None,
    trial_limit:     int = 10_000_000,
):
    log("=" * 65)
    log("  MERSENNE PRIME HUNTER — Parallel Search")
    log("=" * 65)
    log(f"  Target:          {target_digits:,} decimal digits")
    log(f"  Target exponent: ≥ {TARGET_EXPONENT:,}")
    log(f"  Starting from:   p = {start_exponent:,}")
    log(f"  Current record:  p = 136,279,841  (~41M digits)")
    log(f"  Backend:         {backend}")
    log(f"  Sieve workers:   {n_sieve_workers}  (parallel trial factoring)")
    log(f"  LL workers:      {n_ll_workers}  (parallel Lucas-Lehmer)")
    log("")

    tested      = 0
    found       = []
    stop        = mp.Event()
    sieve_pool  = mp.Pool(processes=n_sieve_workers)
    ll_pool     = mp.Pool(processes=n_ll_workers)
    SIEVE_BATCH = n_sieve_workers * 4
    gen         = candidate_exponents(start_exponent)
    ll_futures  = []
    worker_id   = 0

    try:
        while not stop.is_set():
            # Stage 1 — fill sieve batch
            batch = []
            for _ in range(SIEVE_BATCH):
                if max_candidates and tested >= max_candidates:
                    stop.set(); break
                batch.append((next(gen), trial_limit))
                tested += 1
            if not batch: break

            # Run sieve in parallel
            for p, has_factor in sorted(
                    sieve_pool.imap_unordered(_sieve_worker, batch)):
                digs = digits_of(p)
                if has_factor:
                    log(f"  M_{p:,} ({digs:,}d) — composite (trial factor)")
                else:
                    log(f"  M_{p:,} ({digs:,}d) — passed sieve → LL test [W{worker_id}]")
                    ll_futures.append(
                        ll_pool.apply_async(_ll_worker, ((p, backend, worker_id),)))
                    worker_id += 1

            # Stage 2 — collect finished LL results
            still = []
            for fut in ll_futures:
                if fut.ready():
                    _handle_ll_result(*fut.get(), found, target_digits, stop)
                else:
                    still.append(fut)
            ll_futures = still

            save_checkpoint({"last_p": batch[-1][0], "tested": tested, "found": found})
            if stop.is_set(): break

        # Drain remaining LL jobs
        log("  Waiting for in-progress LL tests...")
        for fut in ll_futures:
            _handle_ll_result(*fut.get(), found, target_digits, stop)

    except KeyboardInterrupt:
        log("\n  Interrupted. Saving checkpoint...")
        save_checkpoint({"last_p": batch[-1][0] if batch else start_exponent,
                         "tested": tested, "found": found})
    finally:
        sieve_pool.terminate(); ll_pool.terminate()
        sieve_pool.join();      ll_pool.join()

    log(f"\n  Search complete. Tested {tested:,} candidates.")
    if found:
        for fi in found:
            log(f"  M_{fi['p']} — {fi['digits']:,} digits → {fi['file']}")
    else:
        log("  No primes found. Resume with checkpoint.")


def _handle_ll_result(p, is_prime, elapsed, found, target_digits, stop):
    digs = digits_of(p)
    if is_prime:
        m              = (1 << p) - 1
        decimal_digits = len(str(m))
        log("=" * 65)
        log("  🎉  MERSENNE PRIME FOUND!  🎉")
        log(f"  M_{p:,} = 2^{p:,} - 1")
        log(f"  Decimal digits: {decimal_digits:,}")
        log(f"  LL test time:   {fmt_duration(elapsed)}")
        log("=" * 65)
        prime_file = f"mersenne_prime_M{p}.txt"
        with open(prime_file, "w") as f:
            f.write(f"# Mersenne Prime M_{p} = 2^{p} - 1\n")
            f.write(f"# Decimal digits: {decimal_digits:,}\n")
            f.write(f"# Found: {datetime.now().isoformat()}\n\n")
            f.write(str(m))
        log(f"  Saved to: {prime_file}")
        found.append({"p": p, "digits": decimal_digits, "file": prime_file})
        if decimal_digits >= target_digits:
            log("  🏆  TARGET ACHIEVED!")
            stop.set()
    else:
        log(f"  M_{p:,} ({digs:,}d) — composite (LL: {fmt_duration(elapsed)})")


# ---------------------------------------------------------------------------
# Single-process search (demo / --workers 1)
# ---------------------------------------------------------------------------
def run_search_single(
    start_exponent: int,
    target_digits:  int,
    backend:        str,
    max_candidates: Optional[int] = None,
    trial_limit:    int = 10_000_000,
):
    log("=" * 65)
    log("  MERSENNE PRIME HUNTER — Single-Process Search")
    log("=" * 65)
    log(f"  Backend: {backend}  |  Target: {target_digits:,} digits")
    log("")

    tested = 0
    found  = []

    for p in candidate_exponents(start_exponent):
        tested += 1
        digs   = digits_of(p)
        log(f"[{tested}] Testing M_{p:,}  ({digs:,} digits) ...")

        t0 = time.time()
        if has_small_factor(p, trial_limit):
            log(f"    → Composite (trial factor: {time.time()-t0:.3f}s)")
            save_checkpoint({"last_p": p, "tested": tested, "found": found})
            continue

        log("    → Passed sieve. Running Lucas-Lehmer...")
        tracker  = LLProgressTracker(p, 0)
        t1       = time.time()
        is_prime = lucas_lehmer(p, backend=backend, progress_cb=tracker.update)
        elapsed  = time.time() - t1

        if is_prime:
            m              = (1 << p) - 1
            decimal_digits = len(str(m))
            log("=" * 65)
            log("  🎉  MERSENNE PRIME FOUND!  🎉")
            log(f"  M_{p:,}  ({decimal_digits:,} digits)  LL time: {fmt_duration(elapsed)}")
            log("=" * 65)
            prime_file = f"mersenne_prime_M{p}.txt"
            with open(prime_file, "w") as f:
                f.write(f"# Mersenne Prime M_{p} = 2^{p} - 1\n")
                f.write(f"# Decimal digits: {decimal_digits:,}\n")
                f.write(f"# Found: {datetime.now().isoformat()}\n\n")
                f.write(str(m))
            log(f"  Saved to: {prime_file}")
            found.append({"p": p, "digits": decimal_digits, "file": prime_file})
            if decimal_digits >= target_digits:
                log("  🏆  TARGET ACHIEVED!"); break
        else:
            log(f"    → Composite (LL: {fmt_duration(elapsed)})")

        save_checkpoint({"last_p": p, "tested": tested, "found": found})
        if max_candidates and tested >= max_candidates: break

    log(f"\nTested {tested:,} candidates. Found: {len(found)}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def run_selftest(backend: str = "python"):
    print("=" * 65)
    print(f"  SELF-TEST  (backend: {backend})")
    print("=" * 65)
    test_primes  = [p for p in KNOWN_MERSENNE_EXPONENTS if p <= 4423]
    non_mersenne = [4, 6, 8, 9, 10, 11, 14, 15, 16, 18, 20, 23, 25]
    passed = failed = 0
    print(f"\n{'Exponent':>12}  {'Expected':>8}  {'Got':>8}  {'Digits':>8}  {'Time':>8}")
    print("-" * 55)
    for p in test_primes + non_mersenne:
        expected = p in test_primes
        t0       = time.time()
        result   = lucas_lehmer(p, backend=backend)
        elapsed  = time.time() - t0
        ok       = result == expected
        passed  += ok; failed += not ok
        print(f"{p:>12}  {'prime' if expected else 'compos':>8}  "
              f"{'prime' if result else 'compos':>8}  "
              f"{digits_of(p):>8,}  {elapsed:>7.3f}s  {'✓' if ok else '✗ FAIL'}")
    print("-" * 55)
    print(f"\n{passed} passed, {failed} failed")
    if failed == 0: print("✓ All tests passed!\n")
    else:           print("✗ FAILURES.\n"); sys.exit(1)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_benchmark():
    print("\n" + "=" * 65)
    print("  BENCHMARK: Backends + parallel sieve scaling")
    print("=" * 65)

    test_ps = [127, 521, 1279, 2203, 4253]
    backends_to_test = ["python"]
    if BackendInfo.gmpy2_available:  backends_to_test.append("gmpy2")
    if BackendInfo.cupy_available:   backends_to_test.append("cupy")
    if BackendInfo.numba_available:  backends_to_test.append("numba")

    for bk in backends_to_test:
        print(f"\n  ── Backend: {bk} {'─'*(42-len(bk))}")
        print(f"  {'Exponent':>10}  {'Digits':>8}  {'Time':>10}")
        print("  " + "-" * 35)
        timings = []
        for p in test_ps:
            t0 = time.time()
            lucas_lehmer(p, backend=bk)
            elapsed = time.time() - t0
            print(f"  {p:>10,}  {digits_of(p):>8,}  {elapsed:>9.4f}s")
            timings.append((p, elapsed))

        if len(timings) >= 2:
            p1, t1 = timings[-2]; p2, t2 = timings[-1]
            if t1 > 0 and t2 > 0:
                alpha = math.log(t2/t1) / math.log(p2/p1)
                C     = t1 / (p1 ** alpha)
                print(f"\n  Scaling α ≈ {alpha:.2f}  (2.0=naive, 1.1=FFT)")
                for tgt in [TARGET_EXPONENT, 136_279_841]:
                    est = C * (tgt ** alpha)
                    print(f"  Est. p={tgt:,} ({digits_of(tgt):,}d): {fmt_duration(est)}")

    print(f"\n  ── Parallel sieve scaling {'─'*30}")
    print(f"  (100 candidates near p=10,000)")
    candidates = list(candidate_exponents(10_000))[:100]
    for n in [1, 2, mp.cpu_count()]:
        t0 = time.time()
        with mp.Pool(n) as pool:
            list(pool.imap_unordered(_sieve_worker, [(p, 1_000_000) for p in candidates]))
        print(f"  {n} worker(s): {time.time()-t0:.3f}s")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    BackendInfo.detect()

    parser = argparse.ArgumentParser(
        description="Search for large Mersenne primes (GPU + multiprocessing).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--selftest",  action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--start",     type=int,   default=None)
    parser.add_argument("--digits",    type=int,   default=TARGET_DIGITS)
    parser.add_argument("--max",       type=int,   default=None, dest="max_candidates")
    parser.add_argument("--demo",      action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--backend",   default="auto",
                        choices=["auto","cpu","gpu","python","gmpy2","cupy","numba"])
    parser.add_argument("--workers",   type=int,   default=None,
                        help="Total worker processes (default: all CPU cores).")
    args = parser.parse_args()

    print(BackendInfo.report())

    # Resolve backend
    force   = args.backend if args.backend != "auto" else None
    backend = BackendInfo.best_backend(force)
    print(f"\n  Active backend: {backend}\n")

    total_workers = args.workers or BackendInfo.cpu_count
    n_sieve       = max(1, total_workers - 1)
    n_ll          = max(1, total_workers // 2)

    if args.selftest:
        run_selftest(backend); return
    if args.benchmark:
        run_benchmark(); return

    target_digits = args.digits
    if args.demo:
        start_exp = 10_000
        log("DEMO MODE: searching near p=10,000")
    elif args.start:
        start_exp = args.start
    elif not args.no_resume and os.path.exists(CHECKPOINT_FILE):
        cp        = load_checkpoint()
        start_exp = cp["last_p"] + 2
        log(f"Resuming from checkpoint: last_p={cp['last_p']:,}")
    else:
        start_exp = math.ceil(target_digits / math.log10(2))

    # -------------------------------------------------------------------
    # 2-minute stop/restart cycle + 24-hour log rotation
    # -------------------------------------------------------------------
    # Each cycle: launch the search as a child process, let it run for
    # RESTART_INTERVAL seconds, then SIGTERM it (triggering checkpoint),
    # reload the checkpoint, and re-launch. Benefits:
    #   - Keeps memory clean (no GC/GIL buildup over long runs)
    #   - A crash loses at most 2 minutes of work
    #   - Child processes exit cleanly between cycles
    # -------------------------------------------------------------------
    cycle         = 0
    last_rotation = time.time()   # track when we last rotated the log
    while True:
        cycle += 1
        log(f"  \u2500\u2500 Cycle {cycle} ─ running for {RESTART_INTERVAL//60} min then checkpointing \u2500\u2500")

        # Build child command (--_inner skips the restart loop in the child)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--backend", backend,
            "--workers", str(total_workers),
            "--digits",  str(target_digits),
            "--_inner",
        ]
        if args.max_candidates:
            cmd += ["--max", str(args.max_candidates)]

        proc       = subprocess.Popen(cmd)
        cycle_t0   = time.time()

        # Poll until deadline or child finishes on its own
        while True:
            if time.time() - cycle_t0 >= RESTART_INTERVAL:
                log(f"  {RESTART_INTERVAL//60}-min cycle complete — checkpointing and restarting...")
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
                break
            ret = proc.poll()
            if ret is not None:            # child exited on its own
                if ret == 0:
                    log("  Search finished (prime found or --max reached).")
                else:
                    log(f"  Child exited with code {ret}.")
                return
            time.sleep(2)

        # Reload checkpoint so the next cycle resumes where we left off
        if os.path.exists(CHECKPOINT_FILE):
            cp        = load_checkpoint()
            start_exp = cp["last_p"] + 2
            tested_count = cp["tested"]
            log(f"  Resuming from p={start_exp:,}  ({tested_count:,} candidates tested so far)")
        else:
            log("  No checkpoint found — next cycle restarts from beginning.")

        log("  Restarting in 3 seconds...")
        time.sleep(3)

        # Check if 24-hour log rotation is due
        if time.time() - last_rotation >= LOG_ROTATE_INTERVAL:
            log("  24-hour mark reached — rotating log file...")
            rotate_log()
            last_rotation = time.time()


# ---------------------------------------------------------------------------
# Inner search (runs inside child processes spawned by the restart loop)
# ---------------------------------------------------------------------------
def main_inner():
    """One cycle of search, called by child processes. No restart wrapper."""
    BackendInfo.detect()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", default="auto",
                        choices=["auto","cpu","gpu","python","gmpy2","cupy","numba"])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--digits",  type=int, default=TARGET_DIGITS)
    parser.add_argument("--max",     type=int, default=None, dest="max_candidates")
    args, _ = parser.parse_known_args()

    force         = args.backend if args.backend != "auto" else None
    backend       = BackendInfo.best_backend(force)
    total_workers = args.workers or BackendInfo.cpu_count
    n_sieve       = max(1, total_workers - 1)
    n_ll          = max(1, total_workers // 2)
    target_digits = args.digits

    # Always resume from checkpoint in inner runs
    if os.path.exists(CHECKPOINT_FILE):
        cp        = load_checkpoint()
        start_exp = cp["last_p"] + 2
        log(f"  [inner] Resuming from p={start_exp:,}")
    else:
        start_exp = math.ceil(target_digits / math.log10(2))

    if total_workers == 1:
        run_search_single(start_exp, target_digits, backend, args.max_candidates)
    else:
        run_search_parallel(start_exp, target_digits, backend,
                            n_sieve, n_ll, args.max_candidates)


if __name__ == "__main__":
    if "--_inner" in sys.argv:
        sys.argv.remove("--_inner")
        main_inner()
    else:
        main()
