# 🔢 Mersenne Prime Hunter

> **Targeting a 100,000,000-digit prime — a new world record.**

A distributed, GPU-accelerated search engine for large [Mersenne primes](https://en.wikipedia.org/wiki/Mersenne_prime) using the Lucas–Lehmer primality test. Run it on a single laptop or spread it across a cluster of machines. Every CPU and GPU you add brings us closer to the next discovery.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)]()
[![CUDA](https://img.shields.io/badge/CUDA-optional%20%7E500%C3%97-76b900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

---

## 🌍 Why This Matters

As of 2026, only **52 Mersenne primes** are known. The current world record — found in October 2024 by Luke Durant using thousands of cloud GPUs — has **41 million digits**. We're aiming for **100 million**.

A prime of that size would be:
- A new world record by a factor of **2.5×**
- Eligible for the **$150,000 EFF prize** for the first 100-million-digit prime
- Published in *Mathematics of Computation* (see our [academic paper](paper/mersenne_prime_paper.tex))

```
Current record:  M_136,279,841   (~41 million digits)   ← found Oct 2024
Our target:      M_332,192,810+  (~100 million digits)  ← world record territory
```

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Lucas–Lehmer test** | Gold-standard Mersenne primality algorithm |
| **Trial factor sieve** | Eliminates ~70% of candidates in milliseconds |
| **Auto-backend selection** | Pure Python → gmpy2 → CuPy → Numba CUDA |
| **Multiprocessing** | Parallel sieve + LL pools across all CPU cores |
| **Distributed** | Coordinator/worker architecture via XML-RPC — no extra installs |
| **Checkpointing** | 2-minute restart cycles, zero lost work |
| **Log rotation** | Daily log archiving for long-running deployments |
| **Self-test** | Verified against all 20 known small Mersenne primes |

---

## ⚡ Quick Start

```bash
# Clone
git clone https://github.com/huggablehacker/Mersenne-Prime
cd Mersenne-Prime

# Run (pure Python, no dependencies needed)
python3 mersenne_prime_hunter.py

# Verify the algorithm is correct
python3 mersenne_prime_hunter.py --selftest

# Benchmark your machine
python3 mersenne_prime_hunter.py --benchmark

# Quick demo (finds small primes fast)
python3 mersenne_prime_hunter.py --demo
```

---

## 🚀 Speed Up With Better Backends

The script auto-detects the best available backend. Install any of these to unlock massive speedups:

```bash
# ~50× faster — works on any machine (Mac, Linux, Windows)
pip install gmpy2

# ~500× faster — requires Linux + NVIDIA GPU
pip install cupy-cuda12x

# ~300× faster — requires Linux + NVIDIA GPU
pip install numba
```

On a Mac, `gmpy2` is the best option. Install it with:
```bash
brew install gmp
pip install gmpy2
```

### Estimated time per candidate at the 100M-digit frontier

| Setup | Time per test |
|---|---|
| Mac / pure Python | ~65,000 days |
| Mac + gmpy2 | ~1,300 days |
| Linux + NVIDIA A100 | ~1–2 days ✅ |
| GPU cluster (GIMPS-grade) | Hours |

---

## 🌐 Distributed Search (Multiple Machines)

No extra installs needed — uses Python's built-in XML-RPC. One machine is the **coordinator**; all others are **workers**.

**Step 1 — Start the coordinator** (on one machine):
```bash
python3 mersenne_prime_hunter.py --coordinator
```
It will print its IP address and the exact command to run on workers.

**Step 2 — Start workers** (on every other machine):
```bash
python3 mersenne_prime_hunter.py --worker --host 192.168.1.x
```

Each worker machine tests a different prime candidate simultaneously — just like GIMPS. Workers reconnect automatically if the coordinator restarts, and dead workers' assignments are re-issued after 30 minutes.

```
Coordinator (192.168.1.10)
     │  port 1279
     ├── Worker A (Mac)      → testing M_332,192,813
     ├── Worker B (Linux+GPU) → testing M_332,192,857
     └── Worker C (Linux+GPU) → testing M_332,192,897
```

---

## 🔧 All Options

```
python3 mersenne_prime_hunter.py [options]

Search modes:
  (none)              Standalone search on this machine
  --coordinator       Run as coordinator (assigns work to workers)
  --worker            Run as worker (connects to coordinator)
  --host IP           Coordinator IP address (used with --worker)
  --port PORT         Coordinator port (default: 1279)

Diagnostic:
  --selftest          Verify algorithm against known Mersenne primes
  --benchmark         Benchmark all available backends
  --demo              Quick search near p=10,000

Configuration:
  --backend BACKEND   Force backend: auto|cpu|gpu|python|gmpy2|cupy|numba
  --workers N         Number of CPU worker processes (default: all cores)
  --start P           Start from exponent P
  --digits D          Target digit count (default: 100,000,000)
  --max N             Stop after N candidates (testing)
  --no-resume         Ignore existing checkpoint, start fresh
```

---

## 📐 How It Works

### 1. The Lucas–Lehmer Test

For a prime $p$, define the sequence:

$$s_0 = 4, \qquad s_{i+1} = s_i^2 - 2 \pmod{M_p}$$

Then $M_p = 2^p - 1$ is prime **if and only if** $s_{p-2} \equiv 0 \pmod{M_p}$.

This requires $p - 2$ squarings of a $p$-bit number — O(p²) with naive multiplication, O(p log p log log p) with FFT (which is what gmpy2 and the GPU backends use).

### 2. Trial Factor Sieve

Before the expensive LL test, any prime factor $q$ of $M_p$ must satisfy:
- $q = 2kp + 1$ for some integer $k$
- $q \equiv \pm 1 \pmod{8}$

Testing all such $q$ up to $10^7$ eliminates ~70% of candidates in milliseconds.

### 3. GPU Acceleration (CuPy backend)

Big integers are stored as arrays of 16-bit limbs. Squaring is computed via GPU FFT (equivalent to GIMPS's IBDWT):

```
s² mod M_p:
  1. rfft(s)          → frequency domain  [O(n log n) on GPU]
  2. pointwise square → s² in freq domain [O(n)]
  3. irfft            → back to integers  [O(n log n) on GPU]
  4. carry propagate  → [O(n)]
  5. fold mod M_p     → exploit 2^p ≡ 1 (mod M_p)
```

---

## 📊 Current Search Status

| Field | Value |
|---|---|
| Search started | — |
| Last tested exponent | — |
| Candidates tested | — |
| Primes found | — |
| Current frontier | p ≈ 332,192,810 |

*Results and checkpoints will be updated here as the search progresses.*

---

## 📄 Academic Paper

A full peer-review-ready paper in AMS *Mathematics of Computation* (MCOM) LaTeX format is included in this repository.

---

## 🏆 Prizes

The [Electronic Frontier Foundation](https://www.eff.org/awards/coop) offers cash prizes for large prime discoveries:

| Milestone | Prize |
|---|---|
| First 10,000,000-digit prime | $100,000 ✅ *claimed 2018* |
| **First 100,000,000-digit prime** | **$150,000** ← our target |
| First 1,000,000,000-digit prime | $250,000 |

---

## 🤝 Contributing

Contributions welcome! Ideas especially wanted for:

- IBDWT (irrational-base DWT) implementation for true GIMPS-grade GPU speed
- Integer NTT (number-theoretic transform) to eliminate floating-point error
- Pollard P−1 factoring stage
- Web dashboard for monitoring distributed search progress
- Additional platform testing (Windows, Apple Silicon MPS backend)

Please open an issue or pull request.

---

## 📚 References

1. Lucas, E. (1878). *Théorie des fonctions numériques simplement périodiques.* Amer. J. Math.
2. Lehmer, D. H. (1930). *An extended theory of Lucas' functions.* Ann. of Math.
3. Wagstaff, S. S. (1983). *Divisors of Mersenne numbers.* Math. Comp.
4. Crandall, R. & Fagin, B. (1994). *Discrete weighted transforms and large-integer arithmetic.* Math. Comp.
5. [GIMPS — Great Internet Mersenne Prime Search](https://www.mersenne.org)
6. [OEIS A000043 — Mersenne exponents](https://oeis.org/A000043)

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>The search for the next prime begins with a single squaring.</i><br><br>
  Made with ☕ and a lot of modular arithmetic.
</p>
