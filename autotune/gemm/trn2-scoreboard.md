# NKI Autotune - Scoreboard (trn2)

## Overview

This scoreboard tracks optimal loop orders and tiling configurations for the following GEMM modes:
* **LHS × RHS** - Standard matrix multiplication
* **LHS_T × RHS** - Transposed left-hand side matrix multiplication

All shapes are square matrices of the form (M, K, N). The results show the best-performing configurations discovered through autotuning, including loop iteration order and tile sizes per block dimension.

---

## LHS × RHS Kernel Results

| Shape (M, K, N) | Loop Order | K Tiles | M Tiles | N Tiles | Min (ms) | MFU Estimated (%) |
|-----------------|------------|---------|---------|---------|----------|-------------------|
| (512, 512, 512) | NKM | 2 | 6 | 1 | 0.056 | 6.07 |
| (1024, 1024, 1024) | NMK | 8 | 4 | 2 | 0.079 | 34.15 |
| (1536, 1536, 1536) | NMK | 12 | 1 | 3 | 0.149 | 61.49 |
| (2048, 2048, 2048) | NMK | 2 | 2 | 1 | 0.288 | 75.73 |
| (2560, 2560, 2560) | MNK | 1 | 4 | 1 | 0.509 | 83.81 |
| (3072, 3072, 3072) | MNK | 12 | 4 | 2 | 0.827 | 86.56 |
| (3584, 3584, 3584) | MNK | 14 | 4 | 1 | 1.291 | 90.65 |
| (4608, 4608, 4608) | MNK | 12 | 3 | 3 | 2.653 | 93.77 |
| (4096, 4096, 4096) | MNK | 4 | 4 | 2 | 1.886 | 92.67 |
| (5120, 5120, 5120) | MNK | 10 | 5 | 2 | 3.603 | 94.74 |

---

## LHS_T × RHS Kernel Results

| Shape (M, K, N) | Loop Order | K Tiles | M Tiles | N Tiles | Min (ms) | MFU Estimated (%) |
|-----------------|------------|---------|---------|---------|----------|-------------------|
| (512, 512, 512) | KNM | 4 | 1 | 1 | 0.054 | 6.2 |
| (1024, 1024, 1024) | NMK | 4 | 2 | 1 | 0.076 | 35.83 |
| (1536, 1536, 1536) | NMK | 12 | 1 | 1 | 0.141 | 65.19 |
| (2048, 2048, 2048) | MNK | 2 | 4 | 2 | 0.269 | 81.03 |
| (2560, 2560, 2560) | MNK | 4 | 5 | 1 | 0.483 | 88.34 |
| (3072, 3072, 3072) | NMK | 12 | 2 | 2 | 0.795 | 92.67 |
| (3584, 3584, 3584) | MNK | 14 | 7 | 1 | 1.247 | 93.88 |
| (4608, 4608, 4608) | MNK | 12 | 4 | 3 | 2.573 | 96.68 |
| (4096, 4096, 4096) | MNK | 4 | 4 | 2 | 1.823 | 95.83 |
| (5120, 5120, 5120) | NMK | 20 | 5 | 2 | 3.504 | 97.4 |

---
