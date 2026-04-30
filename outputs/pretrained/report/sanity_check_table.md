| Sanity Check | Expected | Actual | Pass? |
|---|---|---|---|
| Random baseline accuracy | 10.0% | 0.1000 | CHECK |
| Online SGD shows forgetting | BWT < 0 | -0.2025 | PASS |
| E=1 near random | ~10% | 0.2380 | FAIL |
| E=20 significantly better | >30% | 0.8740 | PASS |
| Pi-Transformer > ER | Yes | 0.8810 vs 0.9794 | FAIL |
| Label injection has effect | diff > 0.05 | N/A | N/A |
| Cache persists across chunks | Shape grows | N/A | N/A |
