# autonomous_vit — DETR patch attack & explainable defense (demo)

A self-contained, runnable demonstration of:
1. A pretrained **DETR** transformer object detector,
2. An **adversarial patch** (EOT-lite) that hides objects from it,
3. An **explainable, saliency-based defense** that flags the attack and shows *where* it reacted,
4. A **timing** check for the real-time (edge) constraint.

### Run it in the browser (no local setup)
Open the notebook and click **“Open in Colab”**, then `Runtime → Change runtime type → GPU`, then `Runtime → Run all`:

- [`DETR_patch_attack_defense_demo.ipynb`](DETR_patch_attack_defense_demo.ipynb)

### Files
| File | What it is |
|---|---|
| `DETR_patch_attack_defense_demo.ipynb` | The main notebook (9 steps, with explanations). |
| `detr_patch_demo.py` | Headless version of the same code, for running on a server/CI. |
| `outputs/` | Generated plots + `summary.json` after a run. |

> ⚠️ **This is a learning / proof-of-concept artifact, not a research result.** It runs a *digital* simulation of a physical attack on a *single* image with a *non-adaptive* evaluation. Read **Step 9 (Limitations)** in the notebook before drawing any conclusions.
