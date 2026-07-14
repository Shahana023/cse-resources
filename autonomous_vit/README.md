# autonomous_vit — DETR adversarial-robustness demo

A runnable proof-of-concept for a PhD research proposal on **adversarially robust,
explainable defenses for autonomous-vehicle object detectors (DETR-family)**.

## Run it in the browser (no setup)
Open the notebook and click **Open in Colab**, then `Runtime → Run all`:
- [`AV_adversarial_defense_proposal_demo.ipynb`](AV_adversarial_defense_proposal_demo.ipynb)

## What it shows (on a real KITTI frame)
1. A pretrained **DETR** detector finds the vehicles/objects.
2. A **near-invisible** digital adversarial perturbation (max change ~0.03/pixel) makes DETR detect **nothing**.
3. A cheap **high-frequency probe** flags this (non-adaptive) attack.
4. Honest framing: physical patch attacks + an **adaptive-robust, explainable** defense on
   DETR is the open problem the PhD targets.

`outputs/` holds the generated figures. See **Step 8** in the notebook for honest limitations.
