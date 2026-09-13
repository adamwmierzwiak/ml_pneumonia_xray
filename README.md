# ChestXray — Pneumonia Classification (ResNet-18)

**Educational project — not validated for clinical use.**

Fine-tuning a pretrained ResNet-18 to classify chest X-rays as `NORMAL` or
`PNEUMONIA`. The full workflow lives in `notebook.ipynb`.

**TL;DR for reviewers in a hurry** — the interesting part of this project isn't the
accuracy number. It's the process:
- **Caught its own data leakage** — a naive first training pass (step 3) tuned epoch
  count against the evaluation set; step 5 fixes it with a proper stratified validation
  split, and the diagnosis of what went wrong is kept in the notebook rather than
  quietly deleted.
- **Audited *why* the model works, not just *whether* it works** — a Grad-CAM pass
  (step 7) found a handful of false positives with heat concentrated near the image
  border; step 13 later checks that impression quantitatively across the full
  evaluation set instead of resting on four hand-picked images.
- **Tested that suspicion with real experiments instead of leaving it as a guess** — a
  2×2 ablation (steps 9–11: augmentation × deeper fine-tuning) plus a direct masking
  intervention (step 15) report mostly negative and mixed results, openly, with the
  reasoning for why none of them replace the original model.
- **Went through an external methodological review and fixed what it found** — repeated,
  adaptive use of the "test set," unseeded model comparisons, an unsourced claim about
  the dataset, and overstated clinical framing were all identified and corrected in the
  notebook itself (Conclusions, limitations 1–8) rather than quietly patched over.
- **Then verified the fix on a GPU — twice, at 3 and then 5 seeds — and let each round
  correct the last.** The earlier "best result" (0.89 accuracy) never reproduces across
  5 redraws and is retired as an outlier. More tellingly: a "clean win" seen at 3 seeds
  (one recipe's accuracy beating another's on every seed) turned out to be partly a
  small-sample artifact once the sample grew to 5 — while that same recipe's
  specificity advantage held up with zero overlap. The project treats its own
  first-round conclusion as a hypothesis to re-check, not a result to defend.

## Result

Fine-tuning the head of a frozen* ResNet-18 (513 gradient-trained parameters), with the
checkpoint chosen by validation loss over a fixed 25-epoch budget (best-checkpoint
selection, not early stopping in the stricter sense). On the evaluation set (n=100,
class-balanced), at the untuned default threshold 0.5: **accuracy 0.76 [95% CI
0.66–0.84], F1 0.81 (positive-class), sensitivity 1.00 [0.93–1.00], specificity 0.52
[0.37–0.66]** (exact Clopper-Pearson intervals). A trivial always-predict-PNEUMONIA
baseline scores 0.50 / 1.00 / ~0.67 on this same set for free — sensitivity alone isn't
the story; specificity and F1 are where this model earns its result. This evaluation
set was used repeatedly across the notebook (steps 6, 9–12, 15), not observed once —
see *Conclusions & Limitations* for the full accounting, including the lack of external
validation, unverified patient-level split independence, and why this is not a
screening or triage tool.

*"Frozen" needs a footnote too: gradient-freezing and BatchNorm-statistic-freezing are
different things — see the notebook, step 8 and Conclusions limitation 3.

## Notebook contents

1. **EDA** — label mapping, class balance, sample images (randomly sampled, seeded)
2. **Transfer learning setup** — freeze the ResNet-18 backbone, new `Linear(512→1)` head
3. **Baseline training** — a deliberately naive first pass, kept to illustrate common pitfalls
4. **Diagnosing the baseline** — confusion matrix, sensitivity/specificity, a trivial baseline for scale
5. **Fixing the methodology** — stratified train/val split, smaller batches,
   best-checkpoint selection — no epoch-count leakage from the evaluation set
6. **Evaluation on the held-out set** — and why this notebook no longer calls it
   "observed once"
7. **Grad-CAM** — heatmaps auditing what the model actually relies on
8. **Model persistence** — saving the head only vs. the full state, a deployable
   checkpoint with metadata, and what BatchNorm statistics quietly change
9. **Testing a hypothesis** — framing/crop augmentation targeting the border-attention
   pattern from step 7; a single seed-matched run, mixed result (fewer false alarms,
   a real but more moderate sensitivity cost than the recipe's first, unseeded try)
10. **Deeper fine-tuning** — unfreezing `layer4` with a differential learning rate;
    overfits the training set (same failure mode both times) but this run clearly
    beats step 6 in aggregate, unlike an earlier, unseeded run of the same code that
    landed on an exact tie — a concrete illustration of why single runs mislead
11. **Completing the 2×2** — augmentation × `layer4`, together; the best raw numbers of
    the four, and the clearest demonstration of how much a seed change alone can move
    the "best" result (accuracy 0.79 → 0.89 between two runs of identical code)
12. **ROC/PR curves and threshold calibration** — a validation-chosen threshold for
    95% sensitivity collapses out-of-sample specificity to 0.10
13. **Quantifying the border-attention pattern** — across the full evaluation set, with
    a geometric baseline for comparison
14. **Bootstrap confidence intervals** — cross-checking the closed-form CIs empirically
15. **Removing the border cue at the source** — deterministic masking (train, val, and
    evaluation alike); the steadiest single-lever result in the notebook, with its own
    honest complication
16. **Repeating the 2×2 grid across 5 seeds** (GPU-accelerated, run in two rounds — 3
    seeds, then extended to 5) — settles limitation 4: unfreezing `layer4` stays the
    dominant source of instability regardless of augmentation (worst case: a *fully
    converged* 0.62 accuracy), and step 11's 0.89 never reproduced across 5 redraws
    (best was 0.88). The more useful finding is methodological: the 3-seed round's
    "clean win" for `frozen + aug` on accuracy (every seed beating every
    `frozen + plain` seed) didn't survive to 5 seeds — the distributions now overlap —
    while its specificity advantage did survive, with zero overlap at either sample
    size. A live example of why 3 seeds is a minimum, not a target.

## Data

- `data/chestxrays.zip` — unzipped by the notebook's first cell into `data/chestxrays/`
- 300 training and 100 test images (the 300 are further split 240 train / 60
  validation inside the notebook, step 5a), balanced across classes, preprocessed for
  ResNet-18 (224×224)
- Pediatric (1–5y) chest X-rays from Guangzhou Women and Children's Medical Center
  (Kermany et al.) — single center, routine clinical care, quality-screened,
  physician-graded. The notebook originally claimed the source paper documents
  class-specific acquisition differences; that claim did not check out on re-reading
  the paper and has been corrected throughout (see Conclusions, limitation 2).
- **Patient-level split:** the Mendeley source states the original train/test
  partition uses independent patients. This project's own further split of the
  training pool into 240 train / 60 validation images has no patient-ID field to
  verify — that independence is unconfirmed (Conclusions, limitation 1).
- **Source & license:** Kermany, D., Zhang, K., & Goldbaum, M. (2018). *Labeled
  Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*
  (Version 2) [Dataset]. Mendeley Data. https://doi.org/10.17632/rscbjbr9sj.2 —
  **CC BY 4.0, for research use** (the source explicitly restricts use to research);
  this subset is also mirrored on Kaggle as "Chest X-Ray Images (Pneumonia)."

## Runtime environment (Docker)

Code runs **inside the container**, not on the host. The image is based on
`pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` plus the packages in
`requirements.txt` (`torchvision`, `torchmetrics`, `jupyterlab`, …). Docker on macOS
has no GPU access — training runs on CPU, which is plenty for steps 1–15 (each trains
513–8M parameters for 25 epochs on 240 images). Step 16's 20 repeated trainings were
run on a separate Linux/NVIDIA host instead — see below.

```bash
# start the environment (builds on first run)
docker compose up -d

# quick environment check
docker compose exec chestxray python env_check.py

# stop
docker compose down
```

### GPU (optional)

On a machine with an NVIDIA GPU and `nvidia-container-toolkit` installed, bring the
same image up with GPU passthrough via the additive override in `docker-compose.gpu.yml`
(not applied on Mac/CPU-only hosts):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
docker exec chestxray python3 -c "import torch; print(torch.cuda.is_available())"
```

Nothing in `notebook.ipynb` moves computation onto a GPU by default — steps 1–15 run
on whatever device is implicit (CPU). Step 16 is the one place that explicitly checks
for a GPU (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`) and uses it
when present, since it's the one step expensive enough (12 full training runs) for
that to matter.

## JupyterLab

The `docker-compose.yml` service exposes JupyterLab at:

**http://127.0.0.1:18888/lab**

- host port is `18888` (deliberately not `8888`, to avoid clashing with other
  services; the container's Jupyter listens on 8888 internally),
- bound to `127.0.0.1` only, so the auth token is disabled,
- the project is mounted at `/app`, so the notebook and data are edited directly from the host.

## PyCharm

- **Interpreter:** `Settings → Project → Python Interpreter → Add Interpreter → On Docker Compose…`,
  service `chestxray`, Python `/opt/conda/bin/python3`.
- **Notebook:** "IDE managed" mode works too — PyCharm runs `docker compose run` and
  maps a random host port. The image bakes in `c.ServerApp.ip = '0.0.0.0'` (see
  `Dockerfile`); without it, port mapping can't reach the Jupyter server.
  Alternatively, use "Configured Server" pointing at `http://127.0.0.1:18888`.
- **Packages:** PyCharm won't install packages into Docker-based interpreters (the
  "Cannot modify packages for Docker/Docker Compose SDK…" message is by design, not a
  bug). Change packages through the image instead:

  ```bash
  # after editing requirements.txt
  docker compose up -d --build
  ```

## Structure

| Path                  | Role                                                        |
|------------------------|--------------------------------------------------------------|
| `notebook.ipynb`      | the project — data, training, evaluation, Grad-CAM           |
| `Dockerfile`          | runtime image (PyTorch + requirements + Jupyter config)      |
| `docker-compose.yml`  | JupyterLab service on port 18888                              |
| `docker-compose.gpu.yml` | additive override: GPU passthrough on NVIDIA hosts (step 16) |
| `requirements.txt`    | packages installed into the image                            |
| `env_check.py`        | environment smoke test (versions, CUDA, tensor op)           |
| `data/`               | dataset (zip + unzipped)                                     |
| `models/`             | ImageNet weights (`hub/`) and classifier-head checkpoints    |
