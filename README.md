# ChestXray — Pneumonia Classification (ResNet-18)

Fine-tuning a pretrained ResNet-18 to classify chest X-rays as `NORMAL` or
`PNEUMONIA`. The full workflow lives in `notebook.ipynb`.

**TL;DR for reviewers in a hurry** — the interesting part of this project isn't the
accuracy number. It's the process:
- **Caught its own data leakage** — a naive first training pass (step 3) tuned epoch
  count against the test set; step 5 fixes it with a proper stratified validation
  split, and the diagnosis of what went wrong is kept in the notebook rather than
  quietly deleted.
- **Audited *why* the model works, not just *whether* it works** — a Grad-CAM pass
  (step 7) found false positives correlating with image cropping, consistent with this
  dataset's documented acquisition-protocol difference between classes.
- **Tested that suspicion with a real experiment instead of leaving it as a guess** — a
  2×2 ablation (steps 9–11: augmentation × deeper fine-tuning) reports two negative and
  one unstable-but-encouraging result, openly, with the reasoning for why none of them
  replace the original model.

## Result

Fine-tuned the head of a frozen ResNet-18 (513 trainable parameters), with the
training epoch selected by validation loss (early stopping). On a held-out test set
(n=100, observed once): **accuracy 0.76, F1 0.81, sensitivity 1.00, specificity 0.52**.
Full conclusions and limitations — including the lack of external validation and a
Grad-CAM-detected shortcut-learning risk — are in the last section of `notebook.ipynb`.

## Notebook contents

1. **EDA** — label mapping, class balance, sample images
2. **Transfer learning setup** — freeze the ResNet-18 backbone, new `Linear(512→1)` head
3. **Baseline training** — a deliberately naive first pass, kept to illustrate common pitfalls
4. **Diagnosing the baseline** — confusion matrix, sensitivity/specificity
5. **Fixing the methodology** — stratified train/val split, smaller batches, early
   stopping — no test-set leakage
6. **Final evaluation** — the test set, observed once
7. **Grad-CAM** — heatmaps auditing what the model actually relies on
8. **Model persistence** — saving the head only, and what that leaves out
9. **Testing a hypothesis** — framing/crop augmentation targeting the suspected
   shortcut from step 7; an honest negative result (fewer false alarms, but a large
   sensitivity cost), kept in the notebook and discussed rather than discarded
10. **Deeper fine-tuning** — unfreezing `layer4` with a differential learning rate;
    overfits the training set and ties step 6's test-set result, with the underlying
    per-case predictions redistributed rather than unchanged
11. **Completing the 2×2** — augmentation × `layer4`, together; the best raw numbers
    of the four (accuracy 0.79, F1 0.81, sensitivity 0.92, specificity 0.66), but from
    an unstable run (best checkpoint at epoch 1/25) — read as encouraging, not adopted

## Data

- `data/chestxrays.zip` — unzipped by the notebook's first cell into `data/chestxrays/`
- 300 training and 100 test images, balanced across classes, preprocessed for ResNet-18 (224×224)
- Pediatric chest X-rays (Kermany et al.) — see the notebook's limitations section
  regarding differences in acquisition protocol between classes
- **Source & license:** Kermany, D., Zhang, K., & Goldbaum, M. (2018). *Labeled
  Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*
  (Version 2) [Dataset]. Mendeley Data. https://doi.org/10.17632/rscbjbr9sj.2 —
  **CC BY 4.0**, redistribution and reuse permitted with attribution (given above);
  this subset is also mirrored on Kaggle as "Chest X-Ray Images (Pneumonia)."

## Runtime environment (Docker)

Code runs **inside the container**, not on the host. The image is based on
`pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` plus the packages in
`requirements.txt` (`torchvision`, `torchmetrics`, `jupyterlab`, …). Docker on macOS
has no GPU access — training runs on CPU, which is plenty for this dataset size.

```bash
# start the environment (builds on first run)
docker compose up -d

# quick environment check
docker compose exec chestxray python env_check.py

# stop
docker compose down
```

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
| `requirements.txt`    | packages installed into the image                            |
| `env_check.py`        | environment smoke test (versions, CUDA, tensor op)           |
| `data/`               | dataset (zip + unzipped)                                     |
| `models/`             | ImageNet weights (`hub/`) and classifier-head checkpoints    |
