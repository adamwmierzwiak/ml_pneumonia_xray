# ChestXray — Pneumonia Classification (ResNet-18)

Fine-tuning a pretrained ResNet-18 to classify chest X-rays as `NORMAL` or
`PNEUMONIA`. The full workflow lives in `notebook.ipynb`.

## Result

Fine-tuned the head of a frozen ResNet-18 (513 trainable parameters), with the
training epoch selected by validation loss (early stopping). On a held-out test set
(n=100, observed once): **accuracy 0.80, F1 0.83, sensitivity 1.00, specificity 0.60**.
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

## Data

- `data/chestxrays.zip` — unzipped by the notebook's first cell into `data/chestxrays/`
- 300 training and 100 test images, balanced across classes, preprocessed for ResNet-18 (224×224)
- Pediatric chest X-rays (Kermany et al.) — see the notebook's limitations section
  regarding differences in acquisition protocol between classes

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
