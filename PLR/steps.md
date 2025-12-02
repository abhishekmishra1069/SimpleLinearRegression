# PLR (Polynomial Regression) - Setup and Deployment Guide

## Overview
This document explains how to train the polynomial regression model, run the Flask prediction service locally, containerize the app with Docker or Podman, and troubleshoot frequent failures.

## Prerequisites
- Python 3.13+ (for local testing)
- Docker or Podman (for container runs)
- `pip` or a conda environment
- Files present in `PLR/`: `app.py`, `Train_and_save.ipynb`, `poly_reg_model.pkl`, `poly_features.pkl`, `requirements.txt`

---

## Step 1 — Train the Model (Notebook)
1. Open `PLR/Train_and_save.ipynb` in Jupyter or VS Code.
2. Install dependencies if needed:

```bash
pip install -r requirements.txt
# or using conda
# conda create -n plr-env python=3.10
# conda activate plr-env
# pip install -r requirements.txt
```

3. Run cells in order:
   - Import libraries and load `Salary_Data.csv` (or `data.csv`).
   - Prepare `X` and `y` and split into train/test.
   - Create `PolynomialFeatures` transformer and transform `X`.
   - Train a `LinearRegression` model on the transformed features.
   - Evaluate model (R² or MSE) on test set.
   - Save two artifacts using `pickle`:
     - `poly_reg_model.pkl` (trained regressor)
     - `poly_features.pkl` (fitted PolynomialFeatures transformer)

4. Confirm that `poly_reg_model.pkl` and `poly_features.pkl` exist in `PLR/`.

Example snippet (inside notebook):
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_train.reshape(-1, 1))  # or appropriate shape
regressor = LinearRegression().fit(X_poly, y_train)

with open('poly_reg_model.pkl', 'wb') as f:
    pickle.dump(regressor, f)
with open('poly_features.pkl', 'wb') as f:
    pickle.dump(poly, f)
```

---

## Step 2 — Local API Testing
1. Ensure `PLR/requirements.txt` is installed locally:

```bash
pip install -r PLR/requirements.txt
```

2. Run the Flask app from the `PLR/` directory:

```bash
cd PLR
python app.py
```

3. Test the endpoint (two accepted formats):

- Direct format:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"YearsExperience": 5.5}'
```

- Nested `data` format (the app now supports this):
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [{"YearsExperience": 5.5}]}'
```

Expected response example:
```json
{
  "YearsExperience": 5.5,
  "PredictedSalary": 15000.25
}
```

Notes:
- The app loads `poly_reg_model.pkl` and `poly_features.pkl` from the working directory. Run `python app.py` from `PLR/` or use absolute paths.
- The app applies `poly.transform([[experience]])` before prediction.

---

## Step 3 — Containerize with Docker / Podman
1. Ensure `PLR/requirements.txt` exists and `ContainerFile` is present. If the file is named `ContainerFile`, pass `-f ContainerFile` to `docker build` or `podman build`.

Example `ContainerFile` (recommended):
```dockerfile
FROM python:3.10-slim
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY poly_reg_model.pkl .
COPY poly_features.pkl .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

2. Build image:

```bash
# Docker
docker build -f ContainerFile -t plr-model:latest .

# Podman
podman build -f ContainerFile -t plr-model:latest .
```

3. Run container:

```bash
# Docker
docker run -p 5000:5000 --name plr-app plr-model:latest

# Podman
podman run -p 5000:5000 --name plr-app plr-model:latest
```

4. Run in background (detached):

```bash
# Docker
docker run -d -p 5000:5000 --name plr-app plr-model:latest

# Podman
podman run -d -p 5000:5000 --name plr-app plr-model:latest
```

5. View logs / stop:

```bash
# Docker
docker logs plr-app
docker stop plr-app

# Podman
podman logs plr-app
podman stop plr-app
```

---

## Common Failures and Solutions

### 1) `ModuleNotFoundError: No module named 'sklearn'`
- Cause: dependencies not installed in environment or container.
- Fix:
  - Locally: `pip install -r PLR/requirements.txt` or set up a fresh conda env.
  - Container: ensure `requirements.txt` contains `scikit-learn` and rebuild the image.

### 2) `FileNotFoundError: poly_reg_model.pkl not found` or `poly_features.pkl not found`
- Cause: model/artifact missing or build did not copy files.
- Fix:
  - Confirm files exist in `PLR/`.
  - Run training notebook to produce them.
  - Update `ContainerFile` to `COPY poly_reg_model.pkl .` and `COPY poly_features.pkl .` and rebuild.
  - Alternatively, change `app.py` to use an absolute path to the files.

### 3) `Invalid input format or missing key: 'YearsExperience'`
- Cause: Input JSON uses a different shape or key.
- Fix:
  - Use one of the supported formats (direct or nested `data`).
  - Example nested: `{"data": [{"YearsExperience": 5.5}]}`
  - Example direct: `{"YearsExperience": 5.5}`

### 4) `Address already in use` (port 5000)
- Fix:
```bash
lsof -i :5000
kill -9 <PID>
# or run container mapping to a different host port
docker run -p 5001:5000 plr-model:latest
podman run -p 5001:5000 plr-model:latest
```

### 5) `gunicorn` not found in container
- Cause: `gunicorn` missing from `requirements.txt` or not installed.
- Fix: Ensure `gunicorn` appears in `requirements.txt` and rebuild the image.

### 6) JSON parse errors from `curl`
- Cause: malformed JSON or shell quoting issues.
- Fix: Use single-quoted outer string and double quotes in JSON, or put JSON in a file and use `-d @payload.json`.

Example payload file `payload.json`:
```json
{"data":[{"YearsExperience": 5.5}]}
```
Then run:
```bash
curl -X POST -H "Content-Type: application/json" -d @payload.json http://127.0.0.1:5000/predict
```

---

## Quick Commands

Install deps locally:
```bash
pip install -r PLR/requirements.txt
```

Run locally:
```bash
cd PLR
python app.py
```

Test API (direct):
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"YearsExperience": 5.5}'
```

Test API (nested):
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"data": [{"YearsExperience": 5.5}]}'
```

Build + run container:
```bash
docker build -f ContainerFile -t plr-model:latest .
docker run -p 5000:5000 plr-model:latest
```

---

## File layout (PLR)
```
PLR/
├── app.py
├── Train_and_save.ipynb
├── poly_reg_model.pkl
├── poly_features.pkl
├── requirements.txt
├── ContainerFile
├── Salary_Data.csv
└── steps.md    # <- this file
```

---

If you want, I can also:
- Add a small `README.md` with these commands.
- Update `ContainerFile` to `Dockerfile` name and include multi-stage builds.
- Run a quick local sanity test (if you want me to execute the Flask app here).
