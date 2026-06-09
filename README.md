# Pneumonia Detector

A professional Python project scaffold for detecting pneumonia from chest X-ray images using an efficient deep learning pipeline.

## Project structure

- `src/` — application source code
- `tests/` — unit tests
- `requirements.txt` — Python dependencies
- `README.md` — project overview and usage

## Features

- TensorFlow/Keras transfer learning with EfficientNet.
- Efficient `tf.data` image pipeline.
- Modular design for data loading, model building, training, and inference.
- Configurable training and evaluation process.

## Quickstart

1. Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train the model:

```powershell
python src\train.py --data-dir path\to\dataset --output-model model\pneumonia_detector.h5
```

4. Predict from an image:

```powershell
python src\predict.py --model model\pneumonia_detector.h5 --image path\to\image.png
```

## Deploy to Railway

1. Create a Railway project and connect your GitHub repository or use the Railway CLI.
2. Make sure `requirements.txt` and `Procfile` are present in the project root.
3. Railway will detect `python` and install dependencies automatically.
4. Railway sets `PORT` automatically, and `app.py` now reads it with `os.environ.get('PORT', 5000)`.
5. Deploy the project and Railway will provide a public URL.

## Deploy to Fly.io

1. Install Fly CLI: https://fly.io/docs/getting-started/installing/
2. Login:

```bash
fly auth login
```

3. Create or launch the app from the project root:

```bash
fly launch --name pneumonia-detector --no-deploy
```

4. When prompted, choose:
   - `app` name: `pneumonia-detector` (or your preferred slug)
   - `region`: nearest location
   - `builder`: `Dockerfile`

5. Deploy:

```bash
fly deploy
```

6. Open the app:

```bash
fly open
```

Fly uses the `Dockerfile` in this repository and sets `PORT` automatically.

## Dataset expectations

The dataset directory should contain two subfolders:

- `train/normal`
- `train/pneumonia`
- `val/normal`
- `val/pneumonia`
- `test/normal` (optional)
- `test/pneumonia` (optional)

Images are loaded from those folders and processed automatically.
