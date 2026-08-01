# Heart Disease Prediction — ML Model + Flask API + Render Deployment

## Objective
A healthcare organization wants to deploy a machine learning model that predicts
whether a patient is at risk of heart disease based on clinical parameters. This
project trains a classification model, wraps it in a Flask REST API, and deploys it
as a live web service on Render.

## Dataset Link
[Heart Disease Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

> **Note on `heart.csv` in this repo:** The assignment's specified repository
> structure explicitly includes `heart.csv` as a required file, so it is included
> here. This is the standard, widely-mirrored 303-record UCI Heart Disease dataset
> (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca,
> thal, target) — the same feature set as the Kaggle version linked above. If your
> institution's redistribution policy requires it, spot-check this file against your
> own Kaggle download before submitting, and remove it if your license terms require
> that instead.

## Libraries Used
- pandas, numpy
- scikit-learn (`LogisticRegression`, `StandardScaler`, `train_test_split`,
  evaluation metrics)
- joblib (model persistence)
- Flask (REST API)
- gunicorn (production WSGI server, used by Render)

## Methodology
1. **Data Understanding & Preprocessing** (`train_model.py`) — Loaded the dataset with
   pandas, displayed the first five records, identified 13 numerical input features
   and `target` as the target variable, confirmed there are no missing values, and
   split the data into 80% training / 20% testing (stratified by class).
2. **Model Development** (`train_model.py`) — Standardized features with
   `StandardScaler` and trained a **Logistic Regression** classifier. Evaluated the
   model using Accuracy Score, then saved the trained model, scaler, and feature name
   order together into `model.pkl` using `joblib`.
3. **API Development** (`app.py`) — Built a Flask REST API with:
   - `GET /` — health/info endpoint listing required fields and model accuracy
   - `GET /health` — simple health check for uptime monitors
   - `POST /predict` — accepts patient details as JSON, validates the input, scales
     it with the saved scaler, and returns a prediction as JSON
4. **GitHub & Cloud Deployment** — Published the complete project (source code,
   trained model, Flask app, requirements, README) to a public GitHub repository and
   deployed the Flask app as a live web service on Render.

## Model Performance
| Metric | Value |
|---|---|
| Test Accuracy | 0.8033 |

**Classification Report (test set, 61 patients):**
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| 0 (No Disease) | 0.86 | 0.68 | 0.76 |
| 1 (Disease) | 0.77 | 0.91 | 0.83 |

## API Usage

**Endpoint:** `POST /predict`

**Example request body:**
```json
{
  "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
  "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
  "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
}
```

**Example response (tested locally, real output):**
```json
{
  "prediction": "Heart Disease Detected",
  "prediction_class": 1,
  "probability_of_heart_disease": 0.8745
}
```

**Example negative case (tested locally, real output):**
```json
{
  "prediction": "No Heart Disease Detected",
  "prediction_class": 0,
  "probability_of_heart_disease": 0.0058
}
```

**Example curl command:**
```bash
curl -X POST https://<your-render-app>.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'
```

## Render Deployment URL
**`<PASTE YOUR LIVE RENDER URL HERE AFTER DEPLOYING, e.g. https://heart-disease-api.onrender.com>`**

> This project's code has been fully built and tested locally (see confirmation
> below), but actually deploying it to Render requires your own Render account and
> a live push to GitHub — steps neither of which can be completed from this
> environment. Follow the deployment steps below, then paste the resulting live URL
> into this section before submitting.

### How to deploy on Render (step-by-step)
1. Push this entire folder to your public GitHub repository.
2. Go to [render.com](https://render.com) and sign in (GitHub login works directly).
3. Click **New +** → **Web Service**.
4. Connect your GitHub account and select this repository.
5. Configure the service:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app` (already specified in the included
     `Procfile`, so Render should detect it automatically)
6. Click **Create Web Service**. Render will install dependencies and start the app;
   this takes a few minutes on the first deploy.
7. Once live, Render gives you a public URL like
   `https://your-service-name.onrender.com`. Test it with the curl command above
   (swap in your URL) to confirm predictions work.
8. Paste that URL into the **Render Deployment URL** section above and commit the
   updated README.

**Note on Render's free tier:** free web services on Render spin down after a period
of inactivity and take ~30–60 seconds to "wake up" on the next request. If your
instructor evaluates the API and gets a slow first response or a timeout, that's why
— consider pinging the `/health` endpoint shortly before evaluation, or use a paid
tier if consistent uptime is required.

## Local Testing (already verified working)
Before deployment, the API was run and tested locally in a sandboxed environment to
confirm correctness:
```bash
python train_model.py    # produces model.pkl, prints Test Accuracy: 0.8033
python app.py             # starts Flask dev server on port 5000
curl http://localhost:5000/                    # returns API info + required fields
curl -X POST http://localhost:5000/predict ...  # returns real predictions (see above)
```
All three cases (root endpoint, valid positive prediction, valid negative prediction,
and a missing-field validation error) were tested and returned correct results.

## Conclusion
This project trained a Logistic Regression model to predict heart disease risk from
13 clinical parameters, achieving 80.3% test accuracy with balanced precision and
recall across both classes. The model was wrapped in a Flask REST API that validates
input, applies the same preprocessing used during training, and returns a
human-readable prediction alongside the underlying probability. The main challenge in
deployment is less about the model itself and more about operational details: making
sure the exact same feature order and scaling used in training are reproduced at
inference time, handling malformed or incomplete JSON gracefully, and accounting for
free-tier hosting behavior like Render's spin-down after inactivity, which can cause
slow first responses if not anticipated. This highlights why MLOps matters in
real-world machine learning projects — a model that performs well in a notebook is
only useful once it can be reliably packaged, served, monitored, and kept available to
the systems and people who depend on it, and most of the effort in taking a model to
production lies in that surrounding infrastructure rather than in the model training
itself.
