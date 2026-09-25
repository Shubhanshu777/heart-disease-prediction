# 🫀 Heart Disease Risk Screening

A [Streamlit](https://streamlit.io/) web app that estimates the risk of heart
disease from a handful of clinical inputs, using a trained Support Vector Machine
(SVM) classifier.

> ⚕️ **Disclaimer:** This is a machine-learning demo, **not** a medical device.
> It does not provide a diagnosis. Always consult a qualified healthcare
> professional.

## Features

- Clean, sectioned input form (demographics, vitals & blood work, ECG & exercise)
- Low / high risk prediction with an interactive probability **gauge**
- Sidebar showing the model type and its input features
- Graceful fallback to a progress bar if Plotly isn't installed

## Project structure

| File | Description |
|------|-------------|
| `app.py` | The Streamlit application |
| `heart.ipynb` | Notebook: data cleaning, EDA, encoding & scaling |
| `heart.csv` | Source dataset |
| `SVM_HEART.pkl` | Trained SVM classifier |
| `scaler.pkl` | Fitted `StandardScaler` (18 features) |
| `columns.pkl` | Training-time column order (one-hot layout) |
| `requirements.txt` | Python dependencies |

## Getting started

```bash
# 1. (optional) create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the app
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## How it works

User inputs are one-hot encoded into the exact 18-feature layout stored in
`columns.pkl`, scaled with the fitted `StandardScaler`, and passed to the SVM for
a binary risk prediction. When the model exposes calibrated probabilities
(`predict_proba`), the app renders a color-coded risk gauge.

## Notes

- The `.pkl` artifacts were serialized with a specific scikit-learn version. If
  unpickling raises a version warning/error, install the matching version.
