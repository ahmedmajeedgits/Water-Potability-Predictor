# 💧 AquaScore: Water Potability Predictor

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> Enter 9 water measurements and get a **drinkability score from 0 to 100**, with a safety rating from *Critical* to *Pristine*.

---

## ✨ Features

- 🔬 **Analyzer**: enter water measurements and get an instant score and safety rating
- 🗺️ **Global Map**: compare typical water quality across 13 world regions
- 📈 **Trend Chart**: see how a region's predicted score changes from 2000 to 2024
- 📋 **Reference Ranges**: WHO / EPA guideline values shown next to your inputs

---

## 🧪 Input Measurements

| Parameter | Unit |
|---|---|
| pH | — |
| Hardness | mg/L |
| Solids (TDS) | ppm |
| Chloramines | ppm |
| Sulfate | mg/L |
| Conductivity | μS/cm |
| Organic Carbon | mg/L |
| Trihalomethanes | μg/L |
| Turbidity | NTU |

---

## 🧠 How It Works

1. **Data**: 10,000 water samples, each with 9 measurements and a drinkability score
2. **Model**: a `RandomForestRegressor` from scikit-learn, trained on an 80/20 train/test split
3. **App**: a Streamlit dashboard loads the trained model and scores new inputs live

The regional map uses **illustrative average values** for each region, not official measurements.

---

## 🚀 Run It Yourself

```bash
git clone https://github.com/ahmedmajeedgits/AquaScore.git
cd AquaScore
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
AquaScore/
├── app.py                # Streamlit app
├── WaterNote.ipynb       # Model training notebook
├── water_qualityM.pkl    # Trained Random Forest model
├── scaler.pkl            # Saved scaler
└── requirements.txt      # Dependencies
```

---

## 🛠️ Built With

Python · pandas · NumPy · scikit-learn · Streamlit · Plotly

---

## 👤 Author

**Ahmed Majeed Hameed** ([@EMRIEN](https://github.com/ahmedmajeedgits))

*The model and notebook are my own work; the UI was built with AI assistance.*