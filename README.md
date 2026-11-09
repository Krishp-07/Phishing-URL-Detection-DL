# Phishing URL Detection Using Deep Learning

A fast, lightweight, and 100% offline tool to detect whether a URL is legitimate or a phishing scam using deep learning.

---

## ⚡ Highlights

- **Accuracy:** 98.49% test accuracy (0.995 AUC-ROC)
- **Fast:** Instant predictions in under 1 millisecond
- **100% Offline:** Analyzes the URL string directly without visiting the website (safe & private)
- **Compact:** Lightweight trained model (only 55 KB)

---

## 📦 Installation

1. Make sure you have **Python 3.10+** installed.
2. Open terminal in the project folder and install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Web Application (Interactive UI)
Start the local web dashboard:
```bash
python -m web.app
```
Then open your browser and go to: **`http://127.0.0.1:5000`**

---

### 2. Command Line (CLI)
Test one or more URLs directly from the terminal:
```bash
python predict.py "https://google.com" "http://paypal.com.account-update.xyz/login.php"
```

To output results as JSON:
```bash
python predict.py --json "https://github.com"
```

---

### 3. Use in Python Code
```python
from phishing_detector import PhishingDetector

detector = PhishingDetector()
result = detector.predict("https://www.google.com")

print("Verdict:", "PHISHING" if result.is_phishing else "LEGITIMATE")
print("Probability:", result.probability)
```

---

### 4. (Optional) Retrain the Model
To retrain the neural network on the dataset:
```bash
python train.py
```

---

## 🧠 How It Works

1. **Feature Extraction:** Extracts features purely from the URL text:
   - **Subword N-Grams:** Catches character-level trickery and typosquatting (e.g., `paypa1`, `g00gle`).
   - **Lexical & Structural Indicators:** Checks for raw IP addresses, suspicious TLDs (`.xyz`, `.tk`), `@` symbols, excessive subdomains, and lure keywords (`login`, `verify`).
2. **Neural Network Prediction:** Normalizes features and feeds them into a Keras neural network model (`my_model.keras`) producing a probability score:
   - **Score $\ge$ 0.50** $\rightarrow$ **Phishing**
   - **Score < 0.50** $\rightarrow$ **Legitimate**

---

## 📁 Key Files

- `web/app.py` — Flask web server and UI
- `predict.py` — Command-line prediction tool
- `train.py` — Model training script
- `phishing_detector/` — Core feature extractor and inference engine
- `my_model.keras` — Trained neural network model
- `scaler.pkl` — Feature normalizer

---

## 📊 Results Summary

Tested on a held-out benchmark split of **12,000 URLs**:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **98.49%** |
| **ROC-AUC** | **0.9954** |
| **Phishing Recall** | **99.07%** (catches nearly all attacks) |
| **Legitimate Precision** | **99.06%** (rarely flags safe sites) |
