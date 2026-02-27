# NeuroScan AI — Alzheimer's MRI Detection

A Streamlit web app that classifies brain MRI scans into 4 Alzheimer's severity stages
and generates a personalised supplement/medication care protocol with email alerts.

## Files

```
neuroscan/
├── app.py                    ← Main Streamlit app
├── alzheimer_model.keras     ← Your trained model (add this!)
├── requirements.txt          ← Python dependencies
└── README.md
```

## Deploy on Streamlit Cloud (Free)

1. Push this folder to a **GitHub repository**
2. Add your `alzheimer_model.keras` file to the repo
   - If file > 100MB, use **Git LFS**: `git lfs track "*.keras"`
3. Go to **[share.streamlit.io](https://share.streamlit.io)**
4. Click **New app** → Select your repo → Set main file to `app.py`
5. Click **Deploy** — live in ~2 minutes!

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Email Alerts

Uses Gmail SMTP. Requires a **Gmail App Password** (not your regular password):
1. Google Account → Security → Enable 2-Step Verification
2. Search "App Passwords" → Mail + Other → Name it "NeuroScan"
3. Use the 16-character password in the app

## Model

- Architecture: EfficientNet-B0 (transfer learning)
- Input: 224×224 RGB MRI scan
- Output: 4 classes — NonDemented, VeryMildDemented, MildDemented, ModerateDemented
- Accuracy: ~90%
