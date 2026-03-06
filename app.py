import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
import os
import requests
import json
import pandas as pd

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

html, body, [class*="css"], .css-18e3th9, .css-1d391kg {
    font-family: 'Inter', sans-serif;
    background-color: #0d1526 !important;
    color: #ffffff !important;
    font-weight: 600;
}
.main, .block-container { background-color: transparent !important; }

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
}
.block-container { padding: 2rem 2.5rem; max-width: 1150px; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; }

section[data-testid="stSidebar"] {
    background: #0d1526 !important;
    border-right: 1px solid #1e293b !important;
}
section[data-testid="stSidebar"] * { color: #ffffff !important; }

div[data-testid="stFileUploader"] {
    background: #0d1526 !important;
    border: 2px dashed #9ca3af;
    border-radius: 16px;
    padding: 12px;
    transition: border-color .2s;
}
div[data-testid="stFileUploader"]:hover { border-color: #6b7280; }

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: .03em !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .85 !important; }

.stTextInput > div > div > input {
    background: #0d1526 !important;
    border: 1px solid #1e293b !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
}
.stTextInput > div > div > input:focus { border-color: #6366f1 !important; }


/* ── Chatbot widget ── */
.chat-bubble-user {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    padding: 10px 14px;
    border-radius: 16px 16px 4px 16px;
    margin: 6px 0 6px 40px;
    font-size: 13px;
    line-height: 1.6;
}
.chat-bubble-ai {
    background: #0f1f3d;
    border: 1px solid #1e3a5f;
    color: #e2e8f0;
    padding: 10px 14px;
    border-radius: 16px 16px 16px 4px;
    margin: 6px 40px 6px 0;
    font-size: 13px;
    line-height: 1.6;
}

footer { display: none !important; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────
LABEL_MAP = {
    0: "NonDemented",
    1: "VeryMildDemented",
    2: "MildDemented",
    3: "ModerateDemented",
}

STAGE_DATA = {
    "NonDemented": {
        "label": "Non-Demented", "emoji": "🟢", "color": "#4ade80", "bg": "#052e16",
        "description": "No signs of Alzheimer's detected. Brain function appears normal. Preventive supplementation is recommended to maintain cognitive health.",
        "urgency": "Preventive Care",
        "supplements": [
            {"name": "Omega-3 (Fish Oil)",  "dose": "250 mg",  "times": ["8:00 AM"],  "purpose": "Supports brain cell membrane integrity and cognitive longevity",    "food": "Salmon, Walnuts, Flaxseeds, Sardines",            "type": "nutrient"},
            {"name": "Beta-Carotene",       "dose": "2 mg",    "times": ["8:00 AM"],  "purpose": "Antioxidant precursor to Vitamin A; protects neurons",              "food": "Carrots, Sweet Potato, Pumpkin, Spinach",          "type": "nutrient"},
            {"name": "Vitamin E",           "dose": "100 IU",  "times": ["1:00 PM"],  "purpose": "Neuroprotective antioxidant; protects cell membranes",              "food": "Almonds, Sunflower Seeds, Avocado, Olive Oil",     "type": "nutrient"},
            {"name": "Choline",             "dose": "200 mg",  "times": ["8:00 AM"],  "purpose": "Precursor to acetylcholine; supports memory and learning",          "food": "Eggs, Chicken Liver, Soybeans, Broccoli",          "type": "nutrient"},
            {"name": "Vitamin B12",         "dose": "250 mcg", "times": ["8:00 AM"],  "purpose": "Reduces homocysteine; supports myelin sheath and nerve conduction", "food": "Eggs, Milk, Paneer, Fortified Cereals",            "type": "nutrient"},
            {"name": "Curcumin (Turmeric)", "dose": "250 mg",  "times": ["8:00 PM"],  "purpose": "Mild anti-inflammatory; early amyloid plaque prevention",           "food": "Turmeric Milk, Curry, Golden Paste, Turmeric Tea", "type": "nutrient"},
        ],
    },
    "VeryMildDemented": {
        "label": "Very Mild Demented", "emoji": "🟡", "color": "#facc15", "bg": "#1c1400",
        "description": "Very early-stage cognitive changes detected. Lifestyle modifications and increased nutritional support can significantly slow progression.",
        "urgency": "Early Intervention Recommended",
        "supplements": [
            {"name": "Omega-3 (Fish Oil)",  "dose": "500 mg",  "times": ["8:00 AM"],  "purpose": "Reduces neuroinflammation; supports synaptic plasticity",         "food": "Salmon, Mackerel, Walnuts, Chia Seeds",           "type": "nutrient"},
            {"name": "Vitamin E",           "dose": "200 IU",  "times": ["1:00 PM"],  "purpose": "Slows oxidative damage to neurons in early-stage decline",        "food": "Almonds, Hazelnuts, Spinach, Sunflower Seeds",    "type": "nutrient"},
            {"name": "Choline",             "dose": "250 mg",  "times": ["8:00 AM"],  "purpose": "Boosts acetylcholine production for memory preservation",         "food": "Eggs, Chicken, Soybeans, Kidney Beans",           "type": "nutrient"},
            {"name": "Vitamin B12",         "dose": "500 mcg", "times": ["8:00 AM"],  "purpose": "Slows brain atrophy linked to B12 deficiency in early decline",  "food": "Milk, Curd, Paneer, Fortified Soy Milk",          "type": "nutrient"},
            {"name": "Curcumin (Turmeric)", "dose": "500 mg",  "times": ["8:00 PM"],  "purpose": "Anti-inflammatory; begins inhibiting amyloid-beta aggregation",    "food": "Turmeric Milk, Curry, Turmeric Tea, Haldi Doodh", "type": "nutrient"},
        ],
    },
    "MildDemented": {
        "label": "Mild Demented", "emoji": "🟠", "color": "#fb923c", "bg": "#1c0800",
        "description": "Mild cognitive decline detected. Intensified nutritional protocol recommended alongside neurologist consultation.",
        "urgency": "Medical Attention Required",
        "supplements": [
            {"name": "Omega-3 (Fish Oil)",  "dose": "500 mg",  "times": ["8:00 AM"],  "purpose": "DHA and EPA support for slowing grey matter loss and inflammation", "food": "Salmon, Sardines, Walnuts, Flaxseed Oil",      "type": "nutrient"},
            {"name": "Vitamin E",           "dose": "200 IU",  "times": ["1:00 PM"],  "purpose": "Neuroprotection against free radical damage",                       "food": "Almonds, Avocado, Olive Oil, Sunflower Seeds", "type": "nutrient"},
            {"name": "Choline",             "dose": "250 mg",  "times": ["8:00 AM"],  "purpose": "Supports declining cholinergic neurons; aids recall",               "food": "Eggs, Chicken Liver, Soybeans, Lentils",       "type": "nutrient"},
            {"name": "Vitamin B12",         "dose": "500 mcg", "times": ["8:00 AM"],  "purpose": "Repairs myelin damage; counters neurodegeneration",                "food": "Eggs, Milk, Fish, Fortified Cereals",          "type": "nutrient"},
            {"name": "Curcumin (Turmeric)", "dose": "500 mg",  "times": ["8:00 PM"],  "purpose": "Actively reduces amyloid plaques and tau tangles",                 "food": "Turmeric Milk, Curry, Golden Paste",           "type": "nutrient"},
        ],
    },
    "ModerateDemented": {
        "label": "Moderate Demented", "emoji": "🔴", "color": "#f87171", "bg": "#1c0000",
        "description": "Significant cognitive impairment detected. Maximum nutritional support protocol initiated. Immediate specialist consultation required.",
        "urgency": "URGENT — See Neurologist Immediately",
        "supplements": [
            {"name": "Omega-3 (Fish Oil)",  "dose": "500 mg",  "times": ["8:00 AM"],  "purpose": "DHA and EPA to slow advanced neuronal loss and inflammation",       "food": "Salmon, Mackerel, Sardines, Walnuts",             "type": "nutrient"},
            {"name": "Beta-Carotene",       "dose": "3 mg",    "times": ["8:00 AM"],  "purpose": "Antioxidant coverage for severe oxidative brain damage",            "food": "Carrots, Sweet Potato, Mango, Papaya",            "type": "nutrient"},
            {"name": "Vitamin E",           "dose": "200 IU",  "times": ["1:00 PM"],  "purpose": "Neuroprotective dose; slows advanced neurodegeneration",            "food": "Almonds, Sunflower Seeds, Avocado, Olive Oil",    "type": "nutrient"},
            {"name": "Choline",             "dose": "250 mg",  "times": ["8:00 AM"],  "purpose": "Critical support for severely depleted cholinergic pathways",       "food": "Eggs, Chicken, Soybeans, Broccoli",               "type": "nutrient"},
            {"name": "Vitamin B12",         "dose": "500 mcg", "times": ["8:00 AM"],  "purpose": "B12 support for severely depleted neurological pathways",           "food": "Eggs, Milk, Fish, Fortified Soy Milk",            "type": "nutrient"},
            {"name": "Curcumin (Turmeric)", "dose": "500 mg",  "times": ["8:00 PM"],  "purpose": "Plaque and tangle inhibition in advanced Alzheimer's",             "food": "Turmeric Milk, Curry, Golden Paste, Haldi Doodh", "type": "nutrient"},
        ],
    },
}

# ── MRI Validation ─────────────────────────────────────────────
def is_likely_mri(img: Image.Image) -> tuple:
    """
    Multi-heuristic check to determine if an image is likely a brain MRI scan.
    Returns (is_valid: bool, reason: str, details: dict)
    """
    issues = []
    details = {}

    # Convert to numpy arrays for analysis
    img_rgb  = img.convert("RGB")
    img_gray = img.convert("L")
    arr_rgb  = np.array(img_rgb, dtype=np.float32)
    arr_gray = np.array(img_gray, dtype=np.float32)

    total_pixels = arr_gray.size

    # ── Check 1: Dark background ratio ──────────────────────────
    # Brain MRIs have a large black/near-black background
    dark_pixels = np.sum(arr_gray < 25)
    dark_ratio  = dark_pixels / total_pixels
    details["dark_ratio"] = dark_ratio
    if dark_ratio < 0.20:
        issues.append(f"insufficient dark background ({dark_ratio:.1%} < 20% expected for MRI)")

    # ── Check 2: Colorfulness ────────────────────────────────────
    # MRIs are grayscale; very colorful images are not MRIs
    r = arr_rgb[:, :, 0]
    g = arr_rgb[:, :, 1]
    b = arr_rgb[:, :, 2]
    rg_diff = np.std(r.astype(float) - g.astype(float))
    gb_diff = np.std(g.astype(float) - b.astype(float))
    color_score = rg_diff + gb_diff
    details["color_score"] = color_score
    if color_score > 25:
        issues.append(f"image appears too colorful (score {color_score:.1f} > 25 expected for grayscale MRI)")

    # ── Check 3: Grayscale channel similarity ────────────────────
    # In a true grayscale image R ≈ G ≈ B
    channel_diff = np.mean(np.abs(r - g)) + np.mean(np.abs(g - b))
    details["channel_diff"] = channel_diff
    if channel_diff > 15:
        issues.append(f"channels differ too much ({channel_diff:.1f} > 15) — likely a color photo, not an MRI")

    # ── Check 4: Brightness variance (texture check) ─────────────
    # MRIs have a distinct bright oval (brain) on a dark field
    # A paper/document would be mostly uniform bright
    mean_brightness = np.mean(arr_gray)
    std_brightness  = np.std(arr_gray)
    details["mean_brightness"] = mean_brightness
    details["std_brightness"]  = std_brightness

    if mean_brightness > 200:
        issues.append(f"image is too bright (mean {mean_brightness:.0f} > 200) — looks like a document or photo, not an MRI")

    if std_brightness < 20:
        issues.append(f"image has too little contrast (std {std_brightness:.1f} < 20) — MRIs have strong contrast between brain and background")

    # ── Check 5: Bright region (brain) must exist ────────────────
    # At least some pixels should be bright (the brain tissue)
    bright_pixels = np.sum(arr_gray > 100)
    bright_ratio  = bright_pixels / total_pixels
    details["bright_ratio"] = bright_ratio
    if bright_ratio < 0.05:
        issues.append(f"too few bright pixels ({bright_ratio:.1%} < 5%) — no visible brain structure detected")
    if bright_ratio > 0.80:
        issues.append(f"too many bright pixels ({bright_ratio:.1%} > 80%) — likely a document or overexposed photo")

    is_valid = len(issues) == 0
    reason   = "; ".join(issues) if issues else "Image passed all MRI validation checks."
    return is_valid, reason, details


# ── Model loader ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "alzheimer_model.keras")
    if not os.path.exists(model_path):
        st.error("❌ Model file 'alzheimer_model.keras' not found. Place it in the same folder as app.py")
        st.stop()
    return tf.keras.models.load_model(model_path)

def preprocess(img: Image.Image) -> np.ndarray:
    img    = img.convert("RGB")
    tensor = tf.convert_to_tensor(np.array(img), dtype=tf.float32)
    tensor = tf.image.resize_with_pad(tensor, 224, 224)
    return np.expand_dims(tensor.numpy(), axis=0)

# ── Email helpers ──────────────────────────────────────────────
def build_email_html(name, age, stage_key, conf, all_probs):
    data     = STAGE_DATA[stage_key]
    c        = data["color"]
    date_str = datetime.now().strftime("%d %B %Y")

    rows = ""
    for item in data["supplements"]:
        tc   = c if item["type"] == "medication" else "#a78bfa"
        icon = ""
        sched = "<br>".join([f"{t}" for t in item["times"]])
        rows += f"""<tr>
          <td style="padding:10px 14px;border-bottom:1px solid #1e293b;">
            <strong style="color:{tc};font-size:13px;">{icon} {item['name']}</strong>
            <div style="font-size:11px;color:#cbd5e1;margin-top:2px;">{item['purpose']}</div>
          </td>
          <td style="padding:10px 14px;border-bottom:1px solid #1e293b;color:#e2e8f0;font-weight:600;">{item['dose']}</td>
          <td style="padding:10px 14px;border-bottom:1px solid #1e293b;color:{c};font-size:12px;line-height:1.8;">{sched}</td>
        </tr>"""

    bars = ""
    for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        bc = STAGE_DATA[cls]["color"]
        bars += f"""<div style="margin-bottom:8px;">
          <div style="display:flex;justify-content:space-between;font-size:12px;">
            <span style="color:#cbd5e1;">{STAGE_DATA[cls]['label']}</span>
            <span style="color:{bc};font-weight:600;">{prob:.1f}%</span>
          </div>
          <div style="background:#1e293b;border-radius:4px;height:6px;">
            <div style="width:{min(prob,100):.1f}%;background:{bc};height:6px;border-radius:4px;"></div>
          </div>
        </div>"""

    patient_line = f"""<p style="margin:6px 0 0;color:{c};font-size:13px;">
        <strong>Patient:</strong> {name}{f" &nbsp;·&nbsp; Age: {age}" if age else ""}
    </p>""" if name else ""

    return f"""<!DOCTYPE html><html>
<body style="background:#08090f;font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:0;">
<div style="max-width:640px;margin:0 auto;padding:24px 16px;">
  <div style="background:linear-gradient(135deg,#0f0f1f,#1a1a2e);border:1px solid #1e293b;
              border-radius:14px;padding:20px;margin-bottom:18px;text-align:center;">
    
    <h1 style="margin:4px 0 0;color:#e0e7ff;font-size:20px;">NeuroScan AI</h1>
    <p style="margin:3px 0 0;color:#6366f1;font-size:11px;letter-spacing:.15em;text-transform:uppercase;">
      Alzheimer's Detection & Care Protocol
    </p>
  </div>
  <div style="background:linear-gradient(135deg,{c}0a,{data['bg']});border:1.5px solid {c}55;
              border-radius:14px;padding:18px;margin-bottom:16px;">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
      <span style="font-size:24px;">{data['emoji']}</span>
      <div>
        <div style="color:{c};font-size:18px;font-weight:700;">{data['label']}</div>
        <div style="color:#cbd5e1;font-size:12px;">
          Confidence: <strong style="color:{c};">{conf:.1f}%</strong> &nbsp;·&nbsp; {data['urgency']}
        </div>
      </div>
    </div>
    <p style="color:#cbd5e1;font-size:13px;margin:0;">{data['description']}</p>
    {patient_line}
    <p style="color:#cbd5e1;font-size:11px;margin:6px 0 0;">Report Date: {date_str}</p>
  </div>
  <div style="background:#0f172a;border:1px solid #1e293b;border-radius:14px;padding:16px;margin-bottom:16px;">
    <h3 style="margin:0 0 12px;color:#e2e8f0;font-size:12px;text-transform:uppercase;letter-spacing:.1em;">
      Model Confidence Breakdown
    </h3>{bars}
  </div>
  <div style="background:#0f172a;border:1px solid #1e293b;border-radius:14px;overflow:hidden;margin-bottom:16px;">
    <div style="padding:10px 14px;background:#0a0f1e;color:{c};font-size:12px;font-weight:600;
                text-transform:uppercase;letter-spacing:.12em;border-bottom:1px solid {c}33;">PRESCRIBED PROTOCOL</div>
    <table style="width:100%;border-collapse:collapse;">
      <thead><tr style="background:#080d16;">
        <th style="padding:7px 12px;text-align:left;font-size:10px;color:#cbd5e1;
                   text-transform:uppercase;border-bottom:1px solid #1e293b;">Name</th>
        <th style="padding:7px 12px;text-align:left;font-size:10px;color:#cbd5e1;
                   text-transform:uppercase;border-bottom:1px solid #1e293b;">Dose</th>
        <th style="padding:7px 12px;text-align:left;font-size:10px;color:#cbd5e1;
                   text-transform:uppercase;border-bottom:1px solid #1e293b;">Schedule</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  <div style="background:#0f172a;border:1px solid #1e3a5f;border-radius:12px;padding:14px;margin-bottom:14px;">
    <h3 style="margin:0 0 8px;color:#38bdf8;font-size:12px;text-transform:uppercase;">Daily Reminder Tips</h3>
    <ul style="margin:0;padding-left:16px;color:#cbd5e1;font-size:12px;line-height:2;">
      <li>Set phone alarms for each medication time above.</li>
      <li>Take supplements with food for better absorption.</li>
      <li>Keep a daily medication log to track compliance.</li>
      <li>Follow up with your neurologist every 3 months.</li>
      <li>Combine medication with physical exercise and cognitive activities.</li>
    </ul>
  </div>
  <div style="background:rgba(239,68,68,.06);border:1px solid #7f1d1d;border-radius:10px;padding:10px 14px;">
    <p style="margin:0;color:#f87171;font-size:11px;line-height:1.7;">
      <strong>Note:</strong> AI-generated guidance only.
      Always consult a licensed neurologist before starting any medication or supplement.
    </p>
  </div>
  <p style="text-align:center;color:#e2e8f0;font-size:10px;margin-top:16px;">
    NeuroScan AI · EfficientNetB0 · 96.3% Accuracy · {date_str}
  </p>
</div></body></html>"""


def send_email(sender, password, recipient, subject, html):
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender
        msg["To"]      = recipient
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(sender, password)
            s.sendmail(sender, recipient, msg.as_string())
        return True, "✅ Prescription email sent successfully!"
    except smtplib.SMTPAuthenticationError:
        return False, "❌ Auth failed. Please use a Gmail App Password."
    except Exception as e:
        return False, f"❌ Error: {str(e)}"



# ── Chatbot helper ─────────────────────────────────────────────
def ask_neuroscan_ai(messages: list, stage_context: str = "", api_key: str = "") -> str:
    """Call Groq API for the NeuroScan chatbot."""
    # Try Streamlit secrets first, then env variable
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except:
            pass
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return "⚠️ GROQ_API_KEY not found. Please add it to Streamlit Secrets."

    system_prompt = f"""You are NeuroScan AI Assistant, a compassionate and knowledgeable medical support chatbot embedded in a brain MRI analysis app for Alzheimer's disease detection.

Your role is to:
- Answer questions about Alzheimer's disease, dementia stages, symptoms, and caregiving
- Explain the nutrients and supplements recommended in the app
- Provide lifestyle, diet, and cognitive exercise advice tailored to the patient's stage
- Offer emotional support and guidance to caregivers and family members
- Explain MRI scan results in plain, easy-to-understand language

Current patient context: {stage_context if stage_context else "No scan uploaded yet."}

Guidelines:
- Always be warm, empathetic, and clear
- Use simple language — avoid heavy medical jargon unless asked
- Never provide specific medical diagnoses or replace a neurologist
- Keep responses concise (3-5 sentences) unless a detailed explanation is requested
- If asked something outside your scope, gently redirect to a healthcare professional"""

    # Build messages with system prompt
    groq_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        groq_messages.append({"role": msg["role"], "content": msg["content"]})

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": groq_messages,
        "max_tokens": 1000,
        "temperature": 0.7,
    }

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        data = resp.json()
        if resp.status_code == 200:
            return data["choices"][0]["message"]["content"]
        else:
            return f"⚠️ API error {resp.status_code}: {data.get('error', {}).get('message', 'Unknown error')}"
    except Exception as e:
        return f"⚠️ Connection error: {str(e)}"



# ── Symptom Trend Analysis ─────────────────────────────────────
def analyze_symptom_trends(logs: list, api_key: str = "") -> str:
    """Send symptom logs to Groq for AI trend analysis."""
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except:
            pass
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return "⚠️ GROQ_API_KEY not found. Please add it to Streamlit Secrets."

    log_text = ""
    for i, log in enumerate(logs):
        pid = log.get("patient_id", "")
        pname = log.get("patient_name", "")
        patient_line = f"  Patient: {pname} (ID: {pid})" if pid or pname else ""
        log_text += f"""
Entry {i+1} — Date: {log['date']}{patient_line}
  Memory Loss: {log['memory']}/10
  Confusion: {log['confusion']}/10
  Mood Changes: {log['mood']}/10
  Daily Tasks Difficulty: {log['tasks']}/10
  Sleep Quality: {log['sleep']}/10
  Overall Score: {log['overall']}/10
  Notes: {log.get('notes', 'None')}
"""

    system_prompt = """You are a neurological symptom trend analyst AI embedded in the NeuroScan Alzheimer's detection app.
You analyze daily symptom logs from patients or caregivers to detect cognitive decline patterns.

Your analysis should include:
1. TREND SUMMARY — Is the patient improving, stable, or declining?
2. CONCERNING PATTERNS — Which symptoms are worsening fastest?
3. RISK FLAG — Low / Moderate / High / Critical based on trend
4. RECOMMENDED NEXT STEPS — Specific actionable advice
5. CAREGIVER ALERT — Any urgent flags for the caregiver

Be concise, empathetic, and clear. Use plain language. Format with clear sections."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please analyze these symptom logs and provide a full trend report:\n{log_text}"}
    ]

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "max_tokens": 1200,
        "temperature": 0.4,
    }

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        data = resp.json()
        if resp.status_code == 200:
            return data["choices"][0]["message"]["content"]
        else:
            return f"⚠️ API error {resp.status_code}: {data.get('error', {}).get('message', 'Unknown error')}"
    except Exception as e:
        return f"⚠️ Connection error: {str(e)}"

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 24px;">
      <div style="width:48px;height:48px;background:linear-gradient(135deg,#6366f1,#818cf8);
                  border-radius:12px;margin:0 auto 12px;display:flex;align-items:center;
                  justify-content:center;">
        <div style="width:24px;height:24px;border:3px solid white;border-radius:50%;"></div>
      </div>
      <h2 style="margin:0;color:#e0e7ff !important;font-family:'Playfair Display',serif;font-size:22px;letter-spacing:.02em;">NeuroScan AI</h2>
      <p style="margin:6px 0 0;color:#6366f1 !important;font-size:11px;letter-spacing:.15em;text-transform:uppercase;">
        Alzheimer's Detection
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='font-size:15px;font-weight:700;color:#e2e8f0;'>About</p>", unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:14px;line-height:1.8;color:#e2e8f0 !important;">
    This app uses a deep learning model based on <strong style="color:#e2e8f0 !important;">EfficientNet-B0</strong>
    to classify MRI scans into 4 Alzheimer's severity stages and generates
    a personalised prescription protocol with email alerts.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='font-size:15px;font-weight:700;color:#e2e8f0;'>Classification Stages</p>", unsafe_allow_html=True)
    for key, data in STAGE_DATA.items():
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
          <div style="width:8px;height:8px;border-radius:50%;background:{data['color']};flex-shrink:0;"></div>
          <div>
            <div style="font-size:14px;color:#e2e8f0 !important;font-weight:600;">{data['label']}</div>
            <div style="font-size:12px;color:#cbd5e1 !important;">{data['urgency']}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="background:rgba(239,68,68,.06);border:1px solid #7f1d1d;border-radius:8px;padding:10px 12px;">
      <p style="margin:0;color:#f87171 !important;font-size:13px;line-height:1.7;">
        For educational purposes only. Not a substitute for professional medical advice.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='font-size:15px;font-weight:700;color:#e2e8f0;'>AI Assistant Status</p>", unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:14px;color:#4ade80 !important;margin-top:6px;font-weight:600;">Active</p>
    """, unsafe_allow_html=True)
    gemini_api_key = ""


# ══════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:8px 0 28px;">
  <h1 style="margin:0;background:linear-gradient(135deg,#6366f1,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;font-size:32px;letter-spacing:.02em;">
    Alzheimer's MRI Analysis
  </h1>
  <p style="margin:8px 0 0;color:#cbd5e1;font-size:14px;">
    Upload a brain MRI scan · Get instant classification · Receive personalised care protocol
  </p>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["MRI Analysis", "Symptom Tracker", "Pricing"])

with tab1:
    # ── Patient info ─────────────────────────────────────────
    pc1, pc2 = st.columns([3, 1])
    with pc1:
        patient_name = st.text_input("Patient Name", placeholder="Full name (optional)")
    with pc2:
        patient_age = st.text_input("Age", placeholder="e.g. 68")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Upload ───────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload Brain MRI Scan",
        type=["jpg", "jpeg", "png"],
        help="Upload an axial T1-weighted MRI scan. JPG or PNG format.",
    )

    # ══════════════════════════════════════════════════════════
    #  PREDICTION + RESULTS
    # ══════════════════════════════════════════════════════════
    if uploaded:
        image = Image.open(uploaded)

        # ── MRI Validation ─────────────────────────────────────────
        with st.spinner("Validating image..."):
            is_valid, reason, details = is_likely_mri(image)

        if not is_valid:
            # Show the uploaded image so user can see what was rejected
            col_img, col_msg = st.columns([1, 2])
            with col_img:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            with col_msg:
                st.markdown(f"""
                <div style="background:rgba(239,68,68,.08);border:1.5px solid #ef4444;border-radius:14px;
                            padding:22px;margin-top:8px;">
                  <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                    
                    <div>
                      <div style="color:#f87171;font-size:17px;font-weight:700;">Invalid Image Detected</div>
                      <div style="color:#cbd5e1;font-size:12px;margin-top:2px;">This does not appear to be a brain MRI scan</div>
                    </div>
                  </div>
                  <p style="color:#cbd5e1;font-size:13px;margin:0 0 14px;line-height:1.7;">
                    The uploaded image failed one or more MRI validation checks:
                  </p>
                  <div style="background:#1c0000;border:1px solid #7f1d1d;border-radius:8px;padding:12px;margin-bottom:14px;">
                    <p style="color:#fca5a5;font-size:12px;margin:0;line-height:1.8;">{reason}</p>
                  </div>
                  <div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:8px;padding:12px;">
                    <p style="color:#38bdf8;font-size:12px;font-weight:600;margin:0 0 6px;">
                      What a valid MRI scan looks like:
                    </p>
                    <ul style="color:#cbd5e1;font-size:12px;margin:0;padding-left:16px;line-height:2;">
                      <li>Grayscale image with a large dark (black) background</li>
                      <li>Bright oval/circular brain structure in the center</li>
                      <li>No color — MRI scans are always black & white</li>
                      <li>High contrast between brain tissue and background</li>
                    </ul>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.stop()

        # ── Run model ──────────────────────────────────────────────
        model  = load_model()
        arr    = preprocess(image)

        with st.spinner("Analyzing MRI scan..."):
            probs = model.predict(arr, verbose=0)[0]

        idx       = int(np.argmax(probs))
        label     = LABEL_MAP[idx]
        conf      = float(probs[idx]) * 100
        data      = STAGE_DATA[label]
        all_probs = {LABEL_MAP[i]: float(probs[i]) * 100 for i in range(4)}

        # ── Low-confidence warning ─────────────────────────────────
        LOW_CONF_THRESHOLD = 55.0
        if conf < LOW_CONF_THRESHOLD:
            st.markdown(f"""
            <div style="background:rgba(251,146,60,.08);border:1.5px solid #f97316;border-radius:12px;
                        padding:14px 18px;margin-bottom:16px;">
              <div style="display:flex;align-items:center;gap:8px;">
                
                <div>
                  <strong style="color:#fb923c;font-size:13px;">Low Confidence Result ({conf:.1f}%)</strong>
                  <p style="color:#cbd5e1;font-size:12px;margin:4px 0 0;line-height:1.6;">
                    The model is not confident in this classification. The image may not be a standard
                    Alzheimer's MRI scan, or the scan quality may be poor. Please verify with a qualified neurologist.
                  </p>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns([1, 2])

        # ── Left: image + confidence chart ────────────────────────
        with left:
            st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

            st.markdown("**Confidence Breakdown**")
            for k in LABEL_MAP.values():
                prob    = all_probs[k]
                bc      = STAGE_DATA[k]["color"]
                is_pred = k == label
                st.markdown(f"""
                <div style="margin-bottom:9px;">
                  <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;">
                    <span style="color:{'#e2e8f0' if is_pred else '#64748b'};
                                 font-weight:{'600' if is_pred else '400'};">
                      {STAGE_DATA[k]['label']} {'◀' if is_pred else ''}
                    </span>
                    <span style="color:{bc};font-weight:600;">{prob:.1f}%</span>
                  </div>
                  <div style="background:#1e293b;border-radius:4px;height:7px;">
                    <div style="width:{prob:.1f}%;background:{bc};height:7px;border-radius:4px;
                                {'box-shadow:0 0 8px ' + bc + '88' if is_pred else ''}"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Right: diagnosis + prescription ───────────────────────
        with right:
            date_str     = datetime.now().strftime("%d %B %Y · %I:%M %p")
            patient_line = f"""<p style="margin:6px 0 0;color:{data['color']};font-size:13px;">
                <strong>Patient:</strong> {patient_name}{f" &nbsp;·&nbsp; Age: {patient_age}" if patient_age else ""}
            </p>""" if patient_name else ""

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{data['color']}0d,{data['bg']});
                        border:1.5px solid {data['color']}55;border-radius:14px;padding:20px;
                        margin-bottom:18px;box-shadow:0 8px 32px {data['color']}18;">
              <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
                <span style="font-size:28px;">{data['emoji']}</span>
                <div>
                  <div style="color:{data['color']};font-size:20px;font-weight:700;
                              font-family:'Playfair Display',serif;">{data['label']}</div>
                  <div style="color:#cbd5e1;font-size:12px;margin-top:2px;">
                    Confidence: <strong style="color:{data['color']};">{conf:.1f}%</strong>
                    &nbsp;·&nbsp; <span style="color:{data['color']};">{data['urgency']}</span>
                  </div>
                </div>
              </div>
              <p style="color:#cbd5e1;font-size:13px;margin:0;line-height:1.7;">{data['description']}</p>
              {patient_line}
              <p style="color:#e2e8f0;font-size:11px;margin:8px 0 0;">Analyzed: {date_str}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;padding-bottom:12px;
                        border-bottom:1px solid {data['color']}33;margin-bottom:14px;">
              
              <h3 style="margin:0;color:{data['color']} !important;font-size:14px;text-transform:uppercase;
                         letter-spacing:.1em;font-family:'DM Sans',sans-serif;-webkit-text-fill-color:{data['color']};">Prescribed Protocol</h3>
            </div>
            """, unsafe_allow_html=True)

            for item in data["supplements"]:
                tc       = data["color"] if item["type"] == "medication" else "#a78bfa"
                lb       = data["color"] if item["type"] == "medication" else "#6366f1"
                sched_times = " &nbsp;|&nbsp; ".join([f"<strong>{t}</strong>" for t in item["times"]])
                badge_bg = "#0d1e38" if item["type"] == "medication" else "#1a0a44"
                food_src = item.get("food", "")

                st.markdown(f"""
                <div style="background:#0d1526;border:1px solid #1e293b;border-left:3px solid {lb};
                            border-radius:10px;padding:14px 18px;margin-bottom:10px;">

                  <div style="display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;">
                    <div style="flex:1;min-width:180px;">
                      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                        <strong style="color:{tc};font-size:14px;letter-spacing:.01em;">{item['name']}</strong>
                        <span style="background:{badge_bg};color:{tc};border:1px solid {tc}44;
                                     border-radius:6px;padding:2px 9px;font-size:9px;
                                     text-transform:uppercase;letter-spacing:.1em;font-weight:700;">{item['type']}</span>
                      </div>
                      <div style="font-size:12px;color:#94a3b8;line-height:1.7;">{item['purpose']}</div>
                    </div>

                    <div style="display:flex;gap:28px;flex-shrink:0;align-items:flex-start;padding-top:2px;">
                      <div>
                        <div style="font-size:10px;color:#6366f1;text-transform:uppercase;
                                    letter-spacing:.12em;font-weight:700;margin-bottom:5px;">DOSE</div>
                        <div style="font-size:15px;color:#e2e8f0;font-weight:700;">{item['dose']}</div>
                      </div>
                      <div>
                        <div style="font-size:10px;color:#6366f1;text-transform:uppercase;
                                    letter-spacing:.12em;font-weight:700;margin-bottom:5px;">SCHEDULE</div>
                        <div style="font-size:13px;color:{data['color']};font-weight:600;
                                    line-height:2;">{sched_times}</div>
                      </div>
                    </div>
                  </div>

                  {"" if not food_src else f'''
                  <div style="margin-top:10px;padding-top:10px;border-top:1px solid #1e293b;">
                    <div style="font-size:10px;color:#4ade80;text-transform:uppercase;
                                letter-spacing:.12em;font-weight:700;margin-bottom:5px;">NATURAL FOOD SOURCES</div>
                    <div style="font-size:12px;color:#cbd5e1;line-height:1.7;">{food_src}</div>
                  </div>'''}

                </div>
                """, unsafe_allow_html=True)

        # ── Email Section ──────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<hr style="border-color:#1e293b;margin:8px 0 24px;">""", unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-bottom:18px;">
            <h3 style="margin:0;background:linear-gradient(135deg,#6366f1,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;font-size:17px;font-family:'Playfair Display',serif;">
              Send Prescription Email
            </h3>
            <p style="margin:6px 0 0;color:#cbd5e1;font-size:12px;">
              Email the full care protocol with dosing schedule to patient or caregiver
            </p>
        </div>
        """, unsafe_allow_html=True)

        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            recipient = st.text_input("Recipient Email",    placeholder="patient@example.com")
        with ec2:
            sender_em = st.text_input("Your Gmail Address", placeholder="youremail@gmail.com")
        with ec3:
            sender_pw = st.text_input("Gmail App Password", placeholder="xxxx xxxx xxxx xxxx", type="password")

        if st.button("Send Prescription Email", use_container_width=True):
            if not recipient or "@" not in recipient:
                st.error("⚠️ Please enter a valid recipient email address.")
            elif not sender_em or not sender_pw:
                st.error("⚠️ Please enter your Gmail address and App Password.")
            else:
                with st.spinner("Sending email..."):
                    html    = build_email_html(patient_name, patient_age, label, conf, all_probs)
                    subject = f"NeuroScan AI — {data['label']} Prescription | {datetime.now().strftime('%d %b %Y')}"
                    ok, msg = send_email(sender_em, sender_pw, recipient, subject, html)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        st.markdown("""
        <div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:12px;
                    padding:14px 18px;margin-top:12px;">
          <p style="margin:0 0 8px;color:#38bdf8;font-size:12px;font-weight:600;">
            How to get a Gmail App Password (required)
          </p>
          <ol style="margin:0;padding-left:18px;color:#cbd5e1;font-size:12px;line-height:2.1;">
            <li>Go to <strong style="color:#cbd5e1;">myaccount.google.com</strong> → Security</li>
            <li>Enable <strong style="color:#cbd5e1;">2-Step Verification</strong></li>
            <li>Search <strong style="color:#cbd5e1;">App Passwords</strong> → Mail + Other → Name it "NeuroScan"</li>
            <li>Copy the 16-character password → paste in the field above</li>
          </ol>
        </div>
        """, unsafe_allow_html=True)

        # ── AI Chatbot Section ─────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<hr style="border-color:#1e293b;margin:8px 0 24px;">""", unsafe_allow_html=True)

        # Build context string from current scan result
        stage_context = f"Patient stage: {data['label']} (Confidence: {conf:.1f}%). Urgency: {data['urgency']}."
        if patient_name:
            stage_context += f" Patient name: {patient_name}."
        if patient_age:
            stage_context += f" Age: {patient_age}."

        st.markdown(f"""
        <div style="margin-bottom:18px;">
            <h3 style="margin:0;background:linear-gradient(135deg,#6366f1,#818cf8);-webkit-background-clip:text;
                       -webkit-text-fill-color:transparent;background-clip:text;font-size:17px;
                       font-family:'Playfair Display',serif;">AI Assistant</h3>
            <p style="margin:6px 0 0;color:#cbd5e1;font-size:12px;">
              Ask anything about the diagnosis, nutrients, caregiving, or lifestyle advice
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Suggested questions
        st.markdown(f"""
        <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;">
          <div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:20px;
                      padding:5px 12px;font-size:11px;color:#38bdf8;cursor:pointer;">
            What do these nutrients do?
          </div>
          <div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:20px;
                      padding:5px 12px;font-size:11px;color:#38bdf8;cursor:pointer;">
            What foods should the patient eat?
          </div>
          <div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:20px;
                      padding:5px 12px;font-size:11px;color:#38bdf8;cursor:pointer;">
            What does this stage mean day-to-day?
          </div>
          <div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:20px;
                      padding:5px 12px;font-size:11px;color:#38bdf8;cursor:pointer;">
            How can I help as a caregiver?
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Init chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Chat display container
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown(f"""
                <div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:14px;
                            padding:18px;text-align:center;color:#cbd5e1;">
                  
                  <p style="font-size:13px;margin:0;color:#cbd5e1;">
                    Hi! I'm your NeuroScan AI Assistant. I can see the scan result is
                    <strong style="color:{data['color']};">{data['label']}</strong>.
                    Ask me anything about the diagnosis, recommended nutrients, or how to support the patient.
                  </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div style="display:flex;justify-content:flex-end;margin:6px 0;">
                          <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;
                                      padding:10px 14px;border-radius:16px 16px 4px 16px;
                                      max-width:75%;font-size:13px;line-height:1.6;">
                            {msg["content"]}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="display:flex;justify-content:flex-start;margin:6px 0;">
                          <div style="background:#0f1f3d;border:1px solid #1e3a5f;color:#e2e8f0;
                                      padding:10px 14px;border-radius:16px 16px 16px 4px;
                                      max-width:80%;font-size:13px;line-height:1.6;">
                            <span style="font-size:10px;color:#6366f1;font-weight:600;
                                         display:block;margin-bottom:4px;">NeuroScan AI</span>
                            {msg["content"]}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

        # Input row
        col_input, col_send, col_clear = st.columns([6, 1, 1])
        with col_input:
            user_input = st.text_input(
                "chat_input", label_visibility="collapsed",
                placeholder="Ask about the diagnosis, nutrients, caregiving tips...",
                key="chat_input_field"
            )
        with col_send:
            send_clicked = st.button("Send", use_container_width=True, key="chat_send")
        with col_clear:
            if st.button("Clear", use_container_width=True, key="chat_clear"):
                st.session_state.chat_history = []
                st.rerun()

        if send_clicked and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
            with st.spinner("Analysing..."):
                reply = ask_neuroscan_ai(st.session_state.chat_history, stage_context, gemini_api_key)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()


    else:
        # ── Empty state ──────────────────────────────────────
        st.markdown("""
        <div style="background:#0d1526;border:2px dashed #1e3a5f;border-radius:16px;
                    padding:64px 24px;text-align:center;color:#e2e8f0;margin-top:16px;">
          
          <p style="font-size:16px;letter-spacing:.04em;margin:0;color:#cbd5e1;">
            Upload an MRI scan above to begin analysis
          </p>
          <p style="font-size:12px;margin:8px 0 0;color:#e2e8f0;">
            Supported formats: JPG · JPEG · PNG
          </p>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TAB 2 — SYMPTOM TRACKER
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style="text-align:center;padding:4px 0 24px;">
      <h2 style="margin:0;background:linear-gradient(135deg,#6366f1,#818cf8);-webkit-background-clip:text;
                 -webkit-text-fill-color:transparent;background-clip:text;font-size:26px;">
        Daily Symptom Tracker
      </h2>
      <p style="margin:8px 0 0;color:#cbd5e1;font-size:13px;">
        Log symptoms daily · AI detects patterns · Get early warnings
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Init session state for logs
    if "symptom_logs" not in st.session_state:
        st.session_state.symptom_logs = []

    # ── Log Entry Form ─────────────────────────────────────────
    st.markdown("""
    <div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:14px;
                padding:20px;margin-bottom:20px;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;">
          <h3 style="margin:0;background:linear-gradient(135deg,#6366f1,#818cf8);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent !important;
                   background-clip:text;font-size:16px;letter-spacing:.02em;color:transparent !important;">
          Log Today's Symptoms
        </h3>
      </div>
    """, unsafe_allow_html=True)

    # Patient ID row
    pid_col1, pid_col2, pid_col3 = st.columns([2, 2, 2])
    with pid_col1:
        log_patient_id = st.text_input("Patient ID", placeholder="e.g. PT-001", key="log_pid")
    with pid_col2:
        log_patient_name = st.text_input("Patient Name", placeholder="Full name", key="log_pname")
    with pid_col3:
        log_date = st.date_input("Date", value=datetime.now().date(), key="log_date")

    log_col1, log_col2 = st.columns(2)

    with log_col1:
        log_memory   = st.slider("Memory Loss",                0, 10, 3, key="log_memory", help="0 = None, 10 = Severe")
        log_confusion= st.slider("Confusion / Disorientation", 0, 10, 3, key="log_conf",   help="0 = None, 10 = Severe")
        log_mood     = st.slider("Mood Changes",               0, 10, 3, key="log_mood",   help="0 = None, 10 = Severe")

    with log_col2:
        log_tasks    = st.slider("Daily Tasks Difficulty",     0, 10, 3, key="log_tasks",   help="0 = None, 10 = Severe")
        log_sleep    = st.slider("Sleep Quality",              0, 10, 5, key="log_sleep",   help="0 = Very Poor, 10 = Excellent")
        log_overall  = st.slider("Overall Condition",          0, 10, 5, key="log_overall", help="0 = Very Poor, 10 = Excellent")

    log_notes = st.text_input("Notes (optional)", placeholder="Any observations or changes noticed today...", key="log_notes")

    st.markdown("</div>", unsafe_allow_html=True)

    save_col, clear_col = st.columns([3, 1])
    with save_col:
        if st.button("Save Log", use_container_width=True, key="save_log"):
            new_log = {
                "date":         str(log_date),
                "patient_id":   log_patient_id,
                "patient_name": log_patient_name,
                "memory":       log_memory,
                "confusion":    log_confusion,
                "mood":         log_mood,
                "tasks":        log_tasks,
                "sleep":        log_sleep,
                "overall":      log_overall,
                "notes":        log_notes,
            }
            # Check if entry for this date + patient already exists — update it
            existing = [i for i, l in enumerate(st.session_state.symptom_logs)
                        if l["date"] == str(log_date) and l.get("patient_id","") == log_patient_id]
            if existing:
                st.session_state.symptom_logs[existing[0]] = new_log
                st.success("✅ Log updated for " + str(log_date) + (" — " + log_patient_id if log_patient_id else ""))
            else:
                st.session_state.symptom_logs.append(new_log)
                st.success("✅ Log saved for " + str(log_date) + (" — " + log_patient_id if log_patient_id else ""))
            st.rerun()
    with clear_col:
        if st.button("Clear All", use_container_width=True, key="clear_logs"):
            st.session_state.symptom_logs = []
            st.rerun()

    # ── Logs Table + Chart ─────────────────────────────────────
    if st.session_state.symptom_logs:
        logs = sorted(st.session_state.symptom_logs, key=lambda x: x["date"])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
              <h3 style="margin:0;color:#e2e8f0 !important;-webkit-text-fill-color:#e2e8f0 !important;font-size:15px;">Symptom History</h3>
        </div>
        """, unsafe_allow_html=True)

        # Build DataFrame
        df = pd.DataFrame(logs)
        # Add missing columns if older logs don't have patient_id
        for col in ["patient_id","patient_name"]:
            if col not in df.columns:
                df[col] = ""
        df_display = df[["date","patient_id","patient_name","memory","confusion","mood","tasks","sleep","overall","notes"]].copy()
        df_display.columns = ["Date","Patient ID","Patient Name","Memory","Confusion","Mood","Tasks","Sleep","Overall","Notes"]

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
        )

        # Line chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
              <h3 style="margin:0;color:#e2e8f0 !important;-webkit-text-fill-color:#e2e8f0 !important;font-size:15px;">Trend Chart</h3>
        </div>
        """, unsafe_allow_html=True)

        chart_df = df[["date","memory","confusion","mood","tasks","overall"]].set_index("date")
        chart_df.columns = ["Memory Loss","Confusion","Mood Changes","Task Difficulty","Overall"]
        st.line_chart(chart_df, use_container_width=True)

        # ── AI Analysis ────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
              <div>
            <h3 style="margin:0;background:linear-gradient(135deg,#6366f1,#818cf8);-webkit-background-clip:text;
                       -webkit-text-fill-color:transparent;background-clip:text;font-size:16px;">
              AI Trend Analysis
            </h3>
            <p style="margin:3px 0 0;color:#cbd5e1;font-size:12px;">
              AI reviews all logs and flags patterns, risks, and next steps
            </p>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if len(logs) < 2:
            st.markdown("""
            <div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:12px;
                        padding:16px;text-align:center;color:#cbd5e1;font-size:13px;">
              Log at least <strong>2 days</strong> of symptoms to get AI trend analysis.
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("Analyse Trends with AI", use_container_width=True, key="analyse_trends"):
                with st.spinner("Analysing symptom patterns..."):
                    analysis = analyze_symptom_trends(logs)
                st.session_state.trend_analysis = analysis

            if "trend_analysis" in st.session_state and st.session_state.trend_analysis:
                analysis_text = st.session_state.trend_analysis

                # Determine risk level from analysis text for color
                risk_color = "#4ade80"
                risk_bg    = "#052e16"
                if "critical" in analysis_text.lower():
                    risk_color, risk_bg = "#f87171", "#1c0000"
                elif "high" in analysis_text.lower():
                    risk_color, risk_bg = "#fb923c", "#1c0800"
                elif "moderate" in analysis_text.lower():
                    risk_color, risk_bg = "#facc15", "#1c1400"

                # Format the analysis nicely
                formatted = analysis_text.replace("\n\n", "<br><br>").replace("\n", "<br>")

                st.markdown(f"""
                <div style="background:linear-gradient(135deg,{risk_color}0a,{risk_bg});
                            border:1.5px solid {risk_color}55;border-radius:14px;padding:20px;margin-top:8px;">
                  <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;">
                                  <div>
                      <div style="color:{risk_color};font-size:15px;font-weight:700;">AI Trend Report</div>
                      <div style="color:#cbd5e1;font-size:11px;margin-top:2px;">
                        Based on {len(logs)} logged entries · {logs[0]["date"]} → {logs[-1]["date"]}
                      </div>
                    </div>
                  </div>
                  <div style="color:#e2e8f0;font-size:13px;line-height:1.9;">{formatted}</div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#0d1526;border:2px dashed #1e3a5f;border-radius:16px;
                    padding:48px 24px;text-align:center;margin-top:8px;">
          
          <p style="font-size:15px;margin:0;color:#cbd5e1;">No symptom logs yet</p>
          <p style="font-size:12px;margin:8px 0 0;color:#94a3b8;">
            Use the sliders above to log today's symptoms and start tracking
          </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 3 — PRICING
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style="text-align:center;padding:4px 0 28px;">
      <h2 style="margin:0;background:linear-gradient(135deg,#6366f1,#818cf8);-webkit-background-clip:text;
                 -webkit-text-fill-color:transparent;background-clip:text;font-size:28px;letter-spacing:.02em;">
        Simple, Transparent Pricing
      </h2>
      <p style="margin:10px 0 0;color:#cbd5e1;font-size:13px;">
        Start free · Upgrade as you grow · Cancel anytime
      </p>
    </div>
    """, unsafe_allow_html=True)

    plans = [
        {
            "name": "Free", "emoji": "🆓", "price_inr": "₹0", "price_usd": "$0",
            "period": "forever", "color": "#94a3b8", "bg": "#0d1526", "border": "#1e293b",
            "badge": "",
            "features": [
                ("MRI Classifications", "3 / month"),
                ("Prescription Protocol", "✅"),
                ("AI Chatbot", "❌"),
                ("Email Reports", "❌"),
                ("Symptom Tracker", "❌"),
                ("AI Trend Analysis", "❌"),
                ("Patient Management", "❌"),
                ("Support", "Community"),
            ],
        },
        {
            "name": "Basic", "emoji": "🥈", "price_inr": "₹499", "price_usd": "$5.99",
            "period": "per month", "color": "#38bdf8", "bg": "#0a1e38", "border": "#1e3a5f",
            "badge": "",
            "features": [
                ("MRI Classifications", "30 / month"),
                ("Prescription Protocol", "✅"),
                ("AI Chatbot", "✅"),
                ("Email Reports", "✅"),
                ("Symptom Tracker", "1 patient"),
                ("AI Trend Analysis", "❌"),
                ("Patient Management", "❌"),
                ("Support", "Email"),
            ],
        },
        {
            "name": "Pro", "emoji": "🥇", "price_inr": "₹1,499", "price_usd": "$17.99",
            "period": "per month", "color": "#818cf8", "bg": "#1a0a44", "border": "#6366f1",
            "badge": "MOST POPULAR",
            "features": [
                ("MRI Classifications", "Unlimited"),
                ("Prescription Protocol", "✅"),
                ("AI Chatbot", "✅"),
                ("Email Reports", "✅"),
                ("Symptom Tracker", "10 patients"),
                ("AI Trend Analysis", "✅"),
                ("Patient Management", "✅"),
                ("Support", "Priority"),
            ],
        },
        {
            "name": "Hospital", "emoji": "🏥", "price_inr": "₹9,999", "price_usd": "$119.99",
            "period": "per month", "color": "#4ade80", "bg": "#052e16", "border": "#4ade80",
            "badge": "ENTERPRISE",
            "features": [
                ("MRI Classifications", "Unlimited"),
                ("Prescription Protocol", "✅"),
                ("AI Chatbot", "✅"),
                ("Email Reports", "✅"),
                ("Symptom Tracker", "Unlimited"),
                ("AI Trend Analysis", "✅"),
                ("Patient Management", "✅ + API Access"),
                ("Support", "24/7 Dedicated"),
            ],
        },
    ]

    cols = st.columns(4)
    for col, plan in zip(cols, plans):
        with col:
            badge_html = f"""
            <div style="background:{plan['color']};color:#000;font-size:9px;font-weight:700;
                        letter-spacing:.1em;padding:3px 10px;border-radius:20px;
                        text-align:center;margin-bottom:10px;display:inline-block;">
              {plan['badge']}
            </div>""" if plan['badge'] else "<div style='height:24px;margin-bottom:10px;'></div>"

            feature_rows = ""
            for fname, fval in plan['features']:
                val_color = "#4ade80" if fval == "✅" else "#f87171" if fval == "❌" else "#e2e8f0"
                feature_rows += f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:7px 0;border-bottom:1px solid #1e293b;">
                  <span style="font-size:11px;color:#94a3b8;">{fname}</span>
                  <span style="font-size:11px;color:{val_color};font-weight:600;">{fval}</span>
                </div>"""

            st.markdown(f"""
            <div style="background:{plan['bg']};border:1.5px solid {plan['border']};
                        border-radius:16px;padding:20px;height:100%;position:relative;">
              {badge_html}
              <div style="text-align:center;margin-bottom:16px;">
                <div style="font-size:32px;margin-bottom:6px;">{plan['emoji']}</div>
                <div style="color:{plan['color']};font-size:18px;font-weight:700;
                            font-family:'Playfair Display',serif;">{plan['name']}</div>
                <div style="margin-top:10px;">
                  <span style="color:{plan['color']};font-size:28px;font-weight:700;">{plan['price_inr']}</span>
                  <span style="color:#94a3b8;font-size:11px;"> / {plan['period']}</span>
                </div>
                <div style="color:#94a3b8;font-size:11px;margin-top:2px;">{plan['price_usd']} USD</div>
              </div>
              <div style="margin-top:14px;">{feature_rows}</div>
            </div>
            """, unsafe_allow_html=True)

            btn_label = "Get Started Free" if plan['name'] == "Free" else f"Choose {plan['name']}"
            btn_key   = f"btn_{plan['name'].lower()}"
            st.markdown("<div style='margin-top:12px;'>", unsafe_allow_html=True)
            st.button(btn_label, use_container_width=True, key=btn_key)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Revenue Projection ─────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;">
      
      <h3 style="margin:0;color:#e2e8f0;font-size:15px;">Revenue Projection</h3>
    </div>
    """, unsafe_allow_html=True)

    rev_cols = st.columns(3)
    scenarios = [
        {
            "label": "Conservative", "emoji": "🌱", "color": "#38bdf8",
            "bg": "#0a1e38", "border": "#1e3a5f",
            "users": "50 Basic · 10 Pro · 2 Hospital",
            "inr": "₹59,900 / mo", "usd": "~$720 / mo",
        },
        {
            "label": "Moderate", "emoji": "🚀", "color": "#818cf8",
            "bg": "#1a0a44", "border": "#6366f1",
            "users": "200 Basic · 50 Pro · 10 Hospital",
            "inr": "₹2,74,800 / mo", "usd": "~$3,300 / mo",
        },
        {
            "label": "Optimistic", "emoji": "🌟", "color": "#4ade80",
            "bg": "#052e16", "border": "#4ade80",
            "users": "500 Basic · 200 Pro · 30 Hospital",
            "inr": "₹8,48,300 / mo", "usd": "~$10,200 / mo",
        },
    ]

    for col, sc in zip(rev_cols, scenarios):
        with col:
            st.markdown(f"""
            <div style="background:{sc['bg']};border:1.5px solid {sc['border']};
                        border-radius:14px;padding:18px;text-align:center;">
              <div style="font-size:28px;margin-bottom:8px;">{sc['emoji']}</div>
              <div style="color:{sc['color']};font-size:15px;font-weight:700;margin-bottom:6px;">
                {sc['label']}
              </div>
              <div style="color:#94a3b8;font-size:11px;margin-bottom:12px;">{sc['users']}</div>
              <div style="color:{sc['color']};font-size:22px;font-weight:700;">{sc['inr']}</div>
              <div style="color:#94a3b8;font-size:12px;margin-top:4px;">{sc['usd']}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Running Cost ───────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;">
      
      <h3 style="margin:0;color:#e2e8f0;font-size:15px;">Estimated Running Costs</h3>
    </div>
    """, unsafe_allow_html=True)

    cost_rows = ""
    costs = [
        ("Payment Gateway (Razorpay/Stripe)", "2% per transaction"),
        ("Cloud Hosting (AWS / Streamlit paid)", "₹2,000 – ₹5,000 / mo"),
        ("Database (Supabase / PostgreSQL)",    "🆓 Free tier"),
        ("User Authentication (Firebase)",      "🆓 Free tier"),
        ("Groq AI API",                         "🆓 Free tier"),
        ("Custom Domain",                       "~₹800 / year"),
        ("Total Estimated Cost",                "₹3,000 – ₹8,000 / mo"),
    ]
    for item, val in costs:
        is_total = "Total" in item
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:10px 16px;background:{'linear-gradient(135deg,#6366f10d,#1a0a44)' if is_total else '#0d1526'};
                    border:1px solid {'#6366f155' if is_total else '#1e293b'};
                    border-radius:8px;margin-bottom:6px;">
          <span style="font-size:13px;color:{'#818cf8' if is_total else '#cbd5e1'};
                       font-weight:{'700' if is_total else '400'};">{item}</span>
          <span style="font-size:13px;color:{'#4ade80' if is_total else '#e2e8f0'};
                       font-weight:700;">{val}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Payment note ───────────────────────────────────────────
    st.markdown("""
    <div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:12px;
                padding:14px 18px;margin-top:16px;text-align:center;">
      <p style="margin:0;color:#cbd5e1;font-size:12px;line-height:1.8;">
        Payments via <strong style="color:#38bdf8;">Razorpay</strong> (India) &
        <strong style="color:#818cf8;">Stripe</strong> (International) ·
        Secure & encrypted · Cancel anytime · No hidden fees
      </p>
    </div>
    """, unsafe_allow_html=True)
