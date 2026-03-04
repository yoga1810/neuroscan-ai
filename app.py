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
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"], .css-18e3th9, .css-1d391kg {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1526 !important;
    color: #ffffff !important;
    font-weight: 600;
}
.main, .block-container { background-color: transparent !important; }

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    font-weight: 700 !important;
    color: #000 !important;
}
.block-container { padding: 2rem 2.5rem; max-width: 1150px; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }

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
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: .03em !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .85 !important; }

.stTextInput > div > div > input {
    background: #0d1526 !important;
    border: 1px solid #1e293b !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
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
            {"name": "Omega-3 (Fish Oil)",  "dose": "500 mg",  "times": ["8:00 AM"],            "purpose": "Supports brain cell membrane integrity & cognitive longevity",    "type": "nutrient"},
            {"name": "Beta-Carotene",       "dose": "3 mg",    "times": ["8:00 AM"],            "purpose": "Antioxidant precursor to Vitamin A; protects neurons",            "type": "nutrient"},
            {"name": "Vitamin E",           "dose": "200 IU",  "times": ["9:00 AM"],            "purpose": "Neuroprotective antioxidant; protects cell membranes",            "type": "nutrient"},
            {"name": "Choline",             "dose": "250 mg",  "times": ["8:00 AM"],            "purpose": "Precursor to acetylcholine; supports memory & learning",          "type": "nutrient"},
            {"name": "Vitamin B12",         "dose": "500 mcg", "times": ["8:00 AM"],            "purpose": "Reduces homocysteine; supports myelin sheath & nerve conduction",  "type": "nutrient"},
            {"name": "Curcumin (Turmeric)", "dose": "250 mg",  "times": ["8:00 AM"],            "purpose": "Mild anti-inflammatory; early amyloid plaque prevention",         "type": "nutrient"},
        ],
    },
    "VeryMildDemented": {
        "label": "Very Mild Demented", "emoji": "🟡", "color": "#facc15", "bg": "#1c1400",
        "description": "Very early-stage cognitive changes detected. Lifestyle modifications and increased nutritional support can significantly slow progression.",
        "urgency": "Early Intervention Recommended",
        "supplements": [
            {"name": "Omega-3 (Fish Oil)",  "dose": "1000 mg", "times": ["8:00 AM", "8:00 PM"], "purpose": "Reduces neuroinflammation; supports synaptic plasticity",          "type": "nutrient"},
            {"name": "Vitamin E",           "dose": "400 IU",  "times": ["9:00 AM"],             "purpose": "Slows oxidative damage to neurons in early-stage decline",        "type": "nutrient"},
            {"name": "Choline",             "dose": "375 mg",  "times": ["8:00 AM", "1:00 PM"], "purpose": "Boosts acetylcholine production for memory preservation",         "type": "nutrient"},
            {"name": "Vitamin B12",         "dose": "750 mcg", "times": ["8:00 AM"],             "purpose": "Slows brain atrophy linked to B12 deficiency in early decline",  "type": "nutrient"},
            {"name": "Curcumin (Turmeric)", "dose": "500 mg",  "times": ["8:00 AM", "6:00 PM"], "purpose": "Anti-inflammatory; begins inhibiting amyloid-beta aggregation",    "type": "nutrient"},
        ],
    },
    "MildDemented": {
        "label": "Mild Demented", "emoji": "🟠", "color": "#fb923c", "bg": "#1c0800",
        "description": "Mild cognitive decline detected. Intensified nutritional protocol recommended alongside neurologist consultation.",
        "urgency": "Medical Attention Required",
        "supplements": [
            {"name": "Omega-3 (Fish Oil)",  "dose": "2000 mg",  "times": ["8:00 AM", "1:00 PM", "8:00 PM"], "purpose": "DHA/EPA support for slowing grey matter loss & inflammation",      "type": "nutrient"},
            {"name": "Vitamin E",           "dose": "800 IU",   "times": ["9:00 AM", "9:00 PM"],            "purpose": "High-dose neuroprotection against free radical damage",            "type": "nutrient"},
            {"name": "Choline",             "dose": "500 mg",   "times": ["8:00 AM", "1:00 PM", "6:00 PM"], "purpose": "Supports declining cholinergic neurons; aids recall",              "type": "nutrient"},
            {"name": "Vitamin B12",         "dose": "1000 mcg", "times": ["8:00 AM", "1:00 PM"],            "purpose": "Repairs myelin damage; counters neurodegeneration from deficiency", "type": "nutrient"},
            {"name": "Curcumin (Turmeric)", "dose": "750 mg",   "times": ["8:00 AM", "1:00 PM", "6:00 PM"], "purpose": "Actively reduces amyloid plaques & tau tangles",                  "type": "nutrient"},
        ],
    },
    "ModerateDemented": {
        "label": "Moderate Demented", "emoji": "🔴", "color": "#f87171", "bg": "#1c0000",
        "description": "Significant cognitive impairment detected. Maximum nutritional support protocol initiated. Immediate specialist consultation required.",
        "urgency": "URGENT — See Neurologist Immediately",
        "supplements": [
            {"name": "Omega-3 (Fish Oil)",  "dose": "3000 mg",  "times": ["8:00 AM", "1:00 PM", "8:00 PM"], "purpose": "Maximum DHA dose to slow advanced neuronal loss & inflammation",  "type": "nutrient"},
            {"name": "Beta-Carotene",       "dose": "15 mg",    "times": ["8:00 AM", "1:00 PM", "6:00 PM"], "purpose": "Maximum antioxidant coverage for severe oxidative brain damage",  "type": "nutrient"},
            {"name": "Vitamin E",           "dose": "1000 IU",  "times": ["9:00 AM", "3:00 PM", "9:00 PM"], "purpose": "Peak neuroprotective dose; slows advanced neurodegeneration",     "type": "nutrient"},
            {"name": "Choline",             "dose": "650 mg",   "times": ["8:00 AM", "1:00 PM", "6:00 PM"], "purpose": "Critical support for severely depleted cholinergic pathways",     "type": "nutrient"},
            {"name": "Vitamin B12",         "dose": "1500 mcg", "times": ["8:00 AM", "1:00 PM", "8:00 PM"], "purpose": "Maximum B12 support for severely depleted neurological pathways", "type": "nutrient"},
            {"name": "Curcumin (Turmeric)", "dose": "1000 mg",  "times": ["8:00 AM", "1:00 PM", "6:00 PM"], "purpose": "High-dose plaque & tangle inhibition in advanced Alzheimer's",    "type": "nutrient"},
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
        icon = "💊" if item["type"] == "medication" else "🌿"
        sched = "<br>".join([f"🕐 {t}" for t in item["times"]])
        rows += f"""<tr>
          <td style="padding:10px 14px;border-bottom:1px solid #1e293b;">
            <strong style="color:{tc};font-size:13px;">{icon} {item['name']}</strong>
            <div style="font-size:11px;color:#64748b;margin-top:2px;">{item['purpose']}</div>
          </td>
          <td style="padding:10px 14px;border-bottom:1px solid #1e293b;color:#e2e8f0;font-weight:600;">{item['dose']}</td>
          <td style="padding:10px 14px;border-bottom:1px solid #1e293b;color:{c};font-size:12px;line-height:1.8;">{sched}</td>
        </tr>"""

    bars = ""
    for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        bc = STAGE_DATA[cls]["color"]
        bars += f"""<div style="margin-bottom:8px;">
          <div style="display:flex;justify-content:space-between;font-size:12px;">
            <span style="color:#94a3b8;">{STAGE_DATA[cls]['label']}</span>
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
    <div style="font-size:32px;">🧬</div>
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
        <div style="color:#94a3b8;font-size:12px;">
          Confidence: <strong style="color:{c};">{conf:.1f}%</strong> &nbsp;·&nbsp; {data['urgency']}
        </div>
      </div>
    </div>
    <p style="color:#94a3b8;font-size:13px;margin:0;">{data['description']}</p>
    {patient_line}
    <p style="color:#475569;font-size:11px;margin:6px 0 0;">Report Date: {date_str}</p>
  </div>
  <div style="background:#0f172a;border:1px solid #1e293b;border-radius:14px;padding:16px;margin-bottom:16px;">
    <h3 style="margin:0 0 12px;color:#e2e8f0;font-size:12px;text-transform:uppercase;letter-spacing:.1em;">
      Model Confidence Breakdown
    </h3>{bars}
  </div>
  <div style="background:#0f172a;border:1px solid #1e293b;border-radius:14px;overflow:hidden;margin-bottom:16px;">
    <div style="padding:10px 14px;background:#0a0f1e;color:{c};font-size:12px;font-weight:600;
                text-transform:uppercase;border-bottom:1px solid {c}33;">💊 Prescribed Protocol</div>
    <table style="width:100%;border-collapse:collapse;">
      <thead><tr style="background:#080d16;">
        <th style="padding:7px 12px;text-align:left;font-size:10px;color:#475569;
                   text-transform:uppercase;border-bottom:1px solid #1e293b;">Name</th>
        <th style="padding:7px 12px;text-align:left;font-size:10px;color:#475569;
                   text-transform:uppercase;border-bottom:1px solid #1e293b;">Dose</th>
        <th style="padding:7px 12px;text-align:left;font-size:10px;color:#475569;
                   text-transform:uppercase;border-bottom:1px solid #1e293b;">Schedule</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  <div style="background:#0f172a;border:1px solid #1e3a5f;border-radius:12px;padding:14px;margin-bottom:14px;">
    <h3 style="margin:0 0 8px;color:#38bdf8;font-size:12px;text-transform:uppercase;">💡 Daily Reminder Tips</h3>
    <ul style="margin:0;padding-left:16px;color:#94a3b8;font-size:12px;line-height:2;">
      <li>Set phone alarms for each medication time above.</li>
      <li>Take supplements with food for better absorption.</li>
      <li>Keep a daily medication log to track compliance.</li>
      <li>Follow up with your neurologist every 3 months.</li>
      <li>Combine medication with physical exercise and cognitive activities.</li>
    </ul>
  </div>
  <div style="background:rgba(239,68,68,.06);border:1px solid #7f1d1d;border-radius:10px;padding:10px 14px;">
    <p style="margin:0;color:#f87171;font-size:11px;line-height:1.7;">
      ⚕ <strong>Disclaimer:</strong> AI-generated guidance only.
      Always consult a licensed neurologist before starting any medication or supplement.
    </p>
  </div>
  <p style="text-align:center;color:#334155;font-size:10px;margin-top:16px;">
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
    """Call Gemini API for the NeuroScan chatbot."""
    # Try Streamlit secrets first, then sidebar input, then env variable
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "")
        except:
            pass
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return "⚠️ Please enter your Gemini API key in the sidebar to use the chatbot."

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

    # Build Gemini-format contents from message history
    gemini_contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": gemini_contents,
        "generationConfig": {"maxOutputTokens": 1000, "temperature": 0.7},
    }

    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        data = resp.json()
        if resp.status_code == 200:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return f"⚠️ API error {resp.status_code}: {data.get('error', {}).get('message', 'Unknown error')}"
    except Exception as e:
        return f"⚠️ Connection error: {str(e)}"


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 24px;">
      <div style="font-size:40px;margin-bottom:6px;">🧬</div>
      <h2 style="margin:0;color:#e0e7ff !important;font-family:'DM Serif Display',serif;font-size:20px;">NeuroScan AI</h2>
      <p style="margin:4px 0 0;color:#6366f1 !important;font-size:10px;letter-spacing:.15em;text-transform:uppercase;">
        Alzheimer's Detection
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**ℹ️ About**")
    st.markdown("""
    <p style="font-size:12px;line-height:1.7;color:#64748b !important;">
    This app uses a deep learning model based on <strong style="color:#94a3b8 !important;">EfficientNet-B0</strong>
    to classify MRI scans into 4 Alzheimer's severity stages and generates
    a personalised prescription protocol with email alerts.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📊 Classification Stages**")
    for key, data in STAGE_DATA.items():
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
          <div style="width:8px;height:8px;border-radius:50%;background:{data['color']};flex-shrink:0;"></div>
          <div>
            <div style="font-size:12px;color:#e2e8f0 !important;font-weight:600;">{data['label']}</div>
            <div style="font-size:10px;color:#475569 !important;">{data['urgency']}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="background:rgba(239,68,68,.06);border:1px solid #7f1d1d;border-radius:8px;padding:10px 12px;">
      <p style="margin:0;color:#f87171 !important;font-size:11px;line-height:1.6;">
        ⚕ For educational purposes only. Not a substitute for professional medical advice.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🤖 AI Chatbot**")
    st.markdown("""
    <p style="font-size:11px;color:#4ade80 !important;margin-top:4px;">✅ AI Assistant is active</p>
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
  <p style="margin:8px 0 0;color:#64748b;font-size:14px;">
    Upload a brain MRI scan · Get instant classification · Receive personalised care protocol
  </p>
""", unsafe_allow_html=True)


# ── Patient info ───────────────────────────────────────────────
pc1, pc2 = st.columns([3, 1])
with pc1:
    patient_name = st.text_input("Patient Name", placeholder="Full name (optional)")
with pc2:
    patient_age = st.text_input("Age", placeholder="e.g. 68")

st.markdown("<br>", unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📤 Upload Brain MRI Scan",
    type=["jpg", "jpeg", "png"],
    help="Upload an axial T1-weighted MRI scan. JPG or PNG format.",
)

# ══════════════════════════════════════════════════════════════
#  PREDICTION + RESULTS
# ══════════════════════════════════════════════════════════════
if uploaded:
    image = Image.open(uploaded)

    # ── MRI Validation ─────────────────────────────────────────
    with st.spinner("🔎 Validating image..."):
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
                <span style="font-size:28px;">🚫</span>
                <div>
                  <div style="color:#f87171;font-size:17px;font-weight:700;">Invalid Image Detected</div>
                  <div style="color:#94a3b8;font-size:12px;margin-top:2px;">This does not appear to be a brain MRI scan</div>
                </div>
              </div>
              <p style="color:#94a3b8;font-size:13px;margin:0 0 14px;line-height:1.7;">
                The uploaded image failed one or more MRI validation checks:
              </p>
              <div style="background:#1c0000;border:1px solid #7f1d1d;border-radius:8px;padding:12px;margin-bottom:14px;">
                <p style="color:#fca5a5;font-size:12px;margin:0;line-height:1.8;">⚠️ {reason}</p>
              </div>
              <div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:8px;padding:12px;">
                <p style="color:#38bdf8;font-size:12px;font-weight:600;margin:0 0 6px;">
                  ✅ What a valid MRI scan looks like:
                </p>
                <ul style="color:#64748b;font-size:12px;margin:0;padding-left:16px;line-height:2;">
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

    with st.spinner("🔍 Analyzing MRI scan..."):
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
            <span style="font-size:20px;">⚠️</span>
            <div>
              <strong style="color:#fb923c;font-size:13px;">Low Confidence Result ({conf:.1f}%)</strong>
              <p style="color:#94a3b8;font-size:12px;margin:4px 0 0;line-height:1.6;">
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
                          font-family:'DM Serif Display',serif;">{data['label']}</div>
              <div style="color:#94a3b8;font-size:12px;margin-top:2px;">
                Confidence: <strong style="color:{data['color']};">{conf:.1f}%</strong>
                &nbsp;·&nbsp; <span style="color:{data['color']};">{data['urgency']}</span>
              </div>
            </div>
          </div>
          <p style="color:#94a3b8;font-size:13px;margin:0;line-height:1.7;">{data['description']}</p>
          {patient_line}
          <p style="color:#334155;font-size:11px;margin:8px 0 0;">Analyzed: {date_str}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding-bottom:12px;
                    border-bottom:1px solid {data['color']}33;margin-bottom:14px;">
          <span style="font-size:16px;">💊</span>
          <h3 style="margin:0;color:{data['color']} !important;font-size:14px;text-transform:uppercase;
                     letter-spacing:.1em;font-family:'DM Sans',sans-serif;-webkit-text-fill-color:{data['color']};">Prescribed Protocol</h3>
        </div>
        """, unsafe_allow_html=True)

        for item in data["supplements"]:
            tc       = data["color"] if item["type"] == "medication" else "#a78bfa"
            lb       = data["color"] if item["type"] == "medication" else "#6366f1"
            icon     = "💊" if item["type"] == "medication" else "🌿"
            sched    = " &nbsp;|&nbsp; ".join([f"🕐 {t}" for t in item["times"]])
            badge_bg = "#0d1e38" if item["type"] == "medication" else "#1a0a44"

            st.markdown(f"""
            <div style="background:#0d1526;border:1px solid #1e293b;border-left:3px solid {lb};
                        border-radius:10px;padding:12px 16px;margin-bottom:9px;">
              <div style="display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start;">
                <div style="flex:1;min-width:160px;">
                  <div style="display:flex;align-items:center;gap:7px;margin-bottom:4px;">
                    <span style="font-size:14px;">{icon}</span>
                    <strong style="color:{tc};font-size:13px;">{item['name']}</strong>
                    <span style="background:{badge_bg};color:{tc};border:1px solid {tc}33;
                                 border-radius:10px;padding:1px 8px;font-size:10px;
                                 text-transform:uppercase;letter-spacing:.07em;">{item['type']}</span>
                  </div>
                  <div style="font-size:11px;color:#475569;line-height:1.5;">{item['purpose']}</div>
                </div>
                <div style="display:flex;gap:22px;flex-shrink:0;align-items:center;">
                  <div>
                    <div style="font-size:10px;color:#334155;text-transform:uppercase;letter-spacing:.1em;">Dose</div>
                    <div style="font-size:13px;color:#e2e8f0;font-weight:600;margin-top:2px;">{item['dose']}</div>
                  </div>
                  <div>
                    <div style="font-size:10px;color:#334155;text-transform:uppercase;letter-spacing:.1em;">Schedule</div>
                    <div style="font-size:12px;color:{data['color']};margin-top:3px;line-height:1.9;">{sched}</div>
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Email Section ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<hr style="border-color:#1e293b;margin:8px 0 24px;">""", unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">
      <span style="font-size:22px;">📧</span>
      <div>
        <h3 style="margin:0;background:linear-gradient(135deg,#6366f1,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;font-size:17px;font-family:'DM Serif Display',serif;">
          Send Prescription Email Alert
        </h3>
        <p style="margin:3px 0 0;color:#475569;font-size:12px;">
          Email the full care protocol with dosing schedule to patient or caregiver
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        recipient = st.text_input("📬 Recipient Email",    placeholder="patient@example.com")
    with ec2:
        sender_em = st.text_input("📤 Your Gmail Address", placeholder="youremail@gmail.com")
    with ec3:
        sender_pw = st.text_input("🔑 Gmail App Password", placeholder="xxxx xxxx xxxx xxxx", type="password")

    if st.button("📨 Send Prescription Email", use_container_width=True):
        if not recipient or "@" not in recipient:
            st.error("⚠️ Please enter a valid recipient email address.")
        elif not sender_em or not sender_pw:
            st.error("⚠️ Please enter your Gmail address and App Password.")
        else:
            with st.spinner("📤 Sending email..."):
                html    = build_email_html(patient_name, patient_age, label, conf, all_probs)
                subject = f"🧠 NeuroScan AI — {data['label']} Prescription | {datetime.now().strftime('%d %b %Y')}"
                ok, msg = send_email(sender_em, sender_pw, recipient, subject, html)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    st.markdown("""
    <div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:12px;
                padding:14px 18px;margin-top:12px;">
      <p style="margin:0 0 8px;color:#38bdf8;font-size:12px;font-weight:600;">
        ℹ️ How to get a Gmail App Password (required)
      </p>
      <ol style="margin:0;padding-left:18px;color:#475569;font-size:12px;line-height:2.1;">
        <li>Go to <strong style="color:#94a3b8;">myaccount.google.com</strong> → Security</li>
        <li>Enable <strong style="color:#94a3b8;">2-Step Verification</strong></li>
        <li>Search <strong style="color:#94a3b8;">App Passwords</strong> → Mail + Other → Name it "NeuroScan"</li>
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
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">
      <span style="font-size:22px;">🤖</span>
      <div>
        <h3 style="margin:0;background:linear-gradient(135deg,#6366f1,#818cf8);-webkit-background-clip:text;
                   -webkit-text-fill-color:transparent;background-clip:text;font-size:17px;
                   font-family:'DM Serif Display',serif;">NeuroScan AI Assistant</h3>
        <p style="margin:3px 0 0;color:#475569;font-size:12px;">
          Ask anything about the diagnosis, nutrients, caregiving, or lifestyle advice
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Suggested questions
    st.markdown(f"""
    <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;">
      <div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:20px;
                  padding:5px 12px;font-size:11px;color:#38bdf8;cursor:pointer;">
        💊 What do these nutrients do?
      </div>
      <div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:20px;
                  padding:5px 12px;font-size:11px;color:#38bdf8;cursor:pointer;">
        🍎 What foods should the patient eat?
      </div>
      <div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:20px;
                  padding:5px 12px;font-size:11px;color:#38bdf8;cursor:pointer;">
        🧠 What does this stage mean day-to-day?
      </div>
      <div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:20px;
                  padding:5px 12px;font-size:11px;color:#38bdf8;cursor:pointer;">
        👨‍👩‍👧 How can I help as a caregiver?
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
                        padding:18px;text-align:center;color:#475569;">
              <div style="font-size:32px;margin-bottom:8px;">🧬</div>
              <p style="font-size:13px;margin:0;color:#64748b;">
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
                                     display:block;margin-bottom:4px;">🤖 NeuroScan AI</span>
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
        with st.spinner("🤖 Thinking..."):
            reply = ask_neuroscan_ai(st.session_state.chat_history, stage_context, gemini_api_key)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()


else:
    # ── Empty state ────────────────────────────────────────────
    st.markdown("""
    <div style="background:#0d1526;border:2px dashed #1e3a5f;border-radius:16px;
                padding:64px 24px;text-align:center;color:#334155;margin-top:16px;">
      <div style="font-size:52px;margin-bottom:14px;">🧠</div>
      <p style="font-size:16px;letter-spacing:.04em;margin:0;color:#475569;">
        Upload an MRI scan above to begin analysis
      </p>
      <p style="font-size:12px;margin:8px 0 0;color:#334155;">
        Supported formats: JPG · JPEG · PNG
      </p>
    </div>
    """, unsafe_allow_html=True)
