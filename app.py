import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
import os

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

/* AFTER */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080c14;
    color: #e2e8f0;   ← change this to bright white
}
.main { background-color: #080c14; }
.block-container { padding: 2rem 2.5rem; max-width: 1150px; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0f172a 100%);
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }

div[data-testid="stFileUploader"] {
    background: #0d1526;
    border: 2px dashed #1e3a5f;
    border-radius: 16px;
    padding: 12px;
    transition: border-color .2s;
}
div[data-testid="stFileUploader"]:hover { border-color: #6366f1; }

.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6d28d9) !important;
    border: 1px solid #312e81 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: .03em !important;
}
/* Remove white background from main content area */
.stApp, .main, .block-container, 
div[data-testid="stAppViewContainer"] > section,
div[data-testid="stAppViewContainer"] {
    background-color: #080c14 !important;
    background: #080c14 !important;
}
/* Fix placeholder text color */
.stTextInput > div > div > input::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}

/* Fix typed text color inside inputs */
.stTextInput > div > div > input {
    background: #0d1526 !important;
    border: 1px solid #1e293b !important;
    color: #f1f5f9 !important;
    border-radius: 10px !important;
}

/* Fix email/password fields in the email section too */
input[type="text"], input[type="password"] {
    color: #f1f5f9 !important;
}

input[type="text"]::placeholder, input[type="password"]::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}

/* Remove white card backgrounds */
div[data-testid="stVerticalBlock"] > div,
div[data-testid="element-container"] {
    background: transparent !important;
}
/* Fix all label text to be visible */
.stTextInput label, 
div[data-testid="stFileUploader"] label,
.stFileUploader label,
p, label, span {
    color: #e2e8f0 !important;
}

/* Fix the file uploader text */
div[data-testid="stFileUploader"] {
    color: #e2e8f0 !important;
}

div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] p,
div[data-testid="stFileUploader"] small {
    color: #94a3b8 !important;
}

/* Fix input labels */
.stTextInput > label {
    color: #e2e8f0 !important;
    font-weight: 500 !important;
}
.stButton > button:hover {
    opacity: 0.9;
}
.stButton > button:hover { opacity: .85 !important; }

.stTextInput > div > div > input {
    background: #0d1526 !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus { border-color: #6366f1 !important; }

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
            {"name": "Omega-3 (Fish Oil)",   "dose": "500 mg",  "times": ["8:00 AM"],            "purpose": "Supports brain cell membrane integrity & cognitive longevity",    "type": "nutrient"},
            {"name": "Beta-Carotene",        "dose": "3 mg",    "times": ["8:00 AM"],            "purpose": "Antioxidant precursor to Vitamin A; protects neurons",            "type": "nutrient"},
            {"name": "Vitamin E",            "dose": "200 IU",  "times": ["9:00 AM"],            "purpose": "Neuroprotective antioxidant; protects cell membranes",            "type": "nutrient"},
            {"name": "Choline",              "dose": "250 mg",  "times": ["8:00 AM"],            "purpose": "Precursor to acetylcholine; supports memory & learning",          "type": "nutrient"},
            {"name": "Lycopene",             "dose": "5 mg",    "times": ["1:00 PM"],            "purpose": "Carotenoid antioxidant; reduces oxidative stress in brain tissue", "type": "nutrient"},
            {"name": "Vitamin B12",          "dose": "500 mcg", "times": ["8:00 AM"],            "purpose": "Reduces homocysteine; supports myelin sheath & nerve conduction",  "type": "nutrient"},
            {"name": "Curcumin (Turmeric)",  "dose": "250 mg",  "times": ["8:00 AM"],            "purpose": "Mild anti-inflammatory; early amyloid plaque prevention",         "type": "non-nutrient"},
            {"name": "Resveratrol",          "dose": "100 mg",  "times": ["1:00 PM"],            "purpose": "Polyphenol; activates neuroprotective SIRT1 pathways",            "type": "non-nutrient"},
            {"name": "Ginkgo Biloba",        "dose": "60 mg",   "times": ["8:00 AM"],            "purpose": "Improves cerebral blood flow & oxygen delivery to brain",         "type": "non-nutrient"},
        ],
    },
    "VeryMildDemented": {
        "label": "Very Mild Demented", "emoji": "🟡", "color": "#facc15", "bg": "#1c1400",
        "description": "Very early-stage cognitive changes detected. Lifestyle modifications and increased nutritional support can significantly slow progression.",
        "urgency": "Early Intervention Recommended",
        "supplements": [
            {"name": "Omega-3 (Fish Oil)",   "dose": "1000 mg", "times": ["8:00 AM", "8:00 PM"],            "purpose": "Reduces neuroinflammation; supports synaptic plasticity",         "type": "nutrient"},
            {"name": "Beta-Carotene",        "dose": "6 mg",    "times": ["8:00 AM"],                       "purpose": "Antioxidant defense against early neurodegeneration",             "type": "nutrient"},
            {"name": "Vitamin E",            "dose": "400 IU",  "times": ["9:00 AM"],                       "purpose": "Slows oxidative damage to neurons in early-stage decline",        "type": "nutrient"},
            {"name": "Choline",              "dose": "375 mg",  "times": ["8:00 AM", "1:00 PM"],            "purpose": "Boosts acetylcholine production for memory preservation",         "type": "nutrient"},
            {"name": "Lycopene",             "dose": "10 mg",   "times": ["1:00 PM"],                       "purpose": "Reduces lipid peroxidation in early cognitive decline",           "type": "nutrient"},
            {"name": "Vitamin B12",          "dose": "750 mcg", "times": ["8:00 AM"],                       "purpose": "Slows brain atrophy linked to B12 deficiency in early decline",   "type": "nutrient"},
            {"name": "Curcumin (Turmeric)",  "dose": "500 mg",  "times": ["8:00 AM", "6:00 PM"],            "purpose": "Anti-inflammatory; begins inhibiting amyloid-beta aggregation",   "type": "non-nutrient"},
        
        ],
    },
    "MildDemented": {
        "label": "Mild Demented", "emoji": "🟠", "color": "#fb923c", "bg": "#1c0800",
        "description": "Mild cognitive decline detected. Intensified nutritional protocol recommended alongside neurologist consultation.",
        "urgency": "Medical Attention Required",
        "supplements": [
            {"name": "Omega-3 (Fish Oil)",   "dose": "2000 mg",  "times": ["8:00 AM", "1:00 PM", "8:00 PM"], "purpose": "DHA/EPA support for slowing grey matter loss & inflammation",     "type": "nutrient"},
            {"name": "Beta-Carotene",        "dose": "10 mg",    "times": ["8:00 AM", "6:00 PM"],            "purpose": "Elevated antioxidant load to counter accelerating neuronal damage","type": "nutrient"},
            {"name": "Vitamin E",            "dose": "800 IU",   "times": ["9:00 AM", "9:00 PM"],            "purpose": "High-dose neuroprotection against free radical damage",           "type": "nutrient"},
            {"name": "Choline",              "dose": "500 mg",   "times": ["8:00 AM", "1:00 PM", "6:00 PM"], "purpose": "Supports declining cholinergic neurons; aids recall",             "type": "nutrient"},
            {"name": "Lycopene",             "dose": "15 mg",    "times": ["1:00 PM", "7:00 PM"],            "purpose": "Combats mitochondrial oxidative stress in mild dementia",         "type": "nutrient"},
            {"name": "Vitamin B12",          "dose": "1000 mcg", "times": ["8:00 AM", "1:00 PM"],            "purpose": "Repairs myelin damage; counters neurodegeneration from deficiency","type": "nutrient"},
            {"name": "Curcumin (Turmeric)",  "dose": "750 mg",   "times": ["8:00 AM", "1:00 PM", "6:00 PM"], "purpose": "Actively reduces amyloid plaques & tau tangles",                 "type": "non-nutrient"},
           
        ],
    },
    "ModerateDemented": {
        "label": "Moderate Demented", "emoji": "🔴", "color": "#f87171", "bg": "#1c0000",
        "description": "Significant cognitive impairment detected. Maximum nutritional support protocol initiated. Immediate specialist consultation required.",
        "urgency": "URGENT — See Neurologist Immediately",
        "supplements": [
            {"name": "Omega-3 (Fish Oil)",   "dose": "3000 mg",  "times": ["8:00 AM", "1:00 PM", "8:00 PM"], "purpose": "Maximum DHA dose to slow advanced neuronal loss & inflammation",  "type": "nutrient"},
            {"name": "Beta-Carotene",        "dose": "15 mg",    "times": ["8:00 AM", "1:00 PM", "6:00 PM"], "purpose": "Maximum antioxidant coverage for severe oxidative brain damage",  "type": "nutrient"},
            {"name": "Vitamin E",            "dose": "1000 IU",  "times": ["9:00 AM", "3:00 PM", "9:00 PM"], "purpose": "Peak neuroprotective dose; slows advanced neurodegeneration",     "type": "nutrient"},
            {"name": "Choline",              "dose": "650 mg",   "times": ["8:00 AM", "1:00 PM", "6:00 PM"], "purpose": "Critical support for severely depleted cholinergic pathways",     "type": "nutrient"},
            {"name": "Lycopene",             "dose": "20 mg",    "times": ["8:00 AM", "1:00 PM", "7:00 PM"], "purpose": "Maximum carotenoid protection against advanced brain oxidation",  "type": "nutrient"},
            {"name": "Vitamin B12",          "dose": "1500 mcg", "times": ["8:00 AM", "1:00 PM", "8:00 PM"], "purpose": "Maximum B12 support for severely depleted neurological pathways", "type": "nutrient"},
            {"name": "Curcumin (Turmeric)",  "dose": "1000 mg",  "times": ["8:00 AM", "1:00 PM", "6:00 PM"], "purpose": "High-dose plaque & tangle inhibition in advanced Alzheimer's",    "type": "non-nutrient"},
            
    },
}

# ── Model loader ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Looks for model in same directory as app.py
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
    model  = load_model()
    image  = Image.open(uploaded)
    arr    = preprocess(image)

    with st.spinner("🔍 Analyzing MRI scan..."):
        probs = model.predict(arr, verbose=0)[0]

    idx        = int(np.argmax(probs))
    label      = LABEL_MAP[idx]
    conf       = float(probs[idx]) * 100
    data       = STAGE_DATA[label]
    all_probs  = {LABEL_MAP[i]: float(probs[i]) * 100 for i in range(4)}

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1, 2])

    # ── Left: image + confidence chart ────────────────────────
    with left:
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

        st.markdown("**Confidence Breakdown**")
        for k in LABEL_MAP.values():
            prob     = all_probs[k]
            bc       = STAGE_DATA[k]["color"]
            is_pred  = k == label
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

        # Diagnosis card
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

        # Prescription header
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding-bottom:12px;
                    border-bottom:1px solid {data['color']}33;margin-bottom:14px;">
          <span style="font-size:16px;">💊</span>
          <h3 style="margin:0;color:{data['color']};font-size:14px;text-transform:uppercase;
                     letter-spacing:.1em;font-family:'DM Sans',sans-serif;">Prescribed Protocol</h3>
          <span style="margin-left:auto;font-size:11px;color:#475569;font-style:italic;">
            Consult physician before starting
          </span>
        </div>
        """, unsafe_allow_html=True)

        # Prescription items
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

    # ── Disclaimer ─────────────────────────────────────────────
    st.markdown("""
    <div style="background:rgba(239,68,68,.05);border:1px solid #7f1d1d;border-radius:10px;
                padding:12px 18px;margin-top:8px;">
      <p style="margin:0;color:#f87171;font-size:12px;line-height:1.7;">
        ⚕ <strong>Medical Disclaimer:</strong> All recommendations are AI-generated for informational
        purposes only. They do not constitute medical advice. Always consult a qualified neurologist
        or healthcare provider before starting, stopping, or modifying any medication or supplement.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Email Section ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<hr style="border-color:#1e293b;margin:8px 0 24px;">""", unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">
      <span style="font-size:22px;">📧</span>
      <div>
        <h3 style="margin:0;color:#e2e8f0;font-size:17px;font-family:'DM Serif Display',serif;">
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
