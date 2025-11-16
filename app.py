import os
import io
import fitz
import tempfile
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import extractor
from file_predict import (normalize_units,compute_rule_severity,friendly_explanations,combine_predictions,NORMAL_RANGES,FEATURES,)

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ---------------- Page + THEME ----------------
st.set_page_config(page_title="CBC Urgency Detector", layout="centered")

st.markdown(
    """
<style>
:root {
  --primary-green: #16a34a;
  --green-hover: #0f5c2e;
  --white: #ffffff;
  --black: #000000;
}

/* Black background */
[data-testid="stAppViewContainer"] {
  background-color: var(--black) !important;
}

/* Headings green; text white */
h1, h2, h3, h4, h5, h6 { color: var(--primary-green) !important; }
p, div, span, label { color: var(--white) !important; }

/* File uploader */
[data-testid="stFileUploaderDropzone"]{
  border: 2px dashed var(--primary-green) !important;
  background-color: #111 !important;
}
[data-testid="stFileUploaderDropzone"]:hover{
  border-color: var(--green-hover) !important;
}

/* Buttons = green */
.stButton > button {
  background-color: var(--primary-green) !important;
  color: var(--black) !important;
  border-radius: 8px !important;
  border: none !important;
  padding: 0.6rem 1.2rem !important;
  font-weight: 600 !important;
}
.stButton > button:hover {
  background-color: var(--green-hover) !important;
  color: var(--white) !important;
}

/* Alerts */
.stAlert { border-radius: 8px !important; }

</style>
""",
    unsafe_allow_html=True,
)

MODEL_PATH = "cbc_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

@st.cache_resource(show_spinner=False)
def load_model_and_encoder():
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    return model, le

def prepare_dataframe(values: dict) -> pd.DataFrame:
    row = {f: (values.get(f) if values.get(f) is not None else float("nan")) for f in FEATURES}
    return pd.DataFrame([row], columns=FEATURES)

def align_to_model_columns(df: pd.DataFrame, model):
    """
    If the trained model expects extra columns (e.g., it was trained earlier with RDWCV),
    add them as NaN so sklearn won't error on predict. Then align the column order.
    """
    try:
        expected = list(model.feature_names_in_)
    except Exception:
        return df
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    return df[expected]

def abnormal_list(values: dict):
    items = []
    for k, (lo, hi) in NORMAL_RANGES.items():
        v = values.get(k)
        if v is None:
            items.append(f"{k}: MISSING")
        elif not (lo <= v <= hi):
            items.append(f"{k}: {v} (normal {lo}-{hi})")
    return items

def group_patient_summary(values: dict, final_label: str, model_label: str, rule_label: str):
    core_keys = ["HGB", "WBC", "PLT", "HCT", "RBC"]
    read_lines = []
    for k in core_keys:
        v = values.get(k)
        read_lines.append(f"{k}: {'MISSING' if v is None else v}")

    flags = []
    for k, (lo, hi) in NORMAL_RANGES.items():
        v = values.get(k)
        if v is None:
            flags.append(f"{k} is missing")
        elif not (lo <= v <= hi):
            direction = "low" if v < lo else "high"
            flags.append(f"{k} is {direction} ({v} vs {lo}-{hi})")

    meanings = friendly_explanations(values)

    if final_label == "Normal":
        actions = [
            "No urgent action needed.",
            "If this is a routine check, keep monitoring as advised by your doctor.",
        ]
    elif final_label == "Mild":
        actions = [
            "Schedule a routine consultation to discuss these results.",
            "Consider rechecking the CBC in a few weeks if advised.",
        ]
    elif final_label == "Urgent":
        actions = [
            "Consult a doctor soon to review these findings.",
            "If you have symptoms (fever, fatigue, bleeding, breathlessness), do not delay.",
        ]
    else:  # Emergency
        actions = [
            "Seek immediate medical attention.",
            "If symptoms are severe (heavy bleeding, extreme fatigue, chest pain, confusion), go to the emergency department.",
        ]

    header = f"**Overall Urgency: {final_label}**  \n(Model: {model_label} | Rules: {rule_label})"
    return header, read_lines, flags, meanings, actions

def severity_badge(label: str):
    if label == "Normal":
        st.success("Overall urgency: Normal")
    elif label == "Mild":
        st.info("Overall urgency: Mild")
    elif label == "Urgent":
        st.warning("Overall urgency: Urgent")
    else:
        st.error("Overall urgency: Emergency")

# ---------------- UI ----------------
st.title("CBC Urgency Detector")
st.write("Upload a CBC report (image or PDF). We will extract values, assess urgency and generate a patient-friendly explanation.")

uploaded = st.file_uploader("Upload CBC report (png/jpg/jpeg/pdf)", type=["png", "jpg", "jpeg","pdf"])

file_path = None
import tempfile

if uploaded is not None:
    ext = uploaded.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(uploaded.getbuffer())
        file_path = tmp.name

if file_path:
    st.subheader("Preview")
    try:
        if file_path.lower().endswith(".pdf"):
            doc = fitz.open(file_path)
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=150)

            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

            st.image(img, use_container_width=True)
        else:
            st.image(file_path, use_container_width=True)
    except Exception as e:
        st.warning("Unable to preview PDF/image.")
        st.write(str(e))

    extracted_raw = extractor.extract_from_image_or_pdf(file_path)
    st.write("**Extracted values (raw OCR):**")
    st.json(extracted_raw)
    edit_mode = st.checkbox("Edit values", value=False)

    # CONDITIONAL EDITABLE FIELDS SECTION
    corrected = extracted_raw.copy()

    if edit_mode:
        st.markdown("### Edit extracted values")

        col1, col2 = st.columns(2)

        items = list(extracted_raw.keys())

        with col1:
            for key in items[:len(items)//2]:
                val = extracted_raw.get(key)
                corrected[key] = st.number_input(
                    f"{key}",
                    value=float(val) if val is not None else 0.0,
                    step=0.1,
                    format="%.2f"
                )

        with col2:
            for key in items[len(items)//2:]:
                val = extracted_raw.get(key)
                corrected[key] = st.number_input(
                    f"{key}",
                    value=float(val) if val is not None else 0.0,
                    step=0.1,
                    format="%.2f"
                )

    # ✅ ALWAYS override extracted_raw with corrected
    extracted_raw = corrected


    normalized = normalize_units(extracted_raw)
    df_input = prepare_dataframe(normalized)

    model, le = load_model_and_encoder()
    df_input = align_to_model_columns(df_input, model) 
    y_pred = model.predict(df_input)[0]
    model_label = le.inverse_transform([y_pred])[0]

    # Rules
    rule_label, rule_score, rule_signals = compute_rule_severity(normalized)
    final_label = combine_predictions(model_label, rule_label)
    severity_badge(final_label)

    # Patient-friendly summary
    st.markdown("### Easy Explanation")
    header, read_lines, flags, meanings, actions = group_patient_summary(
        normalized, final_label, model_label, rule_label
    )
    st.markdown(header)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**What we read**")
        for l in read_lines:
            st.markdown(f"- {l}")
    with cols[1]:
        st.markdown("**What stands out**")
        if flags:
            for l in flags:
                st.markdown(f"- {l}")
        else:
            st.markdown("- Nothing abnormal detected.")

    st.markdown("**What it could mean**")
    for l in meanings:
        st.markdown(f"- {l}")

    st.markdown("**What you should do**")
    for l in actions:
        st.markdown(f"- {l}")

    with st.expander("See detailed rule signals"):
        if rule_signals:
            for s in rule_signals:
                st.write(f"- {s}")
        else:
            st.write("No abnormal rule signals.")

    # SHAP graph
    if HAS_SHAP:
        st.markdown("### SHAP Feature Importance")
        try:
           
            try:
                df_train = pd.read_csv("labeled_cbc_new.csv")
                background = df_train[[c for c in FEATURES if c in df_train.columns]].sample(
                    min(80, len(df_train)), random_state=42
                )
            except Exception:
                background = df_input  

            # Build explainer and compute SHAP
            explainer = shap.Explainer(model, background)
            shap_values = explainer(df_input)

            pred_class = y_pred
            class_vals = shap_values.values[0][:, pred_class] 

            fig = plt.figure(figsize=(8, 6))
            pd.Series(class_vals, index=explainer.feature_names).sort_values(
                key=np.abs, ascending=True
            ).plot(kind="barh", edgecolor="black")
            plt.title(f"Feature importance for class: {model_label}")
            plt.xlabel("SHAP value (impact on model output)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"SHAP plot failed: {e}")
    else:
        st.info("SHAP not installed. Run: `pip install shap` to enable explainability.")

    # Download pdf
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)

    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("CBC Report Summary", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))

    # Source
    story.append(Paragraph(f"Source File: {file_path}", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    # Urgency
    story.append(Paragraph(
        f"<b>Final Urgency:</b> {final_label} "
        f"(Model: {model_label}, Rules: {rule_label})",
        styles["Normal"]
    ))
    story.append(Spacer(1, 0.2 * inch))

    # Core readings
    story.append(Paragraph("<b>Core Readings:</b>", styles["Heading3"]))
    for l in read_lines:
        story.append(Paragraph(f"- {l}", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    # Abnormal
    story.append(Paragraph("<b>Abnormal / Missing:</b>", styles["Heading3"]))
    if flags:
        for l in flags:
            story.append(Paragraph(f"- {l}", styles["Normal"]))
    else:
        story.append(Paragraph("- None", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    # Meaning
    story.append(Paragraph("<b>What it could mean:</b>", styles["Heading3"]))
    for l in meanings:
        story.append(Paragraph(f"- {l}", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    # Actions
    story.append(Paragraph("<b>Recommended Actions:</b>", styles["Heading3"]))
    for l in actions:
        story.append(Paragraph(f"- {l}", styles["Normal"]))

    doc.build(story)
    pdf_value = pdf_buffer.getvalue()

    st.download_button(
        "Download Summary (PDF)",
        data=pdf_value,
        file_name="cbc_summary.pdf",
        mime="application/pdf",
    )

else:
    pass
