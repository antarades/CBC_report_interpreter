import joblib
import pandas as pd
import extractor
import os
import math

FEATURES = ['HGB','WBC','RBC','PLT','HCT','MCV','MCH','MCHC','RDWSD','RDWCV']

# Normal lab ranges (adult, generic) used for rule scoring + abnormal flags
NORMAL_RANGES={
    "HGB": (5, 20),
    "WBC": (2, 20),
    "RBC": (3, 7),
    "PLT": (50, 700),
    "HCT": (20, 60),
    "MCV": (60, 120),
    "MCH": (20, 40),
    "MCHC": (25, 40),
    "RDWSD": (20, 80),
    "RDWCV": (8, 25),
}
# Medical importance weights (heavier for core CBC safety signals)
WEIGHTS = {
    "WBC":  1.5,   # infection/inflammation
    "RBC":  1.8,   # oxygen carrying (with HGB/HCT)
    "HGB":  2.0,   # anemia risk
    "HCT":  1.6,
    "PLT":  2.2,   # bleeding risk
    "MCV":  0.8,
    "MCH":  0.8,
    "MCHC": 0.8,
    "RDWCV": 0.6,
    "RDWSD": 0.6,
}

# Map severity to an ordinal to combine with model output
ORDER = ["Normal", "Mild", "Urgent", "Emergency"]
ORDIDX = {k:i for i,k in enumerate(ORDER)}

model = joblib.load("cbc_model.pkl")
le = joblib.load("label_encoder.pkl")

def normalize_units(values: dict) -> dict:
    v = values.copy()
    # WBC: 5100 -> 5.1 (x10^3/µL)
    if v.get("WBC") is not None and v["WBC"] > 100:
        v["WBC"] = round(v["WBC"] / 1000.0, 3)
    return v


# Rule-based severity scoring 
def deviation_score(val, lo, hi):
    if val is None:
        return None
    if lo <= val <= hi:
        return 0.0
    rng = max(1e-9, hi - lo)
    if val < lo:
        return (lo - val) / rng  
    else:
        return (val - hi) / rng 

def compute_rule_severity(values: dict):
    score = 0.0
    details = [] 

    for k, (lo, hi) in NORMAL_RANGES.items():
        w = WEIGHTS.get(k, 1.0)
        v = values.get(k)

        if v is None:
            miss_penalty = w * 0.25
            score += miss_penalty
            details.append(f"{k} missing (penalty {miss_penalty:.1f})")
            continue

        dev = deviation_score(v, lo, hi)
        if dev and dev > 0:
            if dev < 0.15:
                tier = 1.0
            elif dev < 0.35:
                tier = 2.0
            else:
                tier = 3.0
            add = w * tier
            score += add
            direction = "low" if v < lo else "high"
            details.append(f"{k} {direction} ({v} vs {lo}-{hi}) → +{add:.1f}")

    if score == 0:
        rule_label = "Normal"
    elif score <= 2.2:
        rule_label = "Mild"
    elif score <= 5.0:
        rule_label = "Urgent"
    else:
        rule_label = "Emergency"

    return rule_label, score, details


def friendly_explanations(values: dict):
    """
    Returns a list of short layperson messages for each important abnormal/missing value.
    """
    msgs = []
    def add(msg): 
        if msg not in msgs: 
            msgs.append(msg)

    v = values
    def outside(k):
        x = v.get(k)
        if x is None: return None
        lo, hi = NORMAL_RANGES[k]
        if x < lo: return "low"
        if x > hi: return "high"
        return None

    if v.get("HGB") is None or outside("HGB") == "low":
        add("Hemoglobin is low or missing, which may suggest anemia or reduced oxygen-carrying capacity.")
    if v.get("HCT") is None or outside("HCT") == "low":
        add("Hematocrit looks low or missing; this can align with anemia or dehydration context.")
    if v.get("RBC") is None:
        add("RBC count is missing, so anemia risk cannot be fully assessed.")

    # White cells
    o = outside("WBC")
    if o == "high":
        add("White blood cell count appears high, which can be consistent with infection or inflammation.")
    elif o == "low":
        add("White blood cell count appears low, which may reduce infection-fighting capacity.")

    # Platelets
    o = outside("PLT")
    if o == "low":
        add("Platelets are low, which can increase bleeding tendency—please consult a doctor soon.")
    elif o == "high":
        add("Platelets are elevated; consider discussing with a clinician if persistent.")

    # Indices
    if outside("MCV") in ("low","high") or outside("MCH") in ("low","high") or outside("MCHC") in ("low","high"):
        add("Red cell indices (MCV/MCH/MCHC) are outside typical ranges; could relate to iron, B12/folate status, or other causes.")
    if v.get("RDWCV") is None and v.get("RDWSD") is None:
        add("RDW values are missing; variability of red cells cannot be evaluated.")

    if not msgs:
        add("All visible parameters look within expected ranges.")
    return msgs


# ----------------- Combined decision (Rule + ML) -----------------
def combine_predictions(model_label: str, rule_label: str):
    """
    Combine model + rules with intelligent safety logic:
    - If rules say Normal but model says Urgent/Emergency, trust rules.
    - Otherwise take the more severe label.
    """

    # Normalize labels
    ml = model_label if model_label in ORDER else "Mild"
    rl = rule_label if rule_label in ORDER else "Mild"

    # If rules seem safe but model overreacts
    if rl == "Normal" and ml in ["Urgent", "Emergency"]:
        return "Normal"

    # If rules say Mild, model says Urgent, downgrade to Mild
    if rl == "Mild" and ml == "Urgent":
        return "Mild"

    final = ORDER[max(ORDIDX[ml], ORDIDX[rl])]
    return final


def pretty_print_summary(path, extracted, normalized, prediction, abnormal_items, rule_label, rule_score, final_label, summary_lines):
    print("\n============================================================")
    print("                   CBC REPORT SUMMARY")
    print("============================================================\n")
    print(f"Source File: {path}\n")

    print("--------------- EXTRACTED CBC VALUES ----------------")
    for f in FEATURES:
        val = extracted.get(f)
        print(f"{f:<7} : {val if val is not None else 'MISSING'}")

    print("\n--------------- NORMALIZED VALUES -------------------")
    for f in FEATURES:
        val = normalized.get(f)
        print(f"{f:<7} : {val if val is not None else 'MISSING'}")

    print("\n--------------- MODEL PREDICTION --------------------")
    print(f"Model says : {prediction}")

    print("\n--------------- RULE SCORING ------------------------")
    print(f"Rule label : {rule_label}  |  Score: {rule_score:.2f}")
    if abnormal_items:
        print("Signals    :")
        for it in abnormal_items:
            print(f"  - {it}")

    print("\n--------------- FINAL DECISION ----------------------")
    print(f"Final Urgency : {final_label}")

    print("\n--------------- SIMPLE SUMMARY ----------------------")
    for line in summary_lines:
        print(f"- {line}")



def predict_from_file(path: str):
    print("Extracting values from:", path)
    extracted = extractor.extract_from_image_or_pdf(path)
    normalized = normalize_units(extracted)
    row = {f: (values.get(f) if values.get(f) is not None else float("nan"))for f in FEATURES}

    df_input = pd.DataFrame([row], columns=FEATURES)
    pred_idx = model.predict(df_input)[0]
    model_label = le.inverse_transform([pred_idx])[0]

    # Rule-based severity
    rule_label, rule_score, rule_details = compute_rule_severity(normalized)
    abnormal_items = []
    for key, (lo, hi) in NORMAL_RANGES.items():
        val = normalized.get(key)
        if val is None:
            abnormal_items.append(f"{key}: MISSING")
        elif not (lo <= val <= hi):
            abnormal_items.append(f"{key}: {val} (normal {lo}-{hi})")

    # 6) Plain-language summary lines
    summary_lines = friendly_explanations(normalized)
    # Add overall assessment line
    summary_lines.insert(0, f"Overall urgency is **{combine_predictions(model_label, rule_label)}** based on model + medical rules.")

    # 7) Combine (safety-first)
    final_label = combine_predictions(model_label, rule_label)

    # 8) Print terminal summary
    pretty_print_summary(
        path=path,
        extracted=extracted,
        normalized=normalized,
        prediction=model_label,
        abnormal_items=rule_details,  # more specific than raw abnormal list
        rule_label=rule_label,
        rule_score=rule_score,
        final_label=final_label,
        summary_lines=summary_lines
    )

    return final_label


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python file_predict.py <path_to_pdf_or_image>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print("File not found:", path)
        sys.exit(1)

    predict_from_file(path)

