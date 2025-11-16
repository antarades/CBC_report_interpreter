import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

model = joblib.load("cbc_model.pkl")
le = joblib.load("label_encoder.pkl")
print(" Model and encoder loaded successfully!")

df = pd.read_csv("labeled_cbc_new.csv")
features = ['HGB','WBC','RBC','PLT','HCT','MCV','MCH','MCHC','RDWSD','RDWCV']
sample = df[features].iloc[[0]] 
print("\n Sample data to explain:")
print(sample)

prediction = model.predict(sample)
urgency_label = le.inverse_transform(prediction)[0]
print("\nPredicted Urgency Level:", urgency_label)

explainer = shap.Explainer(model, df[features])
shap_values = explainer(sample)

print("\n Generating SHAP force plot...")
pred_class = prediction[0]  # the model's predicted class (index)
shap_value_class = shap_values.values[0][:, pred_class]  

# Plot using SHAP's force plot API
shap.plots.force(
    explainer.expected_value[pred_class],
    shap_value_class,
    sample,
    matplotlib=True
)

plt.savefig("shap_force_plot.png", bbox_inches="tight")
plt.close()
print("SHAP force plot saved as shap_force_plot.png")

print("\n Feature importance based on SHAP values for predicted class...")

feature_importance = pd.Series(shap_value_class, index=features).sort_values(key=abs, ascending=True)

plt.figure(figsize=(8, 6))
feature_importance.plot(kind="barh", color="green", edgecolor="black")
plt.title(f"Feature importance for class: {urgency_label}")
plt.xlabel("SHAP value (impact on model output)")
plt.tight_layout()
plt.savefig("shap_summary_plot.png", bbox_inches="tight")
plt.close()
print("SHAP summary plot saved as shap_summary_plot.png")

print("\n Explainability analysis complete!")
