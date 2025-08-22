import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

# -----------------------
# Load model
# -----------------------
model = joblib.load("diabetes_model.pkl")

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺", layout="wide")

# -----------------------
# Sidebar instructions
# -----------------------
with st.sidebar:
    st.title("ℹ️ How to Use")
    st.markdown("""
    1. Enter patient details in the main panel.
    2. Click **Predict Risk**.
    3. View prediction, probability chart, and lifestyle suggestions.
    4. Download a PDF report if needed.
    """)

# -----------------------
# Custom CSS for dark theme and cards
# -----------------------
st.markdown("""
<style>
body {background-color: #0B0C10; color: #F0F0F0;}
.stButton>button {background: linear-gradient(90deg,#4CAF50,#2E7D32); color:white; font-weight:bold; border-radius:8px; height:3em; width:100%;}
.card {background: linear-gradient(145deg,#1F2937,#272E3B); padding:20px; border-radius:15px; margin-bottom:20px; box-shadow: 0px 6px 15px rgba(0,0,0,0.5);}
.progress-container {background: #333; border-radius: 15px; padding: 3px; margin-top:5px;}
.progress-bar {height:25px; border-radius:15px; text-align:center; color:#fff; font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Title
# -----------------------
st.title("🩺 Diabetes Risk Prediction App")
st.markdown("Estimate your **diabetes risk** based on lab values. This is **educational only**.")

# -----------------------
# User Input Form
# -----------------------
st.header("📋 Patient Details")
with st.form("patient_form"):
    gender_input = st.selectbox("Gender", ["Male", "Female"], help="Select gender")
    gender = "M" if gender_input=="Male" else "F"

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age",1,120,30)
        bmi = st.number_input("BMI",10.0,50.0,25.0)
        urea = st.number_input("Urea (mg/dL)",0.0,200.0,25.0)
        cr = st.number_input("Creatinine (mg/dL)",0.0,10.0,1.0)
        hba1c = st.number_input("HbA1c (%)",3.0,15.0,5.5)
    with col2:
        chol = st.number_input("Cholesterol (mg/dL)",50.0,400.0,180.0)
        tg = st.number_input("Triglycerides (TG)",30.0,500.0,150.0)
        hdl = st.number_input("HDL (mg/dL)",10.0,150.0,50.0)
        ldl = st.number_input("LDL (mg/dL)",20.0,300.0,100.0)
        vldl = st.number_input("VLDL (mg/dL)",5.0,100.0,30.0)

    submitted = st.form_submit_button("🔍 Predict Risk")

# -----------------------
# Prepare data
# -----------------------
data = pd.DataFrame([{
    "Gender": gender,
    "AGE": age,
    "Urea": urea,
    "Cr": cr,
    "HbA1c": hba1c,
    "Chol": chol,
    "TG": tg,
    "HDL": hdl,
    "LDL": ldl,
    "VLDL": vldl,
    "BMI": bmi
}])

# -----------------------
# Prediction
# -----------------------
if submitted:
    prediction = model.predict(data)[0]
    probas = model.predict_proba(data)[0]
    risk_map = {0:"Low Risk",1:"Medium Risk",2:"High Risk"}
    risk_colors = {0:"#4CAF50",1:"#FFA500",2:"#FF3E3E"}
    risk_emoji = {0:"🟢😊",1:"🟠😐",2:"🔴⚠️"}

    # Risk Card
    st.markdown(f"""
    <div class="card">
        <h2>🧾 Risk Level: <span style="color:{risk_colors[prediction]}">{risk_map[prediction]} {risk_emoji[prediction]}</span></h2>
        <div class="progress-container">
            <div class="progress-bar" style="width:{probas[prediction]*100}%;background:linear-gradient(to right,{risk_colors[prediction]},#000)">
                {probas[prediction]*100:.1f}%
            </div>
        </div>
    </div>
    """,unsafe_allow_html=True)

    # Probability chart
    fig, ax = plt.subplots()
    bars = ax.bar(risk_map.values(), probas, color=[risk_colors[i] for i in range(3)])
    ax.set_ylabel("Probability")
    ax.set_ylim(0,1)
    ax.set_title("Risk Probability Distribution")
    for bar, prob in zip(bars, probas):
        ax.text(bar.get_x() + bar.get_width()/2, prob + 0.02, f"{prob*100:.1f}%", ha="center", fontsize=12, color="#fff")
    st.pyplot(fig)

    # -----------------------
    # SHAP Explanation (readable)
    # -----------------------
    st.subheader("🔎 Why this prediction?")
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(data)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)

    # Feature names readable
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name != "remainder":
            if hasattr(transformer, 'get_feature_names_out'):
                feature_names.extend(transformer.get_feature_names_out(columns))
            else:
                feature_names.extend(columns)
    feature_names = [fn.replace("Gender_", "Gender: ") for fn in feature_names]

    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, plot_type="bar", show=False)
    st.pyplot(plt.gcf(), bbox_inches="tight")
    plt.clf()

    # -----------------------
    # Lifestyle suggestions
    # -----------------------
    st.subheader("💡 Lifestyle Suggestions")
    suggestions = {
        0:"✅ Low Risk:\n- Continue healthy diet & exercise\n- Regular checkups",
        1:"ℹ️ Medium Risk:\n- Monitor HbA1c & blood sugar\n- Maintain healthy weight & cholesterol\n- Avoid junk food, alcohol, smoking",
        2:"⚠️ High Risk:\n- Control blood sugar (HbA1c <6.5%)\n- Maintain BMI <25\n- Reduce saturated fats, increase fiber\n- 30 mins daily activity"
    }
    st.markdown(f"<div class='card'>{suggestions[prediction].replace(chr(10),'<br>')}</div>",unsafe_allow_html=True)

    # -----------------------
    # PDF download
    # -----------------------
    st.subheader("📄 Download Report")
    def create_pdf(data,prediction,probas):
        buffer = BytesIO()
        c = canvas.Canvas(buffer,pagesize=letter)
        c.setFont("Helvetica",12)
        c.drawString(30,750,"🩺 Diabetes Risk Prediction Report")
        c.line(30,747,580,747)
        y=720
        for k,v in data.iloc[0].items():
            c.drawString(30,y,f"{k}: {v}")
            y-=20
        c.drawString(30,y-10,f"Prediction: {risk_map[prediction]}")
        c.drawString(30,y-30,f"Probabilities: {probas}")
        c.drawString(30,y-60,"Disclaimer: Educational purposes only. Not medical advice.")
        c.save()
        buffer.seek(0)
        return buffer
    
    pdf_buffer = create_pdf(data,prediction,probas)
    st.download_button("⬇️ Download PDF Report",data=pdf_buffer,
                       file_name="diabetes_report.pdf",mime="application/pdf")

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.caption("⚠️ Educational app only. Not a substitute for medical advice.")
