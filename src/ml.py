from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.db import insert_prediction, fetch_predictions
from src.pdf_export import make_prediction_pdf

FEATURES = [
    "ID","age","gender","height_cm","weight_kg","bmi","waist_cm",
    "diastolic_bp","systolic_bp","fasting_glucose","random_glucose",
    "choles","hba1c","family_history","smoker","physically_active"
]
TARGET = "has_diabetes"

CATEGORICAL = ["gender","family_history","smoker","physically_active"]
NUMERICAL = [c for c in FEATURES if c not in CATEGORICAL and c != "ID"]

def _default_model(model_name: str):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=2000)
    if model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, random_state=42)
    if model_name == "SVM":
        return SVC(probability=True, kernel="rbf", C=2.0, gamma="scale")
    raise ValueError("Unknown model")

def build_pipeline(model_name: str):
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), NUMERICAL),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    clf = _default_model(model_name)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe

def model_path(base_dir: Path, model_name: str):
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir / f"diabetes_{model_name.replace(' ','_').lower()}.joblib"

def train_from_csv(base_dir: Path, csv_path: Path, model_name: str):
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df.dropna(subset=[TARGET]).copy()
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    pipe = build_pipeline(model_name)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)

    path = model_path(base_dir, model_name)
    joblib.dump(pipe, path)

    return {"accuracy": float(acc), "report": classification_report(y_test, preds, zero_division=0), "model_path": str(path)}

def load_or_train_if_missing(base_dir: Path, model_name: str):
    path = model_path(base_dir, model_name)
    if path.exists():
        return joblib.load(path), str(path)

    sample = base_dir / "data" / "sample_diabetes.csv"
    if sample.exists():
        train_from_csv(base_dir, sample, model_name)
        return joblib.load(path), str(path)

    raise FileNotFoundError(f"No trained model found at {path}. Upload CSV and train first.")

def predict_page(base_dir: Path):
    st.markdown("<h1 style='text-align:center'>🧪 Predict Diabetes</h1>", unsafe_allow_html=True)

    colA, colB = st.columns([1.2, 1])

    with colB:
        st.subheader("Model Settings")
        model_name = st.selectbox("Choose ML Model", ["Logistic Regression", "Random Forest", "SVM"])
        st.caption("Train models by uploading CSV in this page.")

        st.markdown("---")
        st.subheader("Train / Retrain")
        uploaded = st.file_uploader("Upload CSV to train", type=["csv"])
        if st.button("Train Now", use_container_width=True):
            if uploaded is None:
                st.error("Upload a CSV first.")
            else:
                tmp = base_dir / "data" / "_uploaded_train.csv"
                tmp.write_bytes(uploaded.getvalue())
                res = train_from_csv(base_dir, tmp, model_name)
                st.success(f"Training done. Accuracy: {res['accuracy']:.3f}")
                st.text(res["report"])

    with colA:
        st.subheader("Patient Inputs")
        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                patient_id = st.text_input("ID (Patient ID)", value="")
                age = st.number_input("age", 0.0, 120.0, 30.0)
                gender = st.selectbox("gender", ["Male", "Female"])
            with c2:
                height_cm = st.number_input("height_cm", 50.0, 250.0, 165.0)
                weight_kg = st.number_input("weight_kg", 10.0, 250.0, 70.0)
                bmi = st.number_input("bmi", 10.0, 80.0, 25.0)
            with c3:
                waist_cm = st.number_input("waist_cm", 30.0, 200.0, 85.0)
                systolic_bp = st.number_input("systolic_bp", 60.0, 250.0, 120.0)
                diastolic_bp = st.number_input("diastolic_bp", 30.0, 150.0, 80.0)

            c4, c5, c6, c7 = st.columns(4)
            with c4:
                fasting_glucose = st.number_input("fasting_glucose", 40.0, 400.0, 95.0)
            with c5:
                random_glucose = st.number_input("random_glucose", 40.0, 500.0, 110.0)
            with c6:
                choles = st.number_input("choles", 50.0, 500.0, 180.0)
            with c7:
                hba1c = st.number_input("hba1c", 3.0, 20.0, 5.5)

            c8, c9, c10 = st.columns(3)
            with c8:
                family_history = st.selectbox("family_history", ["No", "Yes"])
            with c9:
                smoker = st.selectbox("smoker", ["No", "Yes"])
            with c10:
                physically_active = st.selectbox("physically_active", ["No", "Yes"])

            submitted = st.form_submit_button("Predict", use_container_width=True)

        if submitted:
            pipe, model_file = load_or_train_if_missing(base_dir, model_name)

            X = pd.DataFrame([{
                "ID": patient_id if patient_id else "NA",
                "age": age,
                "gender": gender,
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "bmi": bmi,
                "waist_cm": waist_cm,
                "diastolic_bp": diastolic_bp,
                "systolic_bp": systolic_bp,
                "fasting_glucose": fasting_glucose,
                "random_glucose": random_glucose,
                "choles": choles,
                "hba1c": hba1c,
                "family_history": family_history,
                "smoker": smoker,
                "physically_active": physically_active,
            }])

            risk = float(pipe.predict_proba(X)[0][1])
            risk_pct = round(risk * 100.0, 2)
            pred_label = int(risk >= 0.5)

            st.markdown("### Result")
            if pred_label == 1:
                st.error(f"High risk ✅ (AI risk: **{risk_pct}%**)")
            else:
                st.success(f"Low risk ✅ (AI risk: **{risk_pct}%**)")

            insert_prediction(
                user_id=st.session_state.user["id"],
                payload={
                    "patient_id": patient_id,
                    "age": age,
                    "gender": gender,
                    "height_cm": height_cm,
                    "weight_kg": weight_kg,
                    "bmi": bmi,
                    "waist_cm": waist_cm,
                    "diastolic_bp": diastolic_bp,
                    "systolic_bp": systolic_bp,
                    "fasting_glucose": fasting_glucose,
                    "random_glucose": random_glucose,
                    "choles": choles,
                    "hba1c": hba1c,
                    "family_history": family_history,
                    "smoker": smoker,
                    "physically_active": physically_active,
                    "predicted_label": pred_label,
                    "risk_percentage": risk_pct,
                    "model_name": model_name,
                }
            )

            pdf_bytes = make_prediction_pdf(
                username=st.session_state.user["username"],
                patient_id=patient_id,
                model_name=model_name,
                risk_percentage=risk_pct,
                predicted_label=pred_label,
                inputs=X.iloc[0].to_dict(),
            )

            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"diabetes_report_{patient_id or 'patient'}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

def model_info_page(base_dir: Path):
    st.markdown("<h1 style='text-align:center'>ℹ️ Model Info</h1>", unsafe_allow_html=True)
    st.write("Features:")
    st.code(", ".join(FEATURES))
    st.write("Target: has_diabetes (0/1)")

def charts_page(base_dir: Path):
    import matplotlib.pyplot as plt

    st.markdown("<h1 style='text-align:center'>📊 Charts & Visualization</h1>", unsafe_allow_html=True)

    rows = fetch_predictions(st.session_state.user["id"], limit=500)
    if not rows:
        st.info("No predictions yet. Use Predict Diabetes first.")
        return

    df = pd.DataFrame(rows, columns=[
        "patient_id","age","gender","bmi","fasting_glucose","random_glucose","hba1c",
        "predicted_label","risk_percentage","model_name","created_at"
    ])

    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.subheader("Risk Percentage (recent)")
    fig = plt.figure()
    plt.plot(df["risk_percentage"].iloc[::-1].reset_index(drop=True))
    plt.xlabel("Prediction #")
    plt.ylabel("Risk %")
    st.pyplot(fig)

    st.subheader("Distribution of predicted labels")
    fig2 = plt.figure()
    counts = df["predicted_label"].value_counts().sort_index()
    plt.bar([str(i) for i in counts.index], counts.values)
    plt.xlabel("Predicted label (0=Low, 1=Risk)")
    plt.ylabel("Count")
    st.pyplot(fig2)

    st.download_button(
        "⬇️ Download History CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )
