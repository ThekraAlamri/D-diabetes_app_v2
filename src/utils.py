import streamlit as st

def load_css(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

def treatment_page():
    st.markdown("<h1 style='text-align:center'>💊 Treatment</h1>", unsafe_allow_html=True)
    st.warning("Educational only — not a medical diagnosis.")
    st.markdown("""
### If risk is high:
- Visit a clinic for **fasting glucose + HbA1c** tests
- Reduce sugary drinks, refined carbs; increase fiber
- Exercise **150 min/week**
- Sleep 7–9 hours, reduce stress

### If risk is low:
- Maintain healthy habits
- Keep BMI and waist in healthy range
- Repeat checks every 6–12 months (especially with family history)
""")
