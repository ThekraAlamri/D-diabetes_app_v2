from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def make_prediction_pdf(username, patient_id, model_name, risk_percentage, predicted_label, inputs: dict) -> bytes:
    buff = BytesIO()
    c = canvas.Canvas(buff, pagesize=A4)
    width, height = A4

    y = height - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Diabetes Prediction Report")
    y -= 25

    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"User: {username}"); y -= 16
    c.drawString(50, y, f"Patient ID: {patient_id or 'N/A'}"); y -= 16
    c.drawString(50, y, f"Model: {model_name}"); y -= 16
    c.drawString(50, y, f"AI Risk Percentage: {risk_percentage}%"); y -= 16
    c.drawString(50, y, f"Predicted Label: {predicted_label} (1=Risk, 0=Low)"); y -= 28

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Inputs:"); y -= 18

    c.setFont("Helvetica", 10)
    for k, v in inputs.items():
        c.drawString(60, y, f"{k}: {v}"[:120])
        y -= 14
        if y < 80:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 10)

    c.showPage()
    c.save()
    return buff.getvalue()
