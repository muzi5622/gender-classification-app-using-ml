import os
import cv2
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# ===============================
# Load trained model bundle
# ===============================
with open("./model/gender_face_model_tuned2.pkl", "rb") as f:
    bundle = pickle.load(f)

mean_face = bundle["mean_face"]
pca = bundle["pca"]
svm = bundle["svm"]
IMG_SIZE = bundle["img_size"]

haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ===============================
# Preprocess + Predict
# ===============================
def predict_gender(image_path):
    img_color = cv2.imread(image_path)
    if img_color is None:
        return None, "Invalid image"

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, "No face detected"
    if len(faces) > 1:
        return None, "Multiple faces detected"

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]

    face = cv2.resize(face, IMG_SIZE)
    face = face / 255.0

    face_flat = face.flatten()
    face_centered = face_flat - mean_face
    face_pca = pca.transform([face_centered])

    pred = svm.predict(face_pca)[0]
    prob = svm.predict_proba(face_pca).max()

    label = "Male" if pred == 0 else "Female"
    return label, round(prob, 2)

# ===============================
# Routes
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    error = None
    image_name = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            image_name = file.filename
            path = os.path.join(app.config["UPLOAD_FOLDER"], image_name)
            file.save(path)

            output = predict_gender(path)
            if output[0] is None:
                error = output[1]
            else:
                result, confidence = output

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        error=error,
        image=image_name
    )

if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    app.run(debug=True, use_reloader=False)
