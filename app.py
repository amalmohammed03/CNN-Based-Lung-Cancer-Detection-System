from flask import Flask, render_template, request
from predict import predict
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET","POST"])
def home():

    if request.method == "POST":

        if "image" not in request.files:
            return "No file uploaded"

        file = request.files["image"]

        if file.filename == "":
            return "No selected file"

        upload_path = os.path.join(app.config["UPLOAD_FOLDER"],"uploaded.jpg")

        file.save(upload_path)

        result, confidence, heatmap = predict(upload_path)

        return render_template(
            "result1.html",
            result=result,
            confidence=round(confidence,2),
            uploaded_image="uploaded.jpg",
            heatmap_image=heatmap
        )

    return render_template("home.html")
@app.route('/heatmap/<image>')
def heatmap(image):
    return render_template("heatmap.html", image=image)

if __name__ == "__main__":
    app.run(debug=True)