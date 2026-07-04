import os
import io
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_file
from RealESRGAN import RealESRGAN

app = Flask(__name__)

API_KEY = os.getenv("API_KEY", "rakib69")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RealESRGAN(device, scale=4)
model.load_weights("weights/RealESRGAN_x4.pth", download=True)


@app.get("/")
def home():
    return jsonify({
        "status": True,
        "message": "Real-ESRGAN API Running 🚀"
    })


@app.post("/api/upscale")
def upscale():
    if request.form.get("apikey") != API_KEY:
        return jsonify({
            "status": False,
            "message": "Unauthorized"
        }), 401

    if "image" not in request.files:
        return jsonify({
            "status": False,
            "message": "No image uploaded"
        }), 400

    try:
        image = Image.open(request.files["image"]).convert("RGB")

        sr = model.predict(image)

        output = io.BytesIO()
        sr.save(output, format="PNG")
        output.seek(0)

        return send_file(
            output,
            mimetype="image/png",
            download_name="upscaled.png"
        )

    except Exception as e:
        return jsonify({
            "status": False,
            "message": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
