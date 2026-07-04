import os
import io
import requests
import torch
from PIL import Image
from flask import Flask, request, send_file, jsonify
from RealESRGAN import RealESRGAN

app = Flask(__name__)

API_KEY = os.getenv("API_KEY", "rakib69")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RealESRGAN(device, scale=4)
model.load_weights("weights/RealESRGAN_x4.pth", download=True)


@app.route("/")
def home():
    return jsonify({
        "status": True,
        "message": "Real-ESRGAN API Running 🚀"
    })


@app.route("/api/upscale")
def upscale():
    key = request.args.get("apikey")
    if key != API_KEY:
        return jsonify({
            "status": False,
            "message": "Unauthorized"
        }), 401

    url = request.args.get("url")

    if not url:
        return jsonify({
            "status": False,
            "message": "Missing image url"
        }), 400

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        sr = model.predict(image)

        output = io.BytesIO()
        sr.save(output, format="PNG")
        output.seek(0)

        return send_file(output, mimetype="image/png")

    except Exception as e:
        return jsonify({
            "status": False,
            "message": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
