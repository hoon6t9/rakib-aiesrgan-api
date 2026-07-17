import os
import io
import traceback
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_file
from RealESRGAN import RealESRGAN

app = Flask(__name__)

API_KEY = os.getenv("API_KEY", "rakib69")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = RealESRGAN(device, scale=4)
model.load_weights("weights/RealESRGAN_x4.pth", download=True)
print("✅ Model loaded successfully")


@app.get("/")
def home():
    return jsonify({
        "status": True,
        "message": "Real-ESRGAN API Running 🚀"
    })


@app.post("/api/upscale")
def upscale():
    print("\n========== NEW REQUEST ==========")
    print("Form:", request.form)
    print("Files:", request.files)

    try:
        if request.form.get("apikey") != API_KEY:
            print("❌ Invalid API Key")
            return jsonify({
                "status": False,
                "message": "Unauthorized"
            }), 401

        if "image" not in request.files:
            print("❌ Image not found")
            return jsonify({
                "status": False,
                "message": "No image uploaded"
            }), 400

        file = request.files["image"]
        print(f"📷 Image Name: {file.filename}")

        image = Image.open(file).convert("RGB")
        print(f"📐 Original Size: {image.size}")

        print("🚀 Starting upscale...")
        sr = model.predict(image)
        print("✅ Upscale completed")

        output = io.BytesIO()
        sr.save(output, format="PNG")
        output.seek(0)

        print("✅ Sending image")

        return send_file(
            output,
            mimetype="image/png",
            download_name="upscaled.png"
        )

    except Exception:
        print("========== ERROR ==========")
        traceback.print_exc()

        return jsonify({
            "status": False,
            "message": "Internal Server Error"
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
            )
