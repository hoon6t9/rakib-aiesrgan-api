import os
import traceback
import tempfile
import torch
from flask import Flask, request, jsonify, send_file
from RealESRGAN import RealESRGAN

app = Flask(__name__)

API_KEY = os.getenv("API_KEY", "rakib69")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = RealESRGAN(device, scale=4)
model.load_weights("weights/RealESRGAN_x4.pth", download=True)
print("✅ Model loaded successfully")


@app.before_request
def before_request():
    print(f"\n➡️ {request.method} {request.path}")


@app.after_request
def after_request(response):
    print(f"⬅️ Status: {response.status_code}")
    return response


@app.errorhandler(Exception)
def handle_error(e):
    print("\n========== ERROR ==========")
    traceback.print_exc()
    return jsonify({
        "status": False,
        "message": str(e)
    }), 500


@app.get("/")
def home():
    return jsonify({
        "status": True,
        "message": "Real-ESRGAN API Running 🚀"
    })


@app.post("/api/upscale")
def upscale():
    tmp = None

    try:
        print("========== NEW REQUEST ==========")

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

        file = request.files["image"]
        print("📷 File:", file.filename)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        file.save(tmp.name)
        tmp.close()

        print("✅ Saved:", tmp.name)

        image = Image.open(tmp.name).convert("RGB")

        # বড় ছবি হলে ছোট করো (RAM বাঁচানোর জন্য)
        MAX_SIZE = 800
        if max(image.size) > MAX_SIZE:
            image.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)

        print("📐 Size:", image.size)
        print("🚀 Starting ESRGAN...")

        with torch.no_grad():
            sr = model.predict(image)

        print("✅ ESRGAN Finished")

        output = io.BytesIO()
        sr.save(output, format="PNG", optimize=True)
        output.seek(0)

        del sr

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if tmp and os.path.exists(tmp.name):
            os.remove(tmp.name)

        return send_file(
            output,
            mimetype="image/png",
            download_name="upscaled.png"
        )

    except Exception:
        traceback.print_exc()

        if tmp and os.path.exists(tmp.name):
            os.remove(tmp.name)

        return jsonify({
            "status": False,
            "message": "Internal Server Error"
        }), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
)
