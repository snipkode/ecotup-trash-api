import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tflite_runtime.interpreter as tflite
import io

app = Flask(__name__)

# ─────────────────────────────────────────
# Load both models once at startup
# ─────────────────────────────────────────
print("Loading sigmoid model...")
sigmoid_interpreter = tflite.Interpreter(model_path="trashClassification_sigmoid.tflite")
sigmoid_interpreter.allocate_tensors()
sigmoid_input  = sigmoid_interpreter.get_input_details()
sigmoid_output = sigmoid_interpreter.get_output_details()
print("Sigmoid model loaded.")

print("Loading softmax model...")
softmax_interpreter = tflite.Interpreter(model_path="trashClassification_softmax.tflite")
softmax_interpreter.allocate_tensors()
softmax_input  = softmax_interpreter.get_input_details()
softmax_output = softmax_interpreter.get_output_details()
print("Softmax model loaded.")

IMAGE_SIZE = 255

# ─────────────────────────────────────────
# Helper: preprocess image → float32 array
# ─────────────────────────────────────────
def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0   # normalize 0-1
    return np.expand_dims(arr, axis=0)               # add batch dim → (1,255,255,3)

# ─────────────────────────────────────────
# Helper: run sigmoid → Organik/Anorganik
# ─────────────────────────────────────────
def classify_sigmoid(input_data):
    sigmoid_interpreter.set_tensor(sigmoid_input[0]['index'], input_data)
    sigmoid_interpreter.invoke()
    output = sigmoid_interpreter.get_tensor(sigmoid_output[0]['index'])[0]
    # output[0] = organik prob, output[1] = anorganik prob
    labels = ["Organik", "Anorganik"]
    return labels[int(np.argmax(output))], float(np.max(output)), output.tolist()

# ─────────────────────────────────────────
# Helper: run softmax → Glass/Metal/Paper/Plastic
# ─────────────────────────────────────────
def classify_softmax(input_data):
    softmax_interpreter.set_tensor(softmax_input[0]['index'], input_data)
    softmax_interpreter.invoke()
    output = softmax_interpreter.get_tensor(softmax_output[0]['index'])[0]
    # output[0]=Glass, output[1]=Metal, output[2]=Paper, output[3]=Plastic
    labels = ["Glass", "Metal", "Paper", "Plastic"]
    return labels[int(np.argmax(output))], float(np.max(output)), output.tolist()

# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online",
        "message": "Ecotup Trash Classification API",
        "endpoints": {
            "POST /predict": "Classify trash image (multipart/form-data, field: image)"
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": True, "message": "No image field in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": True, "message": "No file selected"}), 400

    allowed_ext = {"jpg", "jpeg", "png", "bmp", "webp"}
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed_ext:
        return jsonify({"error": True, "message": f"File type .{ext} not supported"}), 400

    try:
        image_bytes = file.read()
        input_data  = preprocess(image_bytes)

        # Step 1: Sigmoid — Organik vs Anorganik
        sigmoid_label, sigmoid_confidence, sigmoid_scores = classify_sigmoid(input_data)

        result = {
            "error": False,
            "category": sigmoid_label,
            "confidence": round(sigmoid_confidence * 100, 2),
            "sigmoid_scores": {
                "Organik":    round(sigmoid_scores[0] * 100, 2),
                "Anorganik":  round(sigmoid_scores[1] * 100, 2)
            }
        }

        # Step 2: If Anorganik → run Softmax for sub-category
        if sigmoid_label == "Anorganik":
            softmax_label, softmax_confidence, softmax_scores = classify_softmax(input_data)
            result["sub_category"] = softmax_label
            result["sub_confidence"] = round(softmax_confidence * 100, 2)
            result["softmax_scores"] = {
                "Glass":   round(softmax_scores[0] * 100, 2),
                "Metal":   round(softmax_scores[1] * 100, 2),
                "Paper":   round(softmax_scores[2] * 100, 2),
                "Plastic": round(softmax_scores[3] * 100, 2)
            }
        else:
            result["sub_category"] = None

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": True, "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
