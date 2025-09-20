from flask import Flask, request, jsonify, render_template
import os
from omr_utils import preprocess_omr, detect_answers, load_answer_key_xlsx, evaluate

app = Flask(__name__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Store the answer key in memory
app.config["ANSWER_KEY"] = None  

# -----------------------
# Homepage
# -----------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------
# Upload endpoint
# -----------------------
@app.route("/upload", methods=["POST"])
def upload_files():
    if "omr" not in request.files:
        return jsonify({"error": "Please upload at least the OMR sheet (jpeg)."}), 400

    omr_file = request.files["omr"]
    omr_path = os.path.join(UPLOAD_DIR, omr_file.filename)
    omr_file.save(omr_path)

    # If a new answer key is uploaded, update it
    if "xlsx" in request.files and request.files["xlsx"].filename != "":
        xlsx_file = request.files["xlsx"]
        xlsx_path = os.path.join(UPLOAD_DIR, xlsx_file.filename)
        xlsx_file.save(xlsx_path)
        try:
            app.config["ANSWER_KEY"] = load_answer_key_xlsx(xlsx_path)
        except Exception as e:
            return jsonify({"error": f"Invalid answer key format: {e}"}), 400

    if app.config["ANSWER_KEY"] is None:
        return jsonify({"error": "No answer key uploaded yet. Please upload it once."}), 400

    try:
        # Preprocess OMR
        processed = preprocess_omr(omr_path)

        # Detect answers
        detected_answers = detect_answers(processed)

        # Evaluate
        results = evaluate(detected_answers, app.config["ANSWER_KEY"])

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
