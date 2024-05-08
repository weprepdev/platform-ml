from flask import Flask, request, jsonify
from test import analyze_audio

app = Flask(__name__)


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    p = data.get("p")
    c = data.get("c")
    if p and c:
        results = analyze_audio(p, c)
        return jsonify(results)
    else:
        return jsonify({"error": "Invalid input"})


if __name__ == "__main__":
    app.run(debug=True)
