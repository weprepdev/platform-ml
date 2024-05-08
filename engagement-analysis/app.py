from flask import Flask, request, jsonify
from main import analyze_video

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    path = data.get('path')
    if path:
        result = analyze_video(path)
        return jsonify(result)
    else:
        return jsonify({'error': 'No video found'})

if __name__ == '__main__':
    app.run(debug=True)