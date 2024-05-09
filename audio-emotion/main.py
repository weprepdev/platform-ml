import torch
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import numpy as np
from pydub import AudioSegment
from torch import nn
from flask import Flask, request, jsonify
import os
import json
TRANSFORMERS_NO_ADVISORY_WARNINGS=1
app = Flask(__name__)

def load_model():
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = AutoModelForAudioClassification.from_pretrained("wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("wav2vec2-large-xlsr-53")

    model.projector = nn.Linear(1024, 1024, bias=True)
    model.classifier = nn.Linear(1024, 8, bias=True)

    torch_state_dict = torch.load('wav2vec2-lg-xlsr-en-speech-emotion-recognition/pytorch_model.bin', map_location=torch.device(device))

    model.projector.weight.data = torch_state_dict['classifier.dense.weight']
    model.projector.bias.data = torch_state_dict['classifier.dense.bias']

    model.classifier.weight.data = torch_state_dict['classifier.output.weight']
    model.classifier.bias.data = torch_state_dict['classifier.output.bias']
    model.to(device)
    return model, feature_extractor, device

def predict_emotion(audio_segment, model, feature_extractor, device):
    input = feature_extractor(
        raw_speech=audio_segment,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt"
    ).to(device)

    result = model.forward(input.input_values.float())

    id2label = {
        "0": "Angry",
        "1": "Calm",
        "2": "Disgust",
        "3": "Fear",
        "4": "Happiness",
        "5": "Neutral",
        "6": "Sadness",
        "7": "Surprise"
    }
    interp = dict(zip(id2label.values(), list(round(float(i), 4) for i in result[0][0])))
    return interp


def min_max_scale(emotion_scores):
    min_val = min(emotion_scores.values())
    max_val = max(emotion_scores.values())
    scaled_scores = {}
    for emotion, score in emotion_scores.items():
        scaled_score = (score - min_val) / (max_val - min_val)
        scaled_scores[emotion] = scaled_score

    total_score = sum(scaled_scores.values())
    normalized_scores = {emotion: score / total_score for emotion, score in scaled_scores.items()}
    return normalized_scores

def process_audio_file(audio_file):
    model, feature_extractor, device = load_model()
    sound = AudioSegment.from_file(audio_file)
    sound = sound.set_frame_rate(16000)
    duration_ms = len(sound)
    segment_duration_ms = 1000 / 6  # 5 samples per second
    segments = []
    for start_ms in range(0, int(duration_ms), int(segment_duration_ms)):
        end_ms = min(start_ms + segment_duration_ms, duration_ms)
        segment = sound[start_ms:end_ms]
        segment_array = np.array(segment.get_array_of_samples())
        segments.append(segment_array)

    results = []
    for segment_array in segments:
        results.append(predict_emotion(segment_array,model, feature_extractor, device ))
    results = [min_max_scale(result) for result in results]

    return results



@app.route('/predict_emotion', methods=['POST'])
def predict_emotion_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        file_path = 'temp_audio_file.mp3'
        file.save(file_path)
        results = process_audio_file(file_path)
        json_object = {"emotion_orders": {}}
        for idx, result in enumerate(results):
            json_object["emotion_orders"][str(idx)] = result

    return json.dumps(json_object, indent=4)

if __name__ == '__main__':
    app.run()
    # file_path = 'tests/amir.mp3'
    # results = process_audio_file(file_path)
    # json_object = {"emotion_orders": {}}
    # for idx, result in enumerate(results):
    #     json_object["emotion_orders"][str(idx)] = result

    # with open('res.json', 'w') as f:
    #     json.dump(json_object, f, indent=4)