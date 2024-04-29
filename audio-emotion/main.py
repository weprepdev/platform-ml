import torch
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import numpy as np
from pydub import AudioSegment
from torch import nn
import argparse
import operator
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = AutoModelForAudioClassification.from_pretrained("wav2vec2-lg-xlsr-en-speech-emotion-recognition")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("wav2vec2-large-xlsr-53")

model.projector = nn.Linear(1024, 1024, bias=True)
model.classifier = nn.Linear(1024, 8, bias=True)

torch_state_dict = torch.load('wav2vec2-lg-xlsr-en-speech-emotion-recognition/pytorch_model.bin', map_location=torch.device('cpu'))

model.projector.weight.data = torch_state_dict['classifier.dense.weight']
model.projector.bias.data = torch_state_dict['classifier.dense.bias']

model.classifier.weight.data = torch_state_dict['classifier.output.weight']
model.classifier.bias.data = torch_state_dict['classifier.output.bias']
model.to('cpu')
def predict_emotion(audio_file):
    if not audio_file:
        audio_file = 'dramatic-cry.mp3'
    sound = AudioSegment.from_file(audio_file)
    sound = sound.set_frame_rate(16000)
    sound_array = np.array(sound.get_array_of_samples())

    input = feature_extractor(
        raw_speech=sound_array,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt").to('cpu')

    result = model.forward(input.input_values.float())

    id2label = {
        "0": "angry",
        "1": "calm",
        "2": "disgust",
        "3": "fearful",
        "4": "happy",
        "5": "neutral",
        "6": "sad",
        "7": "surprised"
    }
    interp = dict(zip(id2label.values(), list(round(float(i), 4) for i in result[0][0])))
    return interp
    
parser = argparse.ArgumentParser()
parser.add_argument('--file_path', default='inf.txt', type=str, help='Path to the file')
args = parser.parse_args()
file_path = args.file_path
with open(file_path, "r") as file:
    for file_name in file:
        results = predict_emotion(file_name.strip())

        for key in results:
            results[key] += 4

        total = sum(results.values())

        sorted_percentages = sorted(results.items(), key=lambda x: (x[1] / total) * 100, reverse=True)

        print(f'for file {file_name}')
        print("Emotion\t\tPercentage")
        for emotion, score in sorted_percentages:
            percentage = ((score / total) * 100)
            print(f"{emotion.capitalize()}\t\t{percentage:.2f}%")

        print('\n*********************************')
