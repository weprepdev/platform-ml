import torch
from transformers import AutoProcessor, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import numpy as np
from pydub import AudioSegment

model = AutoModelForAudioClassification.from_pretrained("wav2vec2-lg-xlsr-en-speech-emotion-recognition") 

model.projector = nn.Linear(1024, 1024, bias=True)
model.classifier = nn.Linear(1024, 8, bias=True)

torch_state_dict = torch.load('pytorch_model.bin', map_location=torch.device(0))

model.projector.weight.data = torch_state_dict['classifier.dense.weight']
model.projector.bias.data = torch_state_dict['classifier.dense.bias']

model.classifier.weight.data = torch_state_dict['classifier.output.weight']
model.classifier.bias.data = torch_state_dict['classifier.output.bias']

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("wav2vec2-large")

def predict_emotion(audio_file):
    if not audio_file:
        # I fetched some samples with known emotions from here: https://www.fesliyanstudios.com/royalty-free-sound-effects-download/poeple-crying-252
        audio_file = 'dramatic-cry.mp3'
    sound = AudioSegment.from_file(audio_file)
    sound = sound.set_frame_rate(16000)
    sound_array = np.array(sound.get_array_of_samples())
    # this model is VERY SLOW, so best to pass in small sections that contain 
    # emotional words from the transcript. like 10s or less.
    # how to make sub-chunk  -- this was necessary even with very short audio files 
    # test = torch.tensor(input.input_values.float()[:, :100000])

    input = feature_extractor(
        raw_speech=sound_array,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt")

    result = model.forward(input.input_values.float())
    # making sense of the result 
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
    interp = dict(zip(id2label.values(), list(round(float(i),4) for i in result[0][0])))
    return interp