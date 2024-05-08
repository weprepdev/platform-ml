from scipy.io import wavfile
from pitch_detectors import algorithms
import myspsolution as mysp
import textgrid
import boto3

s3 = boto3.client("s3")

def analyze_audio(p, c):
    key = f"{c}{p}"
    audio_file_path = "/tmp/audio_file.wav"
    s3.download_file("weprep-user-audios", key, audio_file_path)

    # soundfile = p + ".wav"
    soundfile = p

    fs, a = wavfile.read(soundfile)
    pitch = algorithms.Crepe(a, fs)
    times = pitch.t
    f0s = pitch.f0
    pitch_data = {}
    volume_data = {}

    for i in range(len(a)):
        volume_data[i] = str(abs(a[i]))
    vol = {"volume": volume_data}

    for i in range(len(times)):
        if f0s[i] > 0:
            pitch_data[times[i]] = str(f0s[i])
    pit = {"pitch": pitch_data}

    score = mysp.mysppron(p, c)
    general_results = mysp.mysptotal(p, c)
    general_results = general_results[0]
    for key, value in general_results.items():
        if not isinstance(value, str):
            general_results[key] = str(value)
    general_results["pronunciation_score"] = str(score)
    gen_res = {"general": general_results}

    textgridfile = p + ".TextGrid"

    tg = textgrid.TextGrid.fromFile(textgridfile)
    art_data = {}
    for i in range(len(tg[1])):
        if tg[1][i].mark == "silent":
            art_data[tg[1][i].maxTime] = 0
        else:
            art_data[tg[1][i].maxTime] = 1
    art = {"articulation": art_data}

    result = {}
    result.update(gen_res)
    result.update(art)
    result.update(pit)
    # result.update(vol)

    return result
