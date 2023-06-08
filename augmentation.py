import os
import librosa
import soundfile as sf

speed_factors = [0.95, 0.9, 1.05, 1.1]

folders = [d for d in os.listdir() if os.path.isdir(d)]

for folder in folders:
    for filename in os.listdir(folder):
        if filename.endswith('.wav'):
            y, sr = librosa.load(os.path.join(folder, filename))
            
            for speed_factor in speed_factors:
                y_speed = librosa.effects.time_stretch(y, rate=speed_factor)
                sf.write(os.path.join(folder, filename.replace('.wav', f'_speed_{speed_factor}.wav')), y_speed, sr)
            