import pyaudio
import wave
import numpy as np
from hmmlearn import hmm
import preprocessing
import librosa
import joblib
# load model and class_names
model_path = 'trained_model_final'
class_names = ['amluongmottram', 'amluongnammuoi', 'baitieptheo', 'baitruocdo','batnhac','dunglai', 'meimei','phatlaplai','phatngaunhien', 'phattuantu', 'tatam', 'tualui','tuatoi']
model = hmm.GMMHMM(n_components=16, n_mix=16, verbose=True)
model = joblib.load(model_path)

# define audio parameters
audio_format = pyaudio.paInt16
channels = 1
rate = 16000
chunk = 1024

# start recording
record_seconds = 1  # adjust this parameter to increase/decrease recording time
p = pyaudio.PyAudio()
stream = p.open(format=audio_format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
print("Start recording...")
frames = []
for i in range(0, int(rate / chunk * record_seconds)):
    data = stream.read(chunk)
    frames.append(data)
print("Recording finished.")

# save recording to file
wf = wave.open('temp/record.wav', 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(audio_format))
wf.setframerate(rate)
wf.writeframes(b''.join(frames))
wf.close()

# load and preprocess audio file
y, sr = librosa.load('temp/record.wav', sr=16000)
mfcc = preprocessing.get_mfcc('temp/record.wav')
mfcc = mfcc.reshape((1, -1))

# predict class label
log_likelihood = []
for i in range(len(class_names)):
    score = model.score(mfcc)
    log_likelihood.append(score)
predicted_label = np.argmax(log_likelihood)

# print predicted word or "XXX" if unable to predict
if log_likelihood[predicted_label] > -np.inf:
    print("Predicted word:", class_names[predicted_label])
else:
    print("XXX")

# close stream and terminate pyaudio
stream.stop_stream()
stream.close()
p.terminate()
