
# !pip install pyctcdecode transformers datasets
import numpy as np
import tempfile
import shutil
import streamlit as st
from huggingface_hub import notebook_login
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,Wav2Vec2ProcessorWithLM
import subprocess, os
from pydub import AudioSegment

# import subprocess

# # Install FFmpeg using apt-get
# subprocess.run(["apt-get", "install", "-y", "ffmpeg"])


st.title("Hindi Audio Transcription App")


# Get the Hugging Face Hub API token from st secrets
hf_token = st.secrets["HF_TOKEN"]

# Authenticate to the Hugging Face Hub API.
from huggingface_hub import HfApi
api = HfApi(hf_token)

model_id = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"
processor = Wav2Vec2Processor.from_pretrained(model_id, encoding="utf8")
model = Wav2Vec2ForCTC.from_pretrained(model_id)

def read_file_and_process(wav_file):
    filename = wav_file.split('.')[0]
    filename_16k = filename + "_16k.wav"
    resampler(wav_file, filename_16k)
    speech, _ = sf.read(filename_16k)
    inputs = processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)

    return inputs

def resampler(input_file_path, output_file_path):
    command = (
        f"ffmpeg -hide_banner -loglevel panic -i {input_file_path} -ar 16000 -ac 1 -bits_per_raw_sample 16 -vn "
        f"{output_file_path}"
    )
    subprocess.call(command, shell=True)

def parse_transcription(logits):
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    return transcription

def parse(wav_file):
    input_values = read_file_and_process(wav_file)
    with torch.no_grad():
        logits = model(**input_values).logits
    return parse_transcription(logits)

def process_long_audio(audio_object, segment_duration=200, sample_rate=44100):
    transcription = ''
    temp_dir = tempfile.mkdtemp()
    
    # Read audio data in chunks of a multiple of 2 bytes (element size for int16)
    chunk_size = 2 * segment_duration * 1000
    while True:
        chunk = audio_object.read(chunk_size)
        if not chunk:
            break  # Break the loop if there's no more audio to process

        audio_data = np.frombuffer(chunk, dtype=np.int16)

        audio = AudioSegment(
            data=audio_data.tobytes(),
            sample_width=audio_data.itemsize,
            frame_rate=sample_rate,
            channels=1  # Assuming mono audio
        )

        segment_number = 0
        while len(audio) > 0:
            if len(audio) > segment_duration * 1000:
                segment = audio[:segment_duration * 1000]
                audio = audio[segment_duration * 1000:]
            else:
                segment = audio
                audio = AudioSegment.silent(duration=0)

            segment_path = os.path.join(temp_dir, f"segment_{segment_number}.wav")
            segment.export(segment_path, format="wav")

            # Process the saved segment to get a transcript (replace with your parse function)
            transcript = parse(segment_path)
            print(transcript)
            transcription += transcript + " "  # Concatenate the transcript to the transcription string with space

            segment_number += 1

    # Remove temporary segment files
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Remove the temporary directory
    shutil.rmtree(temp_dir)

    return transcription

audio_object = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
if audio_object is not None:
    transcription_button = st.button("Transcribe Audio")
    if transcription_button:
        transcription = process_long_audio(audio_object)
        st.write("Transcription:")
        st.write(transcription)

# audio_path = "/content/hindi-loan.mp3"
# transcription = process_long_audio(audio_path)
# print(transcription)

