
# !pip install pyctcdecode transformers datasets
import tempfile
import streamlit as st
from huggingface_hub import notebook_login
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,Wav2Vec2ProcessorWithLM
import subprocess, os

st.title("Hugging Face Wav2Vec2 Transcription App")


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

def process_long_audio(audio_path, segment_duration=200):
    transcription = ''
    # Open the audio file
    # os.makedirs('/content/temp', exist_ok=True)
    # temp_dir = '/content/temp'
    temp_dir = tempfile.mkdtemp()
    with open(audio_path, 'rb') as audio_file:
        segment_number = 0
        while True:
            segment = audio_file.read(1 * segment_duration * 1000)  # Read segment_duration seconds of audio
            if not segment:
                break  # Break the loop if there's no more audio to process

            segment_path = os.path.join(temp_dir, f"segment_{segment_number}.wav")
            with open(segment_path, 'wb') as segment_file:
                segment_file.write(segment)  # Save the segment as a temporary file

            transcript = parse(segment_path)  # Process the saved segment to get a transcript
            print(transcript)
            transcription += transcript + " "  # Concatenate the transcript to the transcription string with space

            segment_number += 1

    folder_path = temp_dir
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    return transcription


audio_path = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
if audio_path is not None:
    transcription_button = st.button("Transcribe Audio")
    if transcription_button:
        transcription = process_long_audio(audio_path.name)
        st.write("Transcription:")
        st.write(transcription)

# audio_path = "/content/hindi-loan.mp3"
# transcription = process_long_audio(audio_path)
# print(transcription)

