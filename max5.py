from flask import Flask, request, jsonify, render_template
import sounddevice as sd
import wavio
from transformers import pipeline, AutoModel
import threading
import queue
import os
import logging
import requests
import numpy as np
from datetime import datetime
from gtts import gTTS
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor
import librosa
import time
import subprocess
import torch
import torchaudio

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Whisper pipeline
MODEL_PATH = r"C:\Users\npate\OneDrive\Desktop\Webiste\whisper-marathi-final"
try:
    model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)
    logger.info("AI4Bharat ASR model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load AI4Bharat ASR model: {str(e)}")
    raise

# Initialize translators
mr_to_en_translator = GoogleTranslator(source='mr', target='en')
en_to_mr_translator = GoogleTranslator(source='en', target='mr')

# LM Studio API
LM_STUDIO_API = "http://localhost:1234/v1/chat/completions"

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
audio_queue = queue.Queue(maxsize=100)
is_recording = False
audio_file = "audio.wav"
recording_thread = None
executor = ThreadPoolExecutor(max_workers=2)

# Conversation state
conversation = {
    "chat_log": [{"role": "system", "content": """You are a calling agent assisting a user in reporting a scam. Your task is to:
1. Greet the user and ask what kind of scam they encountered.
2. Collect the following details through natural conversation:
   - Full name
   - Age
   - Gender
   - Full address
   - Pincode
   - Nearest police station
   - Aadhar card or PAN card number
   - Email address(ask the victum to spell the email)
   - Bank name involved in the scam
   - Transaction details (for each transaction: transaction ID, time, date)
3. Ask about additional transactions until the user says 'no more', 'that's all', or 'done'.
4. Once all details are collected and no more transactions are reported, say 'Thank you, I have all the details. Goodbye.' to end the conversation.
5. Be patient, polite, and ask one question at a time. If the user's response is unclear, ask for clarification.
6. Do not repeat the entire list of questions in each response; focus on the next piece of information needed."""}],
    "ended": False
}

# Expanded vocabulary dictionary for scam-related terms in Marathi
VOCABULARY_CORRECTIONS = {
    "स्कॅम": "scam",
    "फसवणूक": "fraud",
    "पैसे गमावले": "lost money",
    "खोटं": "fake",
    "धोका": "danger",
    "फसवलं": "cheated",
    "बँक खातं": "bank account",
    "पैसे काढले": "money withdrawn",
    "खोटी ओळख": "fake identity",
    "फोन घोटाळा": "phone scam",
    "ऑनलाइन फसवणूक": "online fraud",
    "जाळं": "trap",
    "हल्ला": "attack",
    "सुरक्षा भंग": "security breach",
    "पैसे परत": "money back",
    # Add more terms as needed
}

def correct_transcription(transcript):
    """Apply vocabulary corrections to the transcribed text."""
    for wrong, correct in VOCABULARY_CORRECTIONS.items():
        transcript = transcript.replace(wrong, correct)
    return transcript

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_listening', methods=['POST'])
def start_listening():
    global is_recording, recording_thread
    if is_recording:
        return jsonify({'error': 'Already listening'}), 400
    
    with audio_queue.mutex:
        audio_queue.queue.clear()
    
    is_recording = True
    recording_thread = threading.Thread(target=record_audio, daemon=True)
    recording_thread.start()
    return jsonify({'status': 'Listening started'})

@app.route('/api/stop_listening', methods=['POST'])
def stop_listening():
    global is_recording, recording_thread
    if not is_recording:
        return jsonify({'error': 'Not listening'}), 400
    
    is_recording = False
    recording_thread.join(timeout=1)
    return process_transcription()

def record_audio():
    global is_recording
    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', blocksize=4096) as stream:
            while is_recording:
                data, _ = stream.read(4096)
                if not audio_queue.full():
                    audio_queue.put(bytes(data), block=False)
    except Exception as e:
        logger.error(f"Recording error: {str(e)}")
        is_recording = False

def process_audio_chunks():
    audio_chunks = []
    while not audio_queue.empty():
        try:
            audio_chunks.append(audio_queue.get_nowait())
        except queue.Empty:
            break
    return np.frombuffer(b''.join(audio_chunks), dtype=np.int16) if audio_chunks else np.array([])

def process_transcription():
    try:
        audio_array = process_audio_chunks()
        if audio_array.size == 0:
            return generate_response("मला काहीही ऐकू आलं नाही. कृपया पुन्हा प्रयत्न करा.")
        
        # Save audio temporarily
        wavio.write(audio_file, audio_array, SAMPLE_RATE, sampwidth=2)
        
        if os.path.getsize(audio_file) < 1000:
            os.remove(audio_file)
            return generate_response("तुमचं ऑडिओ खूपच छोटं होतं. कृपया किमान २ सेकंद बोलावं.")
        
        # Load audio using the same method as in max.py
        try:
            wav, sr = torchaudio.load(audio_file)
            wav = torch.mean(wav, dim=0, keepdim=True)
        except RuntimeError:
            wav, sr = librosa.load(audio_file, sr=None)
            wav = torch.FloatTensor(wav).unsqueeze(0)

        # Resample if necessary
        target_sample_rate = 16000
        if sr != target_sample_rate:
            if isinstance(wav, torch.Tensor):
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
                wav = resampler(wav)
            else:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sample_rate)
                wav = torch.FloatTensor(wav).unsqueeze(0)

        # Get transcription using both CTC and RNNT methods
        transcript_marathi_ctc = model(wav, "mr", "ctc")
        transcript_marathi_rnnt = model(wav, "mr", "rnnt")
        
        # Use RNNT transcription as it's generally more accurate
        transcript_marathi = transcript_marathi_rnnt.strip().lower()
        
        os.remove(audio_file)
        
        if not transcript_marathi:
            return generate_response("मला तुमचं समजलं नाही. कृपया पुन्हा प्रयत्न करा.")
        
        # Apply vocabulary correction
        transcript_marathi = correct_transcription(transcript_marathi)
        
        transcript_english = mr_to_en_translator.translate(transcript_marathi)
        
        if not transcript_english:
            return generate_response("मला तुमचं नीट समजलं नाही. कृपया पुन्हा स्पष्टपणे सांगा.")
        
        conversation["chat_log"].append({"role": "user", "content": transcript_english})
        response_english = get_ai_response(conversation["chat_log"])
        conversation["chat_log"].append({"role": "assistant", "content": response_english})
        
        response_marathi = en_to_mr_translator.translate(response_english)
        synthesize_speech(response_marathi)
        
        if "goodbye" in response_english.lower():
            conversation["ended"] = True
            save_conversation()
            reset_conversation()
        
        return jsonify({'transcript': transcript_marathi, 'response': response_marathi})
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return generate_response("एक त्रुटी आली. कृपया पुन्हा प्रयत्न करा.")

def get_ai_response(chat_history):
    try:
        response = requests.post(LM_STUDIO_API, json={
            "model": "mistral-7b", "messages": chat_history, "temperature": 0.7, "max_tokens": 50, "stream": False
        }, headers={'Content-Type': 'application/json'}, timeout=5)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"AI response error: {str(e)}")
        return "I couldn't process that. Please try again."

def synthesize_speech(text):
    try:
        tts = gTTS(text=text, lang='mr', slow=False)
        temp_audio_file = f"temp_tts_{int(time.time() * 1000)}.mp3"
        tts.save(temp_audio_file)
        subprocess.Popen(['start', '', temp_audio_file], shell=True)
        time.sleep(1.5)
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")

def save_conversation():
    filename = f"scam_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, 'w') as f:
        for entry in conversation["chat_log"]:
            role = "User" if entry["role"] == "user" else "Agent"
            f.write(f"{role}: {entry['content']}\n")
    logger.info(f"Conversation saved to {filename}")

def reset_conversation():
    global conversation
    conversation = {"chat_log": [{"role": "system", "content": "You are a calling agent assisting..."}], "ended": False}

def generate_response(message):
    synthesize_speech(message)
    return jsonify({'transcript': '', 'response': message})

if __name__ == '__main__':
    app.run(debug=True)