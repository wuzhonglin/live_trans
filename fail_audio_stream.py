import pyaudio
import os
from dotenv import load_dotenv
import numpy as np
import whisper
import queue
import threading
from openai import OpenAI
import wave
import time

# Load environment variables and set up OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load Whisper model
model = whisper.load_model("base")

# Audio stream settings
CHUNK = 48000 * 5  # 1 second of data
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 48000

# Queue for audio data
audio_queue = queue.Queue()

def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

# Get audio stream (run once)
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=2,  # BlackHole device index
                stream_callback=audio_callback)

def save_audio_chunk(audio_data, filename):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(audio_data)
    wf.close()

def process_audio():
    chunk_count = 0
    while True:
        # Get audio data from queue
        audio_data = audio_queue.get()
        
        # Convert to NumPy array
        audio_np = np.frombuffer(audio_data, dtype=np.float32)
        
        # Debug: Print max amplitude
        max_amplitude = np.max(np.abs(audio_np))
        print(f"Chunk {chunk_count}: Max amplitude: {max_amplitude}")
        
        # Only process if there's significant audio
        if max_amplitude > 0.01:  # Adjust this threshold as needed
            print(f"Processing chunk {chunk_count}")
            
            # Save this chunk as a WAV file for debugging
            filename = f"debug_chunk_{chunk_count}.wav"
            save_audio_chunk(audio_data, filename)
            print(f"Saved audio chunk to {filename}")
            
            # Transcribe with Whisper
            try:
                result = model.transcribe(audio_np)
                print(f"Whisper result for chunk {chunk_count}: {result}")
                
                # Output results
                if result['text']:
                    print("Whisper output:")
                    print(f"Text: {result['text']}")
                    print(f"Language: {result['language']}")
                    print("--------------")
                else:
                    print(f"No text output for chunk {chunk_count}")
            except Exception as e:
                print(f"Error in Whisper processing for chunk {chunk_count}: {str(e)}")
        else:
            print(f"No significant audio detected in chunk {chunk_count}")
        
        chunk_count += 1

# Start audio processing thread
threading.Thread(target=process_audio, daemon=True).start()

# Main loop
stream.start_stream()
try:
    print("Listening... Press Ctrl+C to stop.")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping audio recognition")

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()