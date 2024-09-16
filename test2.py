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
from scipy import signal
from scipy.fft import fft, ifft

# Load environment variables and set up OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load Whisper model
model = whisper.load_model("base")

# Audio stream settings
CHUNK = 48000 * 5  # 5 seconds of data
FORMAT = pyaudio.paFloat32
CHANNELS = 2
RATE = 48000

# Queue for audio data
audio_queue = queue.Queue()

# Noise reduction parameters
NOISE_REDUCE_STRENGTH = 3.0
NOISE_THRESHOLD = 0.1
NOISE_FLOOR = 0.3

def spectral_gate(audio_data, strength=NOISE_REDUCE_STRENGTH, threshold=NOISE_THRESHOLD, noise_floor=NOISE_FLOOR):
    # Convert to frequency domain
    spectrum = fft(audio_data)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    # Calculate noise floor
    noise_floor = np.mean(magnitude) * noise_floor

    # Apply spectral gating
    mask = (magnitude > (noise_floor + threshold))
    magnitude_reduced = np.maximum(magnitude - threshold, noise_floor) * mask

    # Apply strength factor
    magnitude_reduced = (magnitude_reduced ** (1/strength)) * (magnitude ** ((strength-1)/strength))

    # Convert back to time domain
    spectrum_reduced = magnitude_reduced * np.exp(1j * phase)
    audio_reduced = np.real(ifft(spectrum_reduced)).astype(np.float32)

    return audio_reduced

def noise_reduce(audio_data):
    # Convert to NumPy array and make a copy to ensure it's writable
    audio_np = np.frombuffer(audio_data, dtype=np.float32).copy()
    
    print(f"Noise reduce - Initial shape: {audio_np.shape}")
    print(f"Noise reduce - Initial min: {np.min(audio_np)}, max: {np.max(audio_np)}")
    
    # Apply spectral gating
    audio_reduced = spectral_gate(audio_np)
    
    # Check for invalid values
    if np.isnan(audio_np).any() or np.isinf(audio_np).any():
        print("Warning: Invalid values detected in initial audio data")
        audio_np = np.nan_to_num(audio_np)  # Replace NaN and inf with valid numbers
    
    # Normalize audio
    audio_reduced = audio_reduced / np.max(np.abs(audio_reduced))
    
    print(f"Noise reduce - Final shape: {audio_reduced.shape}")
    print(f"Noise reduce - Final min: {np.min(audio_reduced)}, max: {np.max(audio_reduced)}")
    print(f"Noise reduce - Final dtype: {audio_reduced.dtype}")
    
    return audio_reduced

def audio_callback(in_data, frame_count, time_info, status):
    print(f"Callback received {len(in_data)} bytes of data")
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

def save_audio_chunk(audio_data, filename):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(audio_data)
    wf.close()


def process_audio():
    chunk_count = 0
    while True:
        # Get audio data from queue
        audio_data = audio_queue.get()
        
        # Check if audio_data is empty or None
        if audio_data is None or len(audio_data) == 0:
            print(f"Warning: Empty audio data received in chunk {chunk_count}")
            chunk_count += 1
            continue
        
        print(f"Processing chunk {chunk_count} with {len(audio_data)} bytes of data")
        
        try:
            # Apply noise reduction
            audio_reduced = noise_reduce(audio_data)

            # Convert raw bytes to numpy array for initial check
            initial_np = np.frombuffer(audio_data, dtype=np.float32)
            print(f"Initial data - shape: {initial_np.shape}, min: {np.min(initial_np)}, max: {np.max(initial_np)}")
            
            if np.all(initial_np == 0):
                print(f"Warning: All zero data in chunk {chunk_count}")
                chunk_count += 1
                continue
            
            # Apply noise reduction
            audio_reduced = noise_reduce(audio_data)
            
            # Debug: Print audio_reduced information
            print(f"audio_reduced shape: {audio_reduced.shape}")
            print(f"audio_reduced dtype: {audio_reduced.dtype}")
            print(f"audio_reduced min: {np.min(audio_reduced)}, max: {np.max(audio_reduced)}")
            
            # Calculate max amplitude
            max_amplitude = np.max(np.abs(audio_reduced))
            print(f"Calculated max_amplitude: {max_amplitude}")
            
            # Check for NaN or inf in max_amplitude
            if np.isnan(max_amplitude) or np.isinf(max_amplitude):
                print(f"Warning: Invalid max_amplitude detected in chunk {chunk_count}")
                max_amplitude = 0  # Set to 0 if invalid
            
            print(f"Chunk {chunk_count}: Max amplitude: {max_amplitude}")
            
            # Only process if there's significant audio
            if max_amplitude > 0.01:  # Adjust this threshold as needed
                print(f"Processing chunk {chunk_count}")
                
                # Save this chunk as a WAV file for debugging
                filename = f"debug_chunk_{chunk_count}.wav"
                save_audio_chunk(audio_reduced.tobytes(), filename)
                print(f"Saved audio chunk to {filename}")
                
                # Transcribe with Whisper
                try:
                    result = model.transcribe(audio_reduced)
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
        except Exception as e:
            print(f"Error processing audio chunk {chunk_count}: {str(e)}")
            chunk_count += 1


# Get audio stream (run once)
p = pyaudio.PyAudio()

# Print available audio devices
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    print(f"Device {i}: {dev['name']}")

# Assuming BlackHole is device index 2, but you should verify this
BLACKHOLE_INDEX = 2

# Open the stream with more detailed error checking
try:
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=BLACKHOLE_INDEX,
                    stream_callback=audio_callback)
    print(f"Stream opened successfully. Using device index: {BLACKHOLE_INDEX}")
except Exception as e:
    print(f"Error opening stream: {str(e)}")
    # You might want to exit the program here if the stream can't be opened
    exit(1)

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