import pyaudio
import os
from dotenv import load_dotenv
import numpy as np
import whisper
import queue
import threading
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load environment variables from .env file
load_dotenv()

# OpenAI APIキーの設定

# Whisperモデルの読み込み
model = whisper.load_model("base")

# オーディオストリームの設定
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000

# オーディオストリームの取得
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=1)  # BlackHoleのデバイスインデックスを指定

model = whisper.load_model("base")

def process_audio(audio_data):
    # NumPy配列に変換
    audio_np = np.frombuffer(audio_data, dtype=np.float32)
    
    # Whisperで文字起こし
    result = model.transcribe(audio_np)
    
    # 結果の詳細を出力
    print("Whisper出力の詳細:")
    print(f"テキスト: {result['text']}")
    print(f"言語: {result['language']}")
    print(f"セグメント数: {len(result['segments'])}")
    
    # セグメントの詳細を出力（最初の3つのみ）
    for i, segment in enumerate(result['segments'][:3]):
        print(f"セグメント {i}:")
        print(f"  開始時間: {segment['start']:.2f}秒")
        print(f"  終了時間: {segment['end']:.2f}秒")
        print(f"  テキスト: {segment['text']}")
    
    return result['text']


try:
    while True:
        # オーディオストリームからデータを読み取る
        audio_data = stream.read(CHUNK)
        text = process_audio(audio_data)
        print("--------------")
        print(text)
        
except KeyboardInterrupt:
    print("音声認識と翻訳を停止します")

# クリーンアップ
stream.stop_stream()
stream.close()
p.terminate()