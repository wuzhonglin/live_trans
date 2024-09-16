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

# 音声データを保存するキュー
audio_queue = queue.Queue()

# ChatGPTを使用して翻訳する関数
def translate_text(text, target_language="日本語"):
    response = client.chat.completions.create(model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": f"あなたは翻訳者です。以下のテキストを{target_language}に翻訳してください。"},
        {"role": "user", "content": text}
    ])
    return response.choices[0].message.content

# 音声認識と翻訳を行う関数
def transcribe_and_translate_forever():
    while True:
        # キューから音声データを取得
        audio = np.frombuffer(audio_queue.get(), dtype=np.float32)

        # Whisperで音声認識
        result = model.transcribe(audio)
        original_text = result["text"]
        print(f"元のテキスト: {original_text}")

        # ChatGPTで翻訳
        translated_text = translate_text(original_text)
        print(f"翻訳されたテキスト: {translated_text}")
        print("---")

# 音声認識と翻訳スレッドの開始
threading.Thread(target=transcribe_and_translate_forever, daemon=True).start()

try:
    while True:
        # オーディオストリームからデータを読み取る
        data = stream.read(CHUNK)
        audio_queue.put(data)
except KeyboardInterrupt:
    print("音声認識と翻訳を停止します")

# クリーンアップ
stream.stop_stream()
stream.close()
p.terminate()