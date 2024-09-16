import pyaudio
import numpy as np
import os
from dotenv import load_dotenv
import openai
from queue import Queue
from threading import Thread
import tempfile
import wave
import io
import re

load_dotenv()

# OpenAI APIキーの設定
openai.api_key=os.getenv("OPENAI_API_KEY")

# 不要なフレーズのブラックリスト
BLACKLIST_PHRASES = [
    "請不吝點贊訂閱轉發打賞支持明鏡與點點欄目",
    "字幕由 Amara.org 社群提供",
    "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
    # 他の不要なフレーズをここに追加
]

# オーディオストリームの設定
CHUNK = 1024 * 5
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisperは16kHzのサンプリングレートを推奨
RECORD_SECONDS = 5  # 3秒ごとに文字起こし

# オーディオデータを格納するキュー
audio_queue = Queue()

def is_valid_transcription(text):
    # ブラックリストのフレーズをチェック
    for phrase in BLACKLIST_PHRASES:
        if phrase in text:
            return False
    
    # 最小文字数チェック（例：10文字未満は無効とする）
    if len(text) < 10:
        return False
    
    # 繰り返しパターンのチェック
    if re.search(r'(.)\1{5,}', text):  # 同じ文字が6回以上連続で現れる場合
        return False
    
    return True

def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

def create_audio_file(audio_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        wf = wave.open(temp_file.name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio_data))
        wf.close()
        return temp_file.name

def transcribe_audio(audio_file_path, expected_language='en'):
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    language=expected_language,
                    prompt=f"This is an {expected_language} audio. Transcribe it accurately without adding any extra phrases."
                )
            
            if is_valid_transcription(transcript.text):
                return transcript.text
            else:
                print(f"無効なトランスクリプション（試行 {attempt + 1}/{max_attempts}）: {transcript.text}")
        except Exception as e:
            print(f"音声認識エラー（試行 {attempt + 1}/{max_attempts}）: {e}")
    
    return None

def recognize_worker():
    while True:
        audio_data = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            audio = audio_queue.get()
            if audio is None:
                return
            audio_data.append(audio)

        # 音声セグメントを一時ファイルとして保存
        audio_file_path = create_audio_file(audio_data)

        # 文字起こし
        text = transcribe_audio(audio_file_path, expected_language='ja')
        if text:
            print(f"認識されたテキスト: {text}")
        else:
            print("有効なテキストを認識できませんでした。")

# メイン処理
if __name__ == "__main__":
    p = pyaudio.PyAudio()

    # Blackholeデバイスのインデックスを見つける
    blackhole_index = None
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if "BlackHole" in device_info["name"]:
            blackhole_index = i
            break

    if blackhole_index is None:
        print("Blackholeデバイスが見つかりませんでした。")
        exit(1)

    # ストリームを開始
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=blackhole_index,
                    frames_per_buffer=CHUNK,
                    stream_callback=audio_callback)

    print("ストリーミング文字起こしを開始します...")

    # 認識ワーカースレッドを開始
    recognize_thread = Thread(target=recognize_worker)
    recognize_thread.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("文字起こしを終了します...")

    # クリーンアップ
    stream.stop_stream()
    stream.close()
    p.terminate()
    audio_queue.put(None)
    recognize_thread.join()