import pyaudio
import numpy as np
import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
from queue import Queue
from threading import Thread
import tempfile
import wave
import io
import re

load_dotenv()

# OpenAI APIキーの設定
openai.api_key=os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 不要なフレーズのブラックリスト
BLACKLIST_PHRASES = [
    "請不吝點贊訂閱轉發打賞支持明鏡與點點欄目",
    "字幕由 Amara.org 社群提供",
    "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
    "准确地抄写，不添加任何额外的短语。",
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

def transcribe_audio(audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                language="en"
            )
        return transcript.text
    except Exception as e:
        print(f"音声認識エラー: {e}")
        return None
    finally:
        os.remove(audio_file_path)


def translate_text(text, target_language="日本語"):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a translator. Translate the following English text to {target_language}."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"翻訳エラー: {e}")
        return None


def process_audio(audio_file_path, target_language="zh"):
    # 音声認識
    transcribed_text = transcribe_audio(audio_file_path)
    if not transcribed_text:
        return None, None

    # 翻訳
    translated_text = translate_text(transcribed_text, target_language)

    return transcribed_text, translated_text


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
        target_language = 'zh'

        # 音声認識と翻訳
        original_text, translated_text = process_audio(audio_file_path,target_language)

        if original_text and translated_text:
            print(f"原文 (英語): {original_text}")
            print(f"翻訳後 ({target_language}): {translated_text}")
        else:
            print("音声認識または翻訳に失敗しました。")


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