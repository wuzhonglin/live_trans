import pyaudio
import numpy as np
import os
from dotenv import load_dotenv
import openai
from queue import Queue
from threading import Thread
# import tempfile
from pydub import AudioSegment
import wave

load_dotenv()
# OpenAI APIキーの設定
openai.api_key=os.getenv("OPENAI_API_KEY")

# オーディオストリームの設定
CHUNK = 1024 * 5
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5  # 10秒ごとに文字起こし

# オーディオデータを格納するキュー
audio_queue = Queue()

def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

def save_audio(audio_data, filename):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(audio_data))
    wf.close()

def transcribe_audio(audio_file):
    try:
        with open(audio_file, "rb") as file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1", 
                file=file,
                language="zh"
            )
        return transcript.text
    except Exception as e:
        print(f"音声認識エラー: {e}")
        return None

def recognize_worker():
    # temp_dir = tempfile.mkdtemp()
    temp_dir = '/Users/wuzhonglin/git/live_trans/tmp'
    while True:
        audio_data = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            audio = audio_queue.get()
            if audio is None:
                return
            audio_data.append(audio)

        # 一時的な音声ファイルを保存
        temp_file = os.path.join(temp_dir, "temp_audio.wav")
        save_audio(audio_data, temp_file)

        # 音声ファイルをMP3に変換（OpenAI APIの要件）
        audio = AudioSegment.from_wav(temp_file)
        mp3_file = os.path.join(temp_dir, "temp_audio.mp3")
        audio.export(mp3_file, format="mp3")

        # 文字起こし
        text = transcribe_audio(mp3_file)
        if text:
            print(f"認識されたテキスト: {text}")

        # 一時ファイルの削除
        os.remove(temp_file)
        os.remove(mp3_file)

# メイン処理
if __name__ == "__main__":
    p = pyaudio.PyAudio()

    # Blackholeデバイスのインデックスを見つける
    blackhole_index = None
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        print(device_info["name"])
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

    print("リアルタイム文字起こしを開始します...")

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