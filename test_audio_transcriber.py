import speech_recognition as sr

def transcribe_audio(audio_file):
    # 音声認識オブジェクトの作成
    recognizer = sr.Recognizer()

    # 音声ファイルの読み込み
    with sr.AudioFile(audio_file) as source:
        # ノイズ削減
        recognizer.adjust_for_ambient_noise(source)
        # 音声の読み取り
        audio = recognizer.record(source)

    try:
        # Google Speech Recognition APIを使用してテキストに変換
        text = recognizer.recognize_google(audio, language="ja-JP")
        print(f"認識されたテキスト: {text}")
        return text
    except sr.UnknownValueError:
        print("音声を認識できませんでした")
    except sr.RequestError as e:
        print(f"Google Speech Recognition APIへのリクエストに失敗しました: {e}")

# 使用例
audio_file = "/Users/wuzhonglin/z/221226_2136.wav"  # 音声ファイルのパスを指定
transcribe_audio(audio_file)