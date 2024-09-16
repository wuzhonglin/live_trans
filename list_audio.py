import pyaudio

def list_audio_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:  # Input devices only
            print(f"Input Device id {i} - {device_info.get('name')}")
            print(f"  Max Input Channels: {device_info.get('maxInputChannels')}")
            print(f"  Default Sample Rate: {device_info.get('defaultSampleRate')}")
            print(f"  Is Default Input: {p.get_default_input_device_info()['index'] == i}")
            print()
    p.terminate()

# Display device list
list_audio_devices()