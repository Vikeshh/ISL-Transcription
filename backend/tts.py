from gtts import gTTS
import io
import time

last_spoken = ""
last_time = 0
COOLDOWN = 1.5

def speak(word: str) -> bytes:
    global last_spoken, last_time
    now = time.time()
    if word == last_spoken and (now - last_time) < COOLDOWN:
        return None
    last_spoken = word
    last_time = now
    tts = gTTS(text=word, lang='en')
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()