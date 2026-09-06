import whisper
import time


def transcribe_audio(file_path: str) -> tuple[str, float]:
    """Transcribe audio file to text using Whisper."""
    start = time.time()
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    latency = time.time() - start
    return result["text"], latency
