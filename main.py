import torch
import io
import os
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from pydub import AudioSegment

HF_TOKEN = "your_token"
AUDIO_FILE = "audio/ielts.wav"
WHISPER_SIZE = "large-v3"
LANGUAGE = 'en'
DEVICE = "cuda"
TRANSCRIPT_FILE = "output/transcript.txt"


def main():
    print("\n [0/5] Converting audio to proper WAV format...")

    try:
        audio = AudioSegment.from_file(AUDIO_FILE)

        temp_wav = "temp_converted.wav"
        audio.export(
            temp_wav,
            format="wav",
            parameters=["-ar", "16000", "-ac", "1"]  # 16kHz, mono
        )

        processing_file = temp_wav
        print(f"✅ Converted to proper WAV: {temp_wav}")

    except Exception as e:
        print(f"❌ Error converting audio: {e}")
        return


    print("\n [1/5] Loading Models")

    # Load Pyannote
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN
    ).to(torch.device(DEVICE))

    # Load Whisper
    whisper_model = WhisperModel(WHISPER_SIZE, device=DEVICE, compute_type="float16")

    print(f" [2/5] Scanning '{processing_file}' for speakers...")

    try:
        diarization_result = diarization_pipeline(processing_file)
    except Exception as e:
        print(f" Diarization failed: {e}")
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        return

    print("  [3/5] Loading audio into RAM for cutting...")

    try:
        full_audio = AudioSegment.from_file(AUDIO_FILE)
        print(f" Loaded audio: {len(full_audio) / 1000:.1f} seconds, {full_audio.frame_rate}Hz")
    except Exception as e:
        print(f" Error loading audio: {e}")
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        return

    print(f"\n [4/5] Starting Serial Transcription (The Cutter Method)\n")
    print("=" * 80)
    print(f"{'TIME':<15} | {'SPEAKER':<12} | {'TEXT'}")
    print("=" * 80)

    # Access the speaker_diarization attribute from DiarizeOutput
    diarization = diarization_result.speaker_diarization

    with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:

        f.write("=" * 80 + "\n")
        f.write(f"{'TIME':<15} | {'SPEAKER':<12} | {'TEXT'}\n")
        f.write("=" * 80 + "\n")

        # Now iterate through the annotation
        for segment, track, speaker in diarization.itertracks(yield_label=True):

            # Convert seconds to milliseconds
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)

            # Add safety buffer (100ms before, 100ms after)
            start_ms = max(0, start_ms - 100)
            end_ms = min(len(full_audio), end_ms + 100)

            # Cut the audio chunk
            audio_chunk = full_audio[start_ms:end_ms]

            # Export to BytesIO buffer for Whisper
            buffer = io.BytesIO()
            audio_chunk.export(buffer, format="wav")
            buffer.seek(0)

            # Transcribe the chunk
            try:
                segments_iter, info = whisper_model.transcribe(
                    buffer,
                    language=LANGUAGE,
                    beam_size=10,
                    condition_on_previous_text=False,

                )

                # Collect all text from segments
                text = "".join([s.text for s in segments_iter]).strip()

                # Print result
                if text:
                    timestamp = f"{segment.start:.1f}s-{segment.end:.1f}s"
                    line = f"{timestamp:<15} | {speaker:<12} | {text}"
                    print(line)
                    f.write(line + "\n")  # 👈 ADDED

            except Exception as e:
                print(f"  Transcription error at {segment.start:.1f}s: {e}")
                continue

        f.write("=" * 80 + "\n")

    print("=" * 80)
    print(f"\n Transcript saved to: {TRANSCRIPT_FILE}")
    print("\n [5/5] DONE!.")

    # Clean up temporary file
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
        print(f" Cleaned up temporary file: {temp_wav}")


if __name__ == "__main__":
    main()