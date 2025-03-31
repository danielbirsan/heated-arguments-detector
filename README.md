# 🎷 Audio Argument Detector

Automatically detect heated arguments or tense moments in audio files (MP3/WAV). This Python app uses audio signal analysis, speech-to-text transcription, and emotion detection to locate segments with shouting, high emotion, or overlapping interruptions.

---

##  Features

-  Audio Feature Extraction: Loudness (dBFS), pitch (Hz), and voice overlap
-  Speech-to-Text with Whisper (OpenAI)
-  Emotion Detection with Hugging Face Transformers (e.g. anger)
-  Outputs timestamps of “heated” segments
-  Export results to JSON
-  Chunk-level analysis for debugging

---

##  Requirements

```bash
pip install pydub librosa webrtcvad torch transformers openai-whisper
```

You will also need:
- ffmpeg (for audio processing)
- Python 3.8+

---

##  Usage

```bash
python mp3totext.py \
  --audio your_audio_file.mp3 \
  --chunk_ms 3000 \
  --loudness_thresh -20.0 \
  --pitch_thresh 200.0 \
  --overlap_thresh 0.3 \
  --anger_thresh 0.8 \
  --heated_score_cutoff 4.0 \
  --whisper_model small \
  --whisper_lang en \
  --output_detailed_json detailed_chunks.json \
  --output_segments_json heated_segments.json
```

---

##  How It Works

1. **Chunking**: Audio is split into 3-second chunks.
2. **Audio Feature Analysis**:
   - Loudness (via PyDub)
   - Pitch estimation (via Librosa)
   - Voice Activity Detection (via WebRTC VAD)
3. **Transcription**: Each chunk is transcribed using OpenAI Whisper.
4. **Emotion Detection**: Transcripts are analyzed for anger and negative keywords using Hugging Face’s emotion classification model.
5. **Heated Score Calculation**:
   - +1 for loud audio
   - +1 for high pitch
   - +1 for overlapping speech
   - +1 for anger score > 0.8
   - +1 for presence of “heated” words (e.g., “idiot”, “fight”, etc.)
6. **Merging**: Adjacent heated chunks are merged into full segments and converted to readable timestamps.

---

##  Outputs

- `heated_segments.json`: Final list of detected argument timestamps.
- `detailed_chunks.json`: Debug info for each chunk including transcript, pitch, loudness, and emotion.

---

##  Example Output

```json
[
  {
    "start": "1:49",
    "end": "2:22"
  },
  {
    "start": "3:30",
    "end": "3:47"
  }
]
```

---

##  Applications

- Podcast editing
- Meeting transcription & review
- Conflict detection in customer calls
- Research in discourse/emotion analysis

---

##  Configuration Parameters

| Parameter               | Description                              | Default       |
|------------------------|------------------------------------------|---------------|
| `--chunk_ms`           | Audio chunk size in ms                   | 3000          |
| `--loudness_thresh`    | dBFS threshold for shouting              | -20.0         |
| `--pitch_thresh`       | Pitch threshold in Hz                    | 200.0         |
| `--overlap_thresh`     | Overlap ratio threshold                  | 0.3           |
| `--anger_thresh`       | Anger score threshold                    | 0.8           |
| `--heated_score_cutoff`| Minimum score to consider a chunk heated| 4.0           |

---

##  References

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [PyDub](https://github.com/jiaaro/pydub)
- [Librosa](https://librosa.org/)
- [WebRTC VAD](https://webrtc.org/)

---


> If you like this project, give it a ⭐ and share!
