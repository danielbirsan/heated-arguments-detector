import os
import math
import json
import argparse

import numpy as np
import librosa
import webrtcvad

from pydub import AudioSegment
from pydub.utils import make_chunks

# For Hugging Face sentiment/emotion pipeline
from transformers import pipeline

# For Whisper
import torch
import whisper

# 1. Audio Feature Extraction


class AudioFeatureExtractor:
    """
    Extracts low-level audio features from an audio segment, such as:
     - Loudness (dBFS)
     - Average pitch (using librosa)
     - Voice Activity Detection (VAD) for overlap analysis
    """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(2)  # Medium aggressiveness

    def get_loudness_dbfs(self, audio_segment: AudioSegment) -> float:
        """Return average loudness (dBFS) of the segment."""
        return audio_segment.dBFS

    def get_pitch(self, audio_samples: np.ndarray) -> float:
        """
        Estimate average pitch from non-zero values using librosa's piptrack.
        """
        pitches, magnitudes = librosa.piptrack(
            y=audio_samples, sr=self.sample_rate)
        pitches_nonzero = pitches[magnitudes > np.median(magnitudes)]
        if len(pitches_nonzero) == 0:
            return 0.0
        return float(np.mean(pitches_nonzero))

    def get_vad_segments(self, audio_segment: AudioSegment, chunk_ms=30):
        """
        Use WebRTC VAD to determine which 30ms frames contain speech.
        Returns a list of booleans indicating speech frames.
        """
        raw_data = audio_segment.raw_data
        frame_length = int(self.sample_rate * (chunk_ms /
                           1000.0) * 2)  # 16-bit = 2 bytes
        speech_flags = []

        for i in range(0, len(raw_data), frame_length):
            frame = raw_data[i: i + frame_length]
            if len(frame) < frame_length:
                break
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            speech_flags.append(is_speech)

        return speech_flags

    def estimate_overlap(self, speech_flags_chunk1, speech_flags_chunk2) -> float:
        """
        For single-channel audio, we can't truly do multi-speaker overlap here.
        We'll just compare the same flags as a placeholder.
        """
        if not speech_flags_chunk1 or not speech_flags_chunk2:
            return 0.0
        min_len = min(len(speech_flags_chunk1), len(speech_flags_chunk2))
        overlap_count = sum(
            speech_flags_chunk1[i] and speech_flags_chunk2[i] for i in range(min_len)
        )
        return overlap_count / float(min_len)


# 2. Whisper-Based Speech-to-Text

class WhisperTranscriber:
    """
    Transcribes each chunk of audio using OpenAI Whisper.
    For Romanian, pass language='ro'; for English, 'en', etc.
    """

    def __init__(self, model_name="small", language="en"):
        print(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.language = language

    def transcribe(self, audio_segment: AudioSegment) -> str:
        """
        Transcribe a chunk with Whisper.
        Note: short chunks (~3s) can yield poor accuracy.
        """
        # Export chunk to a temporary WAV file
        temp_wav = "temp_chunk_whisper.wav"
        audio_segment.export(temp_wav, format="wav")

        # Perform transcription
        result = self.model.transcribe(temp_wav, language=self.language)

        # Clean up
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

        return result["text"].strip()

# 3. Text Sentiment/Emotion Analysis


class TextEmotionAnalyzer:

    def __init__(self, model_name="bhadresh-savani/distilbert-base-uncased-emotion"):
        print(f"Loading emotion model: {model_name}")
        self.classifier = pipeline(
            "text-classification", model=model_name, top_k=None)

    def analyze(self, text: str) -> dict:
        """
        A dictionary with emotion/sentiment scores. 
        ex: {"anger": 0.78, "sadness": 0.10, ...}
        """
        if not text.strip():
            return {}

        raw_output = self.classifier(text)

        # If the pipeline returns a list-of-lists, flatten it
        if len(raw_output) == 1 and isinstance(raw_output[0], list):
            raw_output = raw_output[0]

        emotions = {}
        for item in raw_output:
            label = item["label"].lower()
            score = item["score"]
            emotions[label] = score
        return emotions

# 4. Main Detector


class HeatedArgumentDetector:
    """
    Combines all signals to produce a "heated score" for each chunk.
    Then merges adjacent heated chunks into final timestamps.
    Outputs both final segments and chunk-level analysis in JSON.
    """

    def __init__(
        self,
        sample_rate=16000,
        chunk_ms=3000,    # chunk size in ms (3 seconds)
        loudness_thresh=-20.0,
        pitch_thresh=200.0,  # rough pitch threshold for "shout" (Hz)
        overlap_thresh=0.3,  # overlap ratio threshold
        anger_thresh=0.8,    # "anger" sentiment threshold
        heated_score_cutoff=4.0,
        whisper_model_name="small",
        whisper_language="en"
    ):
        self.chunk_ms = chunk_ms
        self.loudness_thresh = loudness_thresh
        self.pitch_thresh = pitch_thresh
        self.overlap_thresh = overlap_thresh
        self.anger_thresh = anger_thresh
        self.heated_score_cutoff = heated_score_cutoff

        # Modules
        self.audio_extractor = AudioFeatureExtractor(sample_rate=sample_rate)
        self.whisper_transcriber = WhisperTranscriber(
            model_name=whisper_model_name,
            language=whisper_language
        )
        self.emotion_analyzer = TextEmotionAnalyzer()

    def detect_arguments(self, audio_path: str):

        audio = AudioSegment.from_file(audio_path)
        # Ensure a common sample rate
        audio = audio.set_frame_rate(
            self.audio_extractor.sample_rate).set_channels(1)

        # Create chunks
        chunks = make_chunks(audio, self.chunk_ms)

        results = []
        for i, chunk in enumerate(chunks):
            start_ms = i * self.chunk_ms
            end_ms = start_ms + len(chunk)

            # (A) Loudness
            loudness = self.audio_extractor.get_loudness_dbfs(chunk)

            # (B) Pitch
            samples = np.array(chunk.get_array_of_samples()).astype(np.float32)
            pitch_val = self.audio_extractor.get_pitch(
                samples / (2**15))  # scale to -1..1

            # (C) Overlap
            speech_flags = self.audio_extractor.get_vad_segments(
                chunk, chunk_ms=30)
            overlap = self.audio_extractor.estimate_overlap(
                speech_flags, speech_flags)

            # (D) Whisper transcript
            transcript = self.whisper_transcriber.transcribe(chunk)

            # (E) Emotion analysis
            emotion_scores = self.emotion_analyzer.analyze(transcript)
            anger_score = emotion_scores.get("anger", 0.0)

            # (F) Weighted "heated" scoring
            heated_score = 0

            # Loudness
            if loudness >= self.loudness_thresh:
                heated_score += 1

            # Pitch
            if pitch_val >= self.pitch_thresh:
                heated_score += 1

            # Overlap
            if overlap >= self.overlap_thresh:
                heated_score += 1

            # Anger
            if anger_score >= self.anger_thresh:
                heated_score += 1

            # Negative/Heated keywords
            heated_keywords = ["fight", "shut up",
                               "idiot", "stupid", "mad", "hate"]  # Add more keywords as needed to imporovre accuracy
            if any(kw in transcript.lower() for kw in heated_keywords):
                heated_score += 1

            # Collect chunk data
            chunk_info = {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "loudness_dbfs": loudness,
                "pitch_hz": pitch_val,
                "overlap_ratio": overlap,
                "transcript": transcript,
                "anger_score": anger_score,
                "heated_score": heated_score
            }
            results.append(chunk_info)

        # Merge consecutive heated chunks
        final_segments = self._merge_heated_chunks(results)

        return final_segments, results

    def _merge_heated_chunks(self, chunk_info_list):
        """
        Merge consecutive chunks that are above the heated_score_cutoff
        into continuous segments.
        """
        heated_segments = []
        current_start = None
        current_end = None

        for info in chunk_info_list:
            if info["heated_score"] >= self.heated_score_cutoff:
                if current_start is None:
                    # start a new segment
                    current_start = info["start_ms"]
                current_end = info["end_ms"]
            else:
                # if we were in a segment, close it
                if current_start is not None:
                    heated_segments.append((current_start, current_end))
                    current_start = None
                    current_end = None

        if current_start is not None:
            heated_segments.append((current_start, current_end))

        # Convert ms to a readable timestamp
        merged_results = []
        for seg in heated_segments:
            start_ts = self._ms_to_timestamp(seg[0])
            end_ts = self._ms_to_timestamp(seg[1])
            merged_results.append({"start": start_ts, "end": end_ts})
        return merged_results

    def _ms_to_timestamp(self, ms: int) -> str:
        secs = ms // 1000
        mm = secs // 60
        ss = secs % 60
        return f"{mm}:{ss:02d}"


# 5. Main Script

def main():
    parser = argparse.ArgumentParser(
        description="Detect heated argument segments from an audio file using chunk-based Whisper STT."
    )
    parser.add_argument("--audio", default="sursa.mp3",
                        help="Path to audio file")
    parser.add_argument("--chunk_ms", type=int, default=3000,
                        help="Chunk size in milliseconds")
    parser.add_argument("--loudness_thresh", type=float,
                        default=-20.0, help="dBFS threshold for loudness")
    parser.add_argument("--pitch_thresh", type=float, default=200.0,
                        help="Hz threshold for pitch (approx for shouting)")
    parser.add_argument("--overlap_thresh", type=float,
                        default=0.3, help="Overlap ratio threshold")
    parser.add_argument("--anger_thresh", type=float,
                        default=0.8, help="Anger probability threshold")
    parser.add_argument("--heated_score_cutoff", type=float, default=3.0,
                        help="Minimum 'heated_score' for a chunk to be considered heated")
    parser.add_argument("--whisper_model", type=str, default="small",
                        help="Which Whisper model (tiny, base, small, medium, large)")
    parser.add_argument("--whisper_lang", type=str, default="en",
                        help="Whisper language code (e.g., 'en', 'ro')")
    # JSON output arguments
    parser.add_argument("--output_detailed_json", default="detailed_chunks.json",
                        help="Path to save chunk-level JSON data")
    parser.add_argument("--output_segments_json", default="heated_segments.json",
                        help="Path to save final heated segments JSON")

    args = parser.parse_args()

    detector = HeatedArgumentDetector(
        chunk_ms=args.chunk_ms,
        loudness_thresh=args.loudness_thresh,
        pitch_thresh=args.pitch_thresh,
        overlap_thresh=args.overlap_thresh,
        anger_thresh=args.anger_thresh,
        heated_score_cutoff=args.heated_score_cutoff,
        whisper_model_name=args.whisper_model,
        whisper_language=args.whisper_lang
    )

    final_segments, chunk_analysis = detector.detect_arguments(args.audio)
    final_segments = [seg for seg in final_segments if ((int(seg["end"].split(":")[0]) * 60 +
                      int(seg["end"].split(":")[1])) - (int(seg["start"].split(":")[0]) *
                                                        60 + int(seg["start"].split(":")[1])) >= 15)]

    # Print final segments over 10s in length
    print("==== Detected Heated Segments (>= 10s) ====")
    for seg in final_segments:
        start_sec = int(seg["start"].split(":")[0]) * \
            60 + int(seg["start"].split(":")[1])
        end_sec = int(seg["end"].split(":")[0]) * 60 + \
            int(seg["end"].split(":")[1])
        if (end_sec - start_sec) > 15:
            print(f"Start: {seg['start']}, End: {seg['end']}")

    print("\n==== Detailed Chunk Analysis (Debug) ====")
    for c in chunk_analysis:
        start_str = detector._ms_to_timestamp(c['start_ms'])
        end_str = detector._ms_to_timestamp(c['end_ms'])
        print(f"[{start_str} - {end_str}] "
              f"Loudness: {c['loudness_dbfs']:.1f} dBFS, "
              f"Pitch: {c['pitch_hz']:.1f} Hz, "
              f"Overlap: {c['overlap_ratio']:.2f}, "
              f"Anger: {c['anger_score']:.2f}, "
              f"Score: {c['heated_score']}")
        if c["transcript"]:
            print(f"  Transcript: {c['transcript']}")
        print("")

    # Save the final heated segments to JSON
    with open(args.output_segments_json, "w", encoding="utf-8") as f:
        json.dump(final_segments, f, indent=2, ensure_ascii=False)

    # Save the detailed chunk analysis to JSON
    with open(args.output_detailed_json, "w", encoding="utf-8") as f:
        json.dump(chunk_analysis, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed chunk data saved to: {args.output_detailed_json}")
    print(f"Heated segments saved to: {args.output_segments_json}")


if __name__ == "__main__":
    main()

print("Finished")
