import librosa
import numpy as np
from pydub import AudioSegment
import whisper
from typing import List, Tuple, Dict
import json
import os

# --- Constants and Music Theory Data ---
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]  # Intervals for major scale
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]  # Intervals for natural minor scale

# Common chord templates (1 for root, 0.8 for major third/fifth, 0.5 for optional notes)
CHORD_TEMPLATES = {
    # Major chords
    'C': [1, 0, 0, 0, 0.8, 0, 0, 0.8, 0, 0, 0, 0],
    'C#': [0, 1, 0, 0, 0, 0.8, 0, 0, 0.8, 0, 0, 0],
    'D': [0, 0, 1, 0, 0, 0, 0.8, 0, 0, 0.8, 0, 0],
    'D#': [0, 0, 0, 1, 0, 0, 0, 0.8, 0, 0, 0.8, 0],
    'E': [0, 0, 0, 0, 1, 0, 0, 0, 0.8, 0, 0, 0.8],
    'F': [0.8, 0, 0, 0, 0, 1, 0, 0, 0, 0.8, 0, 0],
    'F#': [0, 0.8, 0, 0, 0, 0, 1, 0, 0, 0, 0.8, 0],
    'G': [0, 0, 0.8, 0, 0, 0, 0, 1, 0, 0, 0, 0.8],
    'G#': [0.8, 0, 0, 0.8, 0, 0, 0, 0, 1, 0, 0, 0],
    'A': [0, 0.8, 0, 0, 0.8, 0, 0, 0, 0, 1, 0, 0],
    'A#': [0, 0, 0.8, 0, 0, 0.8, 0, 0, 0, 0, 1, 0],
    'B': [0, 0, 0, 0.8, 0, 0, 0.8, 0, 0, 0, 0, 1],
    
    # Minor chords (adjusted third)
    'Cm': [1, 0, 0, 0.8, 0, 0, 0, 0.8, 0, 0, 0, 0],
    'C#m': [0, 1, 0, 0, 0.8, 0, 0, 0, 0.8, 0, 0, 0],
    'Dm': [0, 0, 1, 0, 0, 0.8, 0, 0, 0, 0.8, 0, 0],
    'D#m': [0, 0, 0, 1, 0, 0, 0.8, 0, 0, 0, 0.8, 0],
    'Em': [0, 0, 0, 0, 1, 0, 0, 0.8, 0, 0, 0, 0.8],
    'Fm': [0.8, 0, 0, 0, 0, 1, 0, 0, 0.8, 0, 0, 0],
    'F#m': [0, 0.8, 0, 0, 0, 0, 1, 0, 0, 0.8, 0, 0],
    'Gm': [0, 0, 0.8, 0, 0, 0, 0, 1, 0, 0, 0.8, 0],
    'G#m': [0.8, 0, 0, 0.8, 0, 0, 0, 0, 1, 0, 0, 0],
    'Am': [0, 0.8, 0, 0, 0.8, 0, 0, 0, 0, 1, 0, 0],
    'A#m': [0, 0, 0.8, 0, 0, 0.8, 0, 0, 0, 0, 1, 0],
    'Bm': [0, 0, 0, 0.8, 0, 0, 0.8, 0, 0, 0, 0, 1],
}

class ChordDetector:
    def __init__(self, audio_path: str):
        self.audio_path = audio_path
        self.y, self.sr = librosa.load(audio_path, sr=None)
        self.chroma = None
        self.beats = None
        self.beat_times = None
        self.key = None
        
    def analyze(self):
        """
        Perform complete musical analysis including:
        - Beat detection
        - Key detection
        - Chord detection with musical context
        """
        print("[INFO] Starting musical analysis...")
        
        # Compute chromagram for the entire track
        self.chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr)
        
        # Detect beats
        print("[INFO] Detecting beats...")
        tempo, self.beats = librosa.beat.beat_track(y=self.y, sr=self.sr)
        self.beat_times = librosa.frames_to_time(self.beats, sr=self.sr)
        
        # Detect musical key
        print("[INFO] Detecting musical key...")
        self.key = self._detect_key()
        
        # Detect chords with musical context
        print("[INFO] Detecting chords...")
        chords = self._detect_chords_with_context()
        
        return chords
    
    def _detect_key(self) -> str:
        """
        Detect the musical key of the track using chromagram analysis
        """
        # Average chroma over time
        avg_chroma = np.mean(self.chroma, axis=1)
        
        # Correlate with major and minor scales
        max_correlation = -np.inf
        detected_key = None
        
        for root in range(12):
            # Test major key
            major_corr = self._key_correlation(avg_chroma, root, MAJOR_SCALE)
            minor_corr = self._key_correlation(avg_chroma, root, MINOR_SCALE)
            
            if major_corr > max_correlation:
                max_correlation = major_corr
                detected_key = librosa.midi_to_note(root + 60, octave=False)
            if minor_corr > max_correlation:
                max_correlation = minor_corr
                detected_key = f"{librosa.midi_to_note(root + 60, octave=False)}m"
        
        return detected_key
    
    def _key_correlation(self, chroma: np.ndarray, root: int, scale: List[int]) -> float:
        """
        Calculate correlation between chromagram and a key template
        """
        template = np.zeros(12)
        for interval in scale:
            template[(root + interval) % 12] = 1
        return np.correlate(chroma, template)[0]
    
    def _detect_chords_with_context(self) -> List[Tuple[float, str, float]]:
        """
        Detect chords using beat-synchronized chromagram and musical context
        Returns: List of (time, chord, confidence)
        """
        chords = []
        prev_chord = None
        
        for i in range(len(self.beats) - 1):
            start_frame = self.beats[i]
            end_frame = self.beats[i + 1]
            
            # Get average chroma for this beat segment
            segment_chroma = np.mean(self.chroma[:, start_frame:end_frame], axis=1)
            
            # Find best matching chord
            best_chord = None
            best_score = -np.inf
            
            for chord_name, template in CHORD_TEMPLATES.items():
                # Calculate correlation
                score = np.correlate(segment_chroma, template)[0]
                
                # Apply musical context rules
                if prev_chord is not None:
                    # Favor chord changes that make musical sense
                    if self._is_valid_progression(prev_chord, chord_name):
                        score *= 1.2
                
                if score > best_score:
                    best_score = score
                    best_chord = chord_name
            
            # Only keep chords with good confidence
            if best_score > 0.5:  # Confidence threshold
                chords.append((self.beat_times[i], best_chord, best_score))
                prev_chord = best_chord
        
        return self._smooth_chord_progression(chords)
    
    def _is_valid_progression(self, chord1: str, chord2: str) -> bool:
        """
        Check if chord progression makes musical sense
        """
        # Simple implementation - could be expanded with more music theory
        if chord1 == chord2:
            return True
        return True
    
    def _smooth_chord_progression(self, 
                                chords: List[Tuple[float, str, float]]) -> List[Tuple[float, str, float]]:
        """
        Smooth detected chords to remove unlikely rapid changes
        """
        if len(chords) < 3:
            return chords
            
        smoothed = [chords[0]]  # Keep first chord
        
        # Sliding window smoothing
        window_size = 3
        for i in range(1, len(chords) - 1):
            window = chords[i-1:i+2]
            
            # If middle chord differs from both neighbors and has lower confidence,
            # replace it with the higher confidence neighbor
            if (window[1][1] != window[0][1] and 
                window[1][1] != window[2][1] and
                window[1][2] < max(window[0][2], window[2][2])):
                
                # Choose the neighbor with higher confidence
                if window[0][2] > window[2][2]:
                    smoothed.append((window[1][0], window[0][1], window[0][2]))
                else:
                    smoothed.append((window[1][0], window[2][1], window[2][2]))
            else:
                smoothed.append(window[1])
        
        smoothed.append(chords[-1])  # Keep last chord
        return smoothed

def create_output_folder(song_name: str) -> str:
    """Create output folder for the song if it doesn't exist"""
    # Remove file extension and create folder name
    folder_name = os.path.splitext(song_name)[0]
    output_dir = f"output_{folder_name}"
    
    # Create folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return output_dir

def preprocess_audio(audio_path: str, output_dir: str) -> str:
    """
    Convert audio to WAV format optimized for analysis
    """
    print("[INFO] Preprocessing audio...")
    output_path = os.path.join(output_dir, "processed_audio.wav")
    
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1)        # Convert to mono
    audio = audio.set_frame_rate(22050)  # Standard rate for music analysis
    audio.export(output_path, format="wav")
    return output_path

def transcribe_lyrics(audio_path: str, model_size: str = "base") -> dict:
    """
    Transcribe lyrics using Whisper with word-level timestamps
    """
    print("[INFO] Transcribing lyrics...")
    model = whisper.load_model(model_size)
    
    # Use language detection for better results
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    
    # Detect the language
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    detected_lang = max(probs, key=probs.get)
    
    # Transcribe with word timestamps
    result = model.transcribe(
        audio_path,
        language=detected_lang,
        word_timestamps=True,  # Enable word timestamps
        verbose=False
    )
    
    print(f"[INFO] Detected language: {detected_lang}")
    print("[INFO] Transcription completed!")
    return result

def align_chords_and_lyrics(chords: List[Tuple[float, str, float]], 
                          segments_data: List[dict],
                          output_dir: str) -> None:
    """
    Align detected chords with lyrics, creating a more musical output
    """
    print("[INFO] Aligning chords with lyrics...")
    output_file = os.path.join(output_dir, "music_chord_output.txt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Music Chord Detection Output\n")
        f.write("=========================\n\n")
        
        for segment in segments_data:
            if not 'words' in segment or not segment['words']:
                continue
                
            # Find chords that fall within this segment
            segment_start = segment['start']
            segment_end = segment['end']
            
            # Get all chords in this segment
            segment_chords = []
            for t, c, conf in chords:
                if segment_start <= t < segment_end:
                    segment_chords.append((t - segment_start, c))  # Relative time
            
            # If no chords found for segment, use nearest chord
            if not segment_chords:
                nearest_time = min([(abs(t - segment_start), t, c) 
                                  for t, c, _ in chords])[1:]
                segment_chords = [(0.0, nearest_time[1])]
            
            # Format output
            chord_line = ""
            lyric_line = ""
            
            # First, place the chords with spacing
            last_pos = 0
            scale_factor = 3  # Characters per second for spacing
            
            for chord_time, chord_name in segment_chords:
                pos = int(chord_time * scale_factor)
                if pos > last_pos:
                    chord_line += " " * (pos - last_pos)
                chord_line += chord_name + " "  # Add space after chord
                last_pos = pos + len(chord_name) + 1
            
            # Then, place the lyrics with proper word spacing
            last_pos = 0
            for word, time in segment['words']:
                pos = int((time - segment_start) * scale_factor)
                if pos > last_pos:
                    lyric_line += " " * (pos - last_pos)
                lyric_line += word + " "  # Add space between words
                last_pos = pos + len(word) + 1
            
            # Clean up the lines
            chord_line = chord_line.rstrip()
            lyric_line = lyric_line.rstrip()
            
            # Make sure both lines are same length
            max_len = max(len(chord_line), len(lyric_line))
            chord_line = chord_line.ljust(max_len)
            lyric_line = lyric_line.ljust(max_len)
            
            # Write the aligned lines with timing
            f.write(f"[{segment_start:.2f} - {segment_end:.2f}]\n")
            f.write(chord_line + "\n")
            f.write(lyric_line + "\n\n")

        f.write("\n[INFO] End of chord-lyric alignment.\n")

def main(audio_path: str):
    """
    Main execution function
    """
    # Create output folder
    output_dir = create_output_folder(os.path.basename(audio_path))
    print(f"[INFO] Output will be saved in: {output_dir}")
    
    # Step 1: Preprocess audio
    processed_audio = preprocess_audio(audio_path, output_dir)
    
    # Step 2: Initialize chord detector and analyze
    chord_detector = ChordDetector(processed_audio)
    chords = chord_detector.analyze()
    
    # Save raw chord data
    chord_json = os.path.join(output_dir, "music_chord_data.json")
    with open(chord_json, "w") as f:
        json.dump([(float(t), c, float(conf)) for t, c, conf in chords], f, indent=2)
    
    # Step 3: Transcribe lyrics with word timestamps
    transcription = transcribe_lyrics(processed_audio, model_size="base")
    
    # Step 4: Process transcription into segments with word timing
    segments_data = []
    for seg in transcription["segments"]:
        # Extract word timing information
        words = []
        if "words" in seg:
            for word_info in seg["words"]:
                # Handle different Whisper API versions
                if isinstance(word_info, dict):
                    word = word_info.get("word", "").strip()
                    start = word_info.get("start", 0.0)
                else:
                    word = str(word_info).strip()
                    start = seg["start"]  # Fallback to segment start time
                
                if word:  # Only add non-empty words
                    words.append((word, start))
        
        segments_data.append({
            'start': seg['start'],
            'end': seg['end'],
            'text': seg['text'],
            'words': words
        })
    
    # Step 5: Align chords with lyrics
    align_chords_and_lyrics(chords, segments_data, output_dir)
    
    print(f"[INFO] Analysis complete! Check {output_dir} folder for results.")

if __name__ == "__main__":
    input_audio = "Your_Music.mp3"  # Replace with your audio file
    main(input_audio) 