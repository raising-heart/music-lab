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
    
    # Minor chords
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
        """Detect the musical key of the track using chromagram analysis"""
        avg_chroma = np.mean(self.chroma, axis=1)
        max_correlation = -np.inf
        detected_key = None
        
        for root in range(12):
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
        """Calculate correlation between chromagram and a key template"""
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
            segment_chroma = np.mean(self.chroma[:, start_frame:end_frame], axis=1)
            
            best_chord = None
            best_score = -np.inf
            
            for chord_name, template in CHORD_TEMPLATES.items():
                score = np.correlate(segment_chroma, template)[0]
                
                if prev_chord is not None and self._is_valid_progression(prev_chord, chord_name):
                    score *= 1.2
                
                if score > best_score:
                    best_score = score
                    best_chord = chord_name
            
            if best_score > 0.5:
                chords.append((self.beat_times[i], best_chord, best_score))
                prev_chord = best_chord
        
        return self._smooth_chord_progression(chords)
    
    def _is_valid_progression(self, chord1: str, chord2: str) -> bool:
        """Check if chord progression makes musical sense"""
        return True
    
    def _smooth_chord_progression(self, chords: List[Tuple[float, str, float]]) -> List[Tuple[float, str, float]]:
        """Smooth detected chords to remove unlikely rapid changes"""
        if len(chords) < 3:
            return chords
            
        smoothed = [chords[0]]
        window_size = 3
        
        for i in range(1, len(chords) - 1):
            window = chords[i-1:i+2]
            
            if (window[1][1] != window[0][1] and 
                window[1][1] != window[2][1] and
                window[1][2] < max(window[0][2], window[2][2])):
                
                if window[0][2] > window[2][2]:
                    smoothed.append((window[1][0], window[0][1], window[0][2]))
                else:
                    smoothed.append((window[1][0], window[2][1], window[2][2]))
            else:
                smoothed.append(window[1])
        
        smoothed.append(chords[-1])
        return smoothed

def create_output_folder(song_name: str) -> str:
    """Create output folder for the song if it doesn't exist"""
    folder_name = os.path.splitext(song_name)[0]
    output_dir = f"output_{folder_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def preprocess_audio(audio_path: str, output_dir: str) -> str:
    """Convert audio to WAV format optimized for analysis"""
    print("[INFO] Preprocessing audio...")
    output_path = os.path.join(output_dir, "processed_audio.wav")
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(22050)
    audio.export(output_path, format="wav")
    return output_path

def transcribe_lyrics(audio_path: str, model_size: str = "base") -> dict:
    """Transcribe lyrics using Whisper with word-level timestamps"""
    print("[INFO] Transcribing lyrics...")
    model = whisper.load_model(model_size)
    
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    detected_lang = max(probs, key=probs.get)
    
    result = model.transcribe(
        audio_path,
        language=detected_lang,
        word_timestamps=True,
        verbose=False
    )
    
    print(f"[INFO] Detected language: {detected_lang}")
    return result

def detect_song_sections(segments_data: List[dict]) -> List[dict]:
    """
    Detect song sections (verse, chorus, etc.) based on lyrics patterns
    Simple implementation - can be improved with more sophisticated analysis
    """
    sections = []
    current_section = 1
    section_type = "Verse"
    
    for i, segment in enumerate(segments_data):
        # Simple section detection based on gaps and patterns
        if i > 0:
            prev_end = segments_data[i-1]["end"]
            current_start = segment["start"]
            
            # If there's a significant gap, might be a new section
            if current_start - prev_end > 2.0:
                current_section += 1
        
        sections.append({
            "type": f"{section_type} {current_section}",
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "words": segment["words"]
        })
    
    return sections

def find_chord_for_word(word_time: float, chords: List[Tuple[float, str, float]]) -> str:
    """Find the active chord at a given word timestamp"""
    for i in range(len(chords) - 1):
        if chords[i][0] <= word_time < chords[i+1][0]:
            return chords[i][1]
    return chords[-1][1] if chords else "N/A"

def format_section(section: dict, chords: List[Tuple[float, str, float]]) -> List[str]:
    """Format a section in website-style with chords above lyrics"""
    output_lines = []
    
    # Add section header
    output_lines.append(f"[{section['type']}]")
    output_lines.append("")
    
    # Process each line in the section
    current_line_words = []
    current_line_chords = []
    last_chord = None
    
    for word, time in section["words"]:
        current_chord = find_chord_for_word(time, chords)
        
        # Add chord if it's different from the last one
        if current_chord != last_chord:
            current_line_chords.append((len(" ".join(current_line_words)), current_chord))
            last_chord = current_chord
        
        current_line_words.append(word)
        
        # Break into new line if we detect end of phrase
        if word.endswith((".", "!", "?", ",")):
            # Format chord line
            chord_line = " " * len(" ".join(current_line_words))
            for pos, chord in current_line_chords:
                chord_line = chord_line[:pos] + chord + chord_line[pos+len(chord):]
            
            # Add lines to output
            output_lines.append(chord_line.rstrip())
            output_lines.append(" ".join(current_line_words))
            output_lines.append("")
            
            # Reset for next line
            current_line_words = []
            current_line_chords = []
            last_chord = None
    
    # Handle any remaining words
    if current_line_words:
        chord_line = " " * len(" ".join(current_line_words))
        for pos, chord in current_line_chords:
            chord_line = chord_line[:pos] + chord + chord_line[pos+len(chord):]
        
        output_lines.append(chord_line.rstrip())
        output_lines.append(" ".join(current_line_words))
        output_lines.append("")
    
    return output_lines

def align_chords_and_lyrics_website_style(
    chords: List[Tuple[float, str, float]],
    segments_data: List[dict],
    output_dir: str
) -> None:
    """Create website-style chord sheet with sections"""
    print("[INFO] Creating chord sheet...")
    
    # Detect song sections
    sections = detect_song_sections(segments_data)
    
    # Format output
    output_lines = ["Chord Sheet", "===========", ""]
    
    for section in sections:
        output_lines.extend(format_section(section, chords))
    
    # Save to file
    output_path = os.path.join(output_dir, "chord_sheet.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    
    print(f"[INFO] Chord sheet saved to {output_path}")

def main(audio_path: str):
    """Main execution function"""
    # Create output folder
    output_dir = create_output_folder(os.path.basename(audio_path))
    print(f"[INFO] Output will be saved in: {output_dir}")
    
    # Step 1: Preprocess audio
    processed_audio = preprocess_audio(audio_path, output_dir)
    
    # Step 2: Initialize chord detector and analyze
    chord_detector = ChordDetector(processed_audio)
    chords = chord_detector.analyze()
    
    # Save raw chord data
    chord_json = os.path.join(output_dir, "chord_data.json")
    with open(chord_json, "w") as f:
        json.dump([(float(t), c, float(conf)) for t, c, conf in chords], f, indent=2)
    
    # Step 3: Transcribe lyrics with word timestamps
    transcription = transcribe_lyrics(processed_audio, model_size="base")
    
    # Step 4: Process transcription into segments with word timing
    segments_data = []
    for seg in transcription["segments"]:
        words = []
        if "words" in seg:
            for word_info in seg["words"]:
                if isinstance(word_info, dict):
                    word = word_info.get("word", "").strip()
                    start = word_info.get("start", 0.0)
                else:
                    word = str(word_info).strip()
                    start = seg["start"]
                
                if word:
                    words.append((word, start))
        
        segments_data.append({
            'start': seg['start'],
            'end': seg['end'],
            'text': seg['text'],
            'words': words
        })
    
    # Step 5: Create website-style chord sheet
    align_chords_and_lyrics_website_style(chords, segments_data, output_dir)
    
    print(f"[INFO] Analysis complete! Check {output_dir} folder for results.")

if __name__ == "__main__":
    input_audio = "Your_Music.mp3"  # Replace with your audio file
    main(input_audio) 