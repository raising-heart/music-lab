import whisper
import librosa
import numpy as np
from pydub import AudioSegment

# --- STEP 1: AUDIO PREPROCESSING ---
def preprocess_audio(audio_path, output_path="processed_audio.wav"):
    """
    Convert audio to WAV (mono, 16kHz).
    """
    print("[INFO] Preprocessing audio...")
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1)       # Convert to mono
    audio = audio.set_frame_rate(16000) # Set sample rate to 16k
    audio.export(output_path, format="wav")
    print(f"[INFO] Preprocessed audio saved at {output_path}")
    return output_path

# --- STEP 2A: TRANSCRIBE LYRICS ---
def transcribe_lyrics(audio_path, model_size="base"):
    """
    Use Whisper to transcribe audio. Returns the full transcription result,
    which includes segment-level timestamps.
    """
    print("[INFO] Transcribing lyrics...")
    model = whisper.load_model(model_size)  # e.g. "base", "small", "large", etc.
    result = model.transcribe(audio_path)
    print("[INFO] Transcription completed!")
    return result

# --- STEP 2B: NAIVE WORD-LEVEL TIMESTAMPS ---
def split_segments_into_words(transcription_result):
    """
    Splits each segment into words, distributing segment time evenly 
    across those words. Returns a list of segments, where each segment is:
      {
        'start': float,
        'end': float,
        'text': str,
        'words': [(word, word_timestamp), ...]
      }
    """
    segments_data = []

    for seg in transcription_result["segments"]:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_text = seg["text"].strip()
        
        # Split into words
        words = seg_text.split()
        
        # If no words, skip
        if not words:
            segments_data.append({
                'start': seg_start,
                'end': seg_end,
                'text': seg_text,
                'words': []
            })
            continue

        duration = seg_end - seg_start
        word_duration = duration / len(words)

        # Build (word, time) for each word in this segment
        word_timestamps = []
        for i, w in enumerate(words):
            w_time = seg_start + i * word_duration
            word_timestamps.append((w, w_time))

        segments_data.append({
            'start': seg_start,
            'end': seg_end,
            'text': seg_text,
            'words': word_timestamps
        })

    return segments_data

def save_lyrics_output(segments_data, output_txt="lyrics_output.txt"):
    """
    Save each segment with naive word-level timestamps to a file.
    """
    print(f"[INFO] Saving naive word-level segments to {output_txt}")
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("Lyrics (word-level naive timestamps):\n\n")
        for seg in segments_data:
            seg_start = seg['start']
            seg_end = seg['end']
            seg_text = seg['text']
            f.write(f"[{seg_start:0.2f} - {seg_end:0.2f}] {seg_text}\n")
        f.write("\n[INFO] End of naive word timestamps.\n")

# --- STEP 3: CHORD DETECTION ---
def detect_chords(audio_path, chord_output="chords_output.txt"):
    """
    Detect chords (single-note approximation) using librosa piptrack.
    Groups chords in 0.5 second intervals for smoothing.
    Returns smoothed list of (timestamp, chord).
    """
    print("[INFO] Detecting chords...")
    y, sr = librosa.load(audio_path, sr=None)
    hop_length = 512
    frame_duration = hop_length / sr

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)

    # Frequency to note
    def frequency_to_note_name(freq):
        if freq <= 0:
            return None
        note_number = 12 * np.log2(freq / 440.0) + 69
        note_names = ["C", "C#", "D", "D#", "E", "F", 
                      "F#", "G", "G#", "A", "A#", "B"]
        return note_names[int(round(note_number)) % 12]

    # Collect raw chords with timestamps
    raw_chords = []
    for frame in range(pitches.shape[1]):
        pitch = pitches[:, frame].max()
        note = frequency_to_note_name(pitch)
        timestamp = frame * frame_duration
        raw_chords.append((timestamp, note or "Unknown"))

    # Smooth them in 0.5 second intervals
    interval = 0.5
    smoothed_chords = []
    current_interval = 0
    current_chord_list = []

    for timestamp, chord in raw_chords:
        if timestamp > current_interval + interval:
            if current_chord_list:
                most_frequent_chord = max(set(current_chord_list),
                                          key=current_chord_list.count)
                smoothed_chords.append((current_interval, most_frequent_chord))
            current_interval += interval
            current_chord_list = []
        current_chord_list.append(chord)

    # Handle leftover chords in the last interval
    if current_chord_list:
        most_frequent_chord = max(set(current_chord_list),
                                  key=current_chord_list.count)
        smoothed_chords.append((current_interval, most_frequent_chord))

    # Write chord data to file
    with open(chord_output, "w") as f:
        f.write("Chords (smoothed):\n\n")
        for t, c in smoothed_chords:
            f.write(f"{t:0.2f}\t{c}\n")

    print(f"[INFO] Smoothed chords with timestamps saved to {chord_output}")
    return smoothed_chords

# --- STEP 4A: FIND ACTIVE CHORD FOR A WORD ---
def find_active_chord(chords, word_time):
    """
    Given a word timestamp, return the chord that is 'active' at that time.
    We assume each chord extends until the next chord's start.
    """
    if word_time < chords[0][0]:
        return chords[0][1]
    
    for i in range(len(chords) - 1):
        chord_start_time = chords[i][0]
        chord_name = chords[i][1]
        next_chord_start_time = chords[i+1][0]
        
        if chord_start_time <= word_time < next_chord_start_time:
            return chord_name
    
    # If we exceed all intervals, use the last chord
    return chords[-1][1]

# --- STEP 4B: ALIGN CHORDS & WORDS LINE-BY-LINE ---
def align_chords_line_by_line(segments_data, chords, output_file="lyrics_chords_output.txt"):
    """
    For each segment (which is basically each line in the final transcript),
    1. Build chord_line and lyric_line.
    2. Output them, so each segment is on its own "two-line" chord/lyric block.
    """
    print("[INFO] Aligning chords with words line-by-line...")
    
    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write("Chords above lyrics (naive alignment, line-by-line):\n\n")

        for seg in segments_data:
            words = seg['words']
            
            # If there are no words, just skip or print an empty line
            if not words:
                f_out.write("(no words here)\n\n")
                continue

            chord_line = ""
            lyric_line = ""

            for (word, w_time) in words:
                active_chord = find_active_chord(chords, w_time)
                
                # Append chord with spacing to chord_line
                chord_line += active_chord
                chord_line += " " * (len(word) + 1)

                # Append the word (plus a space) to lyric_line
                lyric_line += word + " "

            # Trim trailing spaces
            chord_line = chord_line.rstrip()
            lyric_line = lyric_line.rstrip()

            # Write both lines for this segment, then a blank line
            f_out.write(chord_line + "\n")
            f_out.write(lyric_line + "\n\n")

    print(f"[INFO] Aligned chords and lyrics saved to {output_file}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1) Preprocess Audio
    input_audio_file = "Your_Music.mp3"  # Replace with your actual file
    processed_audio = preprocess_audio(input_audio_file)

    # 2) Transcription
    transcription_result = transcribe_lyrics(processed_audio, model_size="base")

    # 2B) Word-level naive splitting
    segments_data = split_segments_into_words(transcription_result)
    save_lyrics_output(segments_data, output_txt="lyrics_output.txt")

    # 3) Chord Detection
    chords = detect_chords(processed_audio, chord_output="chords_output.txt")

    # 4) Align chords with lyrics, line-by-line
    align_chords_line_by_line(segments_data, chords, output_file="lyrics_chords_output.txt")

    print("[INFO] Done!")
