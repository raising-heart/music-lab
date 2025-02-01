# Music Chord Detection

A Python script for detecting musical chords and aligning them with lyrics in audio files. This tool uses audio processing and machine learning to detect chords and transcribe lyrics from music files.

**⚠️ IMPORTANT NOTE**: This is an experimental tool and the chord detection accuracy has not been thoroughly tested or verified by the author. The results should be considered approximate and may require manual verification.

## Development History

### Version 1 (chord_lyric_alignment.py)
- Initial implementation with basic chord detection
- Simple lyrics transcription
- Basic alignment of chords with lyrics
- Output format: Raw timestamps with chords and lyrics
- Limitations:
  - No section detection
  - Basic word timing estimation
  - Limited formatting

### Version 2 (music_chord.py)
- Improved chord detection with musical context
- Enhanced lyrics transcription with word-level timestamps
- Better chord-lyric alignment
- Added confidence scores for chord detection
- Organized output structure
- Limitations:
  - Still using basic section detection
  - Repetitive section labeling
  - No handling of instrumental parts

### Version 3 (music_chord_3.py)
- Complete overhaul of section detection:
  - Automatic identification of verses, choruses, and bridges
  - Detection of instrumental sections, intro, and outro
  - Pattern recognition for repeated sections
- Enhanced output formatting:
  - Website-style chord sheet format
  - Proper alignment of chords above lyrics
  - Clear section markers
  - Handling of instrumental breaks
- Improved chord detection:
  - Better smoothing of chord progressions
  - Enhanced musical context awareness
  - More accurate chord placement
- Better organization:
  - Song-specific output folders
  - JSON data storage for further processing
  - Cleaner code structure

## Features

- Chord detection from audio files
- Lyrics transcription with timestamps
- Chord-lyric alignment
- Beat detection and musical key analysis
- Output organized in song-specific folders

## Requirements

### Python Packages
```bash
pip install librosa        # For music/audio analysis
pip install numpy         # For numerical computations
pip install pydub         # For audio file handling
pip install openai-whisper # For lyrics transcription
```

### FFmpeg Requirement
This script requires FFmpeg for audio processing. 

**Windows Installation:**
1. Download FFmpeg from: https://www.gyan.dev/ffmpeg/builds/
2. Extract the downloaded file
3. Place the extracted files in `C:\ffmpeg`
4. Add `C:\ffmpeg` to your system's PATH environment variable

**Linux Installation:**
```bash
sudo apt-get install ffmpeg
```

**macOS Installation:**
```bash
brew install ffmpeg
```

## Usage

1. Place your audio file in the same directory as the script
2. Update the input filename in the script:
   ```python
   input_audio = "your_song.mp3"  # Replace with your audio file
   ```
3. Run the script:
   ```bash
   python music_chord.py
   ```

## Output Structure

For each processed song, the script creates a folder named `output_[songname]` containing:
- `processed_audio.wav` - Preprocessed audio file
- `music_chord_data.json` - Raw chord data with timing and confidence scores
- `music_chord_output.txt` - Formatted output with aligned chords and lyrics

## Pros and Cons

### Pros
- Automated chord detection and lyrics transcription
- Organized output structure
- Handles multiple audio formats
- Includes confidence scores for chord detection
- Supports multiple languages through Whisper
- Beat-synchronized analysis
- Musical context awareness

### Cons
- Chord detection accuracy not thoroughly tested
- High computational requirements
- Requires significant disk space for processed files
- May struggle with complex chord progressions
- Dependent on external services (Whisper API)
- Limited to basic chord types (major/minor)
- May produce incorrect results for:
  - Complex harmonies
  - Non-Western music
  - Poor quality recordings
  - Heavy instrumental sections

## Limitations

1. **Chord Detection Accuracy**
   - The chord detection is based on frequency analysis and may not always be accurate
   - Complex chords (7th, 9th, etc.) are not supported
   - Results may vary based on audio quality

2. **Lyrics Transcription**
   - Accuracy depends on audio clarity
   - May struggle with:
     - Heavy accents
     - Background music
     - Multiple voices
     - Poor audio quality

3. **Resource Usage**
   - Processing large files requires significant memory
   - Whisper transcription can be slow on CPU
   - Requires substantial disk space for output files

## Contributing

This is an experimental project and contributions are welcome. Please note that the chord detection algorithm needs significant validation and improvement.

## License

This project is provided as-is without any warranty. Use at your own risk.

## Acknowledgments

- Uses OpenAI's Whisper for speech recognition
- Built with Librosa for music analysis
- Uses Pydub for audio processing

## Disclaimer

This tool is provided for experimental purposes only. The chord detection results should not be considered definitive and may require manual verification by a musician. The author makes no claims about the accuracy of the chord detection algorithm. 
