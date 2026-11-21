# RAG-Based AI Teaching Assistant

A local Retrieval-Augmented Generation (RAG) assistant that answers questions about a video course using the course transcripts. It transcribes audio with Whisper, creates semantic embeddings with Ollama, retrieves relevant transcript chunks using cosine similarity, and generates timestamp-aware answers with a local Llama model.

## Features

- Transcribe and translate course audio using OpenAI Whisper
- Preserve video number, title, start time, end time, and transcript text
- Generate embeddings with Ollama's `bge-m3` model
- Store transcripts and embeddings in a pandas DataFrame using `joblib`
- Retrieve the five most relevant transcript chunks for each question
- Generate human-readable answers with a local `llama3.2` model
- Save the generated prompt and response for inspection

## How It Works

```text
Video/audio files
      |
      v
Whisper transcription
      |
      v
Timestamped JSON chunks
      |
      v
bge-m3 embeddings through Ollama
      |
      v
Joblib DataFrame (embeddings.joblib)
      |
      v
Cosine-similarity retrieval
      |
      v
Llama 3.2-generated answer
```

The retrieval layer searches the existing course material and passes the most relevant transcript segments to the language model. This helps the assistant cite the correct video and approximate timestamp instead of answering without course context.

## Project Structure

```text
.
|-- README.md
|-- mp3_to_json.py          # Transcribes MP3 files with Whisper
|-- preprocess_json.py      # Converts JSON transcripts into embeddings
|-- process_incoming.py     # Retrieves context and asks the Ollama model
|-- video_to_mp3.py         # Converts video files into MP3 files with FFmpeg
|-- embeddings.joblib       # Prepared transcript DataFrame with embeddings
|-- prompt.txt              # Most recently generated LLM prompt
|-- response.txt            # Most recently generated LLM response
|-- jsons/                  # Timestamped transcript JSON files
|-- Videos/                 # Source video/audio data
|-- whisper/                # Vendored OpenAI Whisper source
`-- RAG BASED AI TEACHING ASSISTANT.docx
```

## Requirements

- Python 3.10 or 3.11 recommended
- FFmpeg
- [Ollama](https://ollama.com/)
- A local Ollama installation with the following models:

```bash
ollama pull bge-m3
ollama pull llama3.2
```

Python packages used by the project:

- `openai-whisper`
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `requests`
- `torch`

## Installation

1. Clone the repository:

```bash
git clone https://github.com/tanvi-bhageshwar/RAGBASEDTEACHING.git
cd RAGBASEDTEACHING
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

3. Install the Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn joblib requests torch
python -m pip install -e ./whisper
```

4. Install FFmpeg and ensure `ffmpeg` is available on your system `PATH`.

5. Start Ollama and verify that both required models are available:

```bash
ollama list
```

## Running the Assistant

### Option 1: Use the existing processed data

The repository already includes `embeddings.joblib` and transcript JSON files for a course dataset. Start Ollama and run:

```bash
python process_incoming.py
```

Enter a course-related question when prompted. The answer will be printed in the terminal and saved to `response.txt`.

Example prompts:

- `Where is HTML concluded?`
- `At what timestamp is CSS explained?`
- `Which video teaches HTML forms?`

### Option 2: Process new course files

The scripts are configured as a local prototype for course folders in this project.

1. Place source videos in a `Videos/` directory.
2. Convert videos to MP3 with `video_to_mp3.py`.
3. Place MP3 files in an `audios/` directory.
4. Run the transcription script:

```bash
python mp3_to_json.py
```

5. Create embeddings from the JSON chunks:

```bash
python preprocess_json.py
```

6. Ask questions:

```bash
python process_incoming.py
```

> **Folder and naming note:** `video_to_mp3.py` and `mp3_to_json.py` use the directory names and filename patterns shown above. Source filenames should contain the course number and title in a format compatible with the script. Before processing a different dataset, update the parsing logic and paths in those scripts as needed.

## Whisper Configuration

`mp3_to_json.py` currently uses:

- Whisper model: `large-v2`
- Language: `hi`
- Task: `translate`
- Word timestamps: disabled

For faster local processing, the model and language can be changed in `mp3_to_json.py`. Smaller Whisper models require less memory but may produce less accurate transcripts.

## Ollama Configuration

Both `preprocess_json.py` and `process_incoming.py` call Ollama at:

```text
http://localhost:11434
```

The embedding model is `bge-m3`, and the response model is `llama3.2`. These values can be changed in the Python scripts if you have suitable Ollama models installed.

## Sample JSON Format

```json
{
  "chunks": [
    {
      "number": "1",
      "title": "Installing VS Code & How Websites Work",
      "start": 0.0,
      "end": 3.5,
      "text": "Transcribed course content..."
    }
  ],
  "text": "Full transcript text"
}
```

## Customization

- Change the course prompt in `process_incoming.py`
- Change the number of retrieved chunks by updating `top_results`
- Replace `bge-m3` with another Ollama embedding model
- Replace `llama3.2` with another Ollama text-generation model
- Adjust retrieval or add filtering by video title or number

## Limitations

- This is a command-line prototype, not a web application
- Answers are limited to course content included in the embedding dataset
- Retrieval quality depends on transcription accuracy and embedding model choice
- Timestamp precision depends on the Whisper segmentation output
- The scripts currently use fixed local paths and Ollama endpoints

## Contributing

Contributions that improve cross-platform setup, automate the pipeline, add a web interface, or generalize the scripts for arbitrary courses are welcome.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run a focused test or smoke test
5. Open a pull request

## License

The project scripts and data are provided for educational use. The vendored Whisper directory retains its own MIT license and upstream package metadata.
