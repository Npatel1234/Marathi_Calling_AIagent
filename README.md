# Scam Report AI Agent

This project is a voice-based AI assistant designed to help users report scams through a natural conversation interface. It utilizes Automatic Speech Recognition (ASR), Natural Language Processing (NLP), and Text-to-Speech (TTS) technologies to collect and process scam-related reports efficiently.

## Features

- **Real-time Voice Recording**: Captures and processes user speech.
- **Marathi to English Translation**: Uses Google Translator to convert Marathi speech to English.
- **AI-Powered Conversation**: Uses LM Studio's API with Mistral-7B for interactive conversations.
- **Speech Synthesis**: Converts AI responses to Marathi speech using Google Text-to-Speech (gTTS).
- **Scam Report Logging**: Saves structured scam reports based on user inputs.
- **AI4Bharat ASR Model**: Implements a multilingual speech recognition model.

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- Virtual environment (optional but recommended)
- Required Python libraries:
  
  ```sh
  pip install flask sounddevice wavio transformers torch torchaudio numpy librosa gtts requests deep_translator concurrent.futures
  ```
- LM Studio running a local Mistral-7B model (Ensure API is accessible at `http://localhost:1234/v1/chat/completions`)

### Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/scam-report-ai.git
   cd scam-report-ai
   ```
2. (Optional) Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # For Linux/macOS
   venv\Scripts\activate     # For Windows
   ```
3. Install dependencies:
   ```sh
   pip install -r Requirements.txt
   ```
4. Set up the AI4Bharat ASR model:
   - Download and place the `indic-conformer-600m-multilingual` model in the appropriate directory.
5. Run the Flask application:
   ```sh
   python app.py
   ```
6. Open a web browser and go to:
   ```
   http://127.0.0.1:5000
   ```

## API Endpoints

- `GET /` - Renders the home page.
- `POST /api/start_listening` - Starts voice recording.
- `POST /api/stop_listening` - Stops recording and processes transcription.

## How It Works

1. User starts a voice conversation.
2. Audio is recorded and transcribed using the AI4Bharat ASR model.
3. Transcription is corrected for scam-related terminology.
4. Transcribed text is translated to English and sent to the AI model.
5. AI responds with relevant questions and instructions.
6. Response is translated back to Marathi and converted to speech.
7. The conversation continues until all necessary details are collected.
8. The report is saved as a text file.

## Technologies Used

- **Flask**: Backend framework.
- **SoundDevice & Wavio**: Audio recording.
- **Transformers (AI4Bharat)**: Automatic Speech Recognition (ASR).
- **Google Translator API**: Marathi-English translation.
- **LM Studio (Mistral-7B)**: Conversational AI.
- **gTTS**: Text-to-Speech in Marathi.
- **Torchaudio & Librosa**: Audio processing.

## Future Enhancements

- Support for more regional languages.
- Integration with law enforcement databases.
- Improved conversation flow with memory handling.



