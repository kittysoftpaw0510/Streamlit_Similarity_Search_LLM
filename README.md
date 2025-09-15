# Similarity Detection System

A professional FastAPI + Streamlit application for real-time text similarity detection using OpenAI LLM.

## Features

- ✅ **4-window layout**: 2 sentence displays + 2 input areas
- ✅ **Red/Blue flashing lamps** for match indication
- ✅ **Real-time similarity detection** with growing prefix analysis (5 words → 6 words → 7 words...)
- ✅ **Scrolling sentence windows** (7 visible of 30 total sentences)
- ✅ **Auto-centering** on matched sentences with red/blue highlighting
- ✅ **File downloads** with user ID prefixes (e.g., u1_a.txt, u1_b.txt)
- ✅ **Multi-user support** with stateless backend (4 FastAPI workers)
- ✅ **Professional, modern UI/UX** with animated lamps and smooth transitions
- ✅ **OpenAI LLM integration** with intelligent fallback heuristics
- ✅ **Start/End typing buttons** for each user with state management

## Quick Start

### 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Optional: Set OpenAI API key for real LLM similarity
cp .env.example .env
# Edit .env with your OpenAI API key and desired model

# Start backend with 4 workers
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 2. Frontend Setup

```bash
cd frontend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start Streamlit app
streamlit run app.py
```

### 3. Access Application

- Frontend: http://localhost:8501
- Backend API: http://localhost:8000

## How It Works

1. **User Setup**: Enter a User ID to generate unique sentence sets
2. **Start Typing**: Click "Start Typing" to enable input for each user
3. **Real-time Detection**: As you type 5+ words, similarity checking begins
4. **Growing Prefix Analysis**: Checks 5 words, then 6, then 7, etc.
5. **Visual Feedback**: Matching sentences highlight and lamps flash
6. **Auto-scroll**: Sentence windows center on matches automatically
7. **File Downloads**: Get personalized a.txt and b.txt files

## Architecture

- **Backend**: FastAPI with 4 workers for concurrent processing
- **Frontend**: Streamlit with modern CSS styling
- **LLM**: OpenAI GPT-4o-mini for similarity detection
- **Fallback**: Heuristic similarity when no API key provided
- **State**: Stateless backend, session state in frontend

## Configuration

- Minimum prefix words: 5
- Sentence window size: 7 visible
- Total sentences per file: 30
- Backend workers: 4
- Request timeout: 30 seconds

## Multi-User Support

Each user gets unique sentence sets based on their User ID. Multiple users can use the system simultaneously without interference.

