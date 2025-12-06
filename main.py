from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import whisperx
import torch
import tempfile
import os
from typing import List, Optional
import time
import json
import gc

app = FastAPI(title="Call Recording AI Service")

# Global configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
batch_size = 16  # Reduce if low on GPU memory

print(f"Device: {device}, Compute type: {compute_type}")

# Load models on startup
print("Loading WhisperX model...")
whisperx_model = whisperx.load_model("base", device, compute_type=compute_type)
print(f"WhisperX model loaded on {device}")

# Optional: Load alignment model (for better timestamps)
alignment_model = None
alignment_metadata = None

class TranscriptionSegment(BaseModel):
    speaker: str
    start: float
    end: float
    text: str
    confidence: Optional[float] = None

class AnalysisRequest(BaseModel):
    transcript: str
    segments: List[dict]

@app.get("/")
async def root():
    return {
        "service": "Call Recording AI Service",
        "status": "running",
        "device": device,
        "compute_type": compute_type,
        "models": {
            "whisperx": "base",
            "alignment": alignment_model is not None,
            "diarization": "available",
            "llm": "placeholder"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": device,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = "en",
    diarize: bool = True
):
    """
    Transcribe audio file using WhisperX with speaker diarization
    """
    start_time = time.time()
    tmp_path = None
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # 1. Transcribe with WhisperX
        audio = whisperx.load_audio(tmp_path)
        result = whisperx_model.transcribe(
            audio, 
            batch_size=batch_size,
            language=language if language != "auto" else None
        )
        
        detected_language = result.get("language", language)
        
        # 2. Align whisper output (for better timestamps)
        global alignment_model, alignment_metadata
        if alignment_model is None:
            try:
                alignment_model, alignment_metadata = whisperx.load_align_model(
                    language_code=detected_language, 
                    device=device
                )
            except Exception as e:
                print(f"Could not load alignment model: {e}")
        
        if alignment_model is not None:
            result = whisperx.align(
                result["segments"], 
                alignment_model, 
                alignment_metadata, 
                audio, 
                device,
                return_char_alignments=False
            )
        
        # 3. Assign speaker labels (diarization)
        if diarize:
            try:
                # Note: Requires HuggingFace token for pyannote models
                # For now, use simple speaker alternation
                # To enable real diarization, set HF_TOKEN env var and install pyannote.audio
                
                # Simple alternating speakers (placeholder)
                current_speaker = 0
                last_end = 0
                speaker_gap_threshold = 2.0  # seconds
                
                for segment in result["segments"]:
                    # Switch speaker if there's a gap
                    if segment["start"] - last_end > speaker_gap_threshold:
                        current_speaker = (current_speaker + 1) % 2
                    
                    segment["speaker"] = f"SPEAKER_{current_speaker:02d}"
                    last_end = segment["end"]
            except Exception as e:
                print(f"Diarization failed: {e}")
                # Fallback: assign single speaker
                for segment in result["segments"]:
                    segment["speaker"] = "SPEAKER_00"
        else:
            for segment in result["segments"]:
                segment["speaker"] = "SPEAKER_00"
        
        # 4. Format response
        full_transcript = " ".join([seg["text"] for seg in result["segments"]])
        
        segments = []
        for segment in result["segments"]:
            segments.append({
                "speaker": segment.get("speaker", "SPEAKER_00"),
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "confidence": segment.get("score", 0.95)
            })
        
        # Clean up
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        # Free memory
        del audio
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        processing_time = time.time() - start_time
        
        return JSONResponse({
            "full_transcript": full_transcript,
            "segments": segments,
            "language": detected_language,
            "confidence": 0.95,  # WhisperX doesn't provide overall confidence
            "model": "whisperx-base",
            "processing_time": round(processing_time, 2)
        })
        
    except Exception as e:
        # Clean up on error
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/analyze")
async def analyze_transcript(request: AnalysisRequest):
    """
    Analyze transcript using LLM
    For now: Simple rule-based analysis
    TODO: Add Llama/Groq integration
    """
    start_time = time.time()
    
    try:
        # Placeholder analysis - replace with actual LLM
        transcript = request.transcript
        segments = request.segments
        
        # Calculate speaker metrics
        speaker_metrics = {}
        for seg in segments:
            speaker = seg.get("speaker", "SPEAKER_00")
            if speaker not in speaker_metrics:
                speaker_metrics[speaker] = {
                    "talk_time": 0,
                    "word_count": 0,
                    "segment_count": 0
                }
            
            duration = seg.get("end", 0) - seg.get("start", 0)
            speaker_metrics[speaker]["talk_time"] += duration
            speaker_metrics[speaker]["word_count"] += len(seg.get("text", "").split())
            speaker_metrics[speaker]["segment_count"] += 1
        
        # Simple sentiment analysis (placeholder)
        positive_words = ["great", "excellent", "good", "happy", "satisfied", "thank", "appreciate"]
        negative_words = ["bad", "issue", "problem", "angry", "frustrated", "complaint", "terrible"]
        
        transcript_lower = transcript.lower()
        pos_count = sum(transcript_lower.count(word) for word in positive_words)
        neg_count = sum(transcript_lower.count(word) for word in negative_words)
        
        sentiment_score = (pos_count - neg_count) * 2  # Scale to -10 to +10
        sentiment_score = max(-10, min(10, sentiment_score))
        
        # Extract simple topics (placeholder)
        words = transcript.lower().split()
        word_freq = {}
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "is", "was", "are"}
        
        for word in words:
            clean_word = word.strip(".,!?;:")
            if len(clean_word) > 3 and clean_word not in stop_words:
                word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
        
        key_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        key_topics = [topic[0] for topic in key_topics]
        
        processing_time = time.time() - start_time
        
        return JSONResponse({
            "summary": f"Call duration approximately {sum(s['talk_time'] for s in speaker_metrics.values()):.0f} seconds with {len(speaker_metrics)} speakers identified.",
            "sentiment_score": sentiment_score,
            "key_topics": key_topics,
            "action_items": [],  # TODO: Extract with LLM
            "quality_score": 75.0,  # Placeholder
            "compliance_flags": [],
            "speaker_metrics": speaker_metrics,
            "model": "rule-based-placeholder",
            "processing_time": round(processing_time, 2)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
