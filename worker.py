"""
Autonomous AI Call Processing Worker
Runs independently - no Laravel queue dependency
"""
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# IMPORTANT: Must set this BEFORE importing torch
# PyTorch 2.6+ changed default to weights_only=True which breaks pyannote/omegaconf
# This env var allows loading trusted model checkpoints
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

import time
import logging
from datetime import datetime
from typing import Optional
import boto3
from botocore.config import Config
import mysql.connector
from mysql.connector import pooling
import torch
import tempfile
import gc

# Monkey-patch lightning to use weights_only=False for PyTorch 2.6+
# This is needed for pyannote.audio models to load correctly
import lightning_fabric.utilities.cloud_io as cloud_io
_original_load = cloud_io._load
def _patched_load(f, map_location=None, weights_only=None):
    return _original_load(f, map_location=map_location, weights_only=False)
cloud_io._load = _patched_load

import whisperx

# Configure logging FIRST before any other imports that use it
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from transformers import pipeline
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("transformers not available, using basic analysis")

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    import soundfile as sf
    import numpy as np
    PRESIDIO_AVAILABLE = True
except ImportError as e:
    PRESIDIO_AVAILABLE = False
    logger.warning(f"Presidio/soundfile not available, PII redaction disabled: {e}")

class CallProcessingWorker:
    def __init__(self):
        # Database configuration
        self.db_config = {
            'host': os.getenv('AI_DB_HOST', 'localhost'),
            'port': int(os.getenv('AI_DB_PORT', 3306)),
            'database': os.getenv('AI_DB_DATABASE', 'ai_calls'),
            'user': os.getenv('AI_DB_USERNAME', 'root'),
            'password': os.getenv('AI_DB_PASSWORD', ''),
            'pool_name': 'ai_pool',
            'pool_size': 5
        }
        
        # R2 configuration
        self.r2_client = boto3.client(
            's3',
            endpoint_url=os.getenv('R2_ENDPOINT'),
            aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
            config=Config(signature_version='s3v4')
        )
        self.r2_bucket = os.getenv('R2_BUCKET', 'call-recordings')
        
        # AI models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.batch_size = 16
        
        # Ensure spaCy model is installed for Presidio
        if PRESIDIO_AVAILABLE:
            self._ensure_spacy_model()
        
        logger.info(f"Initializing WhisperX on {self.device}")
        self.whisperx_model = whisperx.load_model(
            "medium",  # Use medium for better accuracy
            self.device, 
            compute_type=self.compute_type
        )
        
        self.alignment_model = None
        self.alignment_metadata = None
        
        # Initialize Presidio for PII detection
        self.pii_analyzer = None
        self.pii_anonymizer = None
        if PRESIDIO_AVAILABLE and os.getenv('ENABLE_PII_REDACTION', 'true').lower() == 'true':
            try:
                logger.info("Initializing Presidio for PII detection")
                self.pii_analyzer = AnalyzerEngine()
                self.pii_anonymizer = AnonymizerEngine()
                logger.info("Presidio initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Presidio: {e}")
        
        # Initialize LLM for analysis
        self.llm = None
        llm_model_path = os.getenv('LLM_MODEL_PATH', '')
        if LLM_AVAILABLE and llm_model_path:
            # Auto-download if path doesn't exist
            if not os.path.exists(llm_model_path):
                self._download_llm_model(llm_model_path)
            
            if os.path.exists(llm_model_path):
                try:
                    logger.info(f"Loading LLM from {llm_model_path}")
                    self.llm = pipeline(
                        "text-generation",
                        model=llm_model_path,
                        device=0 if self.device == "cuda" else -1,
                        max_new_tokens=512
                    )
                    logger.info("LLM loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load LLM: {e}, using basic analysis")
        else:
            logger.info("No LLM model configured, using basic analysis")
        
        logger.info("Worker initialized successfully")
    
    def _ensure_spacy_model(self):
        """Ensure spaCy English model is installed"""
        try:
            import spacy
            try:
                spacy.load('en_core_web_lg')
                logger.info("spaCy model 'en_core_web_lg' already installed")
            except OSError:
                logger.info("spaCy model 'en_core_web_lg' not found, downloading...")
                import subprocess
                subprocess.check_call([
                    "python", "-m", "spacy", "download", "en_core_web_lg"
                ])
                logger.info("spaCy model downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to ensure spaCy model: {e}")
    
    def _download_llm_model(self, model_path: str):
        """Auto-download LLM model if not present"""
        try:
            # Extract model name from path
            model_name = os.path.basename(model_path)
            
            # Map common model names to HuggingFace repos
            model_repos = {
                "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
                "TinyLlama-1.1B-Chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct"
            }
            
            repo = model_repos.get(model_name)
            if repo:
                logger.info(f"Auto-downloading LLM model: {repo} to {model_path}")
                from huggingface_hub import snapshot_download
                snapshot_download(repo, local_dir=model_path)
                logger.info(f"Model downloaded successfully to {model_path}")
            else:
                logger.warning(f"Unknown model '{model_name}', skipping auto-download")
        except Exception as e:
            logger.error(f"Failed to auto-download LLM model: {e}")
    
    def get_db_connection(self):
        """Get database connection"""
        return mysql.connector.connect(**self.db_config)
    
    def scan_for_pending_recordings(self):
        """Find recordings that need processing"""
        conn = self.get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # Get pending recordings (not already processing or completed)
            cursor.execute("""
                SELECT id, apex_id, r2_path, r2_bucket, file_format
                FROM ai_call_recordings
                WHERE processing_status = 'pending'
                ORDER BY created_at ASC
                LIMIT 10
            """)
            
            recordings = cursor.fetchall()
            return recordings
            
        finally:
            cursor.close()
            conn.close()
    
    def mark_processing(self, recording_id: int):
        """Mark recording as processing"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE ai_call_recordings 
                SET processing_status = 'processing',
                    processing_started_at = NOW()
                WHERE id = %s
            """, (recording_id,))
            conn.commit()
            logger.info(f"Marked recording {recording_id} as processing")
        finally:
            cursor.close()
            conn.close()
    
    def mark_completed(self, recording_id: int):
        """Mark recording as completed"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE ai_call_recordings 
                SET processing_status = 'completed',
                    processing_completed_at = NOW()
                WHERE id = %s
            """, (recording_id,))
            conn.commit()
            logger.info(f"Marked recording {recording_id} as completed")
        finally:
            cursor.close()
            conn.close()
    
    def mark_failed(self, recording_id: int, error: str):
        """Mark recording as failed"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE ai_call_recordings 
                SET processing_status = 'failed',
                    processing_error = %s,
                    processing_completed_at = NOW()
                WHERE id = %s
            """, (error[:500], recording_id))  # Truncate error
            conn.commit()
            logger.error(f"Marked recording {recording_id} as failed: {error}")
        finally:
            cursor.close()
            conn.close()
    
    def download_from_r2(self, r2_path: str, bucket: str) -> str:
        """Download audio file from local folder or R2"""
        logger.info(f"Original r2_path: {r2_path}, bucket: {bucket}")
        
        # Check for local development folder first
        local_storage_path = os.getenv('LOCAL_STORAGE_PATH', '')
        if local_storage_path and os.path.exists(local_storage_path):
            # Construct local file path
            if r2_path.startswith('http://') or r2_path.startswith('https://'):
                # Extract filename from URL
                import urllib.parse
                parsed = urllib.parse.urlparse(r2_path)
                local_file = os.path.join(local_storage_path, os.path.basename(parsed.path))
            else:
                # Use path directly
                local_file = os.path.join(local_storage_path, r2_path)
            
            if os.path.exists(local_file):
                logger.info(f"Using local file: {local_file}")
                return local_file
            else:
                logger.info(f"Local file not found: {local_file}, falling back to R2")
        
        # Download from R2
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.audio')
        temp_file.close()
        
        # Extract path if it's a URL
        if r2_path.startswith('http://') or r2_path.startswith('https://'):
            import urllib.parse
            parsed = urllib.parse.urlparse(r2_path)
            r2_path = parsed.path.lstrip('/')
            logger.info(f"Extracted R2 path from URL: {r2_path}")
        
        # Ensure bucket is valid
        if not bucket or bucket.startswith('http'):
            bucket = self.r2_bucket
        
        logger.info(f"Downloading from R2: bucket={bucket}, path={r2_path}")
        
        try:
            self.r2_client.download_file(bucket, r2_path, temp_file.name)
            logger.info(f"Successfully downloaded from R2 to {temp_file.name}")
            return temp_file.name
        except Exception as e:
            logger.error(f"Failed to download from R2: {e}")
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise
    
    def upload_to_r2(self, local_path: str, r2_path: str, bucket: str):
        """Upload processed audio back to R2"""
        self.r2_client.upload_file(local_path, bucket, r2_path)
        logger.info(f"Uploaded {local_path} to {r2_path}")
    
    def transcribe_audio(self, audio_path: str, language: str = "en"):
        """Transcribe audio using WhisperX"""
        logger.info(f"Starting transcription of {audio_path}")
        
        # Load audio
        audio = whisperx.load_audio(audio_path)
        
        # Transcribe
        result = self.whisperx_model.transcribe(
            audio, 
            batch_size=self.batch_size,
            language=language
        )
        
        detected_language = result.get("language", language)
        
        # Align for better timestamps
        if self.alignment_model is None:
            try:
                self.alignment_model, self.alignment_metadata = whisperx.load_align_model(
                    language_code=detected_language, 
                    device=self.device
                )
            except Exception as e:
                logger.warning(f"Could not load alignment model: {e}")
        
        if self.alignment_model is not None:
            result = whisperx.align(
                result["segments"], 
                self.alignment_model, 
                self.alignment_metadata, 
                audio, 
                self.device,
                return_char_alignments=False
            )
            
            # Extract word-level segments if available
            if "word_segments" in result:
                # Use word-level timestamps
                logger.info(f"Using word-level timestamps: {len(result['word_segments'])} words")
                result["segments"] = result["word_segments"]
            elif "segments" in result and len(result["segments"]) > 0:
                # Check if segments have word-level data
                if "words" in result["segments"][0]:
                    # Flatten words into individual segments
                    word_segments = []
                    for segment in result["segments"]:
                        if "words" in segment:
                            for word in segment["words"]:
                                word_segments.append({
                                    "start": word.get("start", segment["start"]),
                                    "end": word.get("end", segment["end"]),
                                    "text": word.get("word", word.get("text", "")),
                                    "score": word.get("score", segment.get("score", 0.95))
                                })
                    if word_segments:
                        logger.info(f"Extracted {len(word_segments)} word-level segments")
                        result["segments"] = word_segments
        
        # Simple speaker diarization (alternating based on gaps)
        current_speaker = 0
        last_end = 0
        speaker_gap_threshold = 2.0
        
        for segment in result["segments"]:
            if segment["start"] - last_end > speaker_gap_threshold:
                current_speaker = (current_speaker + 1) % 2
            segment["speaker"] = f"SPEAKER_{current_speaker:02d}"
            last_end = segment["end"]
        
        # Clean up memory
        del audio
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return result
    
    def analyze_transcript(self, transcript: dict):
        """Analyze transcript with LLM"""
        logger.info("Analyzing transcript")
        
        # Calculate speaker metrics
        speaker_metrics = {}
        for segment in transcript["segments"]:
            speaker = segment.get("speaker", "SPEAKER_00")
            if speaker not in speaker_metrics:
                speaker_metrics[speaker] = {
                    "talk_time": 0,
                    "word_count": 0,
                    "segment_count": 0
                }
            
            duration = segment.get("end", 0) - segment.get("start", 0)
            text = segment.get("text", segment.get("word", ""))
            speaker_metrics[speaker]["talk_time"] += duration
            speaker_metrics[speaker]["word_count"] += len(text.split())
            speaker_metrics[speaker]["segment_count"] += 1
        
        full_text = " ".join([s.get("text", s.get("word", "")) for s in transcript["segments"]])
        
        # Use LLM if available, otherwise fall back to basic analysis
        if self.llm:
            return self._analyze_with_llm(full_text, speaker_metrics)
        else:
            return self._analyze_basic(full_text, speaker_metrics)
    
    def _analyze_with_llm(self, full_text: str, speaker_metrics: dict):
        """Analyze using LLM"""
        logger.info("Using LLM for analysis")
        
        # Get model name for tracking
        model_name = getattr(self.llm.model, 'name_or_path', 'unknown-llm')
        if '/' in model_name:
            model_name = model_name.split('/')[-1]
        
        # Truncate text if too long
        max_chars = 1500
        text_sample = full_text[:max_chars]
        if len(full_text) > max_chars:
            text_sample += "..."
        
        prompt = """<|system|>
You are a call quality analyst evaluating agent performance during a call to a charity supporter. </s>
<|user|>
Read this call transcript and create a JSON summary, focus on the agent's performance.

TRANSCRIPT:
{}

Create a JSON object with:
1. "summary" - describe what the agent did in 2,3 sentences and their performance
2. "sentiment" - rate agent performance as a number: 10=excellent, 5=good, 0=average, -5=poor, -10=terrible
3. "topics" - list the subjects discussed (billing, product inquiry, complaint, etc)
4. "actions" - list what the agent actually did (sent package, scheduled callback, processed refund, etc)
5. "concerns" - list any issues (rude tone, didn't listen, unclear, etc) or empty array if none

Reply ONLY with the JSON object, no other text.</s>
<|assistant|>
""".format(text_sample)
        
        try:
            result = self.llm(prompt, max_new_tokens=500, temperature=0.5, do_sample=True, top_p=0.9)
            response_text = result[0]['generated_text']
            
            # Log the full response for debugging
            logger.info(f"=== FULL LLM RESPONSE ===")
            logger.info(response_text)
            logger.info(f"=== END LLM RESPONSE ===")
            
            # Extract JSON
            import json
            import re
            
            # Find JSON object
            open_brace = '{'
            close_brace = '}'
            first_brace = response_text.find(open_brace)
            last_brace = response_text.rfind(close_brace)
            
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = response_text[first_brace:last_brace+1]
                try:
                    analysis = json.loads(json_str)
                    logger.info(f"Successfully parsed JSON: {analysis}")
                    
                    # Validate and clean the response
                    summary = analysis.get("summary", f"Call with {len(speaker_metrics)} speakers")
                    if isinstance(summary, str) and len(summary) > 10:
                        summary = summary[:500]
                    else:
                        summary = f"Call with {len(speaker_metrics)} speakers"
                    
                    # Parse sentiment - handle string or number
                    sentiment_raw = analysis.get("sentiment", 0)
                    try:
                        sentiment = float(sentiment_raw)
                    except (ValueError, TypeError):
                        # Handle text sentiments - convert to numbers
                        sentiment_text = str(sentiment_raw).lower()
                        if 'very positive' in sentiment_text or 'excellent' in sentiment_text:
                            sentiment = 8
                        elif 'positive' in sentiment_text or 'good' in sentiment_text:
                            sentiment = 5
                        elif 'neutral' in sentiment_text or 'okay' in sentiment_text:
                            sentiment = 0
                        elif 'negative' in sentiment_text or 'poor' in sentiment_text:
                            sentiment = -5
                        elif 'very negative' in sentiment_text or 'terrible' in sentiment_text:
                            sentiment = -8
                        else:
                            # Try to extract number from string
                            match = re.search(r'-?\d+', str(sentiment_raw))
                            sentiment = float(match.group()) if match else 0
                    
                    # Ensure lists
                    topics = analysis.get("topics", [])
                    if isinstance(topics, str):
                        topics = [t.strip() for t in topics.split(',')]
                    topics = list(topics)[:10] if isinstance(topics, list) else []
                    
                    actions = analysis.get("actions", [])
                    if isinstance(actions, str):
                        actions = [a.strip() for a in actions.split(',')]
                    actions = list(actions)[:10] if isinstance(actions, list) else []
                    
                    concerns = analysis.get("concerns", [])
                    if isinstance(concerns, str):
                        concerns = [c.strip() for c in concerns.split(',')]
                    concerns = list(concerns)[:10] if isinstance(concerns, list) else []
                    
                    return {
                        "summary": summary,
                        "sentiment_score": max(-10, min(10, sentiment)),
                        "speaker_metrics": speaker_metrics,
                        "quality_score": 85.0,
                        "key_topics": topics,
                        "action_items": actions,
                        "compliance_flags": concerns,
                        "model_used": model_name
                    }
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error: {e}, json_str: {json_str[:200]}")
            
            # Fallback
            logger.warning("Using fallback extraction")
            return self._extract_analysis_fallback(response_text, speaker_metrics, model_name)
            
            # If JSON parsing failed, extract what we can
            logger.warning("Failed to parse JSON, using fallback extraction")
            return self._extract_analysis_fallback(response_text, speaker_metrics, model_name)
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}, falling back to basic")
            return self._analyze_basic(full_text, speaker_metrics)
    
    def _extract_analysis_fallback(self, response_text: str, speaker_metrics: dict, model_name: str):
        """Extract analysis from non-JSON LLM response"""
        import re
        
        # Try to extract sentiment
        sentiment = 0
        sentiment_match = re.search(r'sentiment["\']?\s*:\s*(-?\d+)', response_text, re.IGNORECASE)
        if sentiment_match:
            sentiment = int(sentiment_match.group(1))
        
        # Extract topics
        topics = []
        topics_match = re.search(r'topics["\']?\s*:\s*\[(.*?)\]', response_text, re.IGNORECASE | re.DOTALL)
        if topics_match:
            topics = [t.strip(' "\'') for t in topics_match.group(1).split(',') if t.strip()][:5]
        
        # Extract summary (first complete sentence)
        summary_match = re.search(r'summary["\']?\s*:\s*["\']([^"\']+)["\']', response_text, re.IGNORECASE)
        summary = summary_match.group(1) if summary_match else f"Call with {len(speaker_metrics)} speakers"
        
        return {
            "summary": summary[:500],
            "sentiment_score": sentiment,
            "speaker_metrics": speaker_metrics,
            "quality_score": 80.0,
            "key_topics": topics,
            "action_items": [],
            "compliance_flags": [],
            "model_used": model_name
        }
    
    def _analyze_basic(self, full_text: str, speaker_metrics: dict):
        """Basic analysis without LLM"""
        logger.info("Using basic analysis")
        
        # Simple sentiment
        positive_words = ["great", "excellent", "good", "happy", "satisfied", "thank"]
        negative_words = ["bad", "issue", "problem", "angry", "frustrated", "complaint"]
        
        pos_count = sum(full_text.lower().count(w) for w in positive_words)
        neg_count = sum(full_text.lower().count(w) for w in negative_words)
        sentiment_score = (pos_count - neg_count) * 2
        sentiment_score = max(-10, min(10, sentiment_score))
        
        return {
            "summary": f"Call with {len(speaker_metrics)} speakers",
            "sentiment_score": sentiment_score,
            "speaker_metrics": speaker_metrics,
            "quality_score": 75.0,
            "key_topics": [],
            "action_items": [],
            "compliance_flags": [],
            "model_used": "basic-analysis"
        }
    
    def redact_pii(self, audio_path: str, transcript: dict) -> Optional[dict]:
        """Detect and redact PII from transcript and audio"""
        if not self.pii_analyzer or not self.pii_anonymizer:
            logger.info("PII redaction not enabled or Presidio not available")
            return None
        
        try:
            logger.info("Starting PII detection and redaction")
            
            # Get full text
            full_text = " ".join([s.get("text", s.get("word", "")) for s in transcript["segments"]])
            
            # Analyze for PII
            pii_results = self.pii_analyzer.analyze(
                text=full_text,
                language='en',
                entities=[
                    "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
                    "UK_NHS", "IBAN_CODE", "NRP", "MEDICAL_LICENSE",
                    "URL", "UK_BANK_NUMBER",
                    "IP_ADDRESS", "UK_PASSPORT", "UK_DRIVER_LICENSE",
                ]
            )
            
            # Filter out UK from location detections to avoid false positives like "Age UK"
            pii_results = [r for r in pii_results if not (r.entity_type == "LOCATION" and r.text.upper() in ("UK", "UNITED KINGDOM"))]
            
            if not pii_results:
                logger.info("No PII detected")
                return None
            
            logger.info(f"Detected {len(pii_results)} PII entities")
            
            # Anonymize text
            anonymized_result = self.pii_anonymizer.anonymize(
                text=full_text,
                analyzer_results=pii_results
            )
            
            # Map PII locations to word-level segments
            pii_segments = []
            for result in pii_results:
                pii_segments.append({
                    "type": result.entity_type,
                    "start_char": result.start,
                    "end_char": result.end,
                    "score": result.score,
                    "text": full_text[result.start:result.end]
                })
            
            # Find timestamps for PII words and redact word segments
            pii_timestamps, redacted_segments = self._map_pii_to_timestamps(pii_results, transcript["segments"], full_text)
            
            # Beep audio at PII locations
            redacted_audio_path = None
            if pii_timestamps and os.path.exists(audio_path):
                redacted_audio_path = self._beep_audio_pii(audio_path, pii_timestamps)
            
            return {
                "redacted_text": anonymized_result.text,
                "redacted_segments": redacted_segments,
                "pii_count": len(pii_results),
                "pii_types": list(set([r.entity_type for r in pii_results])),
                "pii_segments": pii_segments,
                "pii_timestamps": pii_timestamps,
                "redacted_audio_path": redacted_audio_path
            }
            
        except Exception as e:
            logger.error(f"PII redaction failed: {e}", exc_info=True)
            return None
    
    def _map_pii_to_timestamps(self, pii_results, segments, full_text):
        """Map PII character positions to audio timestamps and redact segments"""
        timestamps = []
        redacted_segments = []
        char_position = 0
        
        for segment in segments:
            text = segment.get("text", segment.get("word", ""))
            segment_start_char = char_position
            segment_end_char = char_position + len(text)
            
            # Check if any PII overlaps with this segment
            pii_found = False
            for pii in pii_results:
                if pii.start < segment_end_char and pii.end > segment_start_char:
                    timestamps.append({
                        "start_time": segment.get("start", 0),
                        "end_time": segment.get("end", 0),
                        "pii_type": pii.entity_type,
                        "text": text
                    })
                    pii_found = True
                    break
            
            # Create redacted version of segment
            if pii_found:
                # Replace text with PII type placeholder
                redacted_segment = segment.copy()
                redacted_segment["text"] = f"[{pii.entity_type}]"
                if "word" in redacted_segment:
                    redacted_segment["word"] = f"[{pii.entity_type}]"
                redacted_segments.append(redacted_segment)
            else:
                # Keep original
                redacted_segments.append(segment.copy())
            
            char_position = segment_end_char + 1  # +1 for space
        
        return timestamps, redacted_segments
    
    def _beep_audio_pii(self, audio_path: str, pii_timestamps: list) -> str:
        """Replace PII sections in audio with beep tone"""
        try:
            logger.info(f"Beeping {len(pii_timestamps)} PII sections in audio")
            
            # Load audio
            audio_data, sample_rate = sf.read(audio_path)
            
            # Handle stereo - convert to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Generate beep tone (1000 Hz sine wave)
            def generate_beep(duration_sec, freq=1000):
                t = np.linspace(0, duration_sec, int(sample_rate * duration_sec))
                return 0.3 * np.sin(2 * np.pi * freq * t)  # 0.3 amplitude
            
            # Replace PII sections with beeps
            for ts in pii_timestamps:
                start_sample = int(ts["start_time"] * sample_rate)
                end_sample = int(ts["end_time"] * sample_rate)
                
                # Ensure bounds are valid
                start_sample = max(0, start_sample)
                end_sample = min(len(audio_data), end_sample)
                
                if end_sample > start_sample:
                    duration = (end_sample - start_sample) / sample_rate
                    beep = generate_beep(duration)
                    
                    # Replace audio with beep
                    audio_data[start_sample:end_sample] = beep[:end_sample - start_sample]
            
            # Determine output path
            local_storage_path = os.getenv('LOCAL_STORAGE_PATH', '')
            if local_storage_path and audio_path.startswith(local_storage_path):
                # Local mode - replace original file
                redacted_path = audio_path
                logger.info(f"Local mode: replacing original file {audio_path}")
            else:
                # Production/R2 mode - create new temp file
                redacted_path = audio_path.replace('.', '_redacted.')
                logger.info(f"Production mode: creating redacted file {redacted_path}")
            
            # Save redacted audio
            sf.write(redacted_path, audio_data, sample_rate)
            
            logger.info(f"Saved redacted audio to {redacted_path}")
            return redacted_path
            
        except Exception as e:
            logger.error(f"Audio beeping failed: {e}", exc_info=True)
            return None
    
    def save_transcription(self, recording_id: int, transcript: dict, language: str, pii_result: Optional[dict] = None, processing_time: float = 0):
        """Save transcription to database"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            import json
            
            # If PII redaction is enabled and we have results, use redacted versions
            if pii_result:
                # Use redacted transcript and segments
                full_transcript = pii_result["redacted_text"]
                segments_to_save = pii_result["redacted_segments"]
                logger.info("Saving REDACTED transcript (PII removed)")
            else:
                # No PII or redaction disabled - use original
                full_transcript = " ".join([s.get("text", s.get("word", "")) for s in transcript["segments"]])
                segments_to_save = transcript["segments"]
                logger.info("Saving original transcript (no PII redaction)")
            
            # Prepare segments JSON
            segments_json = json.dumps(segments_to_save)
            
            # Prepare PII metadata (but don't store the actual PII text)
            pii_detected_json = json.dumps({
                "pii_count": pii_result["pii_count"],
                "pii_types": pii_result["pii_types"],
                "timestamp_count": len(pii_result["pii_timestamps"])
            }) if pii_result else None
            
            # Insert transcription - only redacted versions go to DB
            cursor.execute("""
                INSERT INTO ai_call_transcriptions 
                (ai_call_recording_id, full_transcript, segments, redacted_transcript, 
                 pii_detected, language_detected, confidence_score, model_used, processing_time_seconds)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (recording_id, full_transcript, segments_json, 
                  full_transcript if pii_result else None,  # Mark as redacted if PII found
                  pii_detected_json, language, 0.95, "whisperx-medium", int(processing_time)))
            
            transcription_id = cursor.lastrowid
            
            conn.commit()
            logger.info(f"Saved transcription {transcription_id} with {len(segments_to_save)} segments (processing time: {processing_time:.1f}s)")
            return transcription_id
            
        finally:
            cursor.close()
            conn.close()
    
    def save_analysis(self, recording_id: int, analysis: dict):
        """Save analysis to database"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            import json
            
            # Map sentiment score to label
            score = analysis["sentiment_score"]
            if score >= 6:
                sentiment_label = "very_positive"
            elif score >= 2:
                sentiment_label = "positive"
            elif score >= -2:
                sentiment_label = "neutral"
            elif score >= -6:
                sentiment_label = "negative"
            else:
                sentiment_label = "very_negative"
            
            # Determine model used
            model_used = analysis.get("model_used", "basic-analysis")
            processing_time = int(analysis.get("processing_time", 0))
            
            cursor.execute("""
                INSERT INTO ai_call_analysis
                (ai_call_recording_id, summary, sentiment_score, sentiment_label, 
                 key_topics, action_items, quality_score, compliance_flags, speaker_metrics, 
                 model_used, processing_time_seconds)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                recording_id,
                analysis["summary"],
                analysis["sentiment_score"],
                sentiment_label,
                json.dumps(analysis["key_topics"]),
                json.dumps(analysis["action_items"]),
                analysis["quality_score"],
                json.dumps(analysis["compliance_flags"]),
                json.dumps(analysis["speaker_metrics"]),
                model_used,
                processing_time
            ))
            
            conn.commit()
            logger.info(f"Saved analysis for recording {recording_id} (processing time: {processing_time}s)")
            
        finally:
            cursor.close()
            conn.close()
    
    def process_recording(self, recording: dict):
        """Process a single recording"""
        recording_id = recording['id']
        local_file_used = False
        
        try:
            logger.info(f"Processing recording {recording_id}: {recording['apex_id']}")
            
            # Mark as processing
            self.mark_processing(recording_id)
            
            # Download audio (or use local file)
            audio_path = self.download_from_r2(recording['r2_path'], recording['r2_bucket'])
            
            # Check if it's a local file (don't delete later)
            local_storage_path = os.getenv('LOCAL_STORAGE_PATH', '')
            if local_storage_path and audio_path.startswith(local_storage_path):
                local_file_used = True
            
            try:
                # Transcribe
                import time
                transcribe_start = time.time()
                transcript = self.transcribe_audio(audio_path)
                transcribe_time = time.time() - transcribe_start
                
                # Detect and redact PII
                pii_start = time.time()
                pii_result = self.redact_pii(audio_path, transcript)
                pii_time = time.time() - pii_start
                
                # Save transcription (with PII info and processing time)
                total_transcribe_time = transcribe_time + pii_time
                self.save_transcription(recording_id, transcript, transcript.get("language", "en"), pii_result, total_transcribe_time)
                
                # Analyze
                analysis_start = time.time()
                analysis = self.analyze_transcript(transcript)
                analysis_time = time.time() - analysis_start
                
                # Save analysis (with processing time)
                analysis['processing_time'] = analysis_time
                self.save_analysis(recording_id, analysis)
                
                # Upload redacted audio to R2 if in production mode
                local_storage_path = os.getenv('LOCAL_STORAGE_PATH', '')
                if pii_result and pii_result.get("redacted_audio_path"):
                    redacted_path = pii_result["redacted_audio_path"]
                    
                    # Only upload to R2 if not using local storage
                    if not local_storage_path and os.path.exists(redacted_path) and redacted_path != audio_path:
                        new_r2_path = recording['r2_path'].replace('.', '_redacted.')
                        self.upload_to_r2(redacted_path, new_r2_path, recording['r2_bucket'])
                        os.unlink(redacted_path)  # Clean up temp file
                        logger.info(f"Uploaded redacted audio to R2: {new_r2_path}")
                    elif local_storage_path:
                        logger.info(f"Local mode: audio already redacted in place at {redacted_path}")
                
                # Mark as completed
                self.mark_completed(recording_id)
                
            finally:
                # Clean up temp file (but not local files)
                if not local_file_used and os.path.exists(audio_path):
                    os.unlink(audio_path)
                    
        except Exception as e:
            logger.error(f"Failed to process recording {recording_id}: {e}", exc_info=True)
            self.mark_failed(recording_id, str(e))
    
    def run(self, poll_interval: int = 30):
        """Main worker loop"""
        logger.info(f"Starting worker (poll interval: {poll_interval}s)")
        
        while True:
            try:
                # Find pending recordings
                recordings = self.scan_for_pending_recordings()
                
                if recordings:
                    logger.info(f"Found {len(recordings)} recordings to process")
                    
                    for recording in recordings:
                        self.process_recording(recording)
                else:
                    logger.debug("No pending recordings found")
                
                # Wait before next poll
                time.sleep(poll_interval)
                
            except KeyboardInterrupt:
                logger.info("Worker stopped by user")
                break
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                time.sleep(poll_interval)

if __name__ == "__main__":
    worker = CallProcessingWorker()
    worker.run(poll_interval=int(os.getenv('POLL_INTERVAL', 30)))
