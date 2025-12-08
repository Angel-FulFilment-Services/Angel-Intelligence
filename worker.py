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
import json

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
            try:
                logger.info(f"Loading LLM model: {llm_model_path}")
                
                # Check if it's a local path or HuggingFace repo
                if os.path.exists(llm_model_path):
                    # Local path - use directly
                    logger.info(f"Loading LLM from local path: {llm_model_path}")
                    model_to_load = llm_model_path
                elif '/' in llm_model_path:
                    # HuggingFace repo format (e.g., "Qwen/Qwen2.5-3B-Instruct")
                    logger.info(f"Loading LLM from HuggingFace: {llm_model_path}")
                    model_to_load = llm_model_path  # Pipeline will auto-download
                else:
                    logger.warning(f"Invalid LLM_MODEL_PATH format: {llm_model_path}")
                    model_to_load = None
                
                if model_to_load:
                    self.llm = pipeline(
                        "text-generation",
                        model=model_to_load,
                        device=0 if self.device == "cuda" else -1,
                        max_new_tokens=512,
                        trust_remote_code=True  # Required for some models like Qwen
                    )
                    logger.info(f"LLM loaded successfully: {model_to_load}")
            except Exception as e:
                logger.warning(f"Failed to load LLM: {e}, using basic analysis")
                self.llm = None
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
            
            # Check segmentation preference from environment
            segmentation = os.getenv('TRANSCRIPT_SEGMENTATION', 'word').lower()
            
            # Extract word-level segments if available and requested
            if segmentation == 'word' and "word_segments" in result:
                # Use word-level timestamps
                logger.info(f"Using word-level timestamps: {len(result['word_segments'])} words")
                result["segments"] = result["word_segments"]
            elif segmentation == 'word' and "segments" in result and len(result["segments"]) > 0:
                # Check if segments have word-level data embedded
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
            else:
                # Use sentence-level segments (default from WhisperX)
                logger.info(f"Using sentence-level timestamps: {len(result['segments'])} segments")
        
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
    
    def _analyze_with_llm(self, full_text: str, speaker_metrics: dict, config_path="call_analysis_config.json"):
        """Multi-pass LLM analysis with external JSON config for topics, actions, and rubric"""
        logger.info("Using multi-pass LLM analysis with JSON config")

        # Load JSON config
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return self._extract_analysis_fallback("Config missing", speaker_metrics, "unknown-llm")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        topics = config.get("topics", [])
        actions = config.get("agent_actions", [])
        rubric = config.get("performance_rubric", [])

        # Get model name
        model_name = getattr(self.llm.model, 'name_or_path', 'unknown-llm')
        if '/' in model_name:
            model_name = model_name.split('/')[-1]

        # Trim transcript
        max_chars = 3000
        transcript = full_text[:max_chars]
        if len(full_text) > max_chars:
            transcript += "..."

        # -------------------------------
        # PASS 1 — Extract structured facts
        # -------------------------------
        pass1_prompt = f"""
        <|system|>
        You are a specialist call-quality analyst for charity and supporter-engagement calls.
        Your job is to extract accurate *facts only* from the transcript.
        Do NOT generate opinions, summaries, or scores.
        </s>

        <|user|>
        Extract structured information from this call transcript.
        Use ONLY these topics (You can use multiple if needed): {topics}
        Use ONLY these actions (You can use multiple if needed): {actions}

        TRANSCRIPT:
        {transcript}

        Return ONLY a valid JSON object and add no extra text.

        {{
        "agent_actions": [],
        "call_topics": [],
        "issues_or_concerns": [],
        "objective_outcome": ""
        }}
        </s>
        <|assistant|>
        """

        try:
            logger.info("Running Pass 1 (extraction)")
            pass1_result = self.llm(pass1_prompt, max_new_tokens=400, do_sample=False)
            pass1_text = pass1_result[0]["generated_text"]
            
            # Remove the prompt from the generated text to get only the model's response
            # The model returns prompt + generated tokens, we only want the new part
            if pass1_text.startswith(pass1_prompt):
                pass1_text = pass1_text[len(pass1_prompt):].strip()
            
            logger.info("=== PASS 1 RESPONSE ===")
            logger.info(pass1_text)

            p1_first = pass1_text.find("{")
            p1_last = pass1_text.rfind("}")
            if p1_first == -1 or p1_last == -1:
                raise ValueError("Pass 1 did not return JSON")

            pass1_json_str = pass1_text[p1_first:p1_last+1]
            pass1_data = json.loads(pass1_json_str)
        except Exception as e:
            logger.error(f"Pass 1 failed: {e}")
            return self._extract_analysis_fallback("Pass 1 failed", speaker_metrics, model_name)

        # ------------------------------------------------------------
        # PASS 2 — Score + summarise using Pass 1 output
        # ------------------------------------------------------------

        pass1_json_clean = json.dumps(pass1_data, ensure_ascii=False)

        pass2_prompt = f"""
        <|system|>
        You are a call-quality scoring assistant that evaluates charity-supporter calls.
        You will analyse extracted call events and produce a scored JSON summary.
        Use ONLY the provided rubric: {rubric}
        </s>

        <|user|>
        Create a JSON summary evaluating the agent's performance.

        DATA FROM PASS 1:
        {pass1_json_clean}

        Return ONLY a valid JSON object and add no extra text.

        {{
        "summary": "",            
        "sentiment": 0,           
        "quality_score": 0,       
        "topics": [],             
        "actions": [],            
        "concerns": []            
        }}

        ### SENTIMENT SCORING SCALE
        Rate *agent performance*, not caller mood:

        +10 = excellent: empathetic, efficient, clear, positive, handled objections well  
        +5  = good: generally effective, minor issues  
        0   = average: neutral, basic competence  
        -5  = poor: pushy, unclear, slow, or made mistakes  
        -10 = unacceptable: rude, unprofessional, failed key steps

        ### Quaility Score
        Map sentiment to quality score out of 100 as follows:
        - Sentiment 8 to 10 = 95 to 100
        - Sentiment 5 to 7 = 85 to 94
        - Sentiment 2 to 4 = 70 to 84
        - Sentiment -1 to 1 = 50 to 69
        - Sentiment -4 to -2 = 30 to 49
        - Sentiment -7 to -5 = 10 to 29
        - Sentiment -10 to -8 = 0 to 9

        Reply ONLY with the JSON object.
        </s>
        <|assistant|>
        """

        try:
            logger.info("Running Pass 2 (scoring)")
            pass2_result = self.llm(pass2_prompt, max_new_tokens=400, temperature=0.2)
            pass2_text = pass2_result[0]["generated_text"]
            
            # Remove the prompt from the generated text to get only the model's response
            if pass2_text.startswith(pass2_prompt):
                pass2_text = pass2_text[len(pass2_prompt):].strip()
            
            logger.info("=== PASS 2 RESPONSE ===")
            logger.info(pass2_text)

            p2_first = pass2_text.find("{")
            p2_last = pass2_text.rfind("}")
            if p2_first == -1 or p2_last == -1:
                raise ValueError("Pass 2 did not return JSON")

            pass2_json_str = pass2_text[p2_first:p2_last+1]
            summary_data = json.loads(pass2_json_str)
        except Exception as e:
            logger.error(f"Pass 2 failed: {e}")
            return self._extract_analysis_fallback(pass2_text, speaker_metrics, model_name)

        # -------------------------------
        # Final output
        # -------------------------------
        sentiment_raw = summary_data.get("sentiment", 0)
        try:
            sentiment = float(sentiment_raw)
        except:
            sentiment = 0
        sentiment = max(-10, min(10, sentiment))

        quality_score = 0
        if sentiment >= 8:
            quality_score = 95 + (sentiment - 8) * 2.5
        elif sentiment >= 5:
            quality_score = 85 + (sentiment - 5) * 3
        elif sentiment >= 2:
            quality_score = 70 + (sentiment - 2) * 4.67
        elif sentiment >= -1:
            quality_score = 50 + (sentiment + 1) * 6.33
        elif sentiment >= -4:
            quality_score = 30 + (sentiment + 4) * 6.33
        elif sentiment >= -7:
            quality_score = 10 + (sentiment + 7) * 6.33
        else:
            quality_score = max(0, min(9, (sentiment + 10) * 3))

        return {
            "summary": summary_data.get("summary", ""),
            "sentiment_score": sentiment,
            "speaker_metrics": speaker_metrics,
            "quality_score": quality_score,
            "key_topics": summary_data.get("topics", []),
            "action_items": summary_data.get("actions", []),
            "compliance_flags": summary_data.get("concerns", []),
            "model_used": model_name
        }

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
            
            # Add custom UK bank account recognizer
            from presidio_analyzer import Pattern, PatternRecognizer
            
            # UK bank accounts: 8 digits (account) + 6 digits (sort code)
            # Patterns: 12345678 or 12-34-56 12345678 or 12345678 123456
            uk_bank_patterns = [
                Pattern(name="uk_account_sort", regex=r"\b\d{2}[-\s]?\d{2}[-\s]?\d{2}\s+\d{8}\b", score=0.85),
                Pattern(name="uk_account_number", regex=r"\b\d{8}\b", score=0.4),  # Lower score as it's generic
                Pattern(name="uk_sort_code", regex=r"\b\d{2}[-\s]\d{2}[-\s]\d{2}\b", score=0.5),
            ]
            
            uk_bank_recognizer = PatternRecognizer(
                supported_entity="UK_BANK_ACCOUNT",
                patterns=uk_bank_patterns,
                context=["account", "bank", "sort code", "account number", "banking"]
            )
            
            # Register the custom recognizer
            self.pii_analyzer.registry.add_recognizer(uk_bank_recognizer)
            
            # Analyze for PII - removed unsupported UK entity types
            pii_results = self.pii_analyzer.analyze(
                text=full_text,
                language='en',
                entities=[
                    "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
                    "UK_NHS", "IBAN_CODE", "NRP", "MEDICAL_LICENSE",
                    "URL", "IP_ADDRESS", 
                    "UK_BANK_ACCOUNT"
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
