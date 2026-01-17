"""
Angel Intelligence - Call Analysis Service

AI-powered call analysis using Qwen2.5-Omni-7B.
Supports two modes:
- Audio mode: Analyses audio directly for tone, emotion, sentiment
- Transcript mode: Analyses transcript text only

All outputs use British English (en-GB).
"""

import gc
import json
import logging
import os
import time
from typing import Dict, Any, List, Optional

import torch

from src.config import get_settings

logger = logging.getLogger(__name__)

# Check for model availability
try:
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    QWEN_OMNI_AVAILABLE = True
except ImportError:
    QWEN_OMNI_AVAILABLE = False
    logger.warning("Qwen2.5-Omni not available - install with: pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview")

try:
    from qwen_omni_utils import process_mm_info
    QWEN_OMNI_UTILS_AVAILABLE = True
except ImportError:
    QWEN_OMNI_UTILS_AVAILABLE = False
    logger.warning("qwen_omni_utils not available - install with: pip install qwen-omni-utils[decord]")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


# British English system prompt for all analysis
BRITISH_ENGLISH_INSTRUCTION = """
IMPORTANT: All output text must use British English (en-GB) spelling and conventions:
- Use -ise not -ize (organise, recognise, analyse)
- Use -our not -or (colour, behaviour, favour)
- Use -re not -er (centre, metre)
- Use -ogue not -og (catalogue, dialogue)
- Use -ence not -ense (licence, defence)
- Use DD/MM/YYYY date format
- Use £ (GBP) for currency
"""


class AnalysisService:
    """
    Call analysis service using AI models.
    
    Supports:
    - Audio analysis: Direct audio processing with Qwen2.5-Omni
    - Transcript analysis: Text-based analysis
    - Mock mode: Returns deterministic test data for development
    """
    
    def __init__(self):
        """Initialise the analysis service."""
        settings = get_settings()
        
        # Configuration
        self.analysis_mode = settings.analysis_mode  # 'audio' or 'transcript'
        self.use_mock = settings.use_mock_models
        
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu"
        
        # Model paths
        self.analysis_model_path = settings.get_analysis_model_path()
        self.chat_model_path = settings.get_chat_model_path()
        
        # Models (lazy loaded)
        self._analysis_model = None
        self._analysis_processor = None
        self._chat_model = None
        self._chat_processor = None
        
        # Load configuration
        self._config = self._load_config()
        
        logger.info(f"AnalysisService initialised: mode={self.analysis_mode}, device={self.device}")
    
    def _load_config(self, config_path: str = "call_analysis_config.json") -> Dict[str, Any]:
        """Load analysis configuration."""
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"topics": [], "agent_actions": [], "performance_rubric": []}
    
    def _ensure_analysis_model_loaded(self) -> None:
        """Ensure the analysis model is loaded."""
        if self.use_mock:
            logger.debug("Using mock mode - skipping model load")
            return
        
        if not QWEN_OMNI_AVAILABLE or not QWEN_OMNI_UTILS_AVAILABLE:
            raise RuntimeError("Qwen2.5-Omni or qwen_omni_utils not installed")
        
        if self._analysis_model is None:
            logger.info(f"Loading analysis model: {self.analysis_model_path}")
            
            self._analysis_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.analysis_model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Disable audio output to save VRAM
            self._analysis_model.disable_talker()
            
            self._analysis_processor = Qwen2_5OmniProcessor.from_pretrained(
                self.analysis_model_path,
                trust_remote_code=True
            )
            
            logger.info(f"Analysis model loaded on {self.device}")
    
    def analyse(
        self, 
        audio_path: Optional[str],
        transcript: Dict[str, Any],
        recording_id: int
    ) -> Dict[str, Any]:
        """
        Analyse a call recording.
        
        Args:
            audio_path: Path to audio file (required for audio mode)
            transcript: Transcript dictionary from transcription service
            recording_id: Database recording ID
            
        Returns:
            Analysis results dictionary
        """
        start_time = time.time()
        
        # Calculate speaker metrics (always needed)
        speaker_metrics = self._calculate_speaker_metrics(transcript)
        
        if self.use_mock:
            logger.info("Using mock analysis response")
            return self._mock_analysis(speaker_metrics, start_time)
        
        if self.analysis_mode == "audio" and audio_path:
            return self._analyse_audio(audio_path, transcript, speaker_metrics, start_time)
        else:
            return self._analyse_transcript(transcript, speaker_metrics, start_time)
    
    def _calculate_speaker_metrics(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for each speaker matching specification format."""
        raw_metrics = {}
        total_time = 0
        
        for segment in transcript.get("segments", []):
            speaker = segment.get("speaker", "SPEAKER_00")
            duration = segment.get("end", 0) - segment.get("start", 0)
            total_time += duration
            
            if speaker not in raw_metrics:
                raw_metrics[speaker] = {
                    "talk_time_seconds": 0,
                    "word_count": 0,
                    "segment_count": 0,
                    "interruptions": 0,
                }
            
            text = segment.get("text", "")
            raw_metrics[speaker]["talk_time_seconds"] += duration
            raw_metrics[speaker]["word_count"] += len(text.split())
            raw_metrics[speaker]["segment_count"] += 1
        
        # Calculate percentages and WPM
        formatted_metrics = {}
        for speaker, data in raw_metrics.items():
            talk_time = data["talk_time_seconds"]
            word_count = data["word_count"]
            
            # Determine if agent or supporter (first speaker is usually agent)
            speaker_label = "agent" if speaker == "SPEAKER_00" else "supporter"
            
            formatted_metrics[speaker_label] = {
                "talk_time_seconds": round(talk_time, 1),
                "talk_time_percentage": round((talk_time / total_time * 100) if total_time > 0 else 0, 1),
                "word_count": word_count,
                "average_pace_wpm": round((word_count / (talk_time / 60)) if talk_time > 0 else 0),
                "interruptions": data["interruptions"],
                "silence_percentage": 0,  # Would need more analysis
            }
        
        return formatted_metrics
    
    def _analyse_audio(
        self, 
        audio_path: str, 
        transcript: Dict[str, Any],
        speaker_metrics: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Analyse call using audio directly."""
        try:
            self._ensure_analysis_model_loaded()
            
            # Get full transcript text for context
            full_text = transcript.get("full_transcript", "")
            
            # Preprocess audio
            processed_audio = self._preprocess_audio(audio_path)
            
            # Chunk audio for processing
            chunks = self._chunk_audio(processed_audio, max_duration=30)
            
            # Sample chunks for efficiency
            if len(chunks) > 3:
                sample_chunks = [chunks[0], chunks[len(chunks)//2], chunks[-1]]
            else:
                sample_chunks = chunks
            
            # Analyse each chunk
            chunk_analyses = []
            for chunk_start, chunk_end, chunk_path in sample_chunks:
                analysis = self._analyse_audio_chunk(
                    chunk_path,
                    context=f"Segment from {chunk_start:.0f}s to {chunk_end:.0f}s"
                )
                chunk_analyses.append(analysis)
                
                # Clean up temp chunk
                if chunk_path != processed_audio and os.path.exists(chunk_path):
                    os.unlink(chunk_path)
            
            # Clean up preprocessed audio
            if processed_audio != audio_path and os.path.exists(processed_audio):
                os.unlink(processed_audio)
            
            # Aggregate results
            result = self._aggregate_analyses(chunk_analyses, speaker_metrics)
            result["processing_time"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}", exc_info=True)
            return self._empty_analysis(speaker_metrics, start_time)
    
    def _analyse_transcript(
        self, 
        transcript: Dict[str, Any],
        speaker_metrics: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Analyse call using transcript text only."""
        try:
            self._ensure_analysis_model_loaded()
            
            full_text = transcript.get("full_transcript", "")
            
            # Build prompt
            prompt = self._build_transcript_analysis_prompt(full_text)
            
            # Generate analysis
            response = self._generate_text_response(prompt)
            
            # Parse response
            result = self._parse_analysis_response(response, speaker_metrics)
            result["processing_time"] = time.time() - start_time
            result["analysis_type"] = "transcript"
            
            return result
            
        except Exception as e:
            logger.error(f"Transcript analysis failed: {e}", exc_info=True)
            return self._empty_analysis(speaker_metrics, start_time)
    
    def _preprocess_audio(self, audio_path: str) -> str:
        """Preprocess audio to 16kHz mono WAV."""
        if not SOUNDFILE_AVAILABLE:
            return audio_path
        
        try:
            audio_data, sample_rate = sf.read(audio_path)
            
            # Already good format
            if sample_rate == 16000 and len(audio_data.shape) == 1:
                return audio_path
            
            # Need to convert
            import tempfile
            import subprocess
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.close()
            
            cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-ar', '16000',
                '-ac', '1',
                '-f', 'wav',
                temp_file.name
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            return temp_file.name
            
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}")
            return audio_path
    
    def _chunk_audio(self, audio_path: str, max_duration: int = 30) -> List[tuple]:
        """Split audio into chunks."""
        if not SOUNDFILE_AVAILABLE:
            return [(0, 0, audio_path)]
        
        try:
            audio_data, sample_rate = sf.read(audio_path)
            total_duration = len(audio_data) / sample_rate
            
            if total_duration <= max_duration:
                return [(0, total_duration, audio_path)]
            
            import tempfile
            
            chunks = []
            chunk_samples = max_duration * sample_rate
            
            for i, start_sample in enumerate(range(0, len(audio_data), chunk_samples)):
                end_sample = min(start_sample + chunk_samples, len(audio_data))
                chunk_data = audio_data[start_sample:end_sample]
                
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                sf.write(temp_file.name, chunk_data, sample_rate)
                
                start_time = start_sample / sample_rate
                end_time = end_sample / sample_rate
                chunks.append((start_time, end_time, temp_file.name))
            
            return chunks
            
        except Exception as e:
            logger.warning(f"Audio chunking failed: {e}")
            return [(0, 0, audio_path)]
    
    def _analyse_audio_chunk(self, audio_path: str, context: str = "") -> Dict[str, Any]:
        """Analyse a single audio chunk."""
        topics = self._config.get("topics", [])
        actions = self._config.get("agent_actions", [])
        rubric = self._config.get("performance_rubric", [])
        
        system_prompt = f"""You are an expert call quality analyst for charity and supporter-engagement calls.
{BRITISH_ENGLISH_INSTRUCTION}

Analyse this audio recording segment for:
1. TONE: Agent's voice (warm/neutral/cold/hostile)
2. EMOTION: Detected emotions (calm/frustrated/happy/anxious)
3. PACING: Speech speed (rushed/steady/slow)
4. PROFESSIONALISM: Overall professional conduct
5. CUSTOMER EXPERIENCE: How the customer seems to feel"""

        user_prompt = f"""Analyse this call recording audio.

{context}

Topics to consider: {topics[:15] if topics else 'Standard charity call topics'}
Agent actions to identify: {actions[:15] if actions else 'Standard agent actions'}
Performance criteria: {rubric[:10] if rubric else 'Standard quality metrics'}

Respond with ONLY valid JSON:
{{
    "agent_tone": "warm/neutral/cold/hostile",
    "agent_emotion": "calm/frustrated/happy/anxious/other",
    "customer_emotion": "calm/frustrated/happy/anxious/other",
    "speech_clarity": 1-10,
    "professionalism": 1-10,
    "detected_topics": [],
    "detected_actions": [],
    "performance_scores": {{}},
    "detected_issues": [],
    "positive_indicators": [],
    "sentiment_score": -10 to +10,
    "brief_observation": "One sentence observation in British English"
}}"""

        try:
            # Build conversation with audio
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": system_prompt + "\n\n" + user_prompt}
                    ]
                }
            ]
            
            # Process with Qwen
            text = self._analysis_processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )
            
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            
            inputs = self._analysis_processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False
            )
            
            inputs = inputs.to(self._analysis_model.device).to(self._analysis_model.dtype)
            input_len = inputs.input_ids.shape[1]
            
            with torch.no_grad():
                text_ids = self._analysis_model.generate(
                    **inputs,
                    max_new_tokens=600,
                    do_sample=False,
                    return_audio=False
                )
            
            generated_ids = text_ids[:, input_len:]
            response = self._analysis_processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return self._parse_chunk_response(response)
            
        except Exception as e:
            logger.error(f"Chunk analysis failed: {e}")
            return self._default_chunk_analysis()
    
    def _build_transcript_analysis_prompt(self, transcript: str) -> str:
        """Build prompt for transcript-based analysis."""
        topics = self._config.get("topics", [])
        actions = self._config.get("agent_actions", [])
        rubric = self._config.get("performance_rubric", [])
        
        return f"""You are an expert call quality analyst for charity and supporter-engagement calls.
{BRITISH_ENGLISH_INSTRUCTION}

Analyse this call transcript and provide structured analysis.

TRANSCRIPT:
{transcript[:4000]}

TOPICS (select applicable): {topics[:15]}
AGENT ACTIONS (identify performed): {actions[:15]}
PERFORMANCE RUBRIC (score 1-10 each): {rubric[:10]}

Respond with ONLY valid JSON:
{{
    "summary": "2-3 sentence British English summary of the call",
    "sentiment_score": -10 to +10,
    "quality_score": 0 to 100,
    "key_topics": [{{"name": "topic from config", "confidence": 0.0-1.0}}],
    "agent_actions_performed": [{{"action": "action from config", "timestamp_start": 0.0, "quality": 1-5}}],
    "performance_scores": {{"criterion": 1-10}},
    "action_items": [{{"description": "action needed", "priority": "high/medium/low"}}],
    "compliance_flags": [{{"type": "issue_type", "issue": "description", "severity": "low/medium/high/critical"}}]
}}"""
    
    def _generate_text_response(self, prompt: str) -> str:
        """Generate text response from model."""
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
        
        text = self._analysis_processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        
        inputs = self._analysis_processor(
            text=text,
            return_tensors="pt",
            padding=True
        )
        
        inputs = inputs.to(self._analysis_model.device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            output_ids = self._analysis_model.generate(
                **inputs,
                max_new_tokens=800,
                do_sample=False
            )
        
        generated_ids = output_ids[:, input_len:]
        response = self._analysis_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return response
    
    def _parse_chunk_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from chunk analysis response."""
        try:
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                json_str = response[start:end+1]
                return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
        
        return self._default_chunk_analysis()
    
    def _parse_analysis_response(
        self, 
        response: str, 
        speaker_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse analysis response into structured format."""
        try:
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                json_str = response[start:end+1]
                parsed = json.loads(json_str)
                
                # Ensure all required fields with correct naming
                result = {
                    "summary": parsed.get("summary", "Analysis completed"),
                    "sentiment_score": parsed.get("sentiment_score", 0),
                    "quality_score": parsed.get("quality_score", 50),
                    "key_topics": parsed.get("key_topics", []),
                    "agent_actions_performed": parsed.get("agent_actions_performed", parsed.get("agent_actions", [])),
                    "performance_scores": parsed.get("performance_scores", {}),
                    "action_items": parsed.get("action_items", []),
                    "compliance_flags": parsed.get("compliance_flags", []),
                    "speaker_metrics": speaker_metrics,
                    "model_used": "Qwen2.5-Omni-7B",
                }
                return result
                
        except json.JSONDecodeError:
            pass
        
        return self._empty_analysis(speaker_metrics, 0)
    
    def _aggregate_analyses(
        self, 
        chunk_analyses: List[Dict[str, Any]],
        speaker_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate chunk analyses into final result."""
        if not chunk_analyses:
            return self._empty_analysis(speaker_metrics, 0)
        
        # Average numeric scores
        sentiment_scores = [c.get("sentiment_score", 0) for c in chunk_analyses]
        clarity_scores = [c.get("speech_clarity", 7) for c in chunk_analyses]
        professionalism_scores = [c.get("professionalism", 7) for c in chunk_analyses]
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        avg_clarity = sum(clarity_scores) / len(clarity_scores)
        avg_professionalism = sum(professionalism_scores) / len(professionalism_scores)
        
        # Collect all items
        all_topics = []
        all_actions = []
        all_issues = []
        all_positives = []
        observations = []
        all_performance_scores = {}
        
        for c in chunk_analyses:
            all_topics.extend(c.get("detected_topics", []))
            all_actions.extend(c.get("detected_actions", []))
            all_issues.extend(c.get("detected_issues", []))
            all_positives.extend(c.get("positive_indicators", []))
            if c.get("brief_observation"):
                observations.append(c["brief_observation"])
            
            # Merge performance scores
            for key, value in c.get("performance_scores", {}).items():
                if key not in all_performance_scores:
                    all_performance_scores[key] = []
                all_performance_scores[key].append(value)
        
        # Average performance scores
        final_performance_scores = {
            k: round(sum(v) / len(v)) for k, v in all_performance_scores.items()
        }
        
        # Determine dominant tones
        tones = [c.get("agent_tone", "neutral") for c in chunk_analyses]
        agent_emotions = [c.get("agent_emotion", "calm") for c in chunk_analyses]
        customer_emotions = [c.get("customer_emotion", "calm") for c in chunk_analyses]
        
        dominant_tone = max(set(tones), key=tones.count) if tones else "neutral"
        dominant_agent_emotion = max(set(agent_emotions), key=agent_emotions.count) if agent_emotions else "calm"
        dominant_customer_emotion = max(set(customer_emotions), key=customer_emotions.count) if customer_emotions else "calm"
        
        # Calculate quality score
        quality_score = self._sentiment_to_quality(avg_sentiment)
        
        # Build summary
        summary = " ".join(observations[:3]) if observations else "Audio analysis completed."
        
        # Format topics for database
        key_topics = [
            {"name": t, "confidence": 0.8} 
            for t in list(set(all_topics))[:10]
        ]
        
        # Format actions for database (using spec format)
        agent_actions_performed = [
            {"action": a, "timestamp_start": 0.0, "quality": 4}
            for a in list(set(all_actions))[:10]
        ]
        
        return {
            "summary": summary[:500],
            "sentiment_score": round(avg_sentiment, 1),
            "quality_score": round(quality_score, 1),
            "key_topics": key_topics,
            "agent_actions_performed": agent_actions_performed,
            "performance_scores": final_performance_scores,
            "action_items": [],
            "compliance_flags": [{"type": "issue", "issue": i, "severity": "medium"} for i in list(set(all_issues))[:5]],
            "speaker_metrics": speaker_metrics,
            "audio_observations": {
                "call_quality": "good" if avg_clarity >= 7 else "fair" if avg_clarity >= 5 else "poor",
                "background_noise": "low",
                "agent_tone": dominant_tone,
                "supporter_tone": dominant_customer_emotion,
            },
            "model_used": "Qwen2.5-Omni-7B",
            "analysis_type": "audio",
        }
    
    def _sentiment_to_quality(self, sentiment: float) -> float:
        """Convert sentiment score (-10 to +10) to quality score (0 to 100)."""
        if sentiment >= 8:
            return 95 + (sentiment - 8) * 2.5
        elif sentiment >= 5:
            return 85 + (sentiment - 5) * 3
        elif sentiment >= 2:
            return 70 + (sentiment - 2) * 4.67
        elif sentiment >= -1:
            return 50 + (sentiment + 1) * 6.33
        elif sentiment >= -4:
            return 30 + (sentiment + 4) * 6.33
        elif sentiment >= -7:
            return 10 + (sentiment + 7) * 6.33
        else:
            return max(0, min(9, (sentiment + 10) * 3))
    
    def _default_chunk_analysis(self) -> Dict[str, Any]:
        """Default chunk analysis when processing fails."""
        return {
            "agent_tone": "neutral",
            "agent_emotion": "calm",
            "customer_emotion": "calm",
            "speech_clarity": 7,
            "professionalism": 7,
            "detected_topics": [],
            "detected_actions": [],
            "performance_scores": {},
            "detected_issues": [],
            "positive_indicators": [],
            "sentiment_score": 0,
            "brief_observation": "Analysis unavailable for this segment",
        }
    
    def _empty_analysis(
        self, 
        speaker_metrics: Dict[str, Any], 
        start_time: float
    ) -> Dict[str, Any]:
        """Empty analysis when processing fails."""
        return {
            "summary": "Analysis unavailable",
            "sentiment_score": 0,
            "sentiment_label": "neutral",
            "quality_score": 50.0,
            "key_topics": [],
            "agent_actions_performed": [],
            "performance_scores": {},
            "action_items": [],
            "compliance_flags": [],
            "speaker_metrics": speaker_metrics,
            "audio_observations": None,
            "model_used": "none",
            "model_version": None,
            "analysis_type": "failed",
            "processing_time": time.time() - start_time,
        }
    
    def _mock_analysis(
        self, 
        speaker_metrics: Dict[str, Any], 
        start_time: float
    ) -> Dict[str, Any]:
        """
        Mock analysis for development/testing.
        
        Returns deterministic test data matching the full specification.
        All text uses British English.
        """
        return {
            "summary": "The supporter enquired about regular giving options. The agent explained the monthly donation programme and successfully enrolled the supporter at £10 per month with Gift Aid.",
            "sentiment_score": 7.5,
            "sentiment_label": "positive",
            "quality_score": 85.0,
            "key_topics": [
                {"name": "Regular giving signup", "confidence": 0.95, "timestamp_start": 45.5, "timestamp_end": 78.2},
                {"name": "Gift Aid explanation/enrolment", "confidence": 0.88, "timestamp_start": 120.0, "timestamp_end": 180.5},
                {"name": "Donation completion/payment processing", "confidence": 0.82, "timestamp_start": 200.0, "timestamp_end": 245.0},
            ],
            "agent_actions_performed": [
                {"action": "Greeted supporter", "timestamp_start": 0.0, "quality": 5},
                {"action": "Verified supporter identity", "timestamp_start": 15.2, "quality": 4},
                {"action": "Explained charity mission/programs", "timestamp_start": 45.5, "quality": 5},
                {"action": "Requested regular giving signup", "timestamp_start": 120.5, "quality": 5},
                {"action": "Processed donation payment", "timestamp_start": 200.0, "quality": 4},
                {"action": "Thanked supporter for their time", "timestamp_start": 280.0, "quality": 5},
            ],
            "performance_scores": {
                "Clarity of speech": 8,
                "Tone control": 9,
                "Active listening": 7,
                "Empathy & rapport": 8,
                "Confidence & authority": 7,
                "Accurate information delivery": 9,
                "Script/protocol adherence": 8,
                "Payment and data protection compliance": 10,
                "Recording of mandatory information": 9,
                "Call structure/flow control": 7,
                "Quality of donation ask or conversion attempt": 8,
                "Objection handling skill": 6,
                "Engagement effectiveness": 8,
                "Problem solving": 7,
                "Effective closing": 8,
            },
            "action_items": [
                {"description": "Send confirmation email with Gift Aid declaration", "priority": "high", "due_date": "2026-01-20"},
            ],
            "compliance_flags": [],
            "speaker_metrics": speaker_metrics if speaker_metrics else {
                "agent": {
                    "talk_time_seconds": 245,
                    "talk_time_percentage": 58,
                    "interruptions": 2,
                    "average_pace_wpm": 145,
                    "silence_percentage": 12
                },
                "supporter": {
                    "talk_time_seconds": 180,
                    "talk_time_percentage": 42,
                    "interruptions": 1,
                    "average_pace_wpm": 130,
                    "sentiment_trend": "positive"
                }
            },
            "audio_observations": {
                "call_quality": "good",
                "background_noise": "low",
                "agent_tone": "friendly",
                "supporter_tone": "happy",
            },
            "model_used": "mock",
            "model_version": "mock-v1.0.0",
            "analysis_type": "mock",
            "processing_time": time.time() - start_time,
        }
    
    def is_available(self) -> bool:
        """Check if analysis service is available."""
        if self.use_mock:
            return True
        return QWEN_OMNI_AVAILABLE and QWEN_OMNI_UTILS_AVAILABLE
