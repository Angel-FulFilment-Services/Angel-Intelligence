"""
Angel Intelligence - Live Session Service

Handles real-time meeting transcription and whiteboard updates.
Frontend handles all meeting management - this just does:
- Live transcription (streaming or chunked)
- LLM-generated canvas updates (summaries, action items, etc.)
- Session persistence for disconnect recovery

Architecture:
    Audio Stream → RealtimeSTT/WhisperX → Live transcript
                            ↓
                       Buffer (20s)
                            ↓
                       vLLM → Canvas updates
"""

import asyncio
import logging
import time
import json
import tempfile
import os
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Check for RealtimeSTT availability
REALTIME_STT_AVAILABLE = False
try:
    from RealtimeSTT import AudioToTextRecorder
    REALTIME_STT_AVAILABLE = True
except ImportError:
    logger.warning("RealtimeSTT not available - install with: pip install RealtimeSTT")


class CanvasElementType(str, Enum):
    SUMMARY = "summary"
    ACTION_ITEM = "action_item"
    DECISION = "decision"
    TOPIC = "topic"
    MOOD = "mood"
    CHART = "chart"
    NOTE = "note"


@dataclass
class CanvasElement:
    """Represents an element on the whiteboard canvas."""
    element_id: str
    element_type: CanvasElementType
    content: Any
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0})
    style: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CanvasUpdate:
    """Represents an update to the canvas."""
    action: str  # create, update, delete
    element_id: str
    element_type: str
    content: Any
    position: Optional[Dict[str, float]] = None


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""
    speaker: str
    text: str
    timestamp: float
    is_final: bool = True


class LiveSession:
    """
    Manages a single live transcription session.
    
    The frontend owns meeting management - this just handles:
    - Audio → transcript
    - Transcript → canvas updates via LLM
    - Recovery on disconnect
    """
    
    def __init__(
        self,
        session_id: str,
        on_transcript: Callable[[TranscriptSegment], None],
        on_canvas_update: Callable[[CanvasUpdate], None],
        llm_service: Any,
        update_interval: float = 20.0,
        external_meeting_id: Optional[str] = None,  # Frontend's meeting ID
    ):
        self.session_id = session_id
        self.on_transcript = on_transcript
        self.on_canvas_update = on_canvas_update
        self.llm_service = llm_service
        self.update_interval = update_interval
        self.external_meeting_id = external_meeting_id
        
        # Timestamps
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        
        # State
        self.transcript_buffer: List[TranscriptSegment] = []
        self.full_transcript: List[TranscriptSegment] = []
        self.canvas_state: Dict[str, CanvasElement] = {}
        self.is_running = False
        self.last_llm_update = 0
        
        # Audio processing
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._recorder = None
        self._transcriber = None
        
        # Layout tracking
        self._next_y_position = 50
        self._column_x = {"left": 50, "center": 400, "right": 750}
        
        logger.info(f"LiveSession created: {session_id}" + 
                   (f" (meeting: {external_meeting_id})" if external_meeting_id else ""))
    
    async def start(self):
        """Start the live session."""
        self.is_running = True
        self.last_llm_update = time.time()
        
        asyncio.create_task(self._process_audio_queue())
        asyncio.create_task(self._periodic_llm_update())
        
        logger.info(f"LiveSession started: {self.session_id}")
    
    async def stop(self):
        """Stop the live session."""
        self.is_running = False
        self.end_time = time.time()
        
        # Final LLM update
        await self._run_llm_update(final=True)
        
        # Cleanup RealtimeSTT
        if self._recorder:
            try:
                self._recorder.stop()
                self._recorder.shutdown()
            except Exception as e:
                logger.warning(f"Error stopping RealtimeSTT: {e}")
            self._recorder = None
        
        # Cleanup transcriber
        if self._transcriber:
            try:
                self._transcriber.unload()
            except Exception as e:
                logger.warning(f"Error unloading transcriber: {e}")
            self._transcriber = None
        
        logger.info(f"LiveSession stopped: {self.session_id}")
    
    async def add_audio_chunk(self, audio_data: bytes):
        """Add an audio chunk to the processing queue."""
        await self._audio_queue.put(audio_data)
    
    def _setup_realtime_stt(self):
        """Setup RealtimeSTT for streaming transcription."""
        if not REALTIME_STT_AVAILABLE:
            return None
        
        def on_realtime_text(text: str):
            if text and text.strip():
                segment = TranscriptSegment(
                    speaker="Speaker",
                    text=text.strip(),
                    timestamp=time.time(),
                    is_final=False
                )
                self.on_transcript(segment)
        
        def on_final_text(text: str):
            if text and text.strip():
                segment = TranscriptSegment(
                    speaker="Speaker",
                    text=text.strip(),
                    timestamp=time.time(),
                    is_final=True
                )
                self.transcript_buffer.append(segment)
                self.full_transcript.append(segment)
                self.on_transcript(segment)
        
        try:
            recorder = AudioToTextRecorder(
                model="tiny.en",
                language="en",
                spinner=False,
                silero_sensitivity=0.4,
                webrtc_sensitivity=2,
                post_speech_silence_duration=0.4,
                min_length_of_recording=0.3,
                min_gap_between_recordings=0.1,
                enable_realtime_transcription=True,
                realtime_processing_pause=0.1,
                realtime_model_type="tiny.en",
                on_realtime_transcription_update=on_realtime_text,
            )
            recorder.text = on_final_text
            return recorder
        except Exception as e:
            logger.error(f"Failed to setup RealtimeSTT: {e}")
            return None
    
    async def _process_audio_queue(self):
        """Process audio - streaming if available, else chunked."""
        if REALTIME_STT_AVAILABLE and self._recorder is None:
            self._recorder = self._setup_realtime_stt()
            if self._recorder:
                logger.info("Using RealtimeSTT for streaming")
        
        if self._recorder:
            await self._process_streaming()
        else:
            logger.info("Using chunked WhisperX transcription")
            await self._process_chunked()
    
    async def _process_streaming(self):
        """Process audio via RealtimeSTT."""
        while self.is_running:
            try:
                chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
                pcm_audio = await self._convert_to_pcm(chunk)
                if pcm_audio and self._recorder:
                    self._recorder.feed_audio(pcm_audio)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Streaming error: {e}")
    
    async def _convert_to_pcm(self, webm_data: bytes) -> Optional[bytes]:
        """Convert webm/opus to PCM."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                f.write(webm_data)
                temp_path = f.name
            
            pcm_path = temp_path.replace(".webm", ".raw")
            process = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", temp_path,
                "-ar", "16000", "-ac", "1", "-f", "s16le", pcm_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.wait()
            
            pcm_data = None
            if os.path.exists(pcm_path):
                with open(pcm_path, "rb") as f:
                    pcm_data = f.read()
                os.unlink(pcm_path)
            os.unlink(temp_path)
            return pcm_data
        except Exception as e:
            logger.error(f"PCM conversion error: {e}")
            return None
    
    async def _process_chunked(self):
        """Fallback: Process audio in 3s chunks via WhisperX."""
        accumulated = b""
        
        while self.is_running:
            try:
                chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
                accumulated += chunk
                
                if len(accumulated) > 18000:  # ~3s of webm audio
                    await self._transcribe_chunk(accumulated)
                    accumulated = b""
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Chunked processing error: {e}")
    
    async def _transcribe_chunk(self, audio_data: bytes):
        """Transcribe an audio chunk via WhisperX."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name
            
            wav_path = temp_path.replace(".webm", ".wav")
            process = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", temp_path,
                "-ar", "16000", "-ac", "1", "-f", "wav", wav_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.wait()
            
            if os.path.exists(wav_path):
                text = await self._whisper_transcribe(wav_path)
                
                if text and text.strip():
                    segment = TranscriptSegment(
                        speaker="Speaker",
                        text=text.strip(),
                        timestamp=time.time(),
                        is_final=True
                    )
                    self.transcript_buffer.append(segment)
                    self.full_transcript.append(segment)
                    self.on_transcript(segment)
                
                os.unlink(wav_path)
            os.unlink(temp_path)
        except Exception as e:
            logger.error(f"Transcription error: {e}")
    
    async def _whisper_transcribe(self, audio_path: str) -> str:
        """Transcribe via WhisperX."""
        try:
            if self._transcriber is None:
                from src.services.transcriber import TranscriptionService
                self._transcriber = TranscriptionService()
            
            result = self._transcriber.transcribe(
                audio_path=audio_path,
                language="en",
                diarize=False
            )
            return result.get("full_transcript", "")
        except Exception as e:
            logger.error(f"WhisperX error: {e}")
            return ""
    
    async def _periodic_llm_update(self):
        """Periodically run LLM for canvas updates."""
        while self.is_running:
            await asyncio.sleep(5)
            
            if self.transcript_buffer and (time.time() - self.last_llm_update) >= self.update_interval:
                await self._run_llm_update()
    
    async def _run_llm_update(self, final: bool = False):
        """Generate canvas updates via LLM."""
        if not self.transcript_buffer and not final:
            return
        
        try:
            buffered_text = "\n".join(f"{s.speaker}: {s.text}" for s in self.transcript_buffer)
            existing = {eid: {"type": e.element_type.value, "content": e.content} 
                       for eid, e in self.canvas_state.items()}
            
            prompt = self._build_llm_prompt(buffered_text, existing, final)
            response = await self._call_llm(prompt)
            updates = self._parse_llm_response(response)
            
            for update in updates:
                self._apply_canvas_update(update)
                self.on_canvas_update(update)
            
            self.transcript_buffer.clear()
            self.last_llm_update = time.time()
        except Exception as e:
            logger.error(f"LLM update error: {e}")
    
    def _build_llm_prompt(self, new_transcript: str, existing: Dict, final: bool) -> str:
        existing_json = json.dumps(existing, indent=2) if existing else "{}"
        task = "Generate a final meeting summary." if final else "Update the meeting whiteboard."
        
        return f"""You are a meeting assistant managing a live digital whiteboard.

CURRENT CANVAS STATE:
{existing_json}

NEW TRANSCRIPT SINCE LAST UPDATE:
{new_transcript}

TASK: {task}

Output a JSON array of canvas updates:
{{
  "action": "create" | "update" | "delete",
  "element_id": "unique-id",
  "element_type": "summary" | "action_item" | "decision" | "topic" | "mood" | "note",
  "content": "markdown text content"
}}

RULES:
1. Use "update" for existing elements (same element_id)
2. Use "create" for new elements
3. Keep summaries concise and evolving
4. Extract action items with owner names
5. Note key decisions
6. Track the current topic

CONTENT FORMATTING:
- Use **bold** for emphasis and names
- Use bullet points (- item) for lists within summaries
- Use > for notable quotes
- Keep action items as single lines: "**John**: Review PR #234 by Friday"
- Keep decisions as single lines: "Agreed to **move deadline to Friday**"

{"FINAL: Create a comprehensive summary of the entire meeting with all key points, action items, and decisions clearly formatted." if final else ""}

Respond with ONLY a JSON array."""

    async def _call_llm(self, prompt: str) -> str:
        try:
            if self.llm_service:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.llm_service.chat(message=prompt, max_tokens=1000)
                )
                return response.get("content", "[]")
            return "[]"
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "[]"
    
    def _parse_llm_response(self, response: str) -> List[CanvasUpdate]:
        updates = []
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            data = json.loads(response)
            if not isinstance(data, list):
                data = [data]
            
            for item in data:
                updates.append(CanvasUpdate(
                    action=item.get("action", "create"),
                    element_id=item.get("element_id", f"elem-{time.time()}"),
                    element_type=item.get("element_type", "note"),
                    content=item.get("content", ""),
                    position=item.get("position")
                ))
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
        return updates
    
    def _apply_canvas_update(self, update: CanvasUpdate):
        if update.action == "delete":
            self.canvas_state.pop(update.element_id, None)
        else:
            position = update.position or self._get_next_position(update.element_type)
            self.canvas_state[update.element_id] = CanvasElement(
                element_id=update.element_id,
                element_type=CanvasElementType(update.element_type),
                content=update.content,
                position=position
            )
    
    def _get_next_position(self, element_type: str) -> Dict[str, float]:
        column_map = {
            "summary": "left", "topic": "left",
            "action_item": "center", "decision": "center", "note": "center",
            "mood": "right", "chart": "right",
        }
        column = column_map.get(element_type, "center")
        x = self._column_x[column]
        y = self._next_y_position
        self._next_y_position += 100
        if self._next_y_position > 600:
            self._next_y_position = 50
        return {"x": x, "y": y}
    
    def get_full_transcript(self) -> str:
        return "\n".join(f"[{s.speaker}] {s.text}" for s in self.full_transcript)
    
    def get_session_data(self) -> Dict[str, Any]:
        """
        Get all session data for the frontend to store.
        
        Returns everything needed to persist the meeting.
        """
        duration = (self.end_time or time.time()) - self.start_time
        
        return {
            "session_id": self.session_id,
            "external_meeting_id": self.external_meeting_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": duration,
            "transcript": self.get_full_transcript(),
            "transcript_segments": [
                {"speaker": s.speaker, "text": s.text, "timestamp": s.timestamp}
                for s in self.full_transcript
            ],
            "canvas_elements": {
                eid: {
                    "element_id": e.element_id,
                    "element_type": e.element_type.value,
                    "content": e.content,
                    "position": e.position
                }
                for eid, e in self.canvas_state.items()
            },
            "summary": next(
                (e.content for e in self.canvas_state.values() 
                 if e.element_type == CanvasElementType.SUMMARY), 
                None
            ),
            "action_items": [
                e.content for e in self.canvas_state.values()
                if e.element_type == CanvasElementType.ACTION_ITEM
            ],
            "decisions": [
                e.content for e in self.canvas_state.values()
                if e.element_type == CanvasElementType.DECISION
            ],
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence/recovery."""
        return {
            "session_id": self.session_id,
            "external_meeting_id": self.external_meeting_id,
            "start_time": self.start_time,
            "transcript_buffer": [
                {"speaker": s.speaker, "text": s.text, "timestamp": s.timestamp, "is_final": s.is_final}
                for s in self.transcript_buffer
            ],
            "full_transcript": [
                {"speaker": s.speaker, "text": s.text, "timestamp": s.timestamp, "is_final": s.is_final}
                for s in self.full_transcript
            ],
            "canvas_state": {
                eid: {
                    "element_id": e.element_id,
                    "element_type": e.element_type.value,
                    "content": e.content,
                    "position": e.position,
                }
                for eid, e in self.canvas_state.items()
            },
            "last_llm_update": self.last_llm_update,
            "next_y_position": self._next_y_position,
        }
    
    def restore_from_dict(self, data: Dict[str, Any]):
        """Restore from persisted data."""
        self.external_meeting_id = data.get("external_meeting_id")
        self.start_time = data.get("start_time", time.time())
        
        self.transcript_buffer = [
            TranscriptSegment(s["speaker"], s["text"], s["timestamp"], s.get("is_final", True))
            for s in data.get("transcript_buffer", [])
        ]
        self.full_transcript = [
            TranscriptSegment(s["speaker"], s["text"], s["timestamp"], s.get("is_final", True))
            for s in data.get("full_transcript", [])
        ]
        self.canvas_state = {
            eid: CanvasElement(
                d["element_id"], CanvasElementType(d["element_type"]),
                d["content"], d.get("position", {"x": 0, "y": 0})
            )
            for eid, d in data.get("canvas_state", {}).items()
        }
        self.last_llm_update = data.get("last_llm_update", time.time())
        self._next_y_position = data.get("next_y_position", 50)
        
        logger.info(f"Restored session {self.session_id}: {len(self.full_transcript)} segments")


class LiveSessionManager:
    """Manages multiple live sessions with persistence for recovery."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.sessions: Dict[str, LiveSession] = {}
        self._llm_service = None
        self._persistence_dir = os.path.join(tempfile.gettempdir(), "angel_live_sessions")
        self._save_interval = 10.0
        self._initialized = True
        
        os.makedirs(self._persistence_dir, exist_ok=True)
        logger.info(f"LiveSessionManager initialised (persistence: {self._persistence_dir})")
    
    def _get_llm_service(self):
        if self._llm_service is None:
            try:
                from src.services.interactive import InteractiveService
                self._llm_service = InteractiveService()
            except Exception as e:
                logger.error(f"Failed to load InteractiveService: {e}")
        return self._llm_service
    
    def _get_session_file(self, session_id: str) -> str:
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return os.path.join(self._persistence_dir, f"{safe_id}.json")
    
    def _save_session(self, session: LiveSession):
        try:
            filepath = self._get_session_file(session.session_id)
            data = session.to_dict()
            data["saved_at"] = time.time()
            with open(filepath, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
    
    def _load_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            filepath = self._get_session_file(session_id)
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    data = json.load(f)
                # Discard if > 24h old
                if (time.time() - data.get("saved_at", 0)) > 86400:
                    os.unlink(filepath)
                    return None
                return data
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
        return None
    
    def _delete_session_file(self, session_id: str):
        try:
            filepath = self._get_session_file(session_id)
            if os.path.exists(filepath):
                os.unlink(filepath)
        except Exception as e:
            logger.warning(f"Failed to delete session file: {e}")
    
    async def _periodic_save(self, session: LiveSession):
        while session.is_running:
            await asyncio.sleep(self._save_interval)
            if session.is_running:
                self._save_session(session)
    
    async def create_session(
        self,
        session_id: str,
        on_transcript: Callable[[TranscriptSegment], None],
        on_canvas_update: Callable[[CanvasUpdate], None],
        external_meeting_id: Optional[str] = None,
        update_interval: float = 20.0
    ) -> LiveSession:
        """Create a new live session."""
        if session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")
        
        session = LiveSession(
            session_id=session_id,
            on_transcript=on_transcript,
            on_canvas_update=on_canvas_update,
            llm_service=self._get_llm_service(),
            update_interval=update_interval,
            external_meeting_id=external_meeting_id
        )
        
        self.sessions[session_id] = session
        await session.start()
        asyncio.create_task(self._periodic_save(session))
        
        return session
    
    async def recover_session(
        self,
        session_id: str,
        on_transcript: Callable[[TranscriptSegment], None],
        on_canvas_update: Callable[[CanvasUpdate], None],
        update_interval: float = 20.0
    ) -> Optional[LiveSession]:
        """Recover a session from persistence."""
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        data = self._load_session_data(session_id)
        if not data:
            return None
        
        session = LiveSession(
            session_id=session_id,
            on_transcript=on_transcript,
            on_canvas_update=on_canvas_update,
            llm_service=self._get_llm_service(),
            update_interval=update_interval
        )
        session.restore_from_dict(data)
        
        self.sessions[session_id] = session
        await session.start()
        asyncio.create_task(self._periodic_save(session))
        
        logger.info(f"Recovered session {session_id}")
        return session
    
    async def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """End a session and return all data for frontend to store."""
        session = self.sessions.pop(session_id, None)
        
        if session:
            await session.stop()
            self._delete_session_file(session_id)
            return session.get_session_data()
        
        return None
    
    def get_session(self, session_id: str) -> Optional[LiveSession]:
        return self.sessions.get(session_id)
    
    def has_saved_session(self, session_id: str) -> bool:
        return os.path.exists(self._get_session_file(session_id))
