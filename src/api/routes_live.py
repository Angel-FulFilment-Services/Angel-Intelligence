"""
Angel Intelligence - Live Session WebSocket Routes

WebSocket endpoints for real-time meeting transcription and whiteboard updates.
Frontend handles all meeting management - this just does transcription + canvas.
"""

import logging
import json
import asyncio
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from starlette.websockets import WebSocketState

from src.services.live_session import (
    LiveSessionManager, 
    TranscriptSegment, 
    CanvasUpdate
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/live", tags=["live-session"])


class ConnectionManager:
    """Manages WebSocket connections for live sessions."""
    
    def __init__(self):
        self.session_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.session_connections:
            self.session_connections[session_id] = set()
        self.session_connections[session_id].add(websocket)
        logger.info(f"WebSocket connected: {session_id}")
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.session_connections:
            self.session_connections[session_id].discard(websocket)
            if not self.session_connections[session_id]:
                del self.session_connections[session_id]
        logger.info(f"WebSocket disconnected: {session_id}")
    
    async def broadcast(self, session_id: str, msg_type: str, data: dict):
        if session_id not in self.session_connections:
            return
        
        message = json.dumps({"type": msg_type, "data": data})
        disconnected = set()
        
        for ws in self.session_connections[session_id]:
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(message)
            except Exception:
                disconnected.add(ws)
        
        for ws in disconnected:
            self.session_connections[session_id].discard(ws)


manager = ConnectionManager()
session_manager = LiveSessionManager()


@router.websocket("/session/{session_id}")
async def live_session_websocket(
    websocket: WebSocket,
    session_id: str,
    meeting_id: str = Query(default=None, description="Your frontend's meeting ID"),
):
    """
    WebSocket endpoint for live transcription sessions.
    
    Query params:
        meeting_id: Optional ID from your frontend's meeting system
    
    Client sends:
        {"type": "audio_chunk", "data": "<base64-audio>"}
        {"type": "end_session"}
        {"type": "ping"}
    
    Server sends:
        {"type": "connected", "data": {...}}
        {"type": "transcript", "data": {"speaker": "...", "text": "...", "is_final": bool}}
        {"type": "canvas_update", "data": {"action": "...", "element_id": "...", ...}}
        {"type": "session_ended", "data": {full session data for frontend to store}}
    """
    await manager.connect(websocket, session_id)
    
    def on_transcript(segment: TranscriptSegment):
        asyncio.create_task(manager.broadcast(session_id, "transcript", {
            "speaker": segment.speaker,
            "text": segment.text,
            "timestamp": segment.timestamp,
            "is_final": segment.is_final
        }))
    
    def on_canvas_update(update: CanvasUpdate):
        asyncio.create_task(manager.broadcast(session_id, "canvas_update", {
            "action": update.action,
            "element_id": update.element_id,
            "element_type": update.element_type,
            "content": update.content,
            "position": update.position
        }))
    
    try:
        # Try to recover or create session
        session = session_manager.get_session(session_id)
        recovered = False
        
        if not session:
            session = await session_manager.recover_session(
                session_id, on_transcript, on_canvas_update
            )
            if session:
                recovered = True
        
        if not session:
            session = await session_manager.create_session(
                session_id=session_id,
                on_transcript=on_transcript,
                on_canvas_update=on_canvas_update,
                external_meeting_id=meeting_id
            )
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "data": {
                "session_id": session_id,
                "meeting_id": session.external_meeting_id,
                "recovered": recovered,
                "transcript_count": len(session.full_transcript),
                "canvas_count": len(session.canvas_state)
            }
        })
        
        # If recovered, send existing state
        if recovered:
            for seg in session.full_transcript:
                await websocket.send_json({
                    "type": "transcript",
                    "data": {
                        "speaker": seg.speaker,
                        "text": seg.text,
                        "timestamp": seg.timestamp,
                        "is_final": True,
                        "is_recovery": True
                    }
                })
            
            for elem in session.canvas_state.values():
                await websocket.send_json({
                    "type": "canvas_update",
                    "data": {
                        "action": "create",
                        "element_id": elem.element_id,
                        "element_type": elem.element_type.value,
                        "content": elem.content,
                        "position": elem.position,
                        "is_recovery": True
                    }
                })
            
            await websocket.send_json({
                "type": "recovery_complete",
                "data": {"transcript_count": len(session.full_transcript)}
            })
        
        # Process messages
        while True:
            try:
                message = await websocket.receive()
                
                if message["type"] == "websocket.disconnect":
                    break
                
                if "text" in message:
                    data = json.loads(message["text"])
                    msg_type = data.get("type")
                    
                    if msg_type == "audio_chunk":
                        import base64
                        audio_bytes = base64.b64decode(data.get("data", ""))
                        await session.add_audio_chunk(audio_bytes)
                    
                    elif msg_type == "end_session":
                        # End and return all data for frontend to store
                        session_data = await session_manager.end_session(session_id)
                        await websocket.send_json({
                            "type": "session_ended",
                            "data": session_data
                        })
                        break
                    
                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                
                elif "bytes" in message:
                    await session.add_audio_chunk(message["bytes"])
                    
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                pass
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
    
    except Exception as e:
        logger.error(f"Session error: {e}")
    
    finally:
        manager.disconnect(websocket, session_id)


@router.get("/sessions")
async def list_active_sessions():
    """List active sessions."""
    return {
        "sessions": [
            {
                "session_id": sid,
                "meeting_id": s.external_meeting_id,
                "transcript_count": len(s.full_transcript)
            }
            for sid, s in session_manager.sessions.items()
        ],
        "count": len(session_manager.sessions)
    }


@router.get("/session/{session_id}")
async def get_session_state(session_id: str):
    """Get current session state (transcript + canvas)."""
    session = session_manager.get_session(session_id)
    
    if not session:
        return {"error": "Session not found"}
    
    return {
        "session_id": session_id,
        "meeting_id": session.external_meeting_id,
        "transcript": session.get_full_transcript(),
        "transcript_count": len(session.full_transcript),
        "canvas_elements": {
            eid: {
                "element_type": e.element_type.value,
                "content": e.content,
                "position": e.position
            }
            for eid, e in session.canvas_state.items()
        }
    }


@router.get("/session/{session_id}/recoverable")
async def check_recoverable(session_id: str):
    """Check if a session can be recovered."""
    if session_manager.get_session(session_id):
        return {"session_id": session_id, "status": "active", "recoverable": False}
    
    if session_manager.has_saved_session(session_id):
        return {"session_id": session_id, "status": "saved", "recoverable": True}
    
    return {"session_id": session_id, "status": "not_found", "recoverable": False}
