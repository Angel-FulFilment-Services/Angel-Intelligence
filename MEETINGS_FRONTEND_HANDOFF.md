# Meetings & Live Whiteboard - Frontend Handoff

## Overview

This document details the frontend implementation for the live meetings feature, including real-time transcription, AI-powered whiteboard updates, and meeting management.

### Use Case

This is designed for **in-person meetings** with a device (laptop/PC) set up in a meeting room to:

1. **Capture microphone audio** - For people physically in the room
2. **Capture system/computer audio** - For remote participants on a video call (Teams, Zoom, etc.)
3. **Display on a TV** - The whiteboard and transcript are shown on a large screen for everyone to see

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MEETING ROOM SETUP                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│     ┌─────────────────────┐                                                  │
│     │    CONFERENCE TV    │  ◄── Displays whiteboard + live transcript      │
│     │                     │                                                  │
│     │  ┌───────────────┐  │                                                  │
│     │  │  WHITEBOARD   │  │                                                  │
│     │  │  (Summary,    │  │                                                  │
│     │  │  Actions,     │  │                                                  │
│     │  │  Decisions)   │  │                                                  │
│     │  └───────────────┘  │                                                  │
│     │  ┌───────────────┐  │                                                  │
│     │  │  TRANSCRIPT   │  │                                                  │
│     │  └───────────────┘  │                                                  │
│     └─────────────────────┘                                                  │
│                                                                              │
│     ┌─────────────┐         ┌─────────────┐                                  │
│     │   LAPTOP    │─────────│   🎤 MIC    │  ◄── Captures room audio        │
│     │  (Running   │         └─────────────┘                                  │
│     │   browser)  │                                                          │
│     │             │─────────[System Audio] ◄── Captures remote participants  │
│     └─────────────┘         (Teams/Zoom)                                     │
│                                                                              │
│     👤 👤 👤 👤 👤  ◄── In-person attendees                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │  Meetings List  │───▶│  Meeting Room   │───▶│  Meeting Summary View   │  │
│  │     Page        │    │     Page        │    │        Page             │  │
│  └─────────────────┘    └────────┬────────┘    └─────────────────────────┘  │
│                                  │                                           │
│                         ┌────────▼────────┐                                  │
│                         │   WebSocket     │                                  │
│                         │   Connection    │                                  │
│                         └────────┬────────┘                                  │
└──────────────────────────────────┼──────────────────────────────────────────┘
                                   │
                    ───────────────┼───────────────
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BACKEND                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  WebSocket: /api/v1/live/session/{session_id}                               │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   WhisperX   │───▶│  Transcript  │───▶│    Groq/     │                   │
│  │   (STT)      │    │   Buffer     │    │    vLLM      │                   │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                   │
│                                                  │                           │
│                                          ┌──────▼───────┐                   │
│                                          │    Canvas    │                   │
│                                          │   Updates    │                   │
│                                          └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Database Migration

### ai_meetings Table

```sql
CREATE TABLE ai_meetings (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    
    -- Identification
    meeting_id VARCHAR(36) NOT NULL UNIQUE,  -- UUID, matches session_id in WebSocket
    title VARCHAR(255) DEFAULT NULL,
    description TEXT DEFAULT NULL,
    
    -- Organizer
    organizer_user_id INT UNSIGNED DEFAULT NULL,
    organizer_name VARCHAR(255) DEFAULT NULL,
    organizer_email VARCHAR(255) DEFAULT NULL,
    
    -- Timing
    scheduled_start DATETIME DEFAULT NULL,
    actual_start DATETIME DEFAULT NULL,
    actual_end DATETIME DEFAULT NULL,
    duration_seconds INT UNSIGNED DEFAULT NULL,
    
    -- Status
    status ENUM('scheduled', 'in_progress', 'completed', 'cancelled') DEFAULT 'scheduled',
    
    -- Transcript (full text for search)
    transcript_text LONGTEXT DEFAULT NULL,
    
    -- Transcript segments (JSON array with timestamps)
    transcript_segments JSON DEFAULT NULL,
    -- Format: [{"speaker": "John", "text": "Hello...", "timestamp": 1234567890.123}, ...]
    
    -- Whiteboard/Canvas state (JSON object)
    canvas_state JSON DEFAULT NULL,
    -- Format: {"elem-1": {"element_type": "summary", "content": "...", "position": {"x": 50, "y": 50}}, ...}
    
    -- AI-generated content (extracted from canvas for quick access)
    summary TEXT DEFAULT NULL,
    action_items JSON DEFAULT NULL,       -- ["Action 1", "Action 2", ...]
    decisions JSON DEFAULT NULL,          -- ["Decision 1", "Decision 2", ...]
    topics JSON DEFAULT NULL,             -- ["Topic 1", "Topic 2", ...]
    mood VARCHAR(50) DEFAULT NULL,        -- "positive", "neutral", "tense", etc.
    
    -- Participants (JSON array)
    participants JSON DEFAULT NULL,
    -- Format: [{"user_id": 123, "name": "John Doe", "email": "john@example.com", "role": "organizer|participant", "joined_at": "2026-01-30T10:00:00Z"}, ...]
    
    -- Metadata
    tags JSON DEFAULT NULL,               -- ["weekly-standup", "project-x", ...]
    client_ref VARCHAR(100) DEFAULT NULL, -- Link to client if relevant
    
    -- Recording (optional - if audio is saved)
    recording_url VARCHAR(500) DEFAULT NULL,
    recording_duration_seconds INT UNSIGNED DEFAULT NULL,
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_status (status),
    INDEX idx_organizer (organizer_user_id),
    INDEX idx_scheduled (scheduled_start),
    INDEX idx_actual_start (actual_start),
    INDEX idx_client (client_ref),
    FULLTEXT INDEX idx_transcript_search (transcript_text, summary)
    
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### ai_meeting_participants Table (Optional - for tracking join/leave)

```sql
CREATE TABLE ai_meeting_participants (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    meeting_id VARCHAR(36) NOT NULL,
    user_id INT UNSIGNED DEFAULT NULL,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) DEFAULT NULL,
    role ENUM('organizer', 'participant', 'viewer') DEFAULT 'participant',
    joined_at DATETIME DEFAULT NULL,
    left_at DATETIME DEFAULT NULL,
    duration_seconds INT UNSIGNED DEFAULT NULL,
    
    FOREIGN KEY (meeting_id) REFERENCES ai_meetings(meeting_id) ON DELETE CASCADE,
    INDEX idx_meeting (meeting_id),
    INDEX idx_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

---

## Page 1: Meetings List

### Route
```
/meetings
```

### Features
- List all meetings (paginated)
- Filter by: status, date range, organizer, tags
- Search by: title, transcript content, summary
- Quick actions: Start new meeting, view meeting, delete

### UI Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  MEETINGS                                              [+ New Meeting]       │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ 🔍 Search meetings...                    [Status ▼] [Date ▼] [Tags ▼]  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ 📅 Weekly Team Standup                              🟢 In Progress      ││
│  │ Started 10 minutes ago • 5 participants • John Doe (organizer)          ││
│  │ Current topic: Sprint review                                             ││
│  │                                                    [Join] [View Live]    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ 📋 Project X Planning                               ✅ Completed         ││
│  │ Yesterday at 2:00 PM • 45 min • 3 participants                          ││
│  │ Summary: Discussed Q2 roadmap, assigned sprint tasks...                 ││
│  │ 📌 3 action items • 2 decisions                                         ││
│  │                                            [View Summary] [Transcript]   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ 🗓️ Client Review - ABC Corp                        📆 Scheduled         ││
│  │ Tomorrow at 10:00 AM • 0 participants                                   ││
│  │                                                    [Edit] [Start Now]    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  [< Previous]                    Page 1 of 5                    [Next >]    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### API Calls

**List Meetings (create this endpoint in frontend API layer):**
```
GET /api/meetings?status=all&page=1&limit=20&search=&date_from=&date_to=
```

**Response:**
```json
{
  "meetings": [
    {
      "meeting_id": "550e8400-e29b-41d4-a716-446655440000",
      "title": "Weekly Team Standup",
      "status": "in_progress",
      "actual_start": "2026-01-30T10:00:00Z",
      "duration_seconds": 600,
      "organizer_name": "John Doe",
      "participant_count": 5,
      "summary": "Discussing sprint progress...",
      "action_item_count": 3,
      "decision_count": 2
    }
  ],
  "total": 47,
  "page": 1,
  "pages": 5
}
```

**Create New Meeting:**
```
POST /api/meetings
{
  "title": "Project Planning",
  "description": "Q2 roadmap discussion",
  "scheduled_start": "2026-01-31T10:00:00Z",
  "organizer_user_id": 123,
  "organizer_name": "John Doe",
  "organizer_email": "john@example.com"
}
```

**Response:**
```json
{
  "meeting_id": "550e8400-e29b-41d4-a716-446655440001",
  "title": "Project Planning",
  "status": "scheduled",
  "join_url": "/meetings/550e8400-e29b-41d4-a716-446655440001"
}
```

---

## Page 2: Meeting Room (Live)

### Route
```
/meetings/:meeting_id
```

### Features
- Live transcription display (TV-optimized, large fonts)
- AI-powered whiteboard canvas
- Dual audio capture (microphone + system audio)
- Meeting controls (start/stop)
- Real-time updates via WebSocket
- Designed for display on conference room TV

### UI Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Weekly Team Standup                    🔴 Recording  ⏱️ 00:12:34  [End]   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐│
│  │                         WHITEBOARD CANVAS (TV Display)                   ││
│  │                                                                          ││
│  │  ┌─────────────────────────────────────────────────────────────────────┐ ││
│  │  │ 📝 SUMMARY                                                          │ ││
│  │  │ Team discussed sprint progress. Backend is on track. Frontend      │ ││
│  │  │ needs more time - dashboard 80% complete, ready Thursday.          │ ││
│  │  └─────────────────────────────────────────────────────────────────────┘ ││
│  │                                                                          ││
│  │  ┌────────────────────────────────┐  ┌────────────────────────────────┐  ││
│  │  │ ✅ ACTION ITEMS                │  │ 💡 DECISIONS                   │  ││
│  │  │                                │  │                                │  ││
│  │  │ • John: Review PR #234        │  │ • Move deadline to Friday      │  ││
│  │  │ • Jane: Update designs        │  │ • Add extra QA day             │  ││
│  │  │ • Bob: Fix auth bug           │  │                                │  ││
│  │  └────────────────────────────────┘  └────────────────────────────────┘  ││
│  │                                                                          ││
│  │  ┌────────────────────────────────┐  ┌────────────────────────────────┐  ││
│  │  │ 🎯 TOPIC: Sprint Planning      │  │ 😊 MOOD: Positive              │  ││
│  │  └────────────────────────────────┘  └────────────────────────────────┘  ││
│  │                                                                          ││
│  └──────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                           LIVE TRANSCRIPT                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [10:00:15] John Doe: Alright, let's get started with the standup.         │
│                                                                              │
│  [10:00:23] John Doe: Jane, can you give us an update on the frontend?     │
│                                                                              │
│  [10:00:28] Jane Smith: Sure. We're about 80% done with the new dashboard. │
│             Should be ready for review by Thursday.                          │
│                                                                              │
│  [10:00:45] Bob Wilson: I finished the authentication refactor yesterday.  │
│             Just need to write the tests.                                   │
│                                                                              │
│  [10:01:02] John Doe: Great progress everyone. Let's talk about the...     │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (typing indicator)          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  🎤 Room Mic: ✅ Active    🔊 System Audio: ✅ Active    ⏱️ 00:12:34        │
│                                                                              │
│  [⏸️ Pause]  [⏹️ End Meeting]                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## WebSocket Integration

### Connection

```javascript
// meeting-websocket.js

class MeetingWebSocket {
  constructor(meetingId, callbacks) {
    this.meetingId = meetingId;
    this.callbacks = callbacks;
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.audioContext = null;
    this.mediaRecorder = null;
  }

  connect() {
    const wsUrl = `wss://${window.location.host}/api/v1/live/session/${this.meetingId}?meeting_id=${this.meetingId}`;
    
    this.ws = new WebSocket(wsUrl);
    
    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.callbacks.onConnected?.();
    };
    
    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.handleMessage(message);
    };
    
    this.ws.onclose = () => {
      console.log('WebSocket closed');
      this.callbacks.onDisconnected?.();
      this.attemptReconnect();
    };
    
    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.callbacks.onError?.(error);
    };
  }

  handleMessage(message) {
    switch (message.type) {
      case 'connected':
        this.callbacks.onSessionInfo?.(message.data);
        break;
        
      case 'transcript':
        // { speaker, text, timestamp, is_final, is_recovery? }
        this.callbacks.onTranscript?.(message.data);
        break;
        
      case 'canvas_update':
        // { action, element_id, element_type, content, position, is_recovery? }
        this.callbacks.onCanvasUpdate?.(message.data);
        break;
        
      case 'recovery_complete':
        this.callbacks.onRecoveryComplete?.(message.data);
        break;
        
      case 'session_ended':
        // Full session data - save to database
        this.callbacks.onSessionEnded?.(message.data);
        break;
        
      case 'pong':
        // Heartbeat response
        break;
        
      default:
        console.log('Unknown message type:', message.type);
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
      console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
      setTimeout(() => this.connect(), delay);
    } else {
      this.callbacks.onReconnectFailed?.();
    }
  }

  // Start capturing and sending audio (microphone + system audio)
  async startAudioCapture() {
    try {
      // Capture microphone audio (room participants)
      const micStream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        } 
      });
      
      // Capture system audio (remote participants via Teams/Zoom)
      // Requires user to select "Share system audio" or specific tab/window
      let systemStream = null;
      try {
        systemStream = await navigator.mediaDevices.getDisplayMedia({
          video: true,  // Required but we won't use it
          audio: {
            channelCount: 1,
            sampleRate: 16000,
          }
        });
        // Stop the video track - we only need audio
        systemStream.getVideoTracks().forEach(track => track.stop());
      } catch (e) {
        console.warn('System audio capture not available:', e);
        // Continue with mic only
      }
      
      // Merge both audio streams if system audio is available
      let combinedStream;
      if (systemStream && systemStream.getAudioTracks().length > 0) {
        const audioContext = new AudioContext({ sampleRate: 16000 });
        const destination = audioContext.createMediaStreamDestination();
        
        const micSource = audioContext.createMediaStreamSource(micStream);
        const systemSource = audioContext.createMediaStreamSource(systemStream);
        
        // Mix both sources
        micSource.connect(destination);
        systemSource.connect(destination);
        
        combinedStream = destination.stream;
        this.audioContext = audioContext;
        this.systemStream = systemStream;
      } else {
        combinedStream = micStream;
      }
      
      this.micStream = micStream;
      
      this.mediaRecorder = new MediaRecorder(combinedStream, {
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 16000,
      });
      
      this.mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0 && this.ws?.readyState === WebSocket.OPEN) {
          const base64 = await this.blobToBase64(event.data);
          this.ws.send(JSON.stringify({
            type: 'audio_chunk',
            data: base64
          }));
        }
      };
      
      // Send audio chunks every 250ms
      this.mediaRecorder.start(250);
      
      return { 
        mic: true, 
        system: systemStream?.getAudioTracks().length > 0 
      };
    } catch (error) {
      console.error('Failed to start audio capture:', error);
      this.callbacks.onError?.(error);
      return false;
    }
  }

  stopAudioCapture() {
    if (this.mediaRecorder) {
      this.mediaRecorder.stop();
      this.mediaRecorder = null;
    }
    // Stop microphone stream
    if (this.micStream) {
      this.micStream.getTracks().forEach(track => track.stop());
      this.micStream = null;
    }
    // Stop system audio stream
    if (this.systemStream) {
      this.systemStream.getTracks().forEach(track => track.stop());
      this.systemStream = null;
    }
    // Close audio context
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }

  async blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  endSession() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'end_session' }));
    }
  }

  ping() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'ping' }));
    }
  }

  disconnect() {
    this.stopAudioCapture();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

export default MeetingWebSocket;
```

### React Component Example

```jsx
// MeetingRoom.jsx

import React, { useState, useEffect, useRef, useCallback } from 'react';
import MeetingWebSocket from './meeting-websocket';

import ReactMarkdown from 'react-markdown';

function MeetingRoom({ meetingId }) {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState([]);
  const [canvasElements, setCanvasElements] = useState({});
  const [duration, setDuration] = useState(0);
  const [pendingText, setPendingText] = useState('');
  const [audioStatus, setAudioStatus] = useState({ mic: false, system: false });
  
  const wsRef = useRef(null);
  const timerRef = useRef(null);
  const transcriptEndRef = useRef(null);

  // Scroll to bottom of transcript
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcript]);

  // Connect on mount
  useEffect(() => {
    wsRef.current = new MeetingWebSocket(meetingId, {
      onConnected: () => setIsConnected(true),
      onDisconnected: () => setIsConnected(false),
      
      onSessionInfo: (data) => {
        console.log('Session info:', data);
        if (data.recovered) {
          // Will receive recovery data next
        }
      },
      
      onTranscript: (data) => {
        if (data.is_final) {
          setTranscript(prev => [...prev, {
            speaker: data.speaker,
            text: data.text,
            timestamp: data.timestamp,
          }]);
          setPendingText('');
        } else {
          // Show interim/partial transcription
          setPendingText(data.text);
        }
      },
      
      onCanvasUpdate: (update) => {
        setCanvasElements(prev => {
          if (update.action === 'delete') {
            const { [update.element_id]: _, ...rest } = prev;
            return rest;
          }
          return {
            ...prev,
            [update.element_id]: {
              element_type: update.element_type,
              content: update.content,
              position: update.position,
            }
          };
        });
      },
      
      onRecoveryComplete: (data) => {
        console.log('Recovery complete:', data);
      },
      
      onSessionEnded: async (sessionData) => {
        // Save to database
        await saveMeetingData(meetingId, sessionData);
        // Redirect to summary view
        window.location.href = `/meetings/${meetingId}/summary`;
      },
      
      onError: (error) => {
        console.error('WebSocket error:', error);
      },
      
      onReconnectFailed: () => {
        alert('Lost connection to meeting. Please refresh the page.');
      },
    });
    
    wsRef.current.connect();
    
    // Heartbeat every 30s
    const heartbeat = setInterval(() => wsRef.current?.ping(), 30000);
    
    return () => {
      clearInterval(heartbeat);
      wsRef.current?.disconnect();
    };
  }, [meetingId]);

  // Duration timer
  useEffect(() => {
    if (isRecording) {
      timerRef.current = setInterval(() => {
        setDuration(prev => prev + 1);
      }, 1000);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [isRecording]);

  const startRecording = async () => {
    const result = await wsRef.current?.startAudioCapture();
    if (result) {
      setIsRecording(true);
      setAudioStatus({ mic: result.mic, system: result.system });
      // Update meeting status in database
      await updateMeetingStatus(meetingId, 'in_progress');
      
      if (!result.system) {
        console.warn('System audio not captured - remote participants will not be transcribed');
      }
    }
  };

  const stopRecording = () => {
    wsRef.current?.stopAudioCapture();
    setIsRecording(false);
  };

  const endMeeting = () => {
    if (confirm('Are you sure you want to end this meeting?')) {
      wsRef.current?.endSession();
    }
  };

  const formatTime = (seconds) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const formatTimestamp = (ts) => {
    return new Date(ts * 1000).toLocaleTimeString();
  };

  // Group canvas elements by type
  const summary = Object.values(canvasElements).find(e => e.element_type === 'summary');
  const actionItems = Object.values(canvasElements).filter(e => e.element_type === 'action_item');
  const decisions = Object.values(canvasElements).filter(e => e.element_type === 'decision');
  const currentTopic = Object.values(canvasElements).find(e => e.element_type === 'topic');
  const mood = Object.values(canvasElements).find(e => e.element_type === 'mood');

  return (
    <div className="meeting-room">
      {/* Header */}
      <header className="meeting-header">
        <h1>Weekly Team Standup</h1>
        <div className="meeting-status">
          {isRecording && <span className="recording-indicator">🔴 Recording</span>}
          <span className="duration">⏱️ {formatTime(duration)}</span>
          {isRecording && (
            <>
              <span className="audio-status mic">
                🎤 Room Mic: {audioStatus.mic ? '✅' : '❌'}
              </span>
              <span className="audio-status system">
                🔊 System: {audioStatus.system ? '✅' : '⚠️ Not captured'}
              </span>
            </>
          )}
          <span className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
          </span>
        </div>
      </header>

      <div className="meeting-content">
        {/* Whiteboard Canvas */}
        <div className="whiteboard-panel">
          <h2>Whiteboard</h2>
          
          {summary && (
            <div className="canvas-card summary">
              <h3>📝 Summary</h3>
              <ReactMarkdown>{summary.content}</ReactMarkdown>
            </div>
          )}
          
          {actionItems.length > 0 && (
            <div className="canvas-card action-items">
              <h3>✅ Action Items</h3>
              <ul>
                {actionItems.map((item, i) => (
                  <li key={i}><ReactMarkdown>{item.content}</ReactMarkdown></li>
                ))}
              </ul>
            </div>
          )}
          
          {decisions.length > 0 && (
            <div className="canvas-card decisions">
              <h3>💡 Decisions</h3>
              <ul>
                {decisions.map((item, i) => (
                  <li key={i}><ReactMarkdown>{item.content}</ReactMarkdown></li>
                ))}
              </ul>
            </div>
          )}
          
          {currentTopic && (
            <div className="canvas-card topic">
              <h3>🎯 Current Topic</h3>
              <ReactMarkdown>{currentTopic.content}</ReactMarkdown>
            </div>
          )}
          
          {mood && (
            <div className="canvas-card mood">
              <h3>😊 Mood</h3>
              <ReactMarkdown>{mood.content}</ReactMarkdown>
            </div>
          )}
        </div>

        {/* Participants */}
        <div className="participants-panel">
          <h2>Participants ({participants.length})</h2>
          <ul>
            {participants.map((p, i) => (
              <li key={i} className={`participant ${p.role}`}>
                <span className="avatar">👤</span>
                <span className="name">{p.name}</span>
                {p.role === 'organizer' && <span className="badge">Organizer</span>}
              </li>
            ))}
          </ul>
          <button className="invite-button">🔗 Copy Invite Link</button>
        </div>
      </div>

      {/* Live Transcript */}
      <div className="transcript-panel">
        <h2>Live Transcript</h2>
        <div className="transcript-content">
          {transcript.map((seg, i) => (
            <div key={i} className="transcript-segment">
              <span className="timestamp">[{formatTimestamp(seg.timestamp)}]</span>
              <span className="speaker">{seg.speaker}:</span>
              <span className="text">{seg.text}</span>
            </div>
          ))}
          {pendingText && (
            <div className="transcript-segment pending">
              <span className="text">{pendingText}...</span>
            </div>
          )}
          <div ref={transcriptEndRef} />
        </div>
      </div>

      {/* Controls */}
      <div className="meeting-controls">
        {!isRecording ? (
          <button onClick={startRecording} className="control-button start">
            🎤 Start Recording (Mic + System Audio)
          </button>
        ) : (
          <button onClick={stopRecording} className="control-button pause">
            ⏸️ Pause Recording
          </button>
        )}
        <button onClick={endMeeting} className="control-button end">
          ⏹️ End Meeting
        </button>
      </div>
    </div>
  );
}

// API helpers
async function saveMeetingData(meetingId, sessionData) {
  await fetch(`/api/meetings/${meetingId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      status: 'completed',
      actual_end: new Date().toISOString(),
      duration_seconds: sessionData.duration_seconds,
      transcript_text: sessionData.transcript,
      transcript_segments: sessionData.transcript_segments,
      canvas_state: sessionData.canvas_elements,
      summary: sessionData.summary,
      action_items: sessionData.action_items,
      decisions: sessionData.decisions,
    }),
  });
}

async function updateMeetingStatus(meetingId, status) {
  await fetch(`/api/meetings/${meetingId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      status,
      actual_start: status === 'in_progress' ? new Date().toISOString() : undefined,
    }),
  });
}

export default MeetingRoom;
```

---

## Page 3: Meeting Summary (Completed Meeting)

### Route
```
/meetings/:meeting_id/summary
```

### Features
- Full transcript with search
- Whiteboard snapshot
- Action items (with completion tracking)
- Decisions made
- Export options (PDF, text)

### UI Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ← Back to Meetings                                                          │
│                                                                              │
│  Weekly Team Standup                                         ✅ Completed    │
│  January 30, 2026 • 10:00 AM - 10:45 AM • 45 minutes                        │
│  Organizer: John Doe • 5 participants                                       │
│                                                                              │
│  [📄 Export PDF]  [📋 Copy Transcript]  [🔗 Share]  [🗑️ Delete]             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                              SUMMARY                                     ││
│  │                                                                          ││
│  │  The team discussed Q1 sprint progress. Frontend is 80% complete with   ││
│  │  the new dashboard, expected to be ready for review by Thursday.        ││
│  │  Backend authentication refactor is complete, pending tests. Team       ││
│  │  agreed to move the deadline to Friday to accommodate additional QA.    ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌──────────────────────────────────┐  ┌───────────────────────────────────┐│
│  │        ACTION ITEMS (3)          │  │         DECISIONS (2)             ││
│  │                                  │  │                                   ││
│  │  ☐ John: Review PR #234         │  │  ✓ Move deadline to Friday       ││
│  │  ☐ Jane: Update designs by Thu  │  │  ✓ Add extra QA day              ││
│  │  ☑ Bob: Fix auth bug (done)     │  │                                   ││
│  │                                  │  │                                   ││
│  └──────────────────────────────────┘  └───────────────────────────────────┘│
│                                                                              │
│  ┌──────────────────────────────────┐  ┌───────────────────────────────────┐│
│  │        PARTICIPANTS (5)          │  │         TOPICS DISCUSSED          ││
│  │                                  │  │                                   ││
│  │  👤 John Doe (organizer)        │  │  • Sprint progress                ││
│  │  👤 Jane Smith                   │  │  • Frontend dashboard             ││
│  │  👤 Bob Wilson                   │  │  • Authentication refactor        ││
│  │  👤 Alice Brown                  │  │  • Deadline adjustment            ││
│  │  👤 Charlie Davis                │  │                                   ││
│  └──────────────────────────────────┘  └───────────────────────────────────┘│
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                          FULL TRANSCRIPT                                     │
│  🔍 Search transcript...                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [10:00:15] John Doe                                                        │
│  Alright, let's get started with the standup. I want to go through          │
│  everyone's progress on the Q1 sprint.                                       │
│                                                                              │
│  [10:00:28] Jane Smith                                                       │
│  Sure. So we're about 80% done with the new dashboard. The main             │
│  components are built, we just need to polish the animations and            │
│  fix a few edge cases. Should be ready for review by Thursday.              │
│                                                                              │
│  [10:01:02] John Doe                                                         │
│  Great progress. Bob, how's the auth refactor coming along?                 │
│                                                                              │
│  [10:01:15] Bob Wilson                                                       │
│  Finished it yesterday actually. The new token refresh mechanism            │
│  is working well. Just need to write the unit tests this afternoon.         │
│                                                                              │
│  ... (scrollable)                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## WebSocket Message Reference

### Messages FROM Client → Server

| Type | Payload | Description |
|------|---------|-------------|
| `audio_chunk` | `{ data: "<base64>" }` | Audio chunk (webm/opus, ~250ms) |
| `end_session` | `{}` | End meeting, triggers final summary |
| `ping` | `{}` | Heartbeat |

### Messages FROM Server → Client

| Type | Payload | Description |
|------|---------|-------------|
| `connected` | `{ session_id, meeting_id, recovered, transcript_count, canvas_count }` | Connection established |
| `transcript` | `{ speaker, text, timestamp, is_final, is_recovery? }` | New transcript segment |
| `canvas_update` | `{ action, element_id, element_type, content, position, is_recovery? }` | Whiteboard update |
| `recovery_complete` | `{ transcript_count }` | All recovery data sent |
| `session_ended` | `{ full session data }` | Meeting ended, save this data |
| `pong` | `{}` | Heartbeat response |

### Canvas Element Types

> **Note:** All content fields contain **Markdown text**. Use a markdown renderer (e.g., `react-markdown`, `marked`) to display properly formatted content with bold, lists, links, etc.

| Type | Description | Example Content |
|------|-------------|-----------------|
| `summary` | Evolving meeting summary | "Team discussed sprint progress..." |
| `action_item` | Task with owner | "John: Review PR #234 by Friday" |
| `decision` | Key decision made | "Move deadline to Friday" |
| `topic` | Current discussion topic | "Sprint Planning" |
| `mood` | Meeting sentiment | "Positive", "Focused", "Tense" |
| `note` | General note | "Follow up on this next week" |

### Canvas Actions

| Action | When |
|--------|------|
| `create` | New element added to canvas |
| `update` | Existing element content changed |
| `delete` | Element removed from canvas |

---

## API Endpoints (Frontend Layer)

These should be implemented in your frontend's API layer (calling MySQL):

### Meetings CRUD

```
GET    /api/meetings                    # List meetings (paginated, filtered)
POST   /api/meetings                    # Create new meeting
GET    /api/meetings/:id                # Get meeting details
PATCH  /api/meetings/:id                # Update meeting (status, title, etc.)
PUT    /api/meetings/:id                # Full update (after session ends)
DELETE /api/meetings/:id                # Delete meeting
```

### Participants

```
GET    /api/meetings/:id/participants   # List participants
POST   /api/meetings/:id/participants   # Add participant
DELETE /api/meetings/:id/participants/:user_id  # Remove participant
```

### Action Items

```
PATCH  /api/meetings/:id/action-items/:index  # Mark action item complete
```

---

## CSS Styles (Example)

```css
/* meeting-room.css */

.meeting-room {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: #f5f5f5;
}

.meeting-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  background: white;
  border-bottom: 1px solid #e0e0e0;
}

.meeting-status {
  display: flex;
  gap: 1rem;
  align-items: center;
}

.recording-indicator {
  animation: pulse 1s infinite;
  color: #e53935;
  font-weight: bold;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.meeting-content {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.whiteboard-panel {
  flex: 2;
  padding: 1rem;
  overflow-y: auto;
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  align-content: flex-start;
}

.canvas-card {
  background: white;
  border-radius: 8px;
  padding: 1rem;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  width: calc(50% - 0.5rem);
}

.canvas-card.summary {
  width: 100%;
  border-left: 4px solid #2196F3;
}

.canvas-card.action-items {
  border-left: 4px solid #4CAF50;
}

.canvas-card.decisions {
  border-left: 4px solid #FF9800;
}

.canvas-card.topic {
  border-left: 4px solid #9C27B0;
}

.canvas-card.mood {
  border-left: 4px solid #E91E63;
}

.participants-panel {
  width: 250px;
  background: white;
  border-left: 1px solid #e0e0e0;
  padding: 1rem;
}

.participant {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem;
  border-radius: 4px;
}

.participant.organizer {
  background: #e3f2fd;
}

.transcript-panel {
  height: 250px;
  background: white;
  border-top: 1px solid #e0e0e0;
}

.transcript-content {
  height: calc(100% - 40px);
  overflow-y: auto;
  padding: 1rem;
  font-family: 'Consolas', monospace;
  font-size: 0.9rem;
}

.transcript-segment {
  margin-bottom: 0.75rem;
  line-height: 1.5;
}

.transcript-segment .timestamp {
  color: #666;
  margin-right: 0.5rem;
}

.transcript-segment .speaker {
  font-weight: bold;
  color: #1976D2;
  margin-right: 0.5rem;
}

.transcript-segment.pending {
  color: #999;
  font-style: italic;
}

.meeting-controls {
  display: flex;
  justify-content: center;
  gap: 1rem;
  padding: 1rem;
  background: #263238;
}

.control-button {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 24px;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.2s;
}

.control-button.start {
  background: #4CAF50;
  color: white;
}

.control-button.pause {
  background: #FF9800;
  color: white;
}

.control-button.end {
  background: #f44336;
  color: white;
}

.control-button:hover {
  transform: scale(1.05);
  box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* Markdown content styling */
.canvas-card p {
  margin: 0.5rem 0;
  line-height: 1.6;
}

.canvas-card ul, .canvas-card ol {
  margin: 0.5rem 0;
  padding-left: 1.5rem;
}

.canvas-card li {
  margin: 0.25rem 0;
}

.canvas-card strong {
  font-weight: 600;
  color: #1a1a1a;
}

.canvas-card code {
  background: #f0f0f0;
  padding: 0.1rem 0.3rem;
  border-radius: 3px;
  font-family: 'Consolas', monospace;
  font-size: 0.9em;
}

.canvas-card blockquote {
  border-left: 3px solid #ddd;
  margin: 0.5rem 0;
  padding-left: 1rem;
  color: #666;
}

/* TV Display overrides - larger fonts */
@media (min-width: 1920px) {
  .canvas-card {
    font-size: 1.25rem;
  }
  
  .canvas-card h3 {
    font-size: 1.5rem;
  }
  
  .transcript-segment {
    font-size: 1.1rem;
  }
}
```

---

## Implementation Checklist

### Phase 1: Database & API
- [ ] Run `ai_meetings` table migration
- [ ] Run `ai_meeting_participants` table migration
- [ ] Implement meetings CRUD API endpoints
- [ ] Implement participants API endpoints

### Phase 2: Dependencies & Setup
- [ ] Install `react-markdown` for rendering AI content: `npm install react-markdown`
- [ ] Install `remark-gfm` for GitHub Flavored Markdown (tables, strikethrough): `npm install remark-gfm`

### Phase 3: Meetings List Page
- [ ] Create route `/meetings`
- [ ] Implement meeting list component with filtering
- [ ] Add search functionality (full-text search on transcript)
- [ ] Add "New Meeting" flow
- [ ] Add status badges and quick actions

### Phase 4: Meeting Room Page
- [ ] Create route `/meetings/:id`
- [ ] Implement `MeetingWebSocket` class
- [ ] Implement dual audio capture (mic + system audio)
- [ ] Add system audio permission flow with browser share dialog
- [ ] Handle fallback when system audio unavailable
- [ ] Build transcript display (live + history, TV-friendly fonts)
- [ ] Build whiteboard canvas component (large, readable cards)
- [ ] Add meeting controls (start/stop/end) - for laptop view
- [ ] Add TV display mode (auto-hide controls, large fonts)
- [ ] Handle session recovery on reconnect
- [ ] Save meeting data on session end

### Phase 5: Meeting Summary Page
- [ ] Create route `/meetings/:id/summary`
- [ ] Display full transcript with search
- [ ] Display AI-generated summary
- [ ] Display action items with completion toggle
- [ ] Display decisions and topics
- [ ] Add export functionality (PDF, text)

### Phase 6: Polish
- [ ] Add loading states
- [ ] Add error handling and reconnection UI
- [ ] Add keyboard shortcuts
- [ ] Add mobile responsive design
- [ ] Add meeting invite/share links
- [ ] Add participant join/leave notifications

---

## Notes

### Audio Capture

**Dual Audio Sources:**
1. **Microphone** (`getUserMedia`) - Captures in-room participants
2. **System Audio** (`getDisplayMedia` with audio) - Captures remote participants from Teams/Zoom/etc.

**System Audio Notes:**
- Requires user to click "Share" and select a screen/window
- Must check "Share system audio" checkbox in the browser dialog
- Chrome/Edge: Full support
- Firefox: Limited support (may need about:config tweaks)
- Safari: Not supported

**Audio Format:**
- Both streams mixed to: `audio/webm;codecs=opus`
- Backend converts to: 16kHz mono PCM for WhisperX
- Chunk interval: 250ms recommended

**Fallback:**
- If system audio capture fails, only mic audio is sent
- Display warning to user that remote participants won't be transcribed

### Recovery
- Backend persists session state to disk
- On WebSocket reconnect, check `recovered` flag
- If `true`, receive all previous transcript + canvas via `is_recovery: true` messages
- Wait for `recovery_complete` before showing UI as ready

### Performance
- Canvas updates happen every ~20 seconds (configurable)
- Transcript updates are real-time (RealtimeSTT) or chunked (3s intervals)
- Debounce canvas rendering if updates come in rapidly

### TV Display Considerations

**Design for 10-foot UI (viewing from distance):**
- Minimum font size: 24px for body text, 32px+ for headings
- High contrast colors (dark text on light background or vice versa)
- Large touch targets if using touch-enabled TV
- Auto-scroll transcript to keep latest content visible
- Consider "kiosk mode" (F11 fullscreen)

**Recommended Screen Layout:**
- Whiteboard canvas: 60-70% of screen (top/left)
- Live transcript: 30-40% of screen (bottom/right)
- Minimal controls (operated from laptop, not TV)

**Auto-hide Controls:**
- On TV display, controls should auto-hide after 5 seconds
- Show only: recording status, duration, audio source indicators
- Full controls on the laptop running the capture

### Browser Support
- Requires: WebSocket, getUserMedia, getDisplayMedia (for system audio), MediaRecorder
- Full support: Chrome, Edge (recommended for system audio)
- Partial support: Firefox (system audio may need config)
- Not supported: Safari (no system audio), IE11
