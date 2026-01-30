# Data Formats Reference

Complete JSON schema documentation for all data structures in Angel Intelligence.

## Table of Contents

1. [Transcription Segments](#transcription-segments)
2. [PII Detection](#pii-detection)
3. [Analysis Results](#analysis-results)
4. [Key Topics](#key-topics)
5. [Agent Actions Performed](#agent-actions-performed)
6. [Score Impacts](#score-impacts)
7. [Performance Scores](#performance-scores)
8. [Speaker Metrics](#speaker-metrics)
9. [Audio Observations](#audio-observations)
10. [Compliance Flags](#compliance-flags)
11. [Action Items](#action-items)

---

## Transcription Segments

Word and sentence-level segments for karaoke playback and transcript display.

### Schema

```json
{
  "segment_id": "seg_a1b2c3d4",
  "text": "Hello, thank you for calling Age UK.",
  "start": 0.0,
  "end": 2.45,
  "speaker": "agent",
  "speaker_id": "agent_123",
  "confidence": 0.9542,
  "words": [
    {"word": "Hello", "start": 0.0, "end": 0.35, "confidence": 0.98},
    {"word": "thank", "start": 0.4, "end": 0.62, "confidence": 0.97},
    {"word": "you", "start": 0.65, "end": 0.8, "confidence": 0.99},
    {"word": "for", "start": 0.85, "end": 0.95, "confidence": 0.96},
    {"word": "calling", "start": 1.0, "end": 1.4, "confidence": 0.98},
    {"word": "Age", "start": 1.5, "end": 1.75, "confidence": 0.94},
    {"word": "UK", "start": 1.8, "end": 2.45, "confidence": 0.95}
  ]
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `segment_id` | string | UUID-based unique identifier (e.g., "seg_a1b2c3d4"). Used to trace this segment through AI analysis - same ID appears in agent_actions, score_impacts, and compliance_flags when they reference this segment. |
| `text` | string | Full text of the segment |
| `start` | float | Start time in seconds (3 decimal places) |
| `end` | float | End time in seconds (3 decimal places) |
| `speaker` | string | Speaker label: "agent" or "supporter" |
| `speaker_id` | string | Unique speaker identifier |
| `confidence` | float | Transcription confidence (0.0-1.0) |
| `words` | array | Word-level timestamps for karaoke |

### Segment ID Traceability

The `segment_id` is assigned at transcription time and follows the segment through the entire analysis lifecycle:

1. **Transcription**: Segment created with UUID-based `segment_id` (e.g., "seg_a1b2c3d4")
2. **AI Analysis**: LLM references segments by their `segment_id` in its output
3. **Flagged Items**: `agent_actions`, `score_impacts`, and `compliance_flags` include `segment_ids` array linking back to original transcript segments
4. **Frontend**: Can navigate from any flagged item directly to the transcript segment

This enables full traceability from any flagged item back to the exact moment in the call.

### Word Object

| Field | Type | Description |
|-------|------|-------------|
| `word` | string | The word text |
| `start` | float | Word start time in seconds |
| `end` | float | Word end time in seconds |
| `confidence` | float | Word confidence (0.0-1.0) |

---

## PII Detection

Detected personally identifiable information with timestamps.

### Schema

```json
{
  "type": "national_insurance_number",
  "original": "AB123456C",
  "redacted": "[NI_NUMBER]",
  "timestamp_start": 45.2,
  "timestamp_end": 47.8,
  "confidence": 0.95
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | PII type (see types below) |
| `original` | string | Original detected text |
| `redacted` | string | Redaction placeholder |
| `timestamp_start` | float | Audio start time (seconds) |
| `timestamp_end` | float | Audio end time (seconds) |
| `confidence` | float | Detection confidence (0.0-1.0) |

### PII Types

| Type | Redacted As | Example |
|------|-------------|---------|
| `national_insurance_number` | `[NI_NUMBER]` | AB123456C |
| `nhs_number` | `[NHS_NUMBER]` | 123 456 7890 |
| `postcode` | `[POSTCODE]` | SW1A 1AA |
| `phone_number` | `[PHONE_NUMBER]` | 07700 900123 |
| `sort_code` | `[SORT_CODE]` | 12-34-56 |
| `bank_account` | `[ACCOUNT_NUMBER]` | 12345678 |
| `credit_card` | `[CARD_NUMBER]` | 4111 1111 1111 1111 |
| `card_expiry` | `[CARD_EXPIRY]` | 12/26 |
| `cvv` | `[CVV]` | 123 |
| `date_of_birth` | `[DOB]` | 01/02/1990 |
| `email` | `[EMAIL]` | john@example.com |
| `driving_licence` | `[DRIVING_LICENCE]` | MORGA657054SM9IJ |

---

## Analysis Results

Complete analysis output structure.

### Schema

```json
{
  "summary": "The supporter enquired about regular giving options. The agent explained the monthly donation programme and successfully enrolled the supporter at £10 per month with Gift Aid.",
  "sentiment_score": 7.5,
  "sentiment_label": "positive",
  "quality_score": 85.0,
  "key_topics": [...],
  "agent_actions_performed": [...],
  "performance_scores": {...},
  "action_items": [...],
  "compliance_flags": [...],
  "speaker_metrics": {...},
  "audio_observations": {...},
  "model_used": "Qwen2.5-Omni-7B",
  "model_version": "v1.0.0",
  "processing_time": 45.2
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `summary` | string | 2-3 sentence British English summary |
| `sentiment_score` | float | -10 (very negative) to +10 (very positive) |
| `sentiment_label` | string | "very_negative", "negative", "neutral", "positive", "very_positive" |
| `quality_score` | float | 0-100 overall quality score |
| `key_topics` | array | Topics from config detected in call |
| `agent_actions_performed` | array | Actions from config performed by agent |
| `performance_scores` | object | Rubric scores (1-10 per criterion) |
| `action_items` | array | Follow-up actions needed |
| `compliance_flags` | array | Compliance issues detected |
| `speaker_metrics` | object | Per-speaker statistics |
| `audio_observations` | object | Audio-specific analysis (audio mode only) |
| `model_used` | string | AI model name |
| `model_version` | string | Fine-tuned model version |
| `processing_time` | float | Processing time in seconds |

### Sentiment Score Mapping

| Score Range | Label |
|-------------|-------|
| 6 to 10 | very_positive |
| 2 to 5.99 | positive |
| -2 to 1.99 | neutral |
| -6 to -2.01 | negative |
| -10 to -6.01 | very_negative |

---

## Key Topics

Topics detected from the configured topics list.

### Schema

```json
{
  "name": "Regular giving signup",
  "confidence": 0.95,
  "timestamp_start": 45.5,
  "timestamp_end": 78.2
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Topic name from configuration |
| `confidence` | float | Detection confidence (0.0-1.0) |
| `timestamp_start` | float | When topic discussion started (seconds) |
| `timestamp_end` | float | When topic discussion ended (seconds) |

---

## Agent Actions Performed

Actions from the configured list that the agent performed, linked to transcript segments.

### Schema

```json
{
  "action": "Greeted supporter",
  "segment_ids": ["seg_a1b2c3d4"],
  "timestamp_start": 0.0,
  "timestamp_end": 2.5
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `action` | string | Action name from configuration |
| `segment_ids` | array | Array of segment IDs where this action occurred (links back to transcript segments) |
| `timestamp_start` | float | When action started (seconds) - derived from first segment |
| `timestamp_end` | float | When action ended (seconds) - derived from last segment |

### Segment Linking

The `segment_ids` array contains the UUID-based segment IDs from the transcript that this action references. This allows direct navigation from the action to the exact moment(s) in the call where it occurred.

For actions spanning multiple segments:
```json
{
  "action": "Explained Gift Aid",
  "segment_ids": ["seg_a1b2c3d4", "seg_e5f6g7h8", "seg_i9j0k1l2"],
  "timestamp_start": 45.2,
  "timestamp_end": 78.5
}
```

---

## Score Impacts

Individual moments in the call that positively or negatively impacted the quality score, linked to transcript segments.

### Schema

```json
{
  "key": "uuid-v4-string",
  "segment_ids": ["seg_a1b2c3d4"],
  "impact": 3,
  "category": "Empathy",
  "reason": "Warm, friendly greeting that put supporter at ease",
  "quote": "Good morning, lovely to speak with you today!",
  "timestamp_start": 0.0,
  "timestamp_end": 2.5
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `key` | string | UUID v4 for React list rendering |
| `segment_ids` | array | Array of segment IDs where this moment occurred (links back to transcript) |
| `impact` | int | Score impact from -5 to +5 |
| `category` | string | Performance category affected (from rubric) |
| `reason` | string | Explanation of why this affected the score |
| `quote` | string | Exact quote from transcript as evidence |
| `timestamp_start` | float | When moment started (seconds) - derived from first segment |
| `timestamp_end` | float | When moment ended (seconds) - derived from last segment |

### Impact Scale

| Impact | Meaning |
|--------|---------|
| +5 | Exceptional - exemplary moment, training example quality |
| +3 to +4 | Strong positive - notably good handling |
| +1 to +2 | Minor positive - good practice observed |
| -1 to -2 | Minor negative - could improve |
| -3 to -4 | Significant negative - clear problem |
| -5 | Severe issue - major quality concern |

### Segment Linking

The `segment_ids` array enables direct navigation from a score impact to the exact moment(s) in the transcript. These are the same UUID-based IDs assigned during transcription.

---

## Performance Scores

Scores against the configured performance rubric.

### Schema

```json
{
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
  "Effective closing": 8
}
```

### Fields

Each key is a criterion from `call_analysis_config.json` `performance_rubric` array.

| Value | Description |
|-------|-------------|
| 10 | Perfect - exceptional performance |
| 8-9 | Excellent - above expectations |
| 6-7 | Good - meets expectations |
| 4-5 | Adequate - room for improvement |
| 2-3 | Below standard - training needed |
| 1 | Unacceptable - serious issue |

---

## Speaker Metrics

Statistics for each speaker in the call.

### Schema

```json
{
  "agent": {
    "talk_time_seconds": 245,
    "talk_time_percentage": 58,
    "word_count": 412,
    "average_pace_wpm": 145,
    "interruptions": 2,
    "silence_percentage": 12
  },
  "supporter": {
    "talk_time_seconds": 180,
    "talk_time_percentage": 42,
    "word_count": 298,
    "average_pace_wpm": 130,
    "interruptions": 1,
    "sentiment_trend": "positive"
  }
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `talk_time_seconds` | float | Total speaking time in seconds |
| `talk_time_percentage` | float | Percentage of call spent speaking |
| `word_count` | int | Total words spoken |
| `average_pace_wpm` | int | Words per minute |
| `interruptions` | int | Number of times speaker interrupted |
| `silence_percentage` | float | Percentage of gaps/pauses |
| `sentiment_trend` | string | Optional: overall sentiment direction |

---

## Audio Observations

Audio-specific analysis (only present when `ANALYSIS_MODE=audio`).

### Schema

```json
{
  "call_quality": "good",
  "background_noise": "low",
  "agent_tone": "friendly",
  "supporter_tone": "happy"
}
```

### Fields

| Field | Type | Values |
|-------|------|--------|
| `call_quality` | string | "excellent", "good", "fair", "poor" |
| `background_noise` | string | "none", "low", "moderate", "high" |
| `agent_tone` | string | "friendly", "warm", "neutral", "cold", "hostile" |
| `supporter_tone` | string | "happy", "satisfied", "neutral", "frustrated", "angry" |

---

## Compliance Flags

Issues or concerns that require attention, linked to specific transcript segments.

### Schema

```json
{
  "key": "uuid-v4-string",
  "type": "data_protection",
  "segment_ids": ["seg_a1b2c3d4", "seg_e5f6g7h8"],
  "issue": "Agent read full card number aloud instead of last 4 digits",
  "quote": "Your card number is 4532 1234 5678 9012",
  "severity": "high",
  "timestamp_start": 156.2,
  "timestamp_end": 159.8
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `key` | string | UUID v4 for React list rendering |
| `type` | string | Category of compliance issue |
| `segment_ids` | array | Array of segment IDs where issue occurred (links back to transcript) |
| `issue` | string | Description of the issue |
| `quote` | string | Exact quote from transcript showing the issue |
| `severity` | string | "low", "medium", "high", "critical" |
| `timestamp_start` | float | When issue started (seconds) - derived from first segment |
| `timestamp_end` | float | When issue ended (seconds) - derived from last segment |

### Segment Linking

The `segment_ids` array enables direct navigation from a compliance flag to the exact moment(s) in the transcript where it occurred. The AI copies these IDs directly from the transcript segments.

### Compliance Types

| Type | Description |
|------|-------------|
| `data_protection` | GDPR/PCI compliance issue |
| `script_deviation` | Significant deviation from script |
| `misleading_info` | Inaccurate or misleading information |
| `pressure_tactics` | Inappropriate sales pressure |
| `consent_issue` | Consent not properly obtained |
| `safeguarding` | Vulnerable person concern |

### Severity Levels

| Level | Description | Action |
|-------|-------------|--------|
| `low` | Minor issue | Note for training |
| `medium` | Moderate concern | Supervisor review |
| `high` | Serious issue | Immediate review required |
| `critical` | Severe breach | Escalate immediately |

---

## Action Items

Follow-up actions generated from analysis.

### Schema

```json
{
  "description": "Send confirmation email with Gift Aid declaration",
  "priority": "high",
  "due_date": "2026-01-20"
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `description` | string | What needs to be done |
| `priority` | string | "low", "medium", "high" |
| `due_date` | string | ISO date (YYYY-MM-DD) or null |

---

## Monthly Summary

AI-generated monthly reports.

### Schema

```json
{
  "summary": "In January 2026, 1,247 calls were analysed with an average quality score of 78.5%.",
  "key_insights": [
    "Call quality improved 5% compared to December",
    "Gift Aid conversion rate increased to 45%",
    "Average call duration reduced by 30 seconds"
  ],
  "recommendations": [
    "Continue objection handling training programme",
    "Update scripts for Q1 campaign messaging",
    "Schedule refresher on data protection protocols"
  ],
  "top_agents": [
    {"halo_id": 123, "name": "Jane Smith", "avg_quality": 92.5},
    {"halo_id": 456, "name": "John Doe", "avg_quality": 89.3}
  ],
  "areas_for_improvement": [
    {"area": "Objection handling", "avg_score": 5.8},
    {"area": "Call structure", "avg_score": 6.2}
  ]
}
```

---

## Training Data Export

Format for exporting annotations for model fine-tuning.

### Schema

```json
{
  "id": "annotation_123",
  "transcript": "Full call transcript text...",
  "original_analysis": {
    "sentiment_label": "positive",
    "sentiment_score": 5.2,
    "quality_score": 78.5,
    "key_topics": [...]
  },
  "corrected_analysis": {
    "sentiment_label": "neutral",
    "quality_score": 65.0
  },
  "annotation_metadata": {
    "annotator_id": 42,
    "annotation_type": "sentiment",
    "notes": "Supporter was actually quite hesitant, not truly positive",
    "created_at": "2026-01-15T14:30:00Z"
  }
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique annotation identifier |
| `transcript` | string | Full or redacted transcript |
| `original_analysis` | object | What the model originally predicted |
| `corrected_analysis` | object | Human-corrected values |
| `annotation_metadata` | object | Information about the annotation |

---

## API Response Envelopes

### Success Response

```json
{
  "success": true,
  "data": {...},
  "message": "Operation completed successfully"
}
```

### Error Response

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid date format",
    "details": {
      "field": "call_date",
      "expected": "ISO 8601 format",
      "received": "01-17-2026"
    }
  }
}
```

---

## Date and Time Formats

All dates and times use these formats:

| Context | Format | Example |
|---------|--------|---------|
| Dates | YYYY-MM-DD | 2026-01-17 |
| Timestamps | UNIX epoch (float) | 1737158400.0 |
| API responses | ISO 8601 | 2026-01-17T14:30:00Z |
| Audio positions | Seconds (float) | 45.123 |
