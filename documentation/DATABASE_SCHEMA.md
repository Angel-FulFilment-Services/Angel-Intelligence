# Database Schema

Complete database schema for Angel Intelligence.

## Overview

All tables use the `ai` database and follow these conventions:
- Table prefix: `ai_`
- Character set: `utf8mb4`
- Collation: `utf8mb4_unicode_ci`
- Timestamps: UTC

---

## Entity Relationship Diagram

```
┌─────────────────────┐
│ ai_call_recordings  │
│   (source queue)    │
└─────────┬───────────┘
          │ 1:1
          ├──────────────────────┐
          │                      │
          ▼ 1:1                  ▼ 1:1
┌─────────────────────┐  ┌───────────────────┐
│ai_call_transcriptions│ │  ai_call_analysis │
└─────────────────────┘  └───────────────────┘
          │                      │
          │                      │
          ▼ 1:N                  ▼ 1:N
┌─────────────────────┐  ┌───────────────────┐
│ ai_call_annotations │  │ ai_chat_messages  │
└─────────────────────┘  └───────────────────┘

┌─────────────────────┐  ┌───────────────────┐
│ai_monthly_summaries │  │ai_voice_fingerprints│
└─────────────────────┘  └───────────────────┘

┌─────────────────────┐  ┌───────────────────┐
│ai_chat_conversations│  │  ai_client_configs│
└─────────────────────┘  └───────────────────┘
```

---

## Tables

### ai_call_recordings

The main queue table for call recordings to be processed.

```sql
CREATE TABLE ai_call_recordings (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    
    -- Recording identifiers
    apex_id VARCHAR(255) NOT NULL UNIQUE,
    orderref VARCHAR(100) DEFAULT NULL,      -- Order reference number
    enqref VARCHAR(100) DEFAULT NULL,        -- Enquiry reference number
    obref VARCHAR(100) DEFAULT NULL,         -- Outbound reference number
    client_ref VARCHAR(100) DEFAULT NULL,
    campaign VARCHAR(100) DEFAULT NULL,
    campaign_type VARCHAR(100) DEFAULT NULL, -- Campaign type for config lookup
    halo_id INT UNSIGNED DEFAULT NULL,
    agent_name VARCHAR(255) DEFAULT NULL,
    creative VARCHAR(100) DEFAULT NULL,
    invoicing VARCHAR(100) DEFAULT NULL,
    
    -- Call metadata
    call_date DATE NOT NULL,
    direction ENUM('inbound', 'outbound') DEFAULT 'outbound',
    duration_seconds INT UNSIGNED DEFAULT NULL,
    
    -- Storage paths
    source_path VARCHAR(1024) DEFAULT NULL,
    r2_path VARCHAR(1024) DEFAULT NULL,
    
    -- Processing control
    processing_status ENUM('pending', 'processing', 'completed', 'failed', 'queued') 
        NOT NULL DEFAULT 'pending',
    processing_started_at DATETIME DEFAULT NULL,
    processing_completed_at DATETIME DEFAULT NULL,
    processing_error TEXT DEFAULT NULL,
    processing_worker_id VARCHAR(255) DEFAULT NULL,
    retry_count TINYINT UNSIGNED DEFAULT 0,
    next_retry_at DATETIME DEFAULT NULL,
    
    -- Audio retention
    retain_audio BOOLEAN DEFAULT FALSE,
    audio_deleted_at DATETIME DEFAULT NULL,
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_processing_status (processing_status),
    INDEX idx_call_date (call_date),
    INDEX idx_client_ref (client_ref),
    INDEX idx_campaign_type (campaign_type),
    INDEX idx_halo_id (halo_id),
    INDEX idx_next_retry (next_retry_at, processing_status),
    INDEX idx_worker (processing_worker_id, processing_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

---

### ai_call_transcriptions

Transcription results with PII detection.

Note: `apex_id` allows storing transcription without a recording (for Dojo training).
When full analysis runs, `ai_call_recording_id` is linked via the processor.

```sql
CREATE TABLE ai_call_transcriptions (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    ai_call_recording_id INT UNSIGNED DEFAULT NULL,  -- Nullable: may not have recording yet
    apex_id VARCHAR(255) DEFAULT NULL,               -- Call identifier for lookup
    
    -- Transcript data
    full_transcript TEXT NOT NULL,
    redacted_transcript TEXT DEFAULT NULL,
    segments JSON NOT NULL,
    
    -- PII detection
    pii_detected JSON DEFAULT NULL,
    pii_count INT UNSIGNED DEFAULT 0,
    pii_types JSON DEFAULT NULL,
    
    -- Processing info
    language_detected VARCHAR(10) DEFAULT 'en',
    confidence DECIMAL(5,4) DEFAULT 0.9500,
    model_used VARCHAR(100) DEFAULT NULL,
    processing_time_seconds INT UNSIGNED DEFAULT 0,
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE KEY unique_apex_id (apex_id),
    INDEX idx_recording (ai_call_recording_id),
    FOREIGN KEY (ai_call_recording_id) 
        REFERENCES ai_call_recordings(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### Migration SQL

To add `apex_id` to an existing table:

```sql
ALTER TABLE ai_call_transcriptions
    ADD COLUMN apex_id VARCHAR(255) DEFAULT NULL AFTER ai_call_recording_id,
    ADD UNIQUE KEY unique_apex_id (apex_id),
    MODIFY COLUMN ai_call_recording_id INT UNSIGNED DEFAULT NULL,
    DROP FOREIGN KEY ai_call_transcriptions_ibfk_1,
    ADD FOREIGN KEY (ai_call_recording_id) 
        REFERENCES ai_call_recordings(id) ON DELETE SET NULL;

-- Backfill apex_id from existing recordings
UPDATE ai_call_transcriptions t
JOIN ai_call_recordings r ON t.ai_call_recording_id = r.id
SET t.apex_id = r.apex_id
WHERE t.apex_id IS NULL;
```

#### JSON Column: segments

```json
[
  {
    "text": "Hello, thank you for calling.",
    "start": 0.0,
    "end": 2.5,
    "speaker": "agent",
    "speaker_id": "agent_123",
    "confidence": 0.95,
    "words": [
      {"word": "Hello", "start": 0.0, "end": 0.4, "confidence": 0.98}
    ]
  }
]
```

#### JSON Column: pii_detected

```json
[
  {
    "type": "postcode",
    "original": "SW1A 1AA",
    "redacted": "[POSTCODE]",
    "timestamp_start": 45.2,
    "timestamp_end": 47.1,
    "confidence": 0.92
  }
]
```

---

### ai_call_analysis

AI analysis results.

```sql
CREATE TABLE ai_call_analysis (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    ai_call_recording_id INT UNSIGNED NOT NULL,
    
    -- Core analysis
    summary TEXT NOT NULL,
    sentiment_score DECIMAL(4,1) NOT NULL,
    sentiment_label ENUM('very_negative', 'negative', 'neutral', 'positive', 'very_positive') 
        DEFAULT 'neutral',
    quality_score DECIMAL(5,2) DEFAULT 50.00,
    
    -- Structured analysis
    key_topics JSON DEFAULT NULL,
    agent_actions_performed JSON DEFAULT NULL,
    performance_scores JSON DEFAULT NULL,
    action_items JSON DEFAULT NULL,
    compliance_flags JSON DEFAULT NULL,
    speaker_metrics JSON DEFAULT NULL,
    audio_analysis JSON DEFAULT NULL,
    
    -- Model info
    model_used VARCHAR(100) DEFAULT NULL,
    model_version VARCHAR(50) DEFAULT NULL,
    processing_time_seconds INT UNSIGNED DEFAULT 0,
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE KEY unique_recording (ai_call_recording_id),
    FOREIGN KEY (ai_call_recording_id) 
        REFERENCES ai_call_recordings(id) ON DELETE CASCADE,
    
    -- Indexes
    INDEX idx_sentiment (sentiment_score),
    INDEX idx_quality (quality_score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

---

### ai_call_annotations

Human annotations for model fine-tuning.

```sql
CREATE TABLE ai_call_annotations (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    ai_call_recording_id INT UNSIGNED NOT NULL,
    ai_call_analysis_id INT UNSIGNED DEFAULT NULL,
    
    -- Annotation details
    annotation_type ENUM('sentiment', 'quality', 'topics', 'actions', 'general') NOT NULL,
    original_value JSON DEFAULT NULL,
    corrected_value JSON NOT NULL,
    annotator_id INT UNSIGNED DEFAULT NULL,
    annotator_notes TEXT DEFAULT NULL,
    
    -- Training status
    used_for_training BOOLEAN DEFAULT FALSE,
    training_job_id VARCHAR(100) DEFAULT NULL,
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    FOREIGN KEY (ai_call_recording_id) 
        REFERENCES ai_call_recordings(id) ON DELETE CASCADE,
    
    -- Indexes
    INDEX idx_type (annotation_type),
    INDEX idx_training (used_for_training)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

---

### ai_monthly_summaries

AI-generated monthly reports.

```sql
CREATE TABLE ai_monthly_summaries (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    
    -- Period
    feature VARCHAR(100) NOT NULL,
    summary_month DATE NOT NULL,
    
    -- Filters
    client_ref VARCHAR(100) DEFAULT NULL,
    campaign VARCHAR(100) DEFAULT NULL,
    agent_id INT UNSIGNED DEFAULT NULL,
    
    -- Summary content
    summary_data JSON NOT NULL,
    call_count INT UNSIGNED DEFAULT 0,
    avg_quality_score DECIMAL(5,2) DEFAULT NULL,
    avg_sentiment_score DECIMAL(4,1) DEFAULT NULL,
    
    -- Model info
    model_used VARCHAR(100) DEFAULT NULL,
    model_version VARCHAR(50) DEFAULT NULL,
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE KEY unique_summary (feature, summary_month, client_ref, campaign, agent_id),
    
    -- Indexes
    INDEX idx_month (summary_month),
    INDEX idx_client (client_ref)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

---

### ai_chat_conversations

Chat session records.

```sql
CREATE TABLE ai_chat_conversations (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    
    -- Session info
    user_id INT UNSIGNED DEFAULT NULL,
    feature VARCHAR(100) DEFAULT NULL,
    session_context JSON DEFAULT NULL,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    message_count INT UNSIGNED DEFAULT 0,
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_message_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_user (user_id),
    INDEX idx_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

---

### ai_chat_messages

Individual chat messages.

```sql
CREATE TABLE ai_chat_messages (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    ai_chat_conversation_id INT UNSIGNED NOT NULL,
    
    -- Message content
    role ENUM('user', 'assistant', 'system') NOT NULL,
    content TEXT NOT NULL,
    
    -- Metadata
    tokens_used INT UNSIGNED DEFAULT NULL,
    model_used VARCHAR(100) DEFAULT NULL,
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    FOREIGN KEY (ai_chat_conversation_id) 
        REFERENCES ai_chat_conversations(id) ON DELETE CASCADE,
    
    -- Indexes
    INDEX idx_role (role)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

---

### ai_voice_fingerprints

Agent voice embeddings for speaker identification.

```sql
CREATE TABLE ai_voice_fingerprints (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    
    -- Agent reference
    halo_id INT UNSIGNED NOT NULL,
    agent_name VARCHAR(255) DEFAULT NULL,
    
    -- Fingerprint data
    embedding BLOB NOT NULL,
    embedding_version VARCHAR(50) DEFAULT 'v1.0',
    sample_count INT UNSIGNED DEFAULT 1,
    
    -- Quality metrics
    average_confidence DECIMAL(5,4) DEFAULT 0.9000,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE KEY unique_agent (halo_id),
    
    -- Indexes
    INDEX idx_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

---

### ai_configs

Four-tier configuration system for call analysis.

**Configuration Tiers (load order for analysis):**
1. `global` - Default analysis framework (topics, actions, rubric, quality_signals) - Always loaded, file fallback
2. `campaign` - Campaign-specific goals and requirements (matched by campaign_type)
3. `direction` - Direction-specific requirements (matched by direction: inbound/outbound/sms_handraiser)
4. `client` - Client-specific organisational context (matched by client_ref)

**Override Priority:** client > direction > campaign > global

```sql
CREATE TABLE ai_configs (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    
    -- Configuration tier (four-tier system)
    config_tier ENUM('global', 'universal', 'campaign', 'campaign_type', 'direction', 'client') 
        NOT NULL DEFAULT 'client',
    
    -- Client reference (for 'client' tier)
    client_ref VARCHAR(100) DEFAULT NULL,
    
    -- Campaign (legacy field)
    campaign VARCHAR(100) DEFAULT NULL,
    
    -- Campaign type (for 'campaign' tier)
    campaign_type VARCHAR(100) DEFAULT NULL,
    
    -- Direction (for 'direction' tier)
    direction VARCHAR(50) DEFAULT NULL,
    
    -- Configuration
    config_type VARCHAR(100) NOT NULL,
    config_data JSON NOT NULL,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE KEY unique_config_v3 (config_tier, client_ref, campaign_type, direction, config_type),
    
    -- Indexes
    INDEX idx_config_tier (config_tier),
    INDEX idx_client (client_ref),
    INDEX idx_campaign_type (campaign_type),
    INDEX idx_direction (direction),
    INDEX idx_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

**Example Records:**

```sql
-- Global config (analysis framework) - fallback for all analysis
INSERT INTO ai_configs 
(config_tier, config_type, config_data, is_active)
VALUES 
('global', 'analysis', 
 '{"topics": ["Donation", "Gift Aid"], "agent_actions": ["greeting", "verification"], "performance_rubric": ["Empathy", "Clarity"], "quality_signals": {...}}',
 TRUE);

-- Campaign config (matched by campaign_type in ai_call_recordings)
INSERT INTO ai_configs 
(config_tier, campaign_type, config_type, config_data, is_active)
VALUES 
('campaign', 'regular_giving', 'campaign_info',
 '{"primary_intent": "Sign up to regular giving", "goals": ["Capture donation", "Gift Aid confirmation"], "success_criteria": ["Payment processed"]}',
 TRUE);

-- Direction config (matched by direction in ai_call_recordings)
INSERT INTO ai_configs 
(config_tier, direction, config_type, config_data, is_active)
VALUES 
('direction', 'outbound', 'direction_info',
 '{"expectations": "Agent initiates call, should have clear purpose", "key_differences": ["Agent controls conversation flow", "Must verify supporter identity"]}',
 TRUE);

-- Client config (matched by client_ref in ai_call_recordings)
INSERT INTO ai_configs 
(config_tier, client_ref, config_type, config_data, is_active)
VALUES 
('client', 'CRUK001', 'client_info',
 '{"organisation_name": "Cancer Research UK", "mission": "Beat cancer", "contact": {"email": "support@cruk.org"}}',
 TRUE);
``` 
('universal', 'analysis', 
 '{"topics": ["Donation", "Gift Aid"], "agent_actions": ["greeting", "verification"], "performance_rubric": ["Empathy", "Clarity"]}',
 TRUE);

-- Client config (organisation context)
INSERT INTO ai_configs 
(config_tier, client_ref, config_type, config_data, is_active)
VALUES 
('client', 'CRUK001', 'client_info',
 '{"organisation_name": "Cancer Research UK", "mission": "Beat cancer"}',
 TRUE);

-- Campaign type config (reusable template)
INSERT INTO ai_configs 
(config_tier, campaign_type, config_type, config_data, is_active)
VALUES 
('campaign_type', 'inbound_donation', 'campaign_info',
 '{"goals": ["Capture donation", "Gift Aid confirmation"], "success_criteria": ["Payment processed"]}',
 TRUE);
```

---

## Migration Script

Run this to create all tables:

```sql
-- Create database
CREATE DATABASE IF NOT EXISTS ai 
    CHARACTER SET utf8mb4 
    COLLATE utf8mb4_unicode_ci;

USE ai;

-- Create tables (copy statements above)
-- ...

-- Grant permissions
GRANT ALL PRIVILEGES ON ai.* TO 'angel_ai'@'%';
FLUSH PRIVILEGES;
```

### Migration: Add Three-Tier Config Support

```sql
-- Run scripts/migrate_config_tiers.sql to add three-tier support
-- See documentation/CALL_ANALYSIS_CONFIG.md for details
```

---

## Maintenance

### Clean Up Old Records

```sql
-- Delete completed recordings older than 90 days
DELETE FROM ai_call_recordings 
WHERE processing_status = 'completed' 
  AND created_at < DATE_SUB(NOW(), INTERVAL 90 DAY);

-- Archive old chat conversations
UPDATE ai_chat_conversations 
SET is_active = FALSE 
WHERE last_message_at < DATE_SUB(NOW(), INTERVAL 30 DAY);
```

### Index Maintenance

```sql
-- Analyse tables for query optimisation
ANALYZE TABLE ai_call_recordings;
ANALYZE TABLE ai_call_analysis;

-- Check table health
CHECK TABLE ai_call_recordings;
CHECK TABLE ai_call_transcriptions;
```

---

## Backup

```bash
# Full backup
mysqldump -h $DB_HOST -u $DB_USER -p$DB_PASS ai > backup_$(date +%Y%m%d).sql

# Backup specific tables
mysqldump -h $DB_HOST -u $DB_USER -p$DB_PASS ai \
    ai_call_recordings ai_call_transcriptions ai_call_analysis \
    > backup_core_$(date +%Y%m%d).sql

# Compressed backup
mysqldump -h $DB_HOST -u $DB_USER -p$DB_PASS ai | gzip > backup_$(date +%Y%m%d).sql.gz
```
