-- Migration: Add three-tier configuration support
-- This migration:
-- 1. Renames ai_client_configs to ai_configs
-- 2. Adds config_tier for three-tier system
-- 3. Adds campaign_type field to ai_call_recordings

-- =============================================================================
-- Part 1: Add campaign_type to ai_call_recordings
-- =============================================================================

ALTER TABLE ai_call_recordings
    ADD COLUMN campaign_type VARCHAR(100) DEFAULT NULL AFTER campaign;

-- Add index for campaign_type lookups
ALTER TABLE ai_call_recordings
    ADD INDEX idx_campaign_type (campaign_type);

-- =============================================================================
-- Part 2: Rename ai_client_configs to ai_configs
-- =============================================================================

RENAME TABLE ai_client_configs TO ai_configs;

-- =============================================================================
-- Part 3: Add three-tier config support to ai_configs
-- =============================================================================

-- Add new columns for three-tier config
ALTER TABLE ai_configs
    ADD COLUMN config_tier ENUM('universal', 'client', 'campaign_type') 
        NOT NULL DEFAULT 'client' AFTER id,
    ADD COLUMN campaign_type VARCHAR(100) DEFAULT NULL AFTER campaign,
    MODIFY COLUMN config_type VARCHAR(100) NOT NULL;

-- Update existing configs to use new tier system
-- Global configs (client_ref IS NULL) become universal
UPDATE ai_configs 
SET config_tier = 'universal' 
WHERE client_ref IS NULL AND campaign IS NULL;

-- Client-specific configs stay as 'client' (the default)
UPDATE ai_configs 
SET config_tier = 'client' 
WHERE client_ref IS NOT NULL;

-- Drop the old unique constraint if it exists
-- Note: Run this manually if needed:
-- ALTER TABLE ai_configs DROP INDEX unique_config;

-- Add new composite unique key that handles all three tiers
-- This ensures:
-- - Only one universal config per config_type
-- - Only one client config per client_ref + config_type
-- - Only one campaign_type config per campaign_type + config_type
ALTER TABLE ai_configs
    ADD UNIQUE KEY unique_config_v2 (config_tier, client_ref, campaign_type, config_type);

-- Add indexes for tier-based lookups
ALTER TABLE ai_configs
    ADD INDEX idx_config_tier (config_tier),
    ADD INDEX idx_campaign_type (campaign_type);

-- =============================================================================
-- Example data
-- =============================================================================

-- Example: Insert a universal config
-- INSERT INTO ai_configs (config_tier, config_type, config_data, is_active)
-- VALUES ('universal', 'analysis', '{"topics": [...], "agent_actions": [...], "performance_rubric": [...]}', TRUE);

-- Example: Insert a client config
-- INSERT INTO ai_configs (config_tier, client_ref, config_type, config_data, is_active)
-- VALUES ('client', 'ABC123', 'client_info', '{"organisation_name": "Example Charity", "mission": "Helping people"}', TRUE);

-- Example: Insert a campaign type config
-- INSERT INTO ai_configs (config_tier, campaign_type, config_type, config_data, is_active)
-- VALUES ('campaign_type', 'inbound_donation', 'campaign_info', '{"goals": ["Capture donation", "Maximise Gift Aid"]}', TRUE);
