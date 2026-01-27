"""
Angel Intelligence - Configuration Service

Four-tier configuration system:
1. Global    - Default analysis framework (DB with file fallback)
2. Campaign  - Campaign-specific goals and success criteria (matched by campaign_type)
3. Direction - Direction-specific requirements (inbound/outbound/sms_handraiser)
4. Client    - Client-specific organisational context (matched by client_ref)

Configuration loading order for call analysis:
1. Global config always loads (DB first, file fallback)
2. Campaign config loads when matching campaign_type found
3. Direction config loads when matching direction found
4. Client config loads when matching client_ref found

All configs are bundled and injected into the AI prompt.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.database import get_db_connection

logger = logging.getLogger(__name__)


# Default file paths
GLOBAL_CONFIG_FILE = "call_analysis_config.json"


@dataclass
class GlobalConfig:
    """
    Global configuration - the foundation for all analysis.
    
    Provides the standard framework:
    - topics: What subjects can be discussed
    - agent_actions: What actions agents can perform
    - performance_rubric: How to score agent performance
    - quality_signals: Detailed assessment criteria (universal quality, vulnerability, howlers, etc.)
    
    This is always loaded. Falls back to call_analysis_config.json if not in DB.
    """
    topics: List[str] = field(default_factory=list)
    agent_actions: List[str] = field(default_factory=list)
    performance_rubric: List[str] = field(default_factory=list)
    quality_signals: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topics": self.topics,
            "agent_actions": self.agent_actions,
            "performance_rubric": self.performance_rubric,
            "quality_signals": self.quality_signals,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GlobalConfig":
        return cls(
            topics=data.get("topics", []),
            agent_actions=data.get("agent_actions", []),
            performance_rubric=data.get("performance_rubric", []),
            quality_signals=data.get("quality_signals", {}),
        )
    
    def to_prompt_context(self) -> str:
        """Convert global quality signals to prompt context."""
        if not self.quality_signals:
            return ""
        
        return _build_quality_signals_context(self.quality_signals)


@dataclass
class CampaignConfig:
    """
    Campaign configuration - defines goals and requirements for a campaign type.
    
    Matched by campaign_type field in ai_call_recordings.
    
    Example campaign types:
    - regular_giving: Sign up supporters to monthly donations
    - one_off_donation: Process single donations
    - retention: Save cancellations
    - upgrade: Increase existing donations
    - reactivation: Win back lapsed donors
    """
    campaign_type: str
    config_data: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config_data.get(key, default)
    
    def to_prompt_context(self) -> str:
        """Convert campaign config to prompt context."""
        if not self.config_data:
            return ""
        
        lines = [f"CAMPAIGN TYPE: {self.campaign_type.replace('_', ' ').title()}"]
        
        if intent := self.config_data.get("primary_intent"):
            lines.append(f"Primary Intent: {intent}")
        
        if description := self.config_data.get("description"):
            lines.append(f"Description: {description}")
        
        if goals := self.config_data.get("goals"):
            lines.append("Goals:")
            for goal in (goals if isinstance(goals, list) else [goals]):
                lines.append(f"  - {goal}")
        
        if success := self.config_data.get("success_criteria"):
            lines.append("Success Criteria:")
            for criterion in (success if isinstance(success, list) else [success]):
                lines.append(f"  - {criterion}")
        
        if required := self.config_data.get("required_actions"):
            lines.append("Required Agent Actions:")
            for action in (required if isinstance(required, list) else [required]):
                lines.append(f"  - {action}")
        
        if expectations := self.config_data.get("agent_expectations"):
            lines.append(f"Agent Expectations: {expectations}")
        
        if metrics := self.config_data.get("key_metrics"):
            lines.append("Key Metrics:")
            for metric in (metrics if isinstance(metrics, list) else [metrics]):
                lines.append(f"  - {metric}")
        
        # Include any additional custom fields
        standard_fields = {"primary_intent", "description", "goals", "success_criteria", 
                          "required_actions", "agent_expectations", "key_metrics",
                          "topics", "agent_actions", "performance_rubric"}
        for key, value in self.config_data.items():
            if key not in standard_fields:
                if isinstance(value, list):
                    lines.append(f"{key.replace('_', ' ').title()}:")
                    for item in value:
                        lines.append(f"  - {item if not isinstance(item, dict) else json.dumps(item)}")
                elif isinstance(value, dict):
                    lines.append(f"{key.replace('_', ' ').title()}: {json.dumps(value)}")
                else:
                    lines.append(f"{key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(lines) if len(lines) > 1 else ""


@dataclass
class DirectionConfig:
    """
    Direction configuration - defines requirements based on call direction.
    
    Matched by direction field in ai_call_recordings.
    
    Directions:
    - inbound: Supporter initiated the call
    - outbound: Agent initiated the call
    - sms_handraiser: Supporter responded to SMS campaign
    """
    direction: str
    config_data: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config_data.get(key, default)
    
    def to_prompt_context(self) -> str:
        """Convert direction config to prompt context."""
        if not self.config_data:
            return ""
        
        direction_label = self.direction.replace('_', ' ').title()
        lines = [f"CALL DIRECTION: {direction_label}"]
        
        if expectations := self.config_data.get("expectations"):
            lines.append(f"Expectations: {expectations}")
        
        if differences := self.config_data.get("key_differences"):
            lines.append("Key Differences for this direction:")
            for diff in (differences if isinstance(differences, list) else [differences]):
                lines.append(f"  - {diff}")
        
        if requirements := self.config_data.get("requirements"):
            lines.append("Direction-Specific Requirements:")
            for req in (requirements if isinstance(requirements, list) else [requirements]):
                lines.append(f"  - {req}")
        
        if scoring := self.config_data.get("scoring_adjustments"):
            lines.append(f"Scoring Adjustments: {scoring}")
        
        # Include any additional custom fields
        standard_fields = {"expectations", "key_differences", "requirements", "scoring_adjustments",
                          "topics", "agent_actions", "performance_rubric"}
        for key, value in self.config_data.items():
            if key not in standard_fields:
                if isinstance(value, list):
                    lines.append(f"{key.replace('_', ' ').title()}:")
                    for item in value:
                        lines.append(f"  - {item if not isinstance(item, dict) else json.dumps(item)}")
                elif isinstance(value, dict):
                    lines.append(f"{key.replace('_', ' ').title()}: {json.dumps(value)}")
                else:
                    lines.append(f"{key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(lines) if len(lines) > 1 else ""


@dataclass
class ClientConfig:
    """
    Client configuration - organisational context.
    
    Matched by client_ref field in ai_call_recordings.
    
    Contains client-specific information:
    - Organisation name and type
    - What the charity does
    - Contact information
    - Brand/tone guidelines
    - Client-specific compliance requirements
    """
    client_ref: str
    config_data: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config_data.get(key, default)
    
    def to_prompt_context(self) -> str:
        """Convert client config to prompt context."""
        if not self.config_data:
            return ""
        
        lines = ["CLIENT CONTEXT:"]
        
        if name := self.config_data.get("organisation_name"):
            lines.append(f"- Organisation: {name}")
        
        if org_type := self.config_data.get("organisation_type"):
            lines.append(f"- Type: {org_type}")
        
        if mission := self.config_data.get("mission"):
            lines.append(f"- Mission: {mission}")
        
        if cause := self.config_data.get("cause"):
            lines.append(f"- Cause: {cause}")
        
        if tone := self.config_data.get("tone_guidelines"):
            lines.append(f"- Tone: {tone}")
        
        if compliance := self.config_data.get("compliance_notes"):
            lines.append(f"- Compliance: {compliance}")
        
        if contact := self.config_data.get("contact"):
            if isinstance(contact, dict):
                contact_str = ", ".join([f"{k}: {v}" for k, v in contact.items()])
                lines.append(f"- Contact: {contact_str}")
            else:
                lines.append(f"- Contact: {contact}")
        
        # Include any additional custom fields
        standard_fields = {"organisation_name", "organisation_type", "mission", "cause",
                          "tone_guidelines", "compliance_notes", "contact",
                          "topics", "agent_actions", "performance_rubric"}
        for key, value in self.config_data.items():
            if key not in standard_fields:
                if isinstance(value, (list, dict)):
                    lines.append(f"- {key.replace('_', ' ').title()}: {json.dumps(value)}")
                else:
                    lines.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(lines) if len(lines) > 1 else ""


def _build_quality_signals_context(quality_signals: Dict[str, Any]) -> str:
    """Build detailed quality assessment context from quality_signals config."""
    if not quality_signals:
        return ""
    
    sections = []
    
    # Universal quality signals
    universal = quality_signals.get("universal_quality", [])
    if universal:
        lines = ["QUALITY ASSESSMENT FRAMEWORK:"]
        for item in universal:
            lines.append(f"- {item.get('category', 'Unknown')}/{item.get('signal', 'Unknown')}: {item.get('description', '')}")
        sections.append("\n".join(lines))
    
    # Vulnerability tiers
    vulnerability = quality_signals.get("vulnerability_capacity", [])
    if vulnerability:
        lines = ["VULNERABILITY/CAPACITY ASSESSMENT:"]
        for tier in vulnerability:
            lines.append(f"- {tier.get('tier', 'Unknown')}: {tier.get('definition', '')}")
            if examples := tier.get('examples'):
                lines.append(f"  Examples: {examples}")
            if ai_action := tier.get('ai_action'):
                lines.append(f"  AI Action: {ai_action}")
        sections.append("\n".join(lines))
    
    # Howlers (critical failures)
    howlers = quality_signals.get("howlers", [])
    if howlers:
        lines = ["CRITICAL FAILURES ('HOWLERS') - Flag immediately:"]
        for h in howlers:
            lines.append(f"- {h.get('type', 'Unknown')}: {h.get('examples', '')} → {h.get('action', '')}")
        sections.append("\n".join(lines))
    
    # Gift Aid assessment
    gift_aid = quality_signals.get("gift_aid", {})
    if gift_aid:
        lines = [
            "GIFT AID ASSESSMENT:",
            f"- Trigger terms: {gift_aid.get('trigger_terms', '')}",
            f"- Minimum components: {gift_aid.get('minimum_components', '')}",
            f"- Soft fail: {gift_aid.get('soft_fail_examples', '')}",
            f"- Hard fail: {gift_aid.get('hard_fail_examples', '')}",
            f"- AI handling: {gift_aid.get('ai_handling', '')}"
        ]
        sections.append("\n".join(lines))
    
    # Objection handling framework
    objection_handling = quality_signals.get("objection_handling", [])
    if objection_handling:
        lines = ["OBJECTION HANDLING FRAMEWORK (assess agent against these steps):"]
        for step in objection_handling:
            lines.append(f"- {step.get('step', 'Unknown')}: {step.get('definition', '')}")
        sections.append("\n".join(lines))
    
    # Common objections and expected responses
    objection_library = quality_signals.get("objection_library", [])
    if objection_library:
        lines = ["COMMON OBJECTIONS - Assess agent response quality:"]
        for obj in objection_library:
            lines.append(f"- \"{obj.get('objection', '')}\" → Goal: {obj.get('goal', '')}")
        sections.append("\n".join(lines))
    
    # Rapport cues
    rapport_cues = quality_signals.get("rapport_cues", [])
    if rapport_cues:
        lines = ["RAPPORT BUILDING - When supporter mentions these, assess agent response:"]
        for cue in rapport_cues:
            lines.append(f"- {cue.get('cue', '')} → Expected: {cue.get('expected_response', '')}")
        sections.append("\n".join(lines))
    
    return "\n\n".join(sections)


class ConfigService:
    """
    Service for loading and managing four-tier configuration.
    
    Hierarchy (all loaded and bundled for prompt):
    1. Global (DB) → Global (file fallback) - Always loaded
    2. Campaign (DB) - Loaded when campaign_type matches
    3. Direction (DB) - Loaded when direction matches
    4. Client (DB) - Loaded when client_ref matches
    """
    
    CONFIG_TIER_GLOBAL = "global"
    CONFIG_TIER_CAMPAIGN = "campaign"
    CONFIG_TIER_DIRECTION = "direction"
    CONFIG_TIER_CLIENT = "client"
    
    # Legacy tier names for backwards compatibility
    CONFIG_TIER_UNIVERSAL = "universal"  # Maps to global
    CONFIG_TIER_CAMPAIGN_TYPE = "campaign_type"  # Maps to campaign
    
    def __init__(self, global_config_path: str = GLOBAL_CONFIG_FILE):
        """
        Initialise the config service.
        
        Args:
            global_config_path: Path to fallback global config file
        """
        self.global_config_path = global_config_path
        self._global_cache: Optional[GlobalConfig] = None
        self._campaign_cache: Dict[str, CampaignConfig] = {}
        self._direction_cache: Dict[str, DirectionConfig] = {}
        self._client_cache: Dict[str, ClientConfig] = {}
    
    # =========================================================================
    # Global Config (Tier 1)
    # =========================================================================
    
    def get_global_config(self, use_cache: bool = True) -> GlobalConfig:
        """
        Get global configuration.
        
        Lookup order:
        1. Database (config_tier = 'global' or 'universal')
        2. File fallback (call_analysis_config.json)
        
        Returns:
            GlobalConfig with topics, agent_actions, performance_rubric, quality_signals
        """
        if use_cache and self._global_cache:
            return self._global_cache
        
        config_data = None
        
        # Try database first (check both 'global' and legacy 'universal')
        try:
            db = get_db_connection()
            row = db.fetch_one("""
                SELECT config_data FROM ai_configs
                WHERE config_tier IN (%s, %s) AND is_active = TRUE
                ORDER BY 
                    CASE config_tier WHEN 'global' THEN 0 ELSE 1 END,
                    updated_at DESC
                LIMIT 1
            """, (self.CONFIG_TIER_GLOBAL, self.CONFIG_TIER_UNIVERSAL))
            
            if row:
                config_data = json.loads(row["config_data"])
                logger.debug("Loaded global config from database")
        except Exception as e:
            logger.warning(f"Failed to load global config from DB: {e}")
        
        # Fall back to file
        if not config_data:
            config_data = self._load_global_from_file()
        
        if config_data:
            self._global_cache = GlobalConfig.from_dict(config_data)
        else:
            logger.warning("No global config found in DB or file, using empty defaults")
            self._global_cache = GlobalConfig()
        
        return self._global_cache
    
    def _load_global_from_file(self) -> Optional[Dict[str, Any]]:
        """Load global config from JSON file."""
        if os.path.exists(self.global_config_path):
            try:
                with open(self.global_config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.debug(f"Loaded global config from file: {self.global_config_path}")
                    return data
            except Exception as e:
                logger.error(f"Failed to load global config file: {e}")
        return None
    
    def save_global_config(self, config: GlobalConfig) -> int:
        """Save global configuration to database."""
        db = get_db_connection()
        config_json = json.dumps(config.to_dict())
        
        config_id = db.insert("""
            INSERT INTO ai_configs
            (config_tier, config_type, config_data, is_active, created_at, updated_at)
            VALUES (%s, %s, %s, TRUE, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
                config_data = VALUES(config_data),
                updated_at = NOW()
        """, (self.CONFIG_TIER_GLOBAL, "analysis", config_json))
        
        self._global_cache = None
        logger.info("Saved global config to database")
        return config_id
    
    # =========================================================================
    # Campaign Config (Tier 2)
    # =========================================================================
    
    def get_campaign_config(self, campaign_type: str, use_cache: bool = True) -> Optional[CampaignConfig]:
        """
        Get campaign configuration.
        
        Args:
            campaign_type: Campaign type identifier (matched from ai_call_recordings.campaign_type)
            
        Returns:
            CampaignConfig or None if not found
        """
        if not campaign_type:
            return None
            
        if use_cache and campaign_type in self._campaign_cache:
            return self._campaign_cache[campaign_type]
        
        try:
            db = get_db_connection()
            # Check both 'campaign' and legacy 'campaign_type'
            row = db.fetch_one("""
                SELECT config_data FROM ai_configs
                WHERE config_tier IN (%s, %s) AND campaign_type = %s AND is_active = TRUE
                ORDER BY 
                    CASE config_tier WHEN 'campaign' THEN 0 ELSE 1 END,
                    updated_at DESC
                LIMIT 1
            """, (self.CONFIG_TIER_CAMPAIGN, self.CONFIG_TIER_CAMPAIGN_TYPE, campaign_type))
            
            if row:
                config_data = json.loads(row["config_data"])
                config = CampaignConfig(campaign_type=campaign_type, config_data=config_data)
                self._campaign_cache[campaign_type] = config
                logger.debug(f"Loaded campaign config for '{campaign_type}' from database")
                return config
        except Exception as e:
            logger.warning(f"Failed to load campaign config for '{campaign_type}': {e}")
        
        return None
    
    def save_campaign_config(self, campaign_type: str, config_data: Dict[str, Any]) -> int:
        """Save campaign configuration to database."""
        db = get_db_connection()
        config_json = json.dumps(config_data)
        
        config_id = db.insert("""
            INSERT INTO ai_configs
            (config_tier, campaign_type, config_type, config_data, is_active, created_at, updated_at)
            VALUES (%s, %s, %s, %s, TRUE, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
                config_data = VALUES(config_data),
                updated_at = NOW()
        """, (self.CONFIG_TIER_CAMPAIGN, campaign_type, "campaign_info", config_json))
        
        self._campaign_cache.pop(campaign_type, None)
        logger.info(f"Saved campaign config for '{campaign_type}' to database")
        return config_id
    
    def list_campaign_types(self) -> List[str]:
        """List all campaign types with configurations."""
        try:
            db = get_db_connection()
            rows = db.fetch_all("""
                SELECT DISTINCT campaign_type FROM ai_configs
                WHERE config_tier IN (%s, %s) AND campaign_type IS NOT NULL AND is_active = TRUE
            """, (self.CONFIG_TIER_CAMPAIGN, self.CONFIG_TIER_CAMPAIGN_TYPE))
            return [row["campaign_type"] for row in rows]
        except Exception as e:
            logger.error(f"Failed to list campaign types: {e}")
            return []
    
    # =========================================================================
    # Direction Config (Tier 3)
    # =========================================================================
    
    def get_direction_config(self, direction: str, use_cache: bool = True) -> Optional[DirectionConfig]:
        """
        Get direction configuration.
        
        Args:
            direction: Call direction (inbound/outbound/sms_handraiser)
            
        Returns:
            DirectionConfig or None if not found
        """
        if not direction:
            return None
            
        if use_cache and direction in self._direction_cache:
            return self._direction_cache[direction]
        
        try:
            db = get_db_connection()
            row = db.fetch_one("""
                SELECT config_data FROM ai_configs
                WHERE config_tier = %s AND direction = %s AND is_active = TRUE
                ORDER BY updated_at DESC
                LIMIT 1
            """, (self.CONFIG_TIER_DIRECTION, direction))
            
            if row:
                config_data = json.loads(row["config_data"])
                config = DirectionConfig(direction=direction, config_data=config_data)
                self._direction_cache[direction] = config
                logger.debug(f"Loaded direction config for '{direction}' from database")
                return config
        except Exception as e:
            logger.warning(f"Failed to load direction config for '{direction}': {e}")
        
        return None
    
    def save_direction_config(self, direction: str, config_data: Dict[str, Any]) -> int:
        """Save direction configuration to database."""
        db = get_db_connection()
        config_json = json.dumps(config_data)
        
        config_id = db.insert("""
            INSERT INTO ai_configs
            (config_tier, direction, config_type, config_data, is_active, created_at, updated_at)
            VALUES (%s, %s, %s, %s, TRUE, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
                config_data = VALUES(config_data),
                updated_at = NOW()
        """, (self.CONFIG_TIER_DIRECTION, direction, "direction_info", config_json))
        
        self._direction_cache.pop(direction, None)
        logger.info(f"Saved direction config for '{direction}' to database")
        return config_id
    
    def list_directions(self) -> List[str]:
        """List all directions with configurations."""
        try:
            db = get_db_connection()
            rows = db.fetch_all("""
                SELECT DISTINCT direction FROM ai_configs
                WHERE config_tier = %s AND direction IS NOT NULL AND is_active = TRUE
            """, (self.CONFIG_TIER_DIRECTION,))
            return [row["direction"] for row in rows]
        except Exception as e:
            logger.error(f"Failed to list directions: {e}")
            return []
    
    # =========================================================================
    # Client Config (Tier 4)
    # =========================================================================
    
    def get_client_config(self, client_ref: str, use_cache: bool = True) -> Optional[ClientConfig]:
        """
        Get client-specific configuration.
        
        Args:
            client_ref: Client reference ID (matched from ai_call_recordings.client_ref)
            
        Returns:
            ClientConfig or None if not found
        """
        if not client_ref:
            return None
            
        if use_cache and client_ref in self._client_cache:
            return self._client_cache[client_ref]
        
        try:
            db = get_db_connection()
            row = db.fetch_one("""
                SELECT config_data FROM ai_configs
                WHERE config_tier = %s AND client_ref = %s AND is_active = TRUE
                ORDER BY updated_at DESC
                LIMIT 1
            """, (self.CONFIG_TIER_CLIENT, client_ref))
            
            if row:
                config_data = json.loads(row["config_data"])
                config = ClientConfig(client_ref=client_ref, config_data=config_data)
                self._client_cache[client_ref] = config
                logger.debug(f"Loaded client config for '{client_ref}' from database")
                return config
        except Exception as e:
            logger.warning(f"Failed to load client config for '{client_ref}': {e}")
        
        return None
    
    def save_client_config(self, client_ref: str, config_data: Dict[str, Any]) -> int:
        """Save client configuration to database."""
        db = get_db_connection()
        config_json = json.dumps(config_data)
        
        config_id = db.insert("""
            INSERT INTO ai_configs
            (config_tier, client_ref, config_type, config_data, is_active, created_at, updated_at)
            VALUES (%s, %s, %s, %s, TRUE, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
                config_data = VALUES(config_data),
                updated_at = NOW()
        """, (self.CONFIG_TIER_CLIENT, client_ref, "client_info", config_json))
        
        self._client_cache.pop(client_ref, None)
        logger.info(f"Saved client config for '{client_ref}' to database")
        return config_id
    
    def list_clients(self) -> List[str]:
        """List all client references with configurations."""
        try:
            db = get_db_connection()
            rows = db.fetch_all("""
                SELECT DISTINCT client_ref FROM ai_configs
                WHERE config_tier = %s AND client_ref IS NOT NULL AND is_active = TRUE
            """, (self.CONFIG_TIER_CLIENT,))
            return [row["client_ref"] for row in rows]
        except Exception as e:
            logger.error(f"Failed to list clients: {e}")
            return []
    
    # =========================================================================
    # Merged Config (All Tiers Combined)
    # =========================================================================
    
    def get_merged_config(
        self,
        campaign_type: Optional[str] = None,
        direction: Optional[str] = None,
        client_ref: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get merged configuration for a specific call context.
        
        Loads and combines all applicable tiers:
        1. Global config (always loaded)
        2. Campaign config (if campaign_type provided and found)
        3. Direction config (if direction provided and found)
        4. Client config (if client_ref provided and found)
        
        Override priority for topics/actions/rubric: client > direction > campaign > global
        
        Args:
            campaign_type: Campaign type from ai_call_recordings.campaign_type
            direction: Call direction from ai_call_recordings.direction
            client_ref: Client reference from ai_call_recordings.client_ref
            
        Returns:
            Dictionary with:
            - global_config: GlobalConfig object
            - campaign_config: CampaignConfig object or None
            - direction_config: DirectionConfig object or None
            - client_config: ClientConfig object or None
            - prompt_context: Combined context string for LLM prompts
            - topics: Effective topics (with overrides applied)
            - agent_actions: Effective agent actions (with overrides applied)
            - performance_rubric: Effective rubric (with overrides applied)
            - quality_signals: Quality signals from global config
        """
        global_cfg = self.get_global_config()
        campaign_cfg = self.get_campaign_config(campaign_type) if campaign_type else None
        direction_cfg = self.get_direction_config(direction) if direction else None
        client_cfg = self.get_client_config(client_ref) if client_ref else None
        
        # Build prompt context from all tiers
        context_parts = []
        
        # Global quality signals context
        if global_cfg.quality_signals:
            global_context = global_cfg.to_prompt_context()
            if global_context:
                context_parts.append(global_context)
        
        # Campaign context
        if campaign_cfg:
            campaign_context = campaign_cfg.to_prompt_context()
            if campaign_context:
                context_parts.append(campaign_context)
        
        # Direction context
        if direction_cfg:
            direction_context = direction_cfg.to_prompt_context()
            if direction_context:
                context_parts.append(direction_context)
        
        # Client context
        if client_cfg:
            client_context = client_cfg.to_prompt_context()
            if client_context:
                context_parts.append(client_context)
        
        # Build effective arrays with override priority: client > direction > campaign > global
        effective_topics = global_cfg.topics
        effective_actions = global_cfg.agent_actions
        effective_rubric = global_cfg.performance_rubric
        
        # Campaign overrides global
        if campaign_cfg:
            if campaign_cfg.config_data.get("topics"):
                effective_topics = campaign_cfg.config_data["topics"]
            if campaign_cfg.config_data.get("agent_actions"):
                effective_actions = campaign_cfg.config_data["agent_actions"]
            if campaign_cfg.config_data.get("performance_rubric"):
                effective_rubric = campaign_cfg.config_data["performance_rubric"]
        
        # Direction overrides campaign
        if direction_cfg:
            if direction_cfg.config_data.get("topics"):
                effective_topics = direction_cfg.config_data["topics"]
            if direction_cfg.config_data.get("agent_actions"):
                effective_actions = direction_cfg.config_data["agent_actions"]
            if direction_cfg.config_data.get("performance_rubric"):
                effective_rubric = direction_cfg.config_data["performance_rubric"]
        
        # Client overrides direction
        if client_cfg:
            if client_cfg.config_data.get("topics"):
                effective_topics = client_cfg.config_data["topics"]
            if client_cfg.config_data.get("agent_actions"):
                effective_actions = client_cfg.config_data["agent_actions"]
            if client_cfg.config_data.get("performance_rubric"):
                effective_rubric = client_cfg.config_data["performance_rubric"]
        
        return {
            "global_config": global_cfg,
            "campaign_config": campaign_cfg,
            "direction_config": direction_cfg,
            "client_config": client_cfg,
            "prompt_context": "\n\n".join(context_parts) if context_parts else "",
            # Effective arrays with overrides applied
            "topics": effective_topics,
            "agent_actions": effective_actions,
            "performance_rubric": effective_rubric,
            "quality_signals": global_cfg.quality_signals,
        }
    
    # =========================================================================
    # Legacy Compatibility Methods
    # =========================================================================
    
    def get_universal_config(self, use_cache: bool = True) -> GlobalConfig:
        """Legacy method - redirects to get_global_config."""
        return self.get_global_config(use_cache)
    
    def get_campaign_type_config(self, campaign_type: str, use_cache: bool = True) -> Optional[CampaignConfig]:
        """Legacy method - redirects to get_campaign_config."""
        return self.get_campaign_config(campaign_type, use_cache)
    
    def save_universal_config(self, config: GlobalConfig) -> int:
        """Legacy method - redirects to save_global_config."""
        return self.save_global_config(config)
    
    def save_campaign_type_config(self, campaign_type: str, config_data: Dict[str, Any]) -> int:
        """Legacy method - redirects to save_campaign_config."""
        return self.save_campaign_config(campaign_type, config_data)
    
    # =========================================================================
    # Cache Management
    # =========================================================================
    
    def clear_cache(self) -> None:
        """Clear all cached configurations."""
        self._global_cache = None
        self._campaign_cache.clear()
        self._direction_cache.clear()
        self._client_cache.clear()
        logger.debug("Cleared config cache")


# Singleton instance
_config_service: Optional[ConfigService] = None


def get_config_service() -> ConfigService:
    """Get the singleton ConfigService instance."""
    global _config_service
    if _config_service is None:
        _config_service = ConfigService()
    return _config_service
