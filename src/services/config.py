"""
Angel Intelligence - Configuration Service

Three-tier configuration system:
1. Universal - Standard topics, actions, rubric (rigid structure, DB with file fallback)
2. Client - Client-specific context (dynamic structure, e.g., charity info)
3. Campaign Type - Reusable campaign templates (dynamic structure, e.g., goals, success criteria)

Configuration is merged when building prompts:
- Universal provides the analysis framework
- Client provides organisational context
- Campaign Type provides call-specific goals and expectations
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.database import get_db_connection

logger = logging.getLogger(__name__)


# Default file paths
UNIVERSAL_CONFIG_FILE = "call_analysis_config.json"


@dataclass
class UniversalConfig:
    """
    Universal configuration - rigid structure.
    
    Defines the standard framework for all call analysis:
    - topics: What subjects can be discussed
    - agent_actions: What actions agents can perform
    - performance_rubric: How to score agent performance
    """
    topics: List[str] = field(default_factory=list)
    agent_actions: List[str] = field(default_factory=list)
    performance_rubric: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topics": self.topics,
            "agent_actions": self.agent_actions,
            "performance_rubric": self.performance_rubric,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniversalConfig":
        return cls(
            topics=data.get("topics", []),
            agent_actions=data.get("agent_actions", []),
            performance_rubric=data.get("performance_rubric", []),
        )


@dataclass
class ClientConfig:
    """
    Client configuration - dynamic structure.
    
    Contains client-specific information that helps contextualise analysis:
    - Organisation name and type
    - What the charity does
    - Tone/brand guidelines
    - Any client-specific compliance requirements
    
    Structure is flexible - clients can add any fields they need.
    """
    client_ref: str
    config_data: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value."""
        return self.config_data.get(key, default)
    
    def to_prompt_context(self) -> str:
        """
        Convert client config to prompt context string.
        
        Returns a formatted string suitable for including in LLM prompts.
        """
        if not self.config_data:
            return ""
        
        lines = ["CLIENT CONTEXT:"]
        
        # Handle common fields with nice formatting
        if name := self.config_data.get("organisation_name"):
            lines.append(f"- Organisation: {name}")
        
        if org_type := self.config_data.get("organisation_type"):
            lines.append(f"- Type: {org_type}")
        
        if mission := self.config_data.get("mission"):
            lines.append(f"- Mission: {mission}")
        
        if tone := self.config_data.get("tone_guidelines"):
            lines.append(f"- Tone: {tone}")
        
        if compliance := self.config_data.get("compliance_notes"):
            lines.append(f"- Compliance: {compliance}")
        
        # Include any additional custom fields
        standard_fields = {"organisation_name", "organisation_type", "mission", "tone_guidelines", "compliance_notes"}
        for key, value in self.config_data.items():
            if key not in standard_fields:
                if isinstance(value, (list, dict)):
                    lines.append(f"- {key.replace('_', ' ').title()}: {json.dumps(value)}")
                else:
                    lines.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(lines) if len(lines) > 1 else ""


@dataclass
class CampaignTypeConfig:
    """
    Campaign type configuration - dynamic structure.
    
    Defines goals and expectations for a specific type of campaign.
    Reusable across clients.
    
    Examples:
    - "inbound_donation": Goals for handling incoming donation calls
    - "outbound_upgrade": Goals for upgrade/upsell calls
    - "retention": Goals for cancellation/save calls
    - "complaint": Goals for complaint handling
    """
    campaign_type: str
    config_data: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value."""
        return self.config_data.get(key, default)
    
    def to_prompt_context(self) -> str:
        """
        Convert campaign type config to prompt context string.
        
        Returns a formatted string suitable for including in LLM prompts.
        """
        if not self.config_data:
            return ""
        
        lines = [f"CAMPAIGN TYPE: {self.campaign_type.replace('_', ' ').title()}"]
        
        # Handle common fields
        if description := self.config_data.get("description"):
            lines.append(f"Description: {description}")
        
        if goals := self.config_data.get("goals"):
            if isinstance(goals, list):
                lines.append("Goals:")
                for goal in goals:
                    lines.append(f"  - {goal}")
            else:
                lines.append(f"Goals: {goals}")
        
        if success_criteria := self.config_data.get("success_criteria"):
            if isinstance(success_criteria, list):
                lines.append("Success Criteria:")
                for criterion in success_criteria:
                    lines.append(f"  - {criterion}")
            else:
                lines.append(f"Success Criteria: {success_criteria}")
        
        if key_metrics := self.config_data.get("key_metrics"):
            if isinstance(key_metrics, list):
                lines.append("Key Metrics:")
                for metric in key_metrics:
                    lines.append(f"  - {metric}")
            else:
                lines.append(f"Key Metrics: {key_metrics}")
        
        if expectations := self.config_data.get("agent_expectations"):
            lines.append(f"Agent Expectations: {expectations}")
        
        # Include any additional custom fields
        standard_fields = {"description", "goals", "success_criteria", "key_metrics", "agent_expectations"}
        for key, value in self.config_data.items():
            if key not in standard_fields:
                if isinstance(value, (list, dict)):
                    lines.append(f"{key.replace('_', ' ').title()}: {json.dumps(value)}")
                else:
                    lines.append(f"{key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(lines) if len(lines) > 1 else ""


class ConfigService:
    """
    Service for loading and managing three-tier configuration.
    
    Hierarchy:
    1. Universal (DB) → Universal (file fallback)
    2. Client (DB only)
    3. Campaign Type (DB only)
    """
    
    CONFIG_TIER_UNIVERSAL = "universal"
    CONFIG_TIER_CLIENT = "client"
    CONFIG_TIER_CAMPAIGN_TYPE = "campaign_type"
    
    def __init__(self, universal_config_path: str = UNIVERSAL_CONFIG_FILE):
        """
        Initialise the config service.
        
        Args:
            universal_config_path: Path to fallback universal config file
        """
        self.universal_config_path = universal_config_path
        self._universal_cache: Optional[UniversalConfig] = None
        self._client_cache: Dict[str, ClientConfig] = {}
        self._campaign_type_cache: Dict[str, CampaignTypeConfig] = {}
    
    def get_universal_config(self, use_cache: bool = True) -> UniversalConfig:
        """
        Get universal configuration.
        
        Lookup order:
        1. Database (config_tier = 'universal')
        2. File fallback (call_analysis_config.json)
        
        Args:
            use_cache: Whether to use cached config (default True)
            
        Returns:
            UniversalConfig with topics, agent_actions, performance_rubric
        """
        if use_cache and self._universal_cache:
            return self._universal_cache
        
        config_data = None
        
        # Try database first
        try:
            db = get_db_connection()
            row = db.fetch_one("""
                SELECT config_data FROM ai_configs
                WHERE config_tier = %s AND is_active = TRUE
                ORDER BY updated_at DESC
                LIMIT 1
            """, (self.CONFIG_TIER_UNIVERSAL,))
            
            if row:
                config_data = json.loads(row["config_data"])
                logger.debug("Loaded universal config from database")
        except Exception as e:
            logger.warning(f"Failed to load universal config from DB: {e}")
        
        # Fall back to file
        if not config_data:
            config_data = self._load_universal_from_file()
        
        if config_data:
            self._universal_cache = UniversalConfig.from_dict(config_data)
        else:
            # Return empty config if nothing found
            logger.warning("No universal config found in DB or file, using empty defaults")
            self._universal_cache = UniversalConfig()
        
        return self._universal_cache
    
    def _load_universal_from_file(self) -> Optional[Dict[str, Any]]:
        """Load universal config from JSON file."""
        if os.path.exists(self.universal_config_path):
            try:
                with open(self.universal_config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.debug(f"Loaded universal config from file: {self.universal_config_path}")
                    return data
            except Exception as e:
                logger.error(f"Failed to load universal config file: {e}")
        return None
    
    def get_client_config(self, client_ref: str, use_cache: bool = True) -> Optional[ClientConfig]:
        """
        Get client-specific configuration.
        
        Args:
            client_ref: Client reference ID
            use_cache: Whether to use cached config
            
        Returns:
            ClientConfig or None if not found
        """
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
    
    def get_campaign_type_config(self, campaign_type: str, use_cache: bool = True) -> Optional[CampaignTypeConfig]:
        """
        Get campaign type configuration.
        
        Args:
            campaign_type: Campaign type identifier (e.g., "inbound_donation", "retention")
            use_cache: Whether to use cached config
            
        Returns:
            CampaignTypeConfig or None if not found
        """
        if use_cache and campaign_type in self._campaign_type_cache:
            return self._campaign_type_cache[campaign_type]
        
        try:
            db = get_db_connection()
            row = db.fetch_one("""
                SELECT config_data FROM ai_configs
                WHERE config_tier = %s AND campaign_type = %s AND is_active = TRUE
                ORDER BY updated_at DESC
                LIMIT 1
            """, (self.CONFIG_TIER_CAMPAIGN_TYPE, campaign_type))
            
            if row:
                config_data = json.loads(row["config_data"])
                config = CampaignTypeConfig(campaign_type=campaign_type, config_data=config_data)
                self._campaign_type_cache[campaign_type] = config
                logger.debug(f"Loaded campaign type config for '{campaign_type}' from database")
                return config
        except Exception as e:
            logger.warning(f"Failed to load campaign type config for '{campaign_type}': {e}")
        
        return None
    
    def get_merged_config(
        self,
        client_ref: Optional[str] = None,
        campaign_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get merged configuration for a specific context.
        
        Combines:
        1. Universal config (always included)
        2. Client config (if client_ref provided) - can override universal arrays
        3. Campaign type config (if campaign_type provided) - can override universal arrays
        
        Override priority: campaign_type > client > universal
        
        Args:
            client_ref: Client reference ID (optional)
            campaign_type: Campaign type identifier (optional)
            
        Returns:
            Dictionary with:
            - universal: UniversalConfig data
            - client: ClientConfig data (or None)
            - campaign_type: CampaignTypeConfig data (or None)
            - prompt_context: Combined context string for LLM prompts
            - topics: Effective topics (with overrides applied)
            - agent_actions: Effective agent actions (with overrides applied)
            - performance_rubric: Effective rubric (with overrides applied)
        """
        universal = self.get_universal_config()
        client = self.get_client_config(client_ref) if client_ref else None
        campaign = self.get_campaign_type_config(campaign_type) if campaign_type else None
        
        # Build prompt context
        context_parts = []
        if client:
            client_context = client.to_prompt_context()
            if client_context:
                context_parts.append(client_context)
        
        if campaign:
            campaign_context = campaign.to_prompt_context()
            if campaign_context:
                context_parts.append(campaign_context)
        
        # Build effective arrays with override priority: campaign_type > client > universal
        # Check campaign_type first, then client, then fall back to universal
        effective_topics = universal.topics
        effective_actions = universal.agent_actions
        effective_rubric = universal.performance_rubric
        
        # Client overrides universal
        if client:
            if client.config_data.get("topics"):
                effective_topics = client.config_data["topics"]
            if client.config_data.get("agent_actions"):
                effective_actions = client.config_data["agent_actions"]
            if client.config_data.get("performance_rubric"):
                effective_rubric = client.config_data["performance_rubric"]
        
        # Campaign type overrides client (and universal)
        if campaign:
            if campaign.config_data.get("topics"):
                effective_topics = campaign.config_data["topics"]
            if campaign.config_data.get("agent_actions"):
                effective_actions = campaign.config_data["agent_actions"]
            if campaign.config_data.get("performance_rubric"):
                effective_rubric = campaign.config_data["performance_rubric"]
        
        return {
            "universal": universal,
            "client": client,
            "campaign_type": campaign,
            "prompt_context": "\n\n".join(context_parts) if context_parts else "",
            # Effective arrays with overrides applied
            "topics": effective_topics,
            "agent_actions": effective_actions,
            "performance_rubric": effective_rubric,
        }
    
    def save_universal_config(self, config: UniversalConfig) -> int:
        """
        Save universal configuration to database.
        
        Args:
            config: UniversalConfig to save
            
        Returns:
            Config ID
        """
        db = get_db_connection()
        config_json = json.dumps(config.to_dict())
        
        config_id = db.insert("""
            INSERT INTO ai_configs
            (config_tier, config_type, config_data, is_active, created_at, updated_at)
            VALUES (%s, %s, %s, TRUE, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
                config_data = VALUES(config_data),
                updated_at = NOW()
        """, (self.CONFIG_TIER_UNIVERSAL, "analysis", config_json))
        
        # Invalidate cache
        self._universal_cache = None
        
        logger.info("Saved universal config to database")
        return config_id
    
    def save_client_config(self, client_ref: str, config_data: Dict[str, Any]) -> int:
        """
        Save client configuration to database.
        
        Args:
            client_ref: Client reference ID
            config_data: Dynamic configuration data
            
        Returns:
            Config ID
        """
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
        
        # Invalidate cache
        self._client_cache.pop(client_ref, None)
        
        logger.info(f"Saved client config for '{client_ref}' to database")
        return config_id
    
    def save_campaign_type_config(self, campaign_type: str, config_data: Dict[str, Any]) -> int:
        """
        Save campaign type configuration to database.
        
        Args:
            campaign_type: Campaign type identifier
            config_data: Dynamic configuration data
            
        Returns:
            Config ID
        """
        db = get_db_connection()
        config_json = json.dumps(config_data)
        
        config_id = db.insert("""
            INSERT INTO ai_configs
            (config_tier, campaign_type, config_type, config_data, is_active, created_at, updated_at)
            VALUES (%s, %s, %s, %s, TRUE, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
                config_data = VALUES(config_data),
                updated_at = NOW()
        """, (self.CONFIG_TIER_CAMPAIGN_TYPE, campaign_type, "campaign_info", config_json))
        
        # Invalidate cache
        self._campaign_type_cache.pop(campaign_type, None)
        
        logger.info(f"Saved campaign type config for '{campaign_type}' to database")
        return config_id
    
    def clear_cache(self) -> None:
        """Clear all cached configurations."""
        self._universal_cache = None
        self._client_cache.clear()
        self._campaign_type_cache.clear()
        logger.debug("Cleared config cache")
    
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
    
    def list_campaign_types(self) -> List[str]:
        """List all campaign types with configurations."""
        try:
            db = get_db_connection()
            rows = db.fetch_all("""
                SELECT DISTINCT campaign_type FROM ai_configs
                WHERE config_tier = %s AND campaign_type IS NOT NULL AND is_active = TRUE
            """, (self.CONFIG_TIER_CAMPAIGN_TYPE,))
            return [row["campaign_type"] for row in rows]
        except Exception as e:
            logger.error(f"Failed to list campaign types: {e}")
            return []


# Singleton instance
_config_service: Optional[ConfigService] = None


def get_config_service() -> ConfigService:
    """Get the singleton ConfigService instance."""
    global _config_service
    if _config_service is None:
        _config_service = ConfigService()
    return _config_service
