"""
Angel Intelligence - Model Configuration

Defines model configurations for analysis and chat models.
Analysis model is fine-tunable; chat model uses base weights only.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    base_model: str
    path: str
    quantization: Optional[str] = None
    fine_tunable: bool = False
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "base_model": self.base_model,
            "path": self.path,
            "quantization": self.quantization,
            "fine_tunable": self.fine_tunable,
            "description": self.description,
        }


# Default model configurations
ANALYSIS_MODEL_CONFIG = ModelConfig(
    name="analysis",
    base_model="Qwen/Qwen2.5-Omni-7B",
    path="/models/analysis/current",
    fine_tunable=True,
    description="Call analysis model - fine-tuned on human annotations for sentiment, quality, and topic detection"
)

CHAT_MODEL_CONFIG = ModelConfig(
    name="chat",
    base_model="Qwen/Qwen2.5-Omni-7B",
    path="/models/chat/current",
    fine_tunable=False,
    description="Chat model - base model for conversations, summaries, and ad-hoc queries"
)

WHISPER_MODEL_CONFIG = ModelConfig(
    name="whisper",
    base_model="openai/whisper-medium",
    path="/models/whisper/current",
    fine_tunable=False,
    description="Whisper model for audio transcription"
)


def get_model_config(model_type: str) -> ModelConfig:
    """
    Get model configuration by type.
    
    Args:
        model_type: One of 'analysis', 'chat', or 'whisper'
        
    Returns:
        ModelConfig for the specified model type
    """
    configs = {
        "analysis": ANALYSIS_MODEL_CONFIG,
        "chat": CHAT_MODEL_CONFIG,
        "whisper": WHISPER_MODEL_CONFIG,
    }
    
    if model_type not in configs:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of: {list(configs.keys())}")
    
    return configs[model_type]
