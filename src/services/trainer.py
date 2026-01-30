"""
Model fine-tuning service using LoRA.

Trains the analysis model on human annotations stored in ai_call_annotations.
Designed to run as a nightly job, saving the fine-tuned adapter to NFS.

Supports versioned adapters for zero-downtime deployments:
  - Each training run creates a timestamped adapter (e.g., call-analysis-20260121-0200)
  - A 'current' manifest tracks the active version
  - Old versions are retained for rollback
  
Training lock:
  - A lock file prevents concurrent training runs
  - Lock includes start time and PID for debugging
"""

import logging
import os
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from src.config import get_settings

logger = logging.getLogger(__name__)

# Training dependencies (optional - only needed for training jobs)
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, PeftModel
    from datasets import Dataset
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    logger.warning("Training dependencies not installed. Run: pip install peft datasets")


# Global lock file path for tracking training status
TRAINING_LOCK_FILE = Path("models/adapters/.training_lock.json")


class TrainingService:
    """
    Service for fine-tuning the analysis model with LoRA.
    
    Supports versioned adapters for zero-downtime deployments:
      - Each training run creates: adapters/{adapter_name}/{adapter_name}-{timestamp}/
      - current.json tracks the active version
      - List versions, switch active version, rollback if needed
    """
    
    def __init__(self, adapter_name: str = "call-analysis"):
        """
        Initialise training service.
        
        Args:
            adapter_name: Name for the adapter (used as folder name and vLLM identifier)
                         Default: "call-analysis" for call recording analysis
                         Future: "email-analysis", "chat-analysis", etc.
        """
        self.settings = get_settings()
        self.adapter_name = adapter_name
        self.base_model_path = self.settings.analysis_model_path or self.settings.analysis_model
        
        # Base path for this adapter type (contains all versions)
        self.adapter_base_path = Path(self.settings.models_base_path) / "adapters" / adapter_name
        
        # For backwards compatibility, adapter_output_path points to current version
        # but train() will create versioned paths
        self.adapter_output_path = self.adapter_base_path
        
        # Training hyperparameters (optimised for overnight training)
        self.lora_config = {
            "r": 16,  # LoRA rank
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        
        # Note: output_dir will be set dynamically per training run
        self.training_args = {
            "output_dir": None,  # Set in train() with versioned path
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "save_total_limit": 2,
            "fp16": True,
            "gradient_checkpointing": True,
            "optim": "adamw_torch_fused",
            "report_to": "none",
        }
    
    # =========================================================================
    # Training Lock Management
    # =========================================================================
    
    @staticmethod
    def is_training_in_progress() -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if training is currently in progress.
        
        Returns:
            Tuple of (is_in_progress, lock_info)
            lock_info contains: adapter_name, started_at, pid
        """
        lock_path = Path(get_settings().models_base_path) / "adapters" / ".training_lock.json"
        
        if not lock_path.exists():
            return False, None
        
        try:
            with open(lock_path, "r") as f:
                lock_info = json.load(f)
            
            # Check if the lock is stale (older than 4 hours = likely crashed)
            started_at = lock_info.get("started_at")
            if started_at:
                start_time = datetime.fromisoformat(started_at)
                elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
                if elapsed_hours > 4:
                    logger.warning(f"Training lock is stale ({elapsed_hours:.1f} hours old), removing")
                    lock_path.unlink()
                    return False, None
            
            return True, lock_info
            
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read training lock: {e}")
            return False, None
    
    def _acquire_training_lock(self) -> bool:
        """
        Acquire the training lock.
        
        Returns:
            True if lock acquired, False if training already in progress
        """
        lock_path = Path(self.settings.models_base_path) / "adapters" / ".training_lock.json"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        
        in_progress, lock_info = self.is_training_in_progress()
        if in_progress:
            logger.warning(f"Training already in progress: {lock_info}")
            return False
        
        lock_data = {
            "adapter_name": self.adapter_name,
            "started_at": datetime.now().isoformat(),
            "pid": os.getpid(),
        }
        
        try:
            with open(lock_path, "w") as f:
                json.dump(lock_data, f, indent=2)
            logger.info(f"Acquired training lock for adapter '{self.adapter_name}'")
            return True
        except IOError as e:
            logger.error(f"Failed to acquire training lock: {e}")
            return False
    
    def _release_training_lock(self) -> None:
        """Release the training lock."""
        lock_path = Path(self.settings.models_base_path) / "adapters" / ".training_lock.json"
        
        try:
            if lock_path.exists():
                lock_path.unlink()
                logger.info("Released training lock")
        except IOError as e:
            logger.warning(f"Failed to release training lock: {e}")
    
    # =========================================================================
    # Training Data Preparation
    # =========================================================================
    
    def prepare_training_data(self, limit: int = 2500) -> List[Dict[str, Any]]:
        """
        Load and prepare training data from Dojo session annotations.
        
        Training Approach:
        - Agreement examples (disagreement_reason is NULL): Positive reinforcement, weight 0.5
        - Disagreement examples: Corrections for AI mistakes, weight 1.0
        - False positives: Special handling to reduce over-flagging
        
        Returns list of training samples with weights for fine-tuning.
        """
        from src.database import CallAnnotation
        
        # Get training data from database
        raw_data = CallAnnotation.get_training_data(limit=limit)
        
        if not raw_data:
            logger.warning("No training data found in database")
            return []
        
        training_samples = []
        
        for row in raw_data:
            sample = self._prepare_sample(row)
            if sample:
                training_samples.append(sample)
        
        # Log training data statistics
        agreements = sum(1 for s in training_samples if s.get("is_agreement", False))
        corrections = len(training_samples) - agreements
        logger.info(f"Prepared {len(training_samples)} training samples: "
                   f"{agreements} agreements (weight 0.5), {corrections} corrections (weight 1.0)")
        
        return training_samples
    
    def _prepare_sample(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Prepare a training sample from Dojo session annotation.
        
        Extracts the relevant transcript segment and builds input/output
        based on the trainer's assessment.
        """
        # Get transcript segments for this annotation
        segment_ids = row.get("segment_ids")
        if isinstance(segment_ids, str):
            try:
                segment_ids = json.loads(segment_ids)
            except json.JSONDecodeError:
                segment_ids = []
        
        # Get full transcript segments
        transcript_segments = row.get("transcript_segments")
        if isinstance(transcript_segments, str):
            try:
                transcript_segments = json.loads(transcript_segments)
            except json.JSONDecodeError:
                transcript_segments = []
        
        # Extract relevant segment text based on segment_ids or timestamps
        segment_text = self._extract_segment_text(
            transcript_segments,
            segment_ids,
            row.get("timestamp_start"),
            row.get("timestamp_end")
        )
        
        if not segment_text:
            # Fall back to full transcript if segment extraction fails
            segment_text = row.get("full_transcript") or row.get("redacted_transcript", "")
            if not segment_text:
                return None
        
        # Determine training parameters
        segment_type = row.get("segment_type")  # agent_action, score_impact, compliance_flag
        disagreement_reason = row.get("disagreement_reason")
        is_agreement = disagreement_reason is None
        training_weight = 0.5 if is_agreement else 1.0
        
        trainer_assessment = row.get("trainer_assessment")
        ai_assessment = row.get("ai_assessment")
        
        # For false positives, use special training signal
        is_false_positive = (
            trainer_assessment == "False Positive" or 
            disagreement_reason == "false_positive"
        )
        
        # Build the expected output based on segment type
        if segment_type == "agent_action":
            output = self._build_agent_action_output(row, is_false_positive)
        elif segment_type == "score_impact":
            output = self._build_score_impact_output(row, is_false_positive)
        elif segment_type == "compliance_flag":
            output = self._build_compliance_flag_output(row, is_false_positive)
        else:
            # Generic output for unknown segment types
            output = {
                "assessment": trainer_assessment or ai_assessment,
                "corrected": not is_agreement,
            }
        
        return {
            "transcript": segment_text[:8000],
            "analysis": output,
            "weight": training_weight,
            "is_agreement": is_agreement,
            "is_false_positive": is_false_positive,
            "segment_type": segment_type,
            "disagreement_reason": disagreement_reason,
        }
    
    def _extract_segment_text(
        self,
        transcript_segments: List[Dict[str, Any]],
        segment_ids: Optional[List[int]],
        timestamp_start: Optional[float],
        timestamp_end: Optional[float]
    ) -> str:
        """
        Extract transcript text for the given segment IDs or time range.
        """
        if not transcript_segments:
            return ""
        
        relevant_segments = []
        
        if segment_ids:
            # Use explicit segment IDs
            for i, seg in enumerate(transcript_segments):
                if i in segment_ids or seg.get("id") in segment_ids:
                    relevant_segments.append(seg)
        elif timestamp_start is not None and timestamp_end is not None:
            # Use time range
            for seg in transcript_segments:
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", seg_start)
                # Check for overlap
                if seg_start <= timestamp_end and seg_end >= timestamp_start:
                    relevant_segments.append(seg)
        else:
            # No filtering, use all segments
            relevant_segments = transcript_segments
        
        # Build transcript text
        lines = []
        for seg in relevant_segments:
            speaker = seg.get("speaker", "Unknown")
            text = seg.get("text", "")
            lines.append(f"{speaker}: {text}")
        
        return "\n".join(lines)
    
    def _build_agent_action_output(self, row: Dict[str, Any], is_false_positive: bool) -> Dict[str, Any]:
        """Build training output for agent_action segment type."""
        if is_false_positive:
            return {
                "action_detected": False,
                "action_type": None,
                "quality_impact": "none",
                "explanation": "No significant agent action at this point",
            }
        
        return {
            "action_detected": True,
            "action_type": row.get("segment_key", "").split("-")[1] if row.get("segment_key") else "unknown",
            "quality_impact": row.get("trainer_assessment") or row.get("ai_assessment"),
            "timestamp_start": row.get("timestamp_start"),
            "timestamp_end": row.get("timestamp_end"),
        }
    
    def _build_score_impact_output(self, row: Dict[str, Any], is_false_positive: bool) -> Dict[str, Any]:
        """Build training output for score_impact segment type."""
        if is_false_positive:
            return {
                "has_score_impact": False,
                "impact_zone": None,
                "explanation": "No score impact at this point",
            }
        
        # Map trainer assessment to zone
        assessment = row.get("trainer_assessment") or row.get("ai_assessment")
        return {
            "has_score_impact": True,
            "impact_zone": assessment,  # Excellent, Good, Satisfactory, Below Average, Poor
            "timestamp_start": row.get("timestamp_start"),
            "timestamp_end": row.get("timestamp_end"),
        }
    
    def _build_compliance_flag_output(self, row: Dict[str, Any], is_false_positive: bool) -> Dict[str, Any]:
        """Build training output for compliance_flag segment type."""
        if is_false_positive:
            return {
                "compliance_issue": False,
                "flag_type": None,
                "explanation": "No compliance issue at this point",
            }
        
        return {
            "compliance_issue": True,
            "flag_type": row.get("segment_key", "").split("-")[1] if row.get("segment_key") else "unknown",
            "severity": "high" if row.get("trainer_assessment") == "disagree" else "medium",
            "timestamp_start": row.get("timestamp_start"),
            "timestamp_end": row.get("timestamp_end"),
        }
    
    def format_for_training(self, samples: List[Dict[str, Any]], tokenizer) -> "Dataset":
        """
        Format training samples into tokenized dataset.
        
        Uses chat template format for instruction fine-tuning.
        Applies sample weights for balanced training (0.5 for agreements, 1.0 for corrections).
        """
        formatted = []
        
        system_prompt = """You are an expert call analyst for Angel Fulfilment Services.
Analyse the call transcript and provide structured analysis in JSON format.
Be accurate with sentiment scores (1-10) and quality scores (0-100).
Identify agent actions, score impacts, and compliance issues with precise timestamps."""
        
        for sample in samples:
            user_prompt = self._build_segment_prompt(sample)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json.dumps(sample['analysis'], indent=2)},
            ]
            
            # Apply chat template
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            formatted.append({
                "text": text,
                "weight": sample.get("weight", 1.0),
            })
        
        dataset = Dataset.from_list(formatted)
        
        # Tokenize
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=4096,
                padding="max_length",
            )
            # Preserve weight for weighted loss
            tokenized["weight"] = examples["weight"]
            return tokenized
        
        tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        return tokenized
    
    def _build_segment_prompt(self, sample: Dict[str, Any]) -> str:
        """Build a segment-specific prompt for Dojo training samples."""
        segment_type = sample.get("segment_type")
        transcript = sample.get("transcript", "")
        
        if segment_type == "agent_action":
            return f"""Analyse this call segment for AGENT ACTIONS.
Identify any actions taken by the agent and assess their impact on call quality.

Segment:
{transcript}

Provide your analysis in JSON format with: action_detected, action_type, quality_impact, explanation."""
        
        elif segment_type == "score_impact":
            return f"""Analyse this call segment for SCORE IMPACT.
Determine if this segment impacts the call quality score and rate the zone.

Segment:
{transcript}

Provide your analysis in JSON format with: has_score_impact, impact_zone (Excellent/Good/Satisfactory/Below Average/Poor), explanation."""
        
        elif segment_type == "compliance_flag":
            return f"""Analyse this call segment for COMPLIANCE ISSUES.
Identify any regulatory or policy compliance concerns.

Segment:
{transcript}

Provide your analysis in JSON format with: compliance_issue, flag_type, severity, explanation."""
        
        else:
            return f"Analyse this call transcript:\n\n{transcript}"
    
    def train(
        self,
        max_samples: int = 2500,
        epochs: int = 3,
        resume_from_checkpoint: bool = False,
    ) -> Dict[str, Any]:
        """
        Run fine-tuning on annotations.
        
        Args:
            max_samples: Maximum training samples to use
            epochs: Number of training epochs
            resume_from_checkpoint: Resume from last checkpoint
            
        Returns:
            Training results dict
        """
        if not TRAINING_AVAILABLE:
            raise RuntimeError("Training dependencies not installed. Run: pip install peft datasets")
        
        # Acquire training lock
        if not self._acquire_training_lock():
            in_progress, lock_info = self.is_training_in_progress()
            return {
                "success": False,
                "error": "Training already in progress",
                "lock_info": lock_info,
            }
        
        try:
            start_time = time.time()
            
            # Prepare data
            logger.info("Loading training data from annotations...")
            samples = self.prepare_training_data(limit=max_samples)
            
            if len(samples) < 10:
                return {
                    "success": False,
                    "error": f"Insufficient training data: {len(samples)} samples (minimum 10)",
                    "samples_found": len(samples),
                }
            
            # Load model and tokenizer
            logger.info(f"Loading base model: {self.base_model_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                padding_side="right",
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with quantization for memory efficiency
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                # Use 4-bit for training to fit in memory
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            
            # Prepare model for training
            model.config.use_cache = False
            model = self._prepare_model_for_training(model)
            
            # Apply LoRA
            logger.info("Applying LoRA configuration...")
            lora_config = LoraConfig(**self.lora_config)
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            # Prepare dataset
            logger.info("Tokenizing training data...")
            train_dataset = self.format_for_training(samples, tokenizer)
            
            # Set up versioned checkpoint directory
            checkpoint_dir = self._generate_versioned_output_dir()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Training arguments with versioned output dir
            train_args_dict = {**self.training_args}
            train_args_dict["output_dir"] = str(checkpoint_dir)
            
            training_args = TrainingArguments(
                **train_args_dict,
                num_train_epochs=epochs,
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
            )
            
            # Train
            logger.info("Starting training...")
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            # Generate versioned adapter path
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            version_name = f"{self.adapter_name}-{timestamp}"
            versioned_path = self.adapter_base_path / version_name
            
            # Save adapter to versioned directory
            logger.info(f"Saving adapter to {versioned_path}")
            versioned_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(versioned_path)
            tokenizer.save_pretrained(versioned_path)
            
            elapsed = time.time() - start_time
            
            # Save training metadata
            metadata = {
                "adapter_name": self.adapter_name,
                "version": version_name,
                "trained_at": datetime.now().isoformat(),
                "base_model": self.base_model_path,
                "samples_used": len(samples),
                "epochs": epochs,
                "training_loss": train_result.training_loss,
                "training_time_seconds": elapsed,
                "lora_config": self.lora_config,
            }
            
            with open(versioned_path / "training_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Promote this version to current
            self._set_current_version(version_name, metadata)
            
            logger.info(f"Training completed in {elapsed/60:.1f} minutes")
            logger.info(f"New adapter version: {version_name} (now current)")
            
            return {
                "success": True,
                "adapter_name": self.adapter_name,
                "version": version_name,
                "adapter_path": str(versioned_path),
                "samples_used": len(samples),
                "epochs": epochs,
                "training_loss": train_result.training_loss,
                "training_time_minutes": elapsed / 60,
            }
            
        except Exception as e:
            logger.exception(f"Training failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }
        finally:
            # Always release the lock
            self._release_training_lock()
    
    def _prepare_model_for_training(self, model):
        """Prepare model for LoRA training."""
        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)
        
        # Enable gradient checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        return model
    
    def _generate_versioned_output_dir(self) -> Path:
        """Generate a timestamped output directory for checkpoints."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        return self.adapter_base_path / f"{self.adapter_name}-{timestamp}" / "checkpoints"
    
    def get_adapter_info(self, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get info about an adapter version.
        
        Args:
            version: Specific version name, or None for current version
            
        Returns:
            Metadata dict or None if not found
        """
        if version:
            # Get specific version
            version_path = self.adapter_base_path / version
            metadata_path = version_path / "training_metadata.json"
        else:
            # Get current version
            current = self._get_current_version()
            if not current:
                # Backwards compatibility: check for non-versioned adapter
                metadata_path = self.adapter_base_path / "training_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        return json.load(f)
                return None
            
            version_path = self.adapter_base_path / current["version"]
            metadata_path = version_path / "training_metadata.json"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path) as f:
            return json.load(f)
    
    def _get_current_version(self) -> Optional[Dict[str, Any]]:
        """Get the current (active) version info from manifest."""
        manifest_path = self.adapter_base_path / "current.json"
        
        if not manifest_path.exists():
            return None
        
        with open(manifest_path) as f:
            return json.load(f)
    
    def _set_current_version(self, version: str, metadata: Dict[str, Any]) -> None:
        """
        Set a version as the current active adapter.
        
        Args:
            version: Version name (e.g., "call-analysis-20260121-0200")
            metadata: Training metadata for this version
        """
        self.adapter_base_path.mkdir(parents=True, exist_ok=True)
        
        manifest = {
            "version": version,
            "promoted_at": datetime.now().isoformat(),
            "trained_at": metadata.get("trained_at"),
            "samples_used": metadata.get("samples_used"),
            "training_loss": metadata.get("training_loss"),
        }
        
        manifest_path = self.adapter_base_path / "current.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Create 'current' symlink for vLLM static loading
        # This allows vLLM to load adapters at startup with --lora-modules
        symlink_path = self.adapter_base_path / "current"
        version_path = self.adapter_base_path / version
        
        # Remove existing symlink if present
        if symlink_path.exists() or symlink_path.is_symlink():
            try:
                symlink_path.unlink()
            except Exception as e:
                logger.warning(f"Could not remove old symlink: {e}")
        
        # Create new symlink
        try:
            symlink_path.symlink_to(version_path, target_is_directory=True)
            logger.info(f"Created symlink: current -> {version}")
        except Exception as e:
            logger.warning(f"Could not create symlink (may require admin on Windows): {e}")
        
        logger.info(f"Set current version to: {version}")
    
    def get_current_version_name(self) -> Optional[str]:
        """
        Get the name of the current active version.
        
        Returns:
            Version name string or None if no current version
        """
        current = self._get_current_version()
        return current["version"] if current else None
    
    def get_current_adapter_path(self) -> Optional[Path]:
        """
        Get the filesystem path to the current adapter.
        
        This is the path to pass to vLLM for loading.
        
        Returns:
            Path to current adapter or None
        """
        version = self.get_current_version_name()
        if not version:
            # Backwards compatibility
            if (self.adapter_base_path / "adapter_config.json").exists():
                return self.adapter_base_path
            return None
        return self.adapter_base_path / version
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """
        List all available adapter versions.
        
        Returns:
            List of version info dicts, sorted newest first
        """
        versions = []
        current_version = self.get_current_version_name()
        
        if not self.adapter_base_path.exists():
            return versions
        
        for item in self.adapter_base_path.iterdir():
            if not item.is_dir():
                continue
            
            # Skip checkpoints directories
            if item.name == "checkpoints":
                continue
            
            # Check for adapter config (indicates valid adapter)
            if not (item / "adapter_config.json").exists():
                continue
            
            metadata_path = item / "training_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    versions.append({
                        "version": item.name,
                        "trained_at": metadata.get("trained_at"),
                        "samples_used": metadata.get("samples_used"),
                        "training_loss": metadata.get("training_loss"),
                        "is_current": item.name == current_version,
                    })
        
        # Sort by trained_at descending (newest first)
        versions.sort(key=lambda v: v.get("trained_at", ""), reverse=True)
        return versions
    
    def promote_version(self, version: str) -> bool:
        """
        Promote a specific version to be the current active adapter.
        
        Use this for rollback or to switch to a specific version.
        
        Args:
            version: Version name to promote
            
        Returns:
            True if successful, False if version not found
        """
        version_path = self.adapter_base_path / version
        
        if not version_path.exists():
            logger.error(f"Version not found: {version}")
            return False
        
        if not (version_path / "adapter_config.json").exists():
            logger.error(f"Invalid adapter directory: {version}")
            return False
        
        metadata = self.get_adapter_info(version=version) or {}
        self._set_current_version(version, metadata)
        
        return True
    
    def cleanup_old_versions(self, keep: int = 5) -> int:
        """
        Remove old adapter versions, keeping the newest N.
        
        Never removes the current version.
        
        Args:
            keep: Number of versions to keep (including current)
            
        Returns:
            Number of versions removed
        """
        versions = self.list_versions()
        
        if len(versions) <= keep:
            return 0
        
        removed = 0
        current_version = self.get_current_version_name()
        
        # Versions are sorted newest first, so skip first `keep` entries
        for version in versions[keep:]:
            # Never remove current
            if version["version"] == current_version:
                continue
            
            version_path = self.adapter_base_path / version["version"]
            if version_path.exists():
                shutil.rmtree(version_path)
                logger.info(f"Removed old adapter version: {version['version']}")
                removed += 1
        
        return removed
    
    def count_new_annotations(self, since: Optional[datetime] = None) -> int:
        """
        Count annotations added since the last training run.
        
        Args:
            since: Count annotations after this datetime. If None, uses last training time.
            
        Returns:
            Number of new annotations
        """
        from src.database import get_db_connection
        
        # Get last training time if not specified
        if since is None:
            metadata = self.get_adapter_info()
            if metadata and metadata.get("trained_at"):
                since = datetime.fromisoformat(metadata["trained_at"])
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            if since:
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM ai_call_annotations 
                    WHERE is_training_data = TRUE 
                    AND created_at > %s
                """, (since,))
            else:
                # No previous training - count all
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM ai_call_annotations 
                    WHERE is_training_data = TRUE
                """)
            
            result = cursor.fetchone()
            return result[0] if result else 0
    
    def should_train(self, min_new_annotations: int = 1) -> tuple[bool, str]:
        """
        Check if training should run.
        
        Args:
            min_new_annotations: Minimum new annotations required to trigger training
            
        Returns:
            Tuple of (should_train, reason)
        """
        metadata = self.get_adapter_info()
        
        if metadata:
            last_trained = metadata.get("trained_at")
            new_count = self.count_new_annotations()
            
            if new_count < min_new_annotations:
                return False, f"Only {new_count} new annotations since last training at {last_trained}. Minimum required: {min_new_annotations}"
            
            return True, f"Found {new_count} new annotations since {last_trained}"
        else:
            # No previous training - check if we have enough total annotations
            total = self.count_new_annotations()
            if total < 10:
                return False, f"Only {total} total annotations. Minimum required for first training: 10"
            
            return True, f"First training run with {total} annotations"
    
    def load_adapter(self, base_model) -> Any:
        """
        Load the trained adapter onto a base model.
        
        Args:
            base_model: The base model to apply adapter to
            
        Returns:
            Model with adapter applied
        """
        if not TRAINING_AVAILABLE:
            raise RuntimeError("peft not installed")
        
        adapter_path = self.adapter_output_path
        
        if not (adapter_path / "adapter_config.json").exists():
            logger.info("No trained adapter found, using base model")
            return base_model
        
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        return model


def run_nightly_training():
    """
    Entry point for nightly training cron job.
    
    Usage: python -m src.services.trainer
    """
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info("=" * 60)
    logger.info("Starting nightly training job")
    logger.info("=" * 60)
    
    trainer = TrainingService()
    
    # Check for existing adapter
    existing = trainer.get_adapter_info()
    if existing:
        logger.info(f"Existing adapter found, trained at {existing.get('trained_at')}")
        logger.info(f"Previous training used {existing.get('samples_used')} samples")
    
    # Check if we should train
    should_train, reason = trainer.should_train(min_new_annotations=1)
    logger.info(f"Training check: {reason}")
    
    if not should_train:
        logger.info("=" * 60)
        logger.info("Skipping training - no new annotations")
        logger.info("=" * 60)
        sys.exit(0)
    
    try:
        result = trainer.train(max_samples=2500, epochs=3)
        
        if result["success"]:
            logger.info("=" * 60)
            logger.info("Training completed successfully!")
            logger.info(f"  Version: {result.get('version')}")
            logger.info(f"  Adapter saved to: {result['adapter_path']}")
            logger.info(f"  Samples used: {result['samples_used']}")
            logger.info(f"  Training loss: {result['training_loss']:.4f}")
            logger.info(f"  Time taken: {result['training_time_minutes']:.1f} minutes")
            logger.info("=" * 60)
            
            # Cleanup old versions (keep last 5)
            removed = trainer.cleanup_old_versions(keep=5)
            if removed > 0:
                logger.info(f"Cleaned up {removed} old adapter version(s)")
            
            sys.exit(0)
        else:
            logger.error(f"Training failed: {result.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        logger.exception(f"Training job failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_nightly_training()
