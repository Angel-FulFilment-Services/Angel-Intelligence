#!/usr/bin/env python3
"""
Angel Intelligence - Prompt Tester

Test script that generates the full LLM prompt for a call recording
WITHOUT marking it as processing or completed.

Usage:
    python scripts/test_prompt.py                    # Test next pending recording
    python scripts/test_prompt.py --id 123           # Test specific recording by ID
    python scripts/test_prompt.py --apex ABC123      # Test specific recording by apex_id
    python scripts/test_prompt.py --id 123 --save    # Save prompt to file

This helps verify that:
- Transcript is being formatted correctly
- Enquiry context is loading (calltypes)
- Order context is loading (customer/order data)
- Client/campaign/direction configs are merging correctly
- Quality signals are being included
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.database import CallRecording, CallTranscription, get_db_connection
from src.services.config import get_config_service
from src.services.enquiry_context import get_enquiry_context_service
from src.services.order_context import get_order_context_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class PromptTester:
    """
    Test service that generates LLM prompts without processing.
    
    Replicates the prompt building logic from AnalysisService but
    outputs to console for testing and validation.
    """
    
    def __init__(self):
        """Initialise the prompt tester."""
        self.settings = get_settings()
        self.config_service = get_config_service()
        self.enquiry_service = get_enquiry_context_service()
        self.order_service = get_order_context_service()
        
        # Load global config file as fallback
        self._file_config = self._load_config_file()
        
        logger.info("PromptTester initialised")
    
    def _load_config_file(self, config_path: str = "call_analysis_config.json") -> Dict[str, Any]:
        """Load analysis configuration from file."""
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"topics": [], "agent_actions": [], "performance_rubric": []}
    
    def get_recording(self, recording_id: Optional[int] = None, apex_id: Optional[str] = None) -> Optional[CallRecording]:
        """
        Get a recording by ID, apex_id, or the next pending one.
        
        Does NOT modify the recording status.
        """
        db = get_db_connection()
        
        if recording_id:
            row = db.fetch_one(
                "SELECT * FROM ai_call_recordings WHERE id = %s",
                (recording_id,)
            )
        elif apex_id:
            row = db.fetch_one(
                "SELECT * FROM ai_call_recordings WHERE apex_id = %s",
                (apex_id,)
            )
        else:
            # Get the next pending recording (same logic as worker, but no update)
            row = db.fetch_one("""
                SELECT * FROM ai_call_recordings
                WHERE processing_status IN ('pending', 'queued')
                   OR (processing_status = 'failed' 
                       AND retry_count < 3 
                       AND (next_retry_at IS NULL OR next_retry_at <= NOW()))
                ORDER BY 
                    CASE processing_status 
                        WHEN 'queued' THEN 0 
                        WHEN 'pending' THEN 1 
                        ELSE 2 
                    END,
                    id ASC
                LIMIT 1
            """)
        
        return CallRecording.from_row(row) if row else None
    
    def get_transcription(self, recording: CallRecording) -> Optional[Dict[str, Any]]:
        """
        Get existing transcription for a recording.
        
        Returns transcript in the same format as TranscriptionService.
        """
        transcription = CallTranscription.get_by_recording_id(recording.id)
        
        if not transcription:
            # Try by apex_id (from Dojo pre-transcription)
            transcription = CallTranscription.get_by_apex_id(recording.apex_id)
        
        if transcription:
            return {
                "full_transcript": transcription.full_transcript,
                "segments": transcription.segments or [],
                "language_detected": transcription.language_detected,
                "confidence": transcription.confidence_score,
                "model_used": transcription.model_used,
            }
        
        return None
    
    def _format_transcript_with_timestamps(self, transcript: Dict[str, Any]) -> tuple:
        """
        Format transcript segments with timestamps and IDs for LLM analysis.
        
        Matches the format used by AnalysisService.
        """
        segments = transcript.get("segments", [])
        
        if not segments:
            return transcript.get("full_transcript", ""), {}, 0
        
        formatted_lines = []
        segment_map = {}
        max_timestamp = 0
        
        for i, seg in enumerate(segments):
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            speaker = seg.get("speaker", "Unknown")
            text = seg.get("text", "").strip()
            
            if not text:
                continue
            
            max_timestamp = max(max_timestamp, end)
            
            # Use existing segment_id from transcript, or fall back to index-based
            segment_id = seg.get("segment_id") or f"seg_{i+1:03d}"
            
            # Normalise speaker label for readability
            if speaker in ["SPEAKER_00", "agent"] or speaker.startswith("agent"):
                speaker_label = "Agent"
            elif speaker in ["SPEAKER_01", "supporter"] or speaker.startswith("supporter"):
                speaker_label = "Supporter"
            else:
                speaker_label = speaker.title()
            
            # Build segment map for reference
            segment_map[segment_id] = {
                "start": start,
                "end": end,
                "speaker": speaker_label.lower()
            }
            
            # Format with segment ID and timestamp
            formatted_lines.append(f"[{segment_id}] [{start:.1f}s - {end:.1f}s] {speaker_label}: {text}")
        
        return "\n".join(formatted_lines), segment_map, max_timestamp
    
    def build_prompt(
        self,
        recording: CallRecording,
        transcript: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build the full analysis prompt.
        
        Returns a dictionary with:
        - prompt: The full LLM prompt text
        - context_parts: Individual context sections
        - merged_config: The merged configuration used
        - metadata: Recording metadata
        """
        context_parts = []
        
        # 1. Get enquiry context (calltype validation)
        enquiry_context = None
        if recording.enqref and recording.client_ref:
            try:
                enquiry_context = self.enquiry_service.get_enquiry_context(
                    enqref=recording.enqref,
                    client_ref=recording.client_ref,
                    ddi=recording.ddi
                )
                if enquiry_context.has_context:
                    context_parts.append(("ENQUIRY CONTEXT", enquiry_context.to_prompt_context()))
                    logger.info(f"✓ Enquiry context loaded: {len(enquiry_context.available_calltypes or [])} calltypes")
                elif enquiry_context.error:
                    logger.warning(f"✗ Enquiry context error: {enquiry_context.error}")
            except Exception as e:
                logger.warning(f"✗ Failed to get enquiry context: {e}")
        else:
            logger.info("○ No enqref/client_ref - skipping enquiry context")
        
        # 2. Get order context (data validation)
        order_context = None
        if recording.orderref and recording.client_ref:
            try:
                order_context = self.order_service.get_order_context(
                    orderref=recording.orderref,
                    client_ref=recording.client_ref,
                    ddi=recording.ddi
                )
                if order_context.has_context:
                    context_parts.append(("ORDER CONTEXT", order_context.to_prompt_context()))
                    logger.info(f"✓ Order context loaded: {order_context.customer_name}, {order_context.product}")
                elif order_context.error:
                    logger.warning(f"✗ Order context error: {order_context.error}")
            except Exception as e:
                logger.warning(f"✗ Failed to get order context: {e}")
        else:
            logger.info("○ No orderref/client_ref - skipping order context")
        
        # 3. Get merged configuration (all 4 tiers)
        merged_config = self.config_service.get_merged_config(
            campaign_type=recording.campaign_type,
            direction=recording.direction,
            client_ref=recording.client_ref
        )
        
        logger.info(f"✓ Config merged: campaign={recording.campaign_type}, direction={recording.direction}, client={recording.client_ref}")
        
        # 4. Format transcript
        formatted_transcript, segment_map, max_timestamp = self._format_transcript_with_timestamps(transcript)
        
        logger.info(f"✓ Transcript formatted: {len(segment_map)} segments, {max_timestamp:.1f}s duration")
        
        # 5. Build the prompt context string
        combined_context_parts = [ctx for _, ctx in context_parts]
        if merged_config.get("prompt_context"):
            combined_context_parts.append(merged_config["prompt_context"])
        
        prompt_context = "\n\n".join(combined_context_parts) if combined_context_parts else ""
        
        # 6. Build the actual prompt (using same logic as AnalysisService)
        prompt = self._build_transcript_analysis_prompt(
            formatted_transcript,
            max_timestamp,
            prompt_context,
            merged_config
        )
        
        return {
            "prompt": prompt,
            "context_parts": context_parts,
            "merged_config": merged_config,
            "formatted_transcript": formatted_transcript,
            "segment_count": len(segment_map),
            "max_timestamp": max_timestamp,
            "metadata": {
                "recording_id": recording.id,
                "apex_id": recording.apex_id,
                "client_ref": recording.client_ref,
                "campaign_type": recording.campaign_type,
                "direction": recording.direction,
                "enqref": recording.enqref,
                "orderref": recording.orderref,
                "ddi": recording.ddi,
            }
        }
    
    def _build_transcript_analysis_prompt(
        self,
        transcript: str,
        max_timestamp: float,
        prompt_context: str,
        merged_config: Dict[str, Any]
    ) -> str:
        """Build the full analysis prompt - mirrors AnalysisService._build_transcript_analysis_prompt"""
        
        # Extract config values
        topics = merged_config.get("topics", [])
        actions = merged_config.get("agent_actions", [])
        rubric = merged_config.get("performance_rubric", [])
        
        # Build lists
        topic_list = ", ".join(topics) if topics else "donation, gift aid, regular giving, complaints, updates, account changes, direct debit, legacy giving, event registration, volunteer enquiry"
        action_list = ", ".join(actions) if actions else "greeting, verification, explanation, objection handling, closing, upselling, complaint resolution, payment processing"
        rubric_list = ", ".join([r.get("name", r) if isinstance(r, dict) else r for r in rubric]) if rubric else "Empathy, Clarity, Listening, Script adherence, Product knowledge, Rapport building, Objection handling, Closing effectiveness"
        
        # Convert rubric items to valid JSON keys
        def to_key(name):
            if isinstance(name, dict):
                name = name.get("name", "")
            return name.replace(" ", "_").replace("&", "and").replace("/", "_")
        
        perf_keys = [to_key(r) for r in rubric] if rubric else [
            "Empathy", "Clarity", "Listening", "Script_adherence",
            "Product_knowledge", "Rapport_building", "Objection_handling", "Closing_effectiveness"
        ]
        perf_keys_list = ", ".join(perf_keys)
        perf_scores_json = ",\n        ".join([f'"{k}": 1-10' for k in perf_keys])
        
        # Build context section
        context_section = f"\n\n{prompt_context}\n" if prompt_context else ""
        
        return f"""You are an expert call quality analyst for charity supporter engagement calls.
Analyse this transcript thoroughly and provide detailed coaching insights.
{context_section}
CALL DURATION: {max_timestamp:.1f} seconds

TRANSCRIPT (each line has [segment_id] [start - end] Speaker: text):
{transcript}

TOPICS TO IDENTIFY: {topic_list}
AGENT ACTIONS TO DETECT: {action_list}
PERFORMANCE CRITERIA: {rubric_list}

Return your analysis as valid JSON matching this exact structure:

{{
    "summary": "2-3 sentence British English summary describing the call purpose, outcome, and notable moments",
    "sentiment_score": 0 to 10 (0=hostile, 5=neutral, 10=very friendly),
    "key_topics": [
        {{"name": "Topic Name In Title Case", "confidence": 0.0-1.0}}
    ],
    "agent_actions": [
        {{"action": "specific action the agent took", "segment_ids": ["seg_001", "seg_002"]}}
    ],
    "score_impacts": [
        {{"segment_ids": ["seg_001"], "impact": -5 to +5, "category": "one of: {perf_keys_list}", "reason": "why this affected the score", "quote": "exact quote from transcript"}}
    ],
    "compliance_flags": [
        {{"type": "GDPR|payment_security|misleading_info|rudeness|data_protection", "segment_ids": ["seg_001", "seg_002"], "severity": "low/medium/high/critical", "issue": "detailed description", "quote": "exact quote from transcript"}}
    ],
    "performance_scores": {{
        {perf_scores_json}
    }},
    "action_items": [
        {{"description": "specific follow-up action needed", "priority": "high/medium/low"}}
    ]
}}

SEGMENT IDS - CRITICAL:
- segment_ids is an ARRAY - use ["seg_001"] for single segment or ["seg_001", "seg_002"] for spans
- Copy segment_id values EXACTLY from the transcript
- Do NOT invent segment IDs - only use ones that appear in the transcript

SCORE IMPACT SCALE:
- +5: Exceptional moment (exemplary)
- +3 to +4: Strong positive
- +1 to +2: Minor positive
- -1 to -2: Minor negative
- -3 to -4: Significant negative
- -5: Severe issue (compliance breach, rudeness)

COMPLIANCE FLAGS: Only include if actual issues detected. Use empty array [] if none.

Return ONLY valid JSON - no markdown code fences, no text before or after."""


def main():
    parser = argparse.ArgumentParser(
        description="Test LLM prompt generation for call analysis"
    )
    parser.add_argument(
        "--id",
        type=int,
        help="Recording ID to test"
    )
    parser.add_argument(
        "--apex",
        type=str,
        help="Apex ID to test"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save prompt to file instead of printing"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output the prompt, no metadata"
    )
    
    args = parser.parse_args()
    
    tester = PromptTester()
    
    # Get the recording
    print("\n" + "=" * 80)
    print("PROMPT TESTER - Angel Intelligence")
    print("=" * 80 + "\n")
    
    recording = tester.get_recording(recording_id=args.id, apex_id=args.apex)
    
    if not recording:
        print("❌ No recording found!")
        if args.id:
            print(f"   Could not find recording with ID: {args.id}")
        elif args.apex:
            print(f"   Could not find recording with apex_id: {args.apex}")
        else:
            print("   No pending recordings in queue")
        sys.exit(1)
    
    print(f"📞 Recording: ID={recording.id}, apex_id={recording.apex_id}")
    print(f"   Status: {recording.processing_status.value}")
    print(f"   Client: {recording.client_ref or 'N/A'}")
    print(f"   Campaign: {recording.campaign_type or 'N/A'}")
    print(f"   Direction: {recording.direction}")
    print(f"   EnqRef: {recording.enqref or 'N/A'}")
    print(f"   OrderRef: {recording.orderref or 'N/A'}")
    print(f"   DDI: {recording.ddi or 'N/A'}")
    print()
    
    # Get transcription
    transcript = tester.get_transcription(recording)
    
    if not transcript:
        print("❌ No transcription found for this recording!")
        print("   The recording needs to be transcribed first before testing the prompt.")
        print("   You can either:")
        print("   1. Process the recording normally first")
        print("   2. Use a recording that has already been transcribed")
        sys.exit(1)
    
    print(f"📝 Transcription: {len(transcript.get('segments', []))} segments")
    print()
    
    # Build the prompt
    print("Building prompt...")
    print("-" * 40)
    
    result = tester.build_prompt(recording, transcript)
    
    print("-" * 40)
    print()
    
    if args.save:
        # Save to file
        filename = f"prompt_test_{recording.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# Prompt Test for Recording {recording.id}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Apex ID: {recording.apex_id}\n")
            f.write(f"# Client: {recording.client_ref}\n")
            f.write(f"# Campaign: {recording.campaign_type}\n")
            f.write(f"# Direction: {recording.direction}\n")
            f.write("#" + "=" * 79 + "\n\n")
            f.write(result["prompt"])
        
        print(f"✅ Prompt saved to: {filename}")
        print(f"   Character count: {len(result['prompt']):,}")
        print(f"   Approx tokens: ~{len(result['prompt']) // 4:,}")
    
    elif args.quiet:
        # Just print the prompt
        print(result["prompt"])
    
    else:
        # Print with metadata
        print("=" * 80)
        print("CONTEXT PARTS LOADED")
        print("=" * 80)
        
        if result["context_parts"]:
            for name, context in result["context_parts"]:
                print(f"\n--- {name} ---")
                print(context[:500] + "..." if len(context) > 500 else context)
        else:
            print("(No additional context loaded)")
        
        print("\n" + "=" * 80)
        print("MERGED CONFIG")
        print("=" * 80)
        
        config = result["merged_config"]
        print(f"Topics ({len(config.get('topics', []))}): {config.get('topics', [])[:5]}...")
        print(f"Agent Actions ({len(config.get('agent_actions', []))}): {config.get('agent_actions', [])[:5]}...")
        print(f"Performance Rubric ({len(config.get('performance_rubric', []))}): {[r.get('name', r) if isinstance(r, dict) else r for r in config.get('performance_rubric', [])][:5]}...")
        
        if config.get("prompt_context"):
            print(f"Prompt Context: {len(config['prompt_context'])} chars")
        
        print("\n" + "=" * 80)
        print("FULL PROMPT")
        print("=" * 80 + "\n")
        
        print(result["prompt"])
        
        print("\n" + "=" * 80)
        print("STATISTICS")
        print("=" * 80)
        print(f"Segments: {result['segment_count']}")
        print(f"Duration: {result['max_timestamp']:.1f}s")
        print(f"Prompt length: {len(result['prompt']):,} characters")
        print(f"Approx tokens: ~{len(result['prompt']) // 4:,}")
        print()


if __name__ == "__main__":
    main()
