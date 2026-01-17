"""
Angel Intelligence - PII Detection and Redaction Service

Detects and redacts UK-specific Personally Identifiable Information (PII)
from transcripts and audio files.

Supports UK patterns:
- National Insurance Number
- NHS Number
- UK Postcodes
- UK Phone Numbers
- Bank Sort Codes and Account Numbers
- Credit/Debit Card Numbers
- Dates of Birth
- Email Addresses
- UK Driving Licence Numbers
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from src.config import get_settings

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    logger.warning("Presidio not available - basic PII detection will be used")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logger.warning("soundfile not available - audio redaction disabled")


@dataclass
class PIIMatch:
    """Represents a detected PII match."""
    pii_type: str
    text: str
    start_char: int
    end_char: int
    confidence: float
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None


# UK-specific regex patterns
UK_PII_PATTERNS = {
    "UK_NATIONAL_INSURANCE": {
        "pattern": r"\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-Z]\b",
        "description": "UK National Insurance Number (e.g., AB123456C)",
        "confidence": 0.9,
    },
    "UK_NHS_NUMBER": {
        "pattern": r"\b\d{3}\s?\d{3}\s?\d{4}\b",
        "description": "UK NHS Number (e.g., 123 456 7890)",
        "confidence": 0.7,  # Lower as it matches many 10-digit sequences
        "context": ["nhs", "health", "patient", "medical", "hospital"],
    },
    "UK_POSTCODE": {
        "pattern": r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b",
        "description": "UK Postcode (e.g., SW1A 1AA)",
        "confidence": 0.85,
    },
    "UK_PHONE_MOBILE": {
        "pattern": r"\b(?:\+44\s?|0)7\d{3}\s?\d{6}\b",
        "description": "UK Mobile Phone Number",
        "confidence": 0.9,
    },
    "UK_PHONE_LANDLINE": {
        "pattern": r"\b(?:\+44\s?|0)(?:1|2)\d{2,3}\s?\d{6,7}\b",
        "description": "UK Landline Phone Number",
        "confidence": 0.85,
    },
    "UK_SORT_CODE": {
        "pattern": r"\b\d{2}[-\s]?\d{2}[-\s]?\d{2}\b",
        "description": "UK Bank Sort Code (e.g., 12-34-56)",
        "confidence": 0.6,  # Lower as it's generic
        "context": ["bank", "sort", "account", "payment"],
    },
    "UK_BANK_ACCOUNT": {
        "pattern": r"\b\d{8}\b",
        "description": "UK Bank Account Number (8 digits)",
        "confidence": 0.5,  # Low as it's very generic
        "context": ["account", "bank", "number", "payment", "sort"],
    },
    "CREDIT_CARD": {
        "pattern": r"\b(?:\d{4}[\s-]?){3}\d{4}\b",
        "description": "Credit/Debit Card Number",
        "confidence": 0.95,
    },
    "CARD_EXPIRY": {
        "pattern": r"\b(?:0[1-9]|1[0-2])\/(?:\d{2}|\d{4})\b",
        "description": "Card Expiry Date (e.g., 12/26)",
        "confidence": 0.7,
        "context": ["card", "expiry", "expires", "valid"],
    },
    "CVV": {
        "pattern": r"\b\d{3}\b",
        "description": "Card CVV/CVC (3 digits)",
        "confidence": 0.3,  # Very low as it matches many 3-digit sequences
        "context": ["cvv", "cvc", "security", "code", "back of card"],
    },
    "UK_DOB": {
        "pattern": r"\b(?:0[1-9]|[12]\d|3[01])\/(?:0[1-9]|1[0-2])\/(?:19|20)\d{2}\b",
        "description": "Date of Birth (DD/MM/YYYY)",
        "confidence": 0.75,
        "context": ["born", "birth", "dob", "date of birth", "birthday"],
    },
    "EMAIL": {
        "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "description": "Email Address",
        "confidence": 0.95,
    },
    "UK_DRIVING_LICENCE": {
        "pattern": r"\b[A-Z]{5}\d{6}[A-Z]{2}\d[A-Z]{2}\b",
        "description": "UK Driving Licence Number",
        "confidence": 0.95,
    },
}


class PIIDetector:
    """
    PII detection and redaction service.
    
    Uses Presidio for standard PII detection with custom UK-specific patterns.
    Can redact PII from both text transcripts and audio files.
    """
    
    def __init__(self):
        """Initialise the PII detector."""
        settings = get_settings()
        self.enabled = settings.enable_pii_redaction
        
        # Presidio engines
        self._analyzer = None
        self._anonymizer = None
        
        if self.enabled and PRESIDIO_AVAILABLE:
            self._init_presidio()
        elif self.enabled:
            logger.warning("PII detection enabled but Presidio not installed - using regex fallback")
    
    def _init_presidio(self) -> None:
        """Initialise Presidio engines with UK-specific recognisers."""
        try:
            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            
            # Add UK-specific recognisers
            self._add_uk_recognisers()
            
            logger.info("Presidio PII detector initialised with UK recognisers")
            
        except Exception as e:
            logger.error(f"Failed to initialise Presidio: {e}")
            self._analyzer = None
            self._anonymizer = None
    
    def _add_uk_recognisers(self) -> None:
        """Add UK-specific pattern recognisers to Presidio."""
        if not self._analyzer:
            return
        
        for pii_type, config in UK_PII_PATTERNS.items():
            # Skip types that Presidio already handles well
            if pii_type in ["EMAIL", "CREDIT_CARD"]:
                continue
            
            patterns = [
                Pattern(
                    name=pii_type.lower(),
                    regex=config["pattern"],
                    score=config["confidence"]
                )
            ]
            
            context = config.get("context", [])
            
            recogniser = PatternRecognizer(
                supported_entity=pii_type,
                patterns=patterns,
                context=context if context else None
            )
            
            self._analyzer.registry.add_recognizer(recogniser)
            logger.debug(f"Added UK PII recogniser: {pii_type}")
    
    def detect(
        self, 
        text: str,
        segments: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Detect PII in text.
        
        Args:
            text: Full transcript text
            segments: Optional word/sentence segments with timestamps
            
        Returns:
            Dictionary containing:
            - pii_detected: List of PII matches
            - redacted_text: Text with PII replaced by type placeholders
            - pii_count: Total number of PII items found
            - pii_types: Set of PII types found
        """
        if not self.enabled:
            return {
                "pii_detected": [],
                "redacted_text": text,
                "pii_count": 0,
                "pii_types": [],
            }
        
        if PRESIDIO_AVAILABLE and self._analyzer:
            return self._detect_with_presidio(text, segments)
        else:
            return self._detect_with_regex(text, segments)
    
    def _detect_with_presidio(
        self, 
        text: str,
        segments: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Detect PII using Presidio."""
        # Define entities to detect
        entities = [
            "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
            "UK_NHS", "IBAN_CODE", "URL", "IP_ADDRESS",
            # Custom UK entities
            "UK_NATIONAL_INSURANCE", "UK_NHS_NUMBER", "UK_POSTCODE",
            "UK_PHONE_MOBILE", "UK_PHONE_LANDLINE", "UK_SORT_CODE",
            "UK_BANK_ACCOUNT", "CARD_EXPIRY", "UK_DOB", "UK_DRIVING_LICENCE",
        ]
        
        # Analyse text
        results = self._analyzer.analyze(text=text, language='en', entities=entities)
        
        # Filter false positives (e.g., "Age UK" detected as location)
        results = [r for r in results if not self._is_false_positive(r, text)]
        
        # Convert to PII matches
        pii_detected = []
        for result in results:
            match = PIIMatch(
                pii_type=result.entity_type,
                text=text[result.start:result.end],
                start_char=result.start,
                end_char=result.end,
                confidence=result.score,
            )
            
            # Add timestamps if segments provided
            if segments:
                timestamps = self._find_timestamps(match, segments, text)
                match.timestamp_start = timestamps[0]
                match.timestamp_end = timestamps[1]
            
            pii_detected.append(match)
        
        # Anonymise text
        if results and self._anonymizer:
            anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)
            redacted_text = anonymized.text
        else:
            redacted_text = text
        
        return {
            "pii_detected": [self._match_to_dict(m) for m in pii_detected],
            "redacted_text": redacted_text,
            "pii_count": len(pii_detected),
            "pii_types": list(set(m.pii_type for m in pii_detected)),
        }
    
    def _detect_with_regex(
        self, 
        text: str,
        segments: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Detect PII using regex patterns (fallback when Presidio unavailable)."""
        pii_detected = []
        redacted_text = text
        
        for pii_type, config in UK_PII_PATTERNS.items():
            pattern = re.compile(config["pattern"], re.IGNORECASE)
            
            for match in pattern.finditer(text):
                # Check context if required
                if "context" in config:
                    context_text = text[max(0, match.start()-50):match.end()+50].lower()
                    if not any(ctx in context_text for ctx in config["context"]):
                        continue
                
                pii_match = PIIMatch(
                    pii_type=pii_type,
                    text=match.group(),
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=config["confidence"],
                )
                
                # Add timestamps if segments provided
                if segments:
                    timestamps = self._find_timestamps(pii_match, segments, text)
                    pii_match.timestamp_start = timestamps[0]
                    pii_match.timestamp_end = timestamps[1]
                
                pii_detected.append(pii_match)
        
        # Redact text (replace with placeholders)
        # Sort by position descending to avoid offset issues
        for match in sorted(pii_detected, key=lambda m: m.start_char, reverse=True):
            redacted_text = (
                redacted_text[:match.start_char] + 
                f"[{match.pii_type}]" + 
                redacted_text[match.end_char:]
            )
        
        return {
            "pii_detected": [self._match_to_dict(m) for m in pii_detected],
            "redacted_text": redacted_text,
            "pii_count": len(pii_detected),
            "pii_types": list(set(m.pii_type for m in pii_detected)),
        }
    
    def _find_timestamps(
        self, 
        match: PIIMatch, 
        segments: List[Dict[str, Any]], 
        full_text: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Find audio timestamps for a PII match based on character position."""
        char_pos = 0
        start_time = None
        end_time = None
        
        for segment in segments:
            text = segment.get("text", segment.get("word", ""))
            segment_start = char_pos
            segment_end = char_pos + len(text)
            
            # Check if PII overlaps with this segment
            if match.start_char < segment_end and match.end_char > segment_start:
                if start_time is None:
                    start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
            
            char_pos = segment_end + 1  # +1 for space
        
        return start_time, end_time
    
    def _is_false_positive(self, result, text: str) -> bool:
        """Check if a detection is a likely false positive."""
        detected_text = text[result.start:result.end].upper()
        
        # "UK" alone is often not PII
        if result.entity_type == "LOCATION" and detected_text in ("UK", "UNITED KINGDOM"):
            return True
        
        return False
    
    def _match_to_dict(self, match: PIIMatch) -> Dict[str, Any]:
        """
        Convert PIIMatch to dictionary.
        
        Format matches specification:
        {
            "type": "national_insurance_number",
            "original": "AB123456C",
            "redacted": "[NI_NUMBER]",
            "timestamp_start": 45.2,
            "timestamp_end": 47.8,
            "confidence": 0.95
        }
        """
        # Map internal type names to specification format
        type_mapping = {
            "UK_NATIONAL_INSURANCE": "national_insurance_number",
            "UK_NHS_NUMBER": "nhs_number",
            "UK_POSTCODE": "postcode",
            "UK_PHONE_MOBILE": "phone_number",
            "UK_PHONE_LANDLINE": "phone_number",
            "UK_SORT_CODE": "sort_code",
            "UK_BANK_ACCOUNT": "bank_account",
            "CREDIT_CARD": "credit_card",
            "CARD_EXPIRY": "card_expiry",
            "CVV": "cvv",
            "UK_DOB": "date_of_birth",
            "EMAIL": "email",
            "EMAIL_ADDRESS": "email",
            "UK_DRIVING_LICENCE": "driving_licence",
            "PHONE_NUMBER": "phone_number",
        }
        
        # Map type to redaction placeholder
        redaction_mapping = {
            "national_insurance_number": "[NI_NUMBER]",
            "nhs_number": "[NHS_NUMBER]",
            "postcode": "[POSTCODE]",
            "phone_number": "[PHONE_NUMBER]",
            "sort_code": "[SORT_CODE]",
            "bank_account": "[ACCOUNT_NUMBER]",
            "credit_card": "[CARD_NUMBER]",
            "card_expiry": "[CARD_EXPIRY]",
            "cvv": "[CVV]",
            "date_of_birth": "[DOB]",
            "email": "[EMAIL]",
            "driving_licence": "[DRIVING_LICENCE]",
        }
        
        pii_type = type_mapping.get(match.pii_type, match.pii_type.lower())
        redacted = redaction_mapping.get(pii_type, f"[{match.pii_type}]")
        
        return {
            "type": pii_type,
            "original": match.text,
            "redacted": redacted,
            "timestamp_start": match.timestamp_start,
            "timestamp_end": match.timestamp_end,
            "confidence": match.confidence,
        }
    
    def redact_audio(
        self, 
        audio_path: str, 
        pii_timestamps: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Redact PII from audio by replacing with beep tones.
        
        Args:
            audio_path: Path to input audio file
            pii_timestamps: List of dicts with timestamp_start and timestamp_end
            output_path: Optional output path (defaults to _redacted suffix)
            
        Returns:
            Path to redacted audio file, or None if redaction failed
        """
        if not SOUNDFILE_AVAILABLE:
            logger.warning("soundfile not available - cannot redact audio")
            return None
        
        if not pii_timestamps:
            logger.debug("No PII timestamps to redact")
            return None
        
        try:
            # Load audio
            audio_data, sample_rate = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Generate beep tone
            def generate_beep(duration_sec: float, freq: int = 1000) -> np.ndarray:
                t = np.linspace(0, duration_sec, int(sample_rate * duration_sec))
                return 0.3 * np.sin(2 * np.pi * freq * t)
            
            # Replace PII sections with beeps
            for pii in pii_timestamps:
                start_time = pii.get("timestamp_start", 0)
                end_time = pii.get("timestamp_end", 0)
                
                if start_time is None or end_time is None:
                    continue
                
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                # Ensure bounds are valid
                start_sample = max(0, start_sample)
                end_sample = min(len(audio_data), end_sample)
                
                if end_sample > start_sample:
                    duration = (end_sample - start_sample) / sample_rate
                    beep = generate_beep(duration)
                    
                    # Replace audio with beep
                    audio_data[start_sample:end_sample] = beep[:end_sample - start_sample]
            
            # Determine output path
            if output_path is None:
                base, ext = os.path.splitext(audio_path)
                output_path = f"{base}_redacted{ext}"
            
            # Save redacted audio
            sf.write(output_path, audio_data, sample_rate)
            
            logger.info(f"Saved redacted audio to {output_path} ({len(pii_timestamps)} sections beeped)")
            return output_path
            
        except Exception as e:
            logger.error(f"Audio redaction failed: {e}", exc_info=True)
            return None
    
    def redact_segments(
        self, 
        segments: List[Dict[str, Any]], 
        pii_detected: List[Dict[str, Any]],
        full_text: str
    ) -> List[Dict[str, Any]]:
        """
        Redact PII from transcript segments.
        
        Args:
            segments: Original transcript segments
            pii_detected: List of detected PII items
            full_text: Full transcript text
            
        Returns:
            Segments with PII text replaced by type placeholders
        """
        redacted_segments = []
        char_pos = 0
        
        for segment in segments:
            text = segment.get("text", segment.get("word", ""))
            segment_start = char_pos
            segment_end = char_pos + len(text)
            
            # Check if any PII overlaps with this segment
            pii_found = None
            for pii in pii_detected:
                if pii["start_char"] < segment_end and pii["end_char"] > segment_start:
                    pii_found = pii
                    break
            
            # Create redacted copy
            redacted = segment.copy()
            if pii_found:
                redacted["text"] = f"[{pii_found['type']}]"
                if "word" in redacted:
                    redacted["word"] = f"[{pii_found['type']}]"
            
            redacted_segments.append(redacted)
            char_pos = segment_end + 1  # +1 for space
        
        return redacted_segments
    
    def is_available(self) -> bool:
        """Check if PII detection is available and enabled."""
        return self.enabled and (PRESIDIO_AVAILABLE or True)  # Regex fallback always available
