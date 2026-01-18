"""
Angel Intelligence - Interactive AI Service

Handles real-time AI requests that require immediate responses:
- Chat conversations
- Summary generation
- Future interactive AI features

Runs on dedicated interactive node(s) to ensure fast response times,
separate from batch processing workers.
"""

import logging
import time
from typing import Dict, Any, List, Optional

from src.config import get_settings
from src.database import get_db_connection

logger = logging.getLogger(__name__)

# Check for model availability
QWEN_AVAILABLE = False
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    QWEN_AVAILABLE = True
except ImportError:
    logger.warning("transformers not available - interactive AI disabled")


class InteractiveService:
    """
    Interactive AI service for real-time requests.
    
    This service handles:
    - Chat conversations about call data
    - Monthly summary generation
    - Ad-hoc AI queries
    
    Designed to run on dedicated node(s) separate from batch workers
    to ensure responsive user experience.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure single model instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialise the interactive service."""
        if self._initialized:
            return
        
        self.settings = get_settings()
        self.use_mock = self.settings.use_mock_models
        
        # Model state (lazy loaded)
        self._model = None
        self._tokenizer = None
        self._device = None
        
        self._initialized = True
        logger.info(f"InteractiveService initialised (mock={self.use_mock})")
    
    def _ensure_model_loaded(self) -> None:
        """Ensure the chat model is loaded."""
        if self.use_mock:
            logger.debug("Using mock mode - skipping model load")
            return
        
        if self._model is not None:
            return
        
        if not QWEN_AVAILABLE:
            raise RuntimeError("transformers not installed")
        
        import torch
        
        logger.info("Loading interactive chat model...")
        start_time = time.time()
        
        # Determine device
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get model path
        model_path = self.settings.chat_model_path or self.settings.chat_model
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load model with appropriate settings
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if self._device == "cuda" else None,
        }
        
        # Apply quantization if configured
        if self.settings.chat_model_quantization == "int4":
            load_kwargs["load_in_4bit"] = True
        elif self.settings.chat_model_quantization == "int8":
            load_kwargs["load_in_8bit"] = True
        else:
            load_kwargs["torch_dtype"] = torch.float16 if self._device == "cuda" else torch.float32
        
        self._model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        
        if self._device == "cpu":
            self._model = self._model.to(self._device)
        
        elapsed = time.time() - start_time
        logger.info(f"Chat model loaded in {elapsed:.1f}s on {self._device}")
    
    def chat(
        self,
        message: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 512,  # Reduced default for faster responses
    ) -> Dict[str, Any]:
        """
        Generate a chat response.
        
        Args:
            message: User's message
            context: Optional context (e.g., call transcript, analysis)
            conversation_history: Previous messages in conversation
            max_tokens: Maximum response tokens (default 512 for speed)
            
        Returns:
            Dict with response text and metadata
        """
        start_time = time.time()
        
        if self.use_mock:
            return self._mock_chat(message, context)
        
        self._ensure_model_loaded()
        
        # Build conversation
        messages = []
        
        # System prompt
        system_prompt = """You are an AI assistant for Angel Fulfilment Services, helping staff analyse charity call recordings.

Key guidelines:
- Use British English spelling and conventions
- **IMPORTANT**: Always format dates as DD/MM/YYYY (e.g., 17/01/2026, not 2026-01-17)
- Format responses using Markdown for better readability when providing detailed information
- **IMPORTANT**: Match your response length and detail to the user's question:
  * Simple greetings ("hello", "hi", "thanks") → Brief, friendly response without data dumps
  * General questions → Concise overview with key points
  * Specific questions → Detailed analysis with relevant data
- You have access to REAL DATA in the context - only reference it when the user asks for analysis or insights
- If the user is just greeting you or chatting casually, don't overwhelm them with metrics
- Keep responses concise unless specifically asked for detailed analysis
- Be data-driven when appropriate, conversational when appropriate
- Professional and supportive tone

**CRITICAL - Stay On Topic:**
- Your ONLY purpose is to help analyse charity call recordings and donor interactions
- ONLY answer questions related to: call quality, transcripts, sentiment, agent performance, donor interactions, call metrics, trends, or related analysis
- REFUSE politely if asked about: general knowledge, personal advice, coding help, unrelated topics, or anything outside call/donor analysis
- If unsure whether a question is relevant, err on the side of keeping it focused on call analysis
- Example refusal: "I'm specifically designed to help with call quality analysis and donor interactions insights. I can't assist with that topic, but I'd be happy to help you analyse call recordings or review performance metrics instead."

**Relevant topics:** call transcripts, quality scores, sentiment analysis, agent performance, donor behaviour, compliance checks, call topics, Gift Aid explanations, objection handling, conversion rates, campaign effectiveness
**Off-limits:** general chat, recipes, weather, news, personal advice, technical support for unrelated systems"""
        
        if context:
            system_prompt += f"\n\n**Available Data Context (use only when relevant to user's question):**\n{context}"
        
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Generate response
        try:
            inputs = self._tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(self._device)
            
            with torch.no_grad():
                outputs = self._model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.6,  # Reduced for faster, more focused responses
                    top_p=0.85,        # Reduced for faster generation
                    pad_token_id=self._tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Prevent repetition
                )
            
            # Decode response (only new tokens)
            response_tokens = outputs[0][inputs.shape[1]:]
            response_text = self._tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            return {
                "response": response_text.strip(),
                "tokens_used": len(response_tokens),
                "processing_time": time.time() - start_time,
                "model": self.settings.chat_model,
            }
            
        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            raise
    
    def generate_summary(
        self,
        data: Dict[str, Any],
        summary_type: str = "monthly",
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate an AI summary of call data.
        
        Args:
            data: Aggregated call data (metrics, counts, etc.)
            summary_type: Type of summary ("monthly", "campaign", "agent")
            filters: Applied filters (date range, client, etc.)
            
        Returns:
            Dict with summary text and insights
        """
        start_time = time.time()
        
        if self.use_mock:
            return self._mock_summary(data, summary_type, filters)
        
        self._ensure_model_loaded()
        
        # Build prompt based on summary type
        filter_desc = ""
        if filters:
            filter_parts = []
            if filters.get("client_ref"):
                filter_parts.append(f"client {filters['client_ref']}")
            if filters.get("campaign"):
                filter_parts.append(f"campaign '{filters['campaign']}'")
            if filters.get("start_date") and filters.get("end_date"):
                filter_parts.append(f"period {filters['start_date']} to {filters['end_date']}")
            if filter_parts:
                filter_desc = f" for {', '.join(filter_parts)}"
        
        prompt = f"""Generate a {summary_type} summary{filter_desc} based on the following call quality data:

Call Count: {data.get('call_count', 0)}
Average Quality Score: {data.get('avg_quality_score', 0):.1f}%
Average Sentiment Score: {data.get('avg_sentiment_score', 0):.1f}
Total Call Duration: {data.get('total_duration_seconds', 0) // 60} minutes

Provide in Markdown format:
1. A brief executive summary (2-3 sentences with **bold** metrics)
2. ### Key Insights - 3-5 bullet points with specific data
3. ### Recommendations - 2-3 actionable bullet points

Use British English. Be specific and data-driven. Format numbers clearly."""

        messages = [
            {"role": "system", "content": "You are an AI analyst generating call quality reports for a charity fundraising company."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            inputs = self._tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(self._device)
            
            with torch.no_grad():
                outputs = self._model.generate(
                    inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            
            response_tokens = outputs[0][inputs.shape[1]:]
            response_text = self._tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Parse response into structured format
            return self._parse_summary_response(response_text, data, start_time)
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            raise
    
    def _parse_summary_response(
        self,
        response_text: str,
        data: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Parse AI response into structured summary format."""
        # Simple parsing - could be enhanced with more sophisticated extraction
        lines = response_text.strip().split('\n')
        
        summary = ""
        insights = []
        recommendations = []
        
        current_section = "summary"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            lower_line = line.lower()
            if "insight" in lower_line or "key finding" in lower_line:
                current_section = "insights"
                continue
            elif "recommendation" in lower_line or "action" in lower_line:
                current_section = "recommendations"
                continue
            
            # Remove bullet points and numbering
            clean_line = line.lstrip('•-*123456789.').strip()
            if not clean_line:
                continue
            
            if current_section == "summary":
                summary += clean_line + " "
            elif current_section == "insights":
                insights.append(clean_line)
            elif current_section == "recommendations":
                recommendations.append(clean_line)
        
        return {
            "summary": summary.strip() or f"Analysis of {data.get('call_count', 0)} calls completed.",
            "key_insights": insights[:5] if insights else ["Analysis complete - review individual calls for details"],
            "recommendations": recommendations[:3] if recommendations else ["Continue monitoring call quality"],
            "metrics": data,
            "processing_time": time.time() - start_time,
            "model": self.settings.chat_model,
        }
    
    def _mock_chat(self, message: str, context: Optional[str]) -> Dict[str, Any]:
        """Generate mock chat response for testing."""
        # Check if it's just a greeting
        greeting_words = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon"]
        if any(word in message.lower() for word in greeting_words) and len(message.split()) <= 3:
            return {
                "response": "Hello! I'm here to help you analyse call recordings and data. What would you like to know?",
                "tokens_used": 20,
                "processing_time": 0.1,
                "model": "mock",
            }
        
        return {
            "response": f"""Thank you for your question. Based on the call quality data:

**Performance Overview:**
- Average quality score indicates **good adherence** to scripts
- Positive supporter interactions detected
- Call handling time within expected ranges

**Key Observations:**
- Sentiment analysis shows consistently positive tone
- Gift Aid explanations meeting compliance standards

Is there a specific aspect of the calls you'd like me to analyse further?""",
            "tokens_used": 50,
            "processing_time": 0.1,
            "model": "mock",
        }
    
    def _mock_summary(
        self,
        data: Dict[str, Any],
        summary_type: str,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate mock summary for testing."""
        call_count = data.get('call_count', 0)
        avg_quality = data.get('avg_quality_score', 75)
        
        return {
            "summary": f"In the reporting period, **{call_count} calls** were analysed with an average quality score of **{avg_quality:.1f}%**. Overall performance remains strong with positive sentiment detected in the majority of interactions.",
            "key_insights": [
                "Call quality remained **consistent** throughout the period",
                "Positive sentiment detected in **78%** of calls",
                "Gift Aid explanation compliance improved by **12%**",
                "Average call duration optimised at **4.5 minutes**",
            ],
            "recommendations": [
                "Consider additional training on **objection handling** techniques",
                "Update scripts based on **high-performing call patterns**",
                "Implement **peer review sessions** for knowledge sharing",
            ],
            "metrics": data,
            "processing_time": 0.1,
            "model": "mock",
        }
    
    def is_available(self) -> bool:
        """Check if interactive service is available."""
        if self.use_mock:
            return True
        return QWEN_AVAILABLE
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status for health checks."""
        return {
            "available": self.is_available(),
            "model_loaded": self._model is not None,
            "device": self._device,
            "mock_mode": self.use_mock,
            "model": self.settings.chat_model if not self.use_mock else "mock",
        }


# Singleton instance
_interactive_service: Optional[InteractiveService] = None


def get_interactive_service() -> InteractiveService:
    """Get the interactive service singleton."""
    global _interactive_service
    if _interactive_service is None:
        _interactive_service = InteractiveService()
    return _interactive_service
