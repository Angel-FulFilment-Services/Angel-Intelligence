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
        
        .. deprecated::
            Use chat_with_functions() instead. This method does not have
            SQL Agent capabilities and uses a legacy system prompt.
            Kept for backwards compatibility with /internal/chat endpoint.
        
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
- **IMPORTANT**: Always use the full client name (client_name) when referring to clients, never show the shorthand code (client_ref) to users
- Format responses using Markdown for better readability when providing detailed information
- **IMPORTANT**: Match your response length and detail to the user's question:
  * Simple greetings ("hello", "hi", "thanks") → Brief, friendly response (e.g., "Hi! How can I help you today?")
  * Follow-up questions about previous responses → Answer based on conversation context
  * General questions → Concise overview with key points
  * Specific questions → Detailed analysis with relevant data
- You have access to REAL DATA in the context - only reference it when the user asks for analysis or insights
- If the user is just greeting you or chatting casually, don't overwhelm them with metrics
- Keep responses concise unless specifically asked for detailed analysis
- Be data-driven when appropriate, conversational when appropriate
- Professional and supportive tone

**Conversation Behaviour:**
- **ALWAYS respond to greetings and casual messages** - never ignore the user
- **Remember previous messages** - if the user asks a follow-up question, use conversation context
- **Don't repeat yourself** - if you already provided data, refer to it rather than querying again
- **Ask clarifying questions** when the user's intent is unclear
- **Never auto-select or filter data** unless the user specifically asks you to

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
    
    def chat_with_functions(
        self,
        message: str,
        user_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 512,
        max_function_calls: int = 5,
    ) -> Dict[str, Any]:
        """
        Chat with SQL Agent function calling support.
        
        This method enables the AI to query the database to answer questions.
        It handles the function calling loop, executing queries and feeding
        results back to the AI until it has enough information to respond.
        
        Args:
            message: User's message
            user_name: User's name for personalization
            filters: Active filters from UI (date range, client, etc.)
            conversation_history: Previous messages
            max_tokens: Maximum response tokens
            max_function_calls: Maximum function calls allowed per request
            
        Returns:
            Dict with response text and metadata
        """
        from src.services.sql_agent import (
            get_sql_agent_system_prompt,
            SQL_AGENT_FUNCTIONS,
            handle_function_call,
        )
        import json
        
        start_time = time.time()
        function_calls_made = []
        
        if self.use_mock:
            return self._mock_chat_with_functions(message, user_name, filters)
        
        self._ensure_model_loaded()
        
        # Build messages with SQL Agent system prompt
        messages = []
        system_prompt = get_sql_agent_system_prompt(user_name, filters)
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Function calling loop
        for iteration in range(max_function_calls + 1):
            try:
                # Generate response
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
                        temperature=0.6,
                        top_p=0.85,
                        pad_token_id=self._tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                    )
                
                response_tokens = outputs[0][inputs.shape[1]:]
                response_text = self._tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
                
                # Check if the response contains a function call
                # Look for patterns like: FUNCTION_CALL: execute_sql_query(...)
                # or JSON-like function calls
                function_call = self._extract_function_call(response_text)
                
                if function_call and iteration < max_function_calls:
                    # Execute the function
                    func_name = function_call.get("name")
                    func_args = function_call.get("arguments", {})
                    
                    logger.info(f"SQL Agent calling function: {func_name}")
                    result = handle_function_call(func_name, func_args)
                    
                    function_calls_made.append({
                        "function": func_name,
                        "arguments": func_args,
                        "result_rows": result.get("row_count", 0),
                        "success": result.get("success", False),
                    })
                    
                    # Add the function call and result to messages
                    messages.append({
                        "role": "assistant",
                        "content": f"FUNCTION_CALL: {func_name}\nARGUMENTS: {json.dumps(func_args)}"
                    })
                    
                    # Format result for the model
                    if result.get("success"):
                        result_text = f"FUNCTION_RESULT:\n{json.dumps(result.get('data', []), indent=2, default=str)[:4000]}"
                        if result.get("row_count", 0) > 10:
                            result_text += f"\n... ({result['row_count']} total rows)"
                    else:
                        result_text = f"FUNCTION_ERROR: {result.get('error', 'Unknown error')}"
                    
                    messages.append({
                        "role": "user",
                        "content": result_text
                    })
                    
                    # Continue the loop to let the model process the result
                    continue
                else:
                    # No function call or max iterations reached - return the response
                    # Clean up any function call syntax that might have leaked through
                    clean_response = self._clean_function_syntax(response_text)
                    
                    return {
                        "response": clean_response,
                        "tokens_used": len(response_tokens),
                        "processing_time": time.time() - start_time,
                        "model": self.settings.chat_model,
                        "function_calls": function_calls_made,
                    }
                    
            except Exception as e:
                logger.error(f"Chat with functions failed: {e}")
                raise
        
        # Shouldn't reach here, but just in case
        return {
            "response": "I apologize, but I had trouble processing your request. Please try again.",
            "tokens_used": 0,
            "processing_time": time.time() - start_time,
            "model": self.settings.chat_model,
            "function_calls": function_calls_made,
            "error": "Max function calls exceeded",
        }
    
    def _extract_function_call(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Extract function call from response text.
        
        Looks for patterns like:
        - FUNCTION_CALL: execute_sql_query {"query": "...", "purpose": "..."}
        - {"function": "execute_sql_query", "arguments": {...}}
        - execute_sql_query(query="...", purpose="...")
        """
        import json
        import re
        
        # Pattern 1: FUNCTION_CALL: name {json}
        match = re.search(r'FUNCTION_CALL:\s*(\w+)\s*(\{.*\})', response_text, re.DOTALL)
        if match:
            try:
                func_name = match.group(1)
                args = json.loads(match.group(2))
                return {"name": func_name, "arguments": args}
            except json.JSONDecodeError:
                pass
        
        # Pattern 2: Look for execute_sql_query or search_calls with JSON
        for func_name in ["execute_sql_query", "search_calls"]:
            pattern = rf'{func_name}\s*[:\(]\s*(\{{.*?\}})'
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                try:
                    args = json.loads(match.group(1))
                    return {"name": func_name, "arguments": args}
                except json.JSONDecodeError:
                    pass
        
        # Pattern 3: Look for SQL query patterns and convert to function call
        sql_match = re.search(r'```sql\s*(SELECT.*?)```', response_text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            query = sql_match.group(1).strip()
            return {
                "name": "execute_sql_query",
                "arguments": {"query": query, "purpose": "User requested query"}
            }
        
        # Pattern 4: Look for "I'll search for call X" patterns
        search_match = re.search(r"(?:search|find|look up).*?(?:call|recording).*?['\"]?(\d+)['\"]?", response_text, re.IGNORECASE)
        if search_match:
            return {
                "name": "search_calls",
                "arguments": {"reference": search_match.group(1), "reference_type": "any"}
            }
        
        return None
    
    def _clean_function_syntax(self, text: str) -> str:
        """Remove any function call syntax from the final response."""
        import re
        
        # Remove FUNCTION_CALL blocks
        text = re.sub(r'FUNCTION_CALL:.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)
        
        # Remove JSON blocks that look like function calls
        text = re.sub(r'\{"function":\s*".*?\}', '', text, flags=re.DOTALL)
        
        # Remove SQL code blocks if they're the only content
        if text.strip().startswith('```sql') and text.strip().endswith('```'):
            text = re.sub(r'```sql.*?```', '', text, flags=re.DOTALL)
        
        return text.strip()
    
    def _mock_chat_with_functions(
        self,
        message: str,
        user_name: Optional[str],
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Mock chat with functions for testing."""
        from src.services.sql_agent import handle_function_call
        import json
        
        greeting = f"Hi {user_name}! " if user_name else "Hello! "
        function_calls_made = []
        
        # Build filter context for mock responses
        filter_context = ""
        if filters:
            filter_parts = []
            if filters.get("client_ref"):
                filter_parts.append(f"client {filters['client_ref']}")
            if filters.get("start_date") and filters.get("end_date"):
                filter_parts.append(f"{filters['start_date']} to {filters['end_date']}")
            elif filters.get("start_date"):
                filter_parts.append(f"from {filters['start_date']}")
            if filter_parts:
                filter_context = f" (filtered by {', '.join(filter_parts)})"
        
        # Check for call search patterns
        import re
        call_search = re.search(r'(?:call|recording).*?(\d{5,})', message, re.IGNORECASE)
        if call_search:
            ref = call_search.group(1)
            result = handle_function_call("search_calls", {"reference": ref, "reference_type": "any"})
            function_calls_made.append({
                "function": "search_calls",
                "arguments": {"reference": ref},
                "result_rows": result.get("row_count", 0),
                "success": result.get("success", False),
            })
            
            if result.get("success") and result.get("data"):
                call_data = result["data"][0]
                return {
                    "response": f"""{greeting}I found that call! Here are the details:

**Call ID:** {call_data.get('apex_id', 'N/A')}
**Date:** {call_data.get('call_date', 'N/A')}
**Agent:** {call_data.get('agent_name', 'Unknown')}
**Duration:** {call_data.get('duration_seconds', 0) // 60} minutes
**Quality Score:** {call_data.get('quality_score', 'N/A')}%
**Sentiment:** {call_data.get('sentiment_label', 'N/A')}

**Summary:** {call_data.get('summary', 'No summary available.')}

Would you like me to show you the full transcript or any other details?""",
                    "tokens_used": 100,
                    "processing_time": 0.5,
                    "model": "mock",
                    "function_calls": function_calls_made,
                }
            else:
                return {
                    "response": f"{greeting}I couldn't find a call with reference '{ref}'. Please check the reference number and try again.",
                    "tokens_used": 20,
                    "processing_time": 0.2,
                    "model": "mock",
                    "function_calls": function_calls_made,
                }
        
        # Check for agent performance queries
        if "worst" in message.lower() and "agent" in message.lower():
            result = handle_function_call("execute_sql_query", {
                "query": """SELECT agent_name, AVG(quality_score) as avg_quality, COUNT(*) as call_count
                           FROM ai_call_recordings r
                           JOIN ai_call_analysis a ON r.id = a.ai_call_recording_id
                           WHERE call_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                           GROUP BY agent_name
                           ORDER BY avg_quality ASC
                           LIMIT 5""",
                "purpose": "Find worst performing agents"
            })
            function_calls_made.append({
                "function": "execute_sql_query",
                "arguments": {"query": "...", "purpose": "Find worst performing agents"},
                "result_rows": result.get("row_count", 0),
                "success": result.get("success", False),
            })
            
            if result.get("success") and result.get("data"):
                agents = result["data"]
                agent_list = "\n".join([
                    f"{i+1}. **{a.get('agent_name', 'Unknown')}** - {a.get('avg_quality', 0):.1f}% quality ({a.get('call_count', 0)} calls)"
                    for i, a in enumerate(agents)
                ])
                return {
                    "response": f"""{greeting}Here are the 5 lowest performing agents in the last 30 days:

{agent_list}

Would you like me to analyse specific calls from any of these agents?""",
                    "tokens_used": 100,
                    "processing_time": 0.5,
                    "model": "mock",
                    "function_calls": function_calls_made,
                }
        
        # Default response
        return {
            "response": f"""{greeting}I can help you analyse call data. Here are some things I can do:

- **Find a specific call:** "Find call 12345" or "Show me call with order reference 67890"
- **Agent performance:** "Who are the worst performing agents?" or "Show me John's calls"
- **Call statistics:** "How many calls were processed last week?"
- **Transcript search:** "What did the customer say about billing in call 12345?"

What would you like to know?""",
            "tokens_used": 50,
            "processing_time": 0.1,
            "model": "mock",
            "function_calls": [],
        }
    
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
