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
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
        
        # vLLM/external API configuration
        self.llm_api_url = self.settings.llm_api_url
        self.llm_api_key = self.settings.llm_api_key
        
        # Model state (lazy loaded - only used if not using API)
        self._model = None
        self._tokenizer = None
        self._device = None
        
        self._initialized = True
        
        mode_desc = "mock" if self.use_mock else ("API" if self.llm_api_url else "local")
        logger.info(f"InteractiveService initialised (mode={mode_desc})")
    
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
        
        # Apply quantization if configured (GPU only - no CPU offload for int4)
        if self.settings.chat_model_quantization == "int4" and self._device == "cuda":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif self.settings.chat_model_quantization == "int8" and self._device == "cuda":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        else:
            load_kwargs["torch_dtype"] = torch.float16 if self._device == "cuda" else torch.float32
        
        try:
            self._model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        except Exception as e:
            # Fallback to CPU with float32 if quantized loading fails
            logger.warning(f"Quantized model loading failed: {e}")
            logger.info("Falling back to CPU with float32 (slower but compatible)")
            load_kwargs = {
                "trust_remote_code": True,
                "device_map": None,
                "torch_dtype": torch.float32,
            }
            self._model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            self._model = self._model.to("cpu")
            self._device = "cpu"
        
        if self._device == "cpu" and not hasattr(self._model, '_hf_hook'):
            self._model = self._model.to(self._device)
        
        elapsed = time.time() - start_time
        logger.info(f"Chat model loaded in {elapsed:.1f}s on {self._device}")
    
    def _generate_via_api(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.6,
    ) -> str:
        """
        Generate text via external vLLM/OpenAI-compatible API.
        
        Args:
            messages: Chat messages in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        import httpx
        
        headers = {"Content-Type": "application/json"}
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"
        
        # Use the chat model name for the API request
        model_name = self.settings.chat_model
        
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.85,
        }
        
        # Only add repetition_penalty for vLLM (not supported by OpenAI/Groq)
        if "groq.com" not in self.llm_api_url and "openai.com" not in self.llm_api_url:
            payload["repetition_penalty"] = 1.1
        
        api_endpoint = f"{self.llm_api_url.rstrip('/')}/chat/completions"
        
        with httpx.Client(timeout=300.0) as client:
            response = client.post(api_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
        
        return result["choices"][0]["message"]["content"]
    
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
        
        # Generate response - via API or local model
        try:
            if self.llm_api_url:
                # Use external vLLM/OpenAI-compatible API
                response_text = self._generate_via_api(messages, max_tokens=max_tokens)
                tokens_used = len(response_text.split())  # Approximate
            else:
                # Use local model
                self._ensure_model_loaded()
                
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
                tokens_used = len(response_tokens)
            
            return {
                "response": response_text.strip(),
                "tokens_used": tokens_used,
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
            if filters.get("client_name"):
                filter_parts.append(f"client {filters['client_name']}")
            elif filters.get("client_ref"):
                filter_parts.append(f"client {filters['client_ref']}")
            if filters.get("campaign"):
                filter_parts.append(f"campaign '{filters['campaign']}'")
            if filters.get("start_date") and filters.get("end_date"):
                # Format dates as UK format
                try:
                    from datetime import datetime
                    start = datetime.strptime(filters['start_date'], '%Y-%m-%d').strftime('%d/%m/%Y')
                    end = datetime.strptime(filters['end_date'], '%Y-%m-%d').strftime('%d/%m/%Y')
                    filter_parts.append(f"period {start} to {end}")
                except:
                    filter_parts.append(f"period {filters['start_date']} to {filters['end_date']}")
            if filter_parts:
                filter_desc = f" for {', '.join(filter_parts)}"
        
        # Build rich data section
        data_sections = []
        
        # Core metrics
        data_sections.append(f"""## Core Metrics
- **Total Calls:** {data.get('call_count', 0):,}
- **Average Quality Score:** {data.get('avg_quality_score', 0):.1f}%
- **Average Sentiment Score:** {data.get('avg_sentiment_score', 0):.1f}/10
- **Total Call Duration:** {data.get('total_duration_seconds', 0) // 60:,} minutes""")
        
        # Quality distribution
        if data.get('quality_distribution'):
            qd = data['quality_distribution']
            data_sections.append(f"""## Quality Distribution
- Excellent (80%+): {qd.get('excellent_80_plus', 0)} calls
- Good (60-79%): {qd.get('good_60_79', 0)} calls
- Average (40-59%): {qd.get('average_40_59', 0)} calls
- Poor (<40%): {qd.get('poor_below_40', 0)} calls""")
        
        # Top agents
        if data.get('top_agents'):
            top_list = "\n".join([f"- {a['name']}: {a['quality']:.1f}% quality ({a['calls']} calls)" 
                                   for a in data['top_agents'][:5]])
            data_sections.append(f"""## Top Performing Agents
{top_list}""")
        
        # Agents needing coaching
        if data.get('agents_needing_coaching'):
            bottom_list = "\n".join([f"- {a['name']}: {a['quality']:.1f}% quality ({a['calls']} calls)" 
                                      for a in data['agents_needing_coaching'][:5]])
            data_sections.append(f"""## Agents Needing Coaching
{bottom_list}""")
        
        # Common improvement areas
        if data.get('common_improvement_areas'):
            areas_list = "\n".join([f"- {a['area']}: {a['count']} occurrences" 
                                     for a in data['common_improvement_areas'][:7]])
            data_sections.append(f"""## Common Improvement Areas
{areas_list}""")
        
        # Compliance issues
        if data.get('calls_with_compliance_issues', 0) > 0:
            data_sections.append(f"""## Compliance
- Calls with compliance issues: {data['calls_with_compliance_issues']}""")
        
        # Top topics
        if data.get('top_topics'):
            topics_list = "\n".join([f"- {t['topic']}: {t['count']} calls" 
                                      for t in data['top_topics'][:7]])
            data_sections.append(f"""## Top Call Topics
{topics_list}""")
        
        data_text = "\n\n".join(data_sections)
        
        prompt = f"""Write a SHORT {summary_type} call quality summary{filter_desc}.

{data_text}

FORMAT (use exactly these headers, keep each section brief):

## Summary
One paragraph, 2-3 sentences max. Bold the key numbers.

## Agents Needing Coaching
Bullet list: name, score, calls. Maximum 5 agents. No commentary.

## Actions
3-4 bullet points. One line each. Actionable and specific.

RULES:
- Maximum 150 words total
- No introductions or conclusions
- No filler phrases
- Do not repeat data already shown
- British English"""

        messages = [
            {"role": "system", "content": "You write extremely concise reports. No fluff. Bullet points only. Maximum 150 words."},
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
                    max_new_tokens=400,  # Strict limit for concise output
                    do_sample=True,
                    temperature=0.5,  # Lower temp for more focused output
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
        """
        Parse AI response into structured summary format.
        
        The AI generates full Markdown text. We extract sections for backward 
        compatibility but return the full markdown in 'summary' field.
        """
        import re
        
        # Clean up any markdown code fences if present
        response_text = response_text.strip()
        response_text = re.sub(r'^```markdown\s*', '', response_text, flags=re.IGNORECASE)
        response_text = re.sub(r'\s*```$', '', response_text)
        response_text = response_text.strip()
        
        insights = []
        recommendations = []
        
        # Try to extract key insights section (look for bullet points under relevant headers)
        insights_match = re.search(
            r'###?\s*(?:Key\s+)?(?:Insights?|Findings?|Highlights?)[:\s]*\n((?:[-*•]\s*.+\n?)+)',
            response_text,
            re.IGNORECASE | re.MULTILINE
        )
        if insights_match:
            insight_text = insights_match.group(1)
            insights = [
                line.strip().lstrip('•-*').strip()
                for line in insight_text.split('\n')
                if line.strip() and line.strip().startswith(('•', '-', '*'))
            ]
        
        # Try to extract recommendations section
        rec_match = re.search(
            r'###?\s*(?:Recommendations?|Action\s+Items?)[:\s]*\n((?:[-*•]\s*.+\n?)+)',
            response_text,
            re.IGNORECASE | re.MULTILINE
        )
        if rec_match:
            rec_text = rec_match.group(1)
            recommendations = [
                line.strip().lstrip('•-*').strip()
                for line in rec_text.split('\n')
                if line.strip() and line.strip().startswith(('•', '-', '*'))
            ]
        
        # Fallback: extract any bullet points from the whole text if sections not found
        if not insights:
            # Get first 5 bullet points
            all_bullets = re.findall(r'^[-*•]\s*(.+)$', response_text, re.MULTILINE)
            insights = all_bullets[:5] if all_bullets else ["Analysis complete - review individual calls for details"]
        
        if not recommendations:
            # Look for numbered lists which often contain recommendations
            numbered = re.findall(r'^\d+\.\s*(.+)$', response_text, re.MULTILINE)
            recommendations = numbered[-3:] if len(numbered) >= 3 else numbered
            if not recommendations:
                recommendations = ["Continue monitoring call quality"]
        
        return {
            "summary": response_text,  # Return full markdown text
            "key_insights": insights[:5],
            "recommendations": recommendations[:5],
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
        # Available if using API or if transformers is installed
        return bool(self.llm_api_url) or QWEN_AVAILABLE
    
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
        
        # Only load local model if not using API
        if not self.llm_api_url:
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
                # Generate response - via API or local model
                if self.llm_api_url:
                    response_text = self._generate_via_api(messages, max_tokens=max_tokens).strip()
                    response_tokens_count = len(response_text.split())  # Approximate
                else:
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
                    response_tokens_count = len(response_tokens)
                
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
                        error_msg = result.get('error', 'Unknown error')
                        result_text = f"FUNCTION_ERROR: {error_msg}\n\n"
                        
                        # Provide schema hints based on the error
                        if "Unknown column" in error_msg or "doesn't exist" in error_msg:
                            result_text += """REMINDER - Here are the ACTUAL tables and columns available:

**ai_call_recordings**: id, apex_id, orderref, enqref, obref, client_ref, client_name, campaign, halo_id, agent_name, creative, invoicing, call_date, direction, duration_seconds, processing_status, created_at

**ai_call_transcriptions**: id, ai_call_recording_id, full_transcript, redacted_transcript, segments (JSON), pii_detected (JSON), pii_count, language_detected, confidence, model_used, processing_time_seconds, created_at

**ai_call_analysis**: id, ai_call_recording_id, summary, sentiment_score, sentiment_label, quality_score, key_topics (JSON), agent_actions_performed (JSON), performance_scores (JSON), action_items (JSON), compliance_flags (JSON), improvement_areas (JSON), speaker_metrics (JSON), audio_analysis (JSON), model_used, processing_time_seconds, created_at

**ai_monthly_summaries**: id, feature, summary_month, client_ref, campaign, agent_id, summary_data (JSON), call_count, avg_quality_score, avg_sentiment_score, created_at

Please try again with a corrected query using ONLY these tables and columns."""
                        else:
                            result_text += "Please try a different query to answer the user's question."
                    
                    messages.append({
                        "role": "user",
                        "content": result_text
                    })
                    
                    # If we've had multiple failures, give up and return an error message
                    failed_calls = sum(1 for fc in function_calls_made if not fc.get("success", False))
                    if failed_calls >= 3:
                        logger.warning(f"Too many failed function calls ({failed_calls}), giving up")
                        return {
                            "response": "I'm having trouble querying the database. The queries I tried didn't work - I may have used incorrect table or column names. Could you try rephrasing your question, or ask for something more specific like 'how many calls were completed this month?'",
                            "tokens_used": response_tokens_count,
                            "processing_time": time.time() - start_time,
                            "model": self.settings.chat_model,
                            "function_calls": function_calls_made,
                            # Don't set "error" - this is a graceful failure with a message
                        }
                    
                    # Continue the loop to let the model process the result
                    continue
                else:
                    # No function call or max iterations reached - return the response
                    # Clean up any function call syntax that might have leaked through
                    clean_response = self._clean_function_syntax(response_text)
                    
                    # If the response is empty but we had function calls, provide a fallback
                    if not clean_response.strip() and function_calls_made:
                        # Check if any succeeded
                        successful = [fc for fc in function_calls_made if fc.get("success", False)]
                        if successful:
                            clean_response = "I found some data but had trouble formatting a response. Please try asking your question again."
                        else:
                            clean_response = "I tried to query the database but encountered errors. Please try rephrasing your question."
                    
                    return {
                        "response": clean_response,
                        "tokens_used": response_tokens_count,
                        "processing_time": time.time() - start_time,
                        "model": self.settings.chat_model,
                        "function_calls": function_calls_made,
                    }
                    
            except Exception as e:
                logger.error(f"Chat with functions failed: {e}")
                raise
        
        # Shouldn't reach here, but just in case
        return {
            "response": "I made several attempts to query the data but couldn't complete the request. Please try a simpler question.",
            "tokens_used": 0,
            "processing_time": time.time() - start_time,
            "model": self.settings.chat_model,
            "function_calls": function_calls_made,
            # Don't set "error" - this is a graceful failure with a message
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
        
        logger.debug(f"Extracting function call from: {response_text[:500]}...")
        
        # Helper to fix common JSON issues from LLM output
        def fix_json_string(json_str: str) -> str:
            # Replace unescaped newlines inside strings with spaces
            # This handles multi-line SQL that the model generates
            fixed = json_str.replace('\r\n', ' ').replace('\n', ' ')
            # Collapse multiple spaces
            fixed = re.sub(r'\s+', ' ', fixed)
            return fixed
        
        # Helper to extract balanced JSON object
        def extract_json_object(text: str, start_pos: int = 0) -> Optional[str]:
            """Extract a balanced JSON object starting from start_pos."""
            idx = text.find('{', start_pos)
            if idx == -1:
                return None
            
            depth = 0
            in_string = False
            escape_next = False
            
            for i, char in enumerate(text[idx:], start=idx):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        return text[idx:i+1]
            return None
        
        # Pattern 1: FUNCTION_CALL: name {json}
        match = re.search(r'FUNCTION_CALL:\s*(\w+)\s*', response_text)
        if match:
            func_name = match.group(1)
            json_str = extract_json_object(response_text, match.end())
            if json_str:
                try:
                    json_str = fix_json_string(json_str)
                    args = json.loads(json_str)
                    logger.info(f"Extracted function call (pattern 1): {func_name}")
                    return {"name": func_name, "arguments": args}
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse function args: {e}")
        
        # Pattern 2: Look for execute_sql_query or search_calls with JSON
        for func_name in ["execute_sql_query", "search_calls"]:
            match = re.search(rf'{func_name}\s*[:\(]\s*', response_text)
            if match:
                json_str = extract_json_object(response_text, match.end())
                if json_str:
                    try:
                        json_str = fix_json_string(json_str)
                        args = json.loads(json_str)
                        logger.info(f"Extracted function call (pattern 2): {func_name}")
                        return {"name": func_name, "arguments": args}
                    except json.JSONDecodeError:
                        pass
        
        # Pattern 3: Look for SQL query patterns in code blocks
        sql_match = re.search(r'```sql\s*(SELECT.*?)```', response_text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            query = sql_match.group(1).strip()
            logger.info(f"Extracted SQL from code block")
            return {
                "name": "execute_sql_query",
                "arguments": {"query": query, "purpose": "User requested query"}
            }
        
        # Pattern 4: Look for inline SELECT query (not in code block)
        inline_sql = re.search(r'(SELECT\s+.+?(?:FROM|LIMIT)\s+.+?)(?:\n\n|$)', response_text, re.DOTALL | re.IGNORECASE)
        if inline_sql:
            query = inline_sql.group(1).strip()
            if len(query) > 20:  # Avoid false positives
                logger.info(f"Extracted inline SQL query")
                return {
                    "name": "execute_sql_query",
                    "arguments": {"query": query, "purpose": "User requested query"}
                }
        
        # Pattern 5: Look for "I'll search for call X" patterns
        search_match = re.search(r"(?:search|find|look up).*?(?:call|recording).*?['\"]?(\d+)['\"]?", response_text, re.IGNORECASE)
        if search_match:
            logger.info(f"Extracted search call pattern")
            return {
                "name": "search_calls",
                "arguments": {"reference": search_match.group(1), "reference_type": "any"}
            }
        
        logger.debug("No function call found in response")
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
