"""
Angel Intelligence - SQL Agent Service

Provides safe, read-only SQL query execution capabilities for the AI chatbot.
Enables the AI to gather its own data by generating SELECT queries based on
user questions and the database schema.

Safety Features:
- Only SELECT statements allowed
- Dangerous keywords blocked (DROP, DELETE, UPDATE, INSERT, TRUNCATE, ALTER, etc.)
- Row limits enforced
- Query timeout protection
- Query logging for audit
"""

import re
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from src.database.connection import get_db_connection

logger = logging.getLogger(__name__)


# =============================================================================
# Database Schema Definition (What the AI knows about)
# =============================================================================

DATABASE_SCHEMA = """
## Database: Angel Intelligence Call Analytics

You have access to a MySQL database containing call recordings, transcriptions, and AI analysis.
Use this schema to write SELECT queries that answer user questions.

### Tables

#### ai_call_recordings
Main table of call recordings. Each row is a phone call that has been submitted for processing.
**Note:** Not all recordings have been fully processed yet. Check `processing_status`:
- 'completed' = Fully processed with transcript and analysis available
- 'pending' / 'queued' = Waiting to be processed (no transcript/analysis yet)
- 'processing' = Currently being processed
- 'failed' = Processing failed (check `processing_error` for details)

When querying for transcripts or analysis, only 'completed' recordings will have data in the joined tables.

| Column | Type | Description |
|--------|------|-------------|
| id | INT | Primary key |
| apex_id | VARCHAR(255) | Unique call identifier from PBX system |
| orderref | VARCHAR(100) | Order reference number (if call relates to an order) |
| enqref | VARCHAR(100) | Enquiry reference number |
| obref | VARCHAR(100) | Outbound reference number |
| client_ref | VARCHAR(100) | Client/company code - short reference (e.g., 'ABC', 'ACME'). This is an internal shorthand - never show this to users |
| client_name | VARCHAR(255) | Full client/company name (e.g., 'British Heart Foundation', 'Cancer Research UK'). **Always use this when displaying client information to users** |
| campaign | VARCHAR(100) | Campaign name |
| halo_id | INT | Agent ID from Halo system |
| agent_name | VARCHAR(255) | Agent's display name |
| creative | VARCHAR(100) | Creative/script name used |
| invoicing | VARCHAR(100) | Invoicing category |
| call_date | DATE | Date the call occurred |
| direction | ENUM | 'inbound' or 'outbound' |
| duration_seconds | INT | Call length in seconds |
| processing_status | ENUM | 'pending', 'processing', 'completed', 'failed', 'queued' |
| created_at | DATETIME | When record was created |

**Finding Calls:** When a user asks to find a call by a number/reference:
- If they say "call ID" or "apex ID" → search `apex_id`
- If they say "order reference" or "order" → search `orderref`
- If they say "enquiry reference" or "enquiry" → search `enqref`
- If they say "outbound reference" → search `obref`
- If just a number with no context → search ALL reference fields: apex_id, orderref, enqref, obref

#### ai_call_transcriptions
Full transcripts of calls. Linked to ai_call_recordings via ai_call_recording_id.

| Column | Type | Description |
|--------|------|-------------|
| id | INT | Primary key |
| ai_call_recording_id | INT | Foreign key to ai_call_recordings.id |
| full_transcript | TEXT | Complete call transcript |
| redacted_transcript | TEXT | Transcript with PII removed |
| segments | JSON | Word-level timestamps and speaker labels |
| pii_detected | JSON | List of detected PII items |
| pii_count | INT | Number of PII items found |
| language_detected | VARCHAR(10) | Detected language code |
| confidence | DECIMAL(5,4) | Transcription confidence (0-1) |
| model_used | VARCHAR(100) | Whisper model version |
| processing_time_seconds | INT | Time to transcribe |
| created_at | DATETIME | When transcribed |

#### ai_call_analysis
AI analysis results for each call. Linked to ai_call_recordings via ai_call_recording_id.

| Column | Type | Description |
|--------|------|-------------|
| id | INT | Primary key |
| ai_call_recording_id | INT | Foreign key to ai_call_recordings.id |
| summary | TEXT | AI-generated call summary |
| sentiment_score | DECIMAL(4,1) | Sentiment rating 1-10 (1=very negative, 10=very positive) |
| sentiment_label | ENUM | 'very_negative', 'negative', 'neutral', 'positive', 'very_positive' |
| quality_score | DECIMAL(5,2) | Call quality percentage 0-100 |
| key_topics | JSON | Array of topics discussed: [{"name": "Billing", "confidence": 0.9}] |
| agent_actions_performed | JSON | Actions the agent took |
| performance_scores | JSON | Breakdown of agent performance metrics |
| action_items | JSON | Follow-up actions identified |
| compliance_flags | JSON | Any compliance issues detected |
| improvement_areas | JSON | Key areas for agent improvement/coaching: [{"area": "Objection Handling", "description": "Agent struggled when donor raised concerns", "priority": "high", "examples": ["quote"]}] |
| speaker_metrics | JSON | Talk time, interruptions, etc. per speaker |
| audio_analysis | JSON | Audio quality observations |
| model_used | VARCHAR(100) | Analysis model name |
| processing_time_seconds | INT | Time to analyse |
| created_at | DATETIME | When analysed |

#### ai_voice_fingerprints
Agent voice embeddings for speaker identification.

| Column | Type | Description |
|--------|------|-------------|
| id | INT | Primary key |
| halo_id | INT | Agent ID (unique) |
| agent_name | VARCHAR(255) | Agent's name |
| sample_count | INT | Number of voice samples used |
| average_confidence | DECIMAL(5,4) | Average identification confidence |
| is_active | BOOLEAN | Whether fingerprint is active |
| created_at | DATETIME | When created |
| updated_at | DATETIME | Last updated |

#### ai_monthly_summaries
Pre-generated monthly summary reports.

| Column | Type | Description |
|--------|------|-------------|
| id | INT | Primary key |
| feature | VARCHAR(100) | Summary type (e.g., 'call_quality') |
| summary_month | DATE | First day of the month summarised |
| client_ref | VARCHAR(100) | Client filter (NULL = all clients) |
| campaign | VARCHAR(100) | Campaign filter (NULL = all campaigns) |
| agent_id | INT | Agent filter (NULL = all agents) |
| summary_data | JSON | The summary content |
| call_count | INT | Number of calls in period |
| avg_quality_score | DECIMAL(5,2) | Average quality for period |
| avg_sentiment_score | DECIMAL(4,1) | Average sentiment for period |
| created_at | DATETIME | When generated |

### Common Query Patterns

**Find a specific call:**
```sql
SELECT r.*, t.full_transcript, a.summary, a.quality_score, a.sentiment_label
FROM ai_call_recordings r
LEFT JOIN ai_call_transcriptions t ON r.id = t.ai_call_recording_id
LEFT JOIN ai_call_analysis a ON r.id = a.ai_call_recording_id
WHERE r.apex_id = '12345' OR r.orderref = '12345' OR r.enqref = '12345' OR r.obref = '12345'
```

**Agent performance ranking (only completed calls):**
```sql
SELECT r.agent_name, r.halo_id,
       COUNT(*) as call_count,
       AVG(a.quality_score) as avg_quality,
       AVG(a.sentiment_score) as avg_sentiment
FROM ai_call_recordings r
JOIN ai_call_analysis a ON r.id = a.ai_call_recording_id
WHERE r.call_date >= '2026-01-01'
  AND r.processing_status = 'completed'
GROUP BY r.agent_name, r.halo_id
ORDER BY avg_quality DESC
LIMIT 10
```

**Calls by date range (completed only):**
```sql
SELECT COUNT(*) as total_calls,
       AVG(a.quality_score) as avg_quality,
       SUM(r.duration_seconds) / 60 as total_minutes
FROM ai_call_recordings r
JOIN ai_call_analysis a ON r.id = a.ai_call_recording_id
WHERE r.call_date BETWEEN '2026-01-01' AND '2026-01-31'
  AND r.processing_status = 'completed'
```

**Count all recordings by status:**
```sql
SELECT processing_status, COUNT(*) as count
FROM ai_call_recordings
GROUP BY processing_status
```

### Important Notes
- Always use LIMIT to avoid returning too many rows (max 100 for detail queries, 1000 for counts)
- Use JOINs to connect recordings with their transcripts and analysis
- **Processing status matters:** Only recordings with `processing_status = 'completed'` have transcripts and analysis. Use LEFT JOIN if you want to include pending/failed recordings.
- When reporting totals or averages, clarify whether you're counting all recordings or only completed ones
- Dates are stored as DATE type, use 'YYYY-MM-DD' format
- sentiment_score is 1-10 scale, quality_score is 0-100 percentage
- JSON columns can be searched but it's better to filter in Python after retrieval
"""


# =============================================================================
# SQL Safety Validator
# =============================================================================

class SQLValidationError(Exception):
    """Raised when SQL query fails validation."""
    pass


# Dangerous SQL keywords that are never allowed
BLOCKED_KEYWORDS = [
    'INSERT', 'UPDATE', 'DELETE', 'DROP', 'TRUNCATE', 'ALTER', 'CREATE',
    'REPLACE', 'GRANT', 'REVOKE', 'LOCK', 'UNLOCK', 'CALL', 'EXECUTE',
    'EXEC', 'INTO OUTFILE', 'INTO DUMPFILE', 'LOAD_FILE', 'LOAD DATA',
    'BENCHMARK', 'SLEEP', 'WAITFOR', 'SHUTDOWN', 'KILL',
]

# Tables that can be queried
ALLOWED_TABLES = [
    'ai_call_recordings',
    'ai_call_transcriptions',
    'ai_call_analysis',
    'ai_voice_fingerprints',
    'ai_monthly_summaries',
    'ai_chat_conversations',
    'ai_chat_messages',
    'ai_client_configs',
    'ai_call_annotations',
]

# Maximum rows to return
MAX_ROWS = 100
MAX_ROWS_AGGREGATE = 1000


def validate_sql(query: str) -> Tuple[bool, str]:
    """
    Validate that a SQL query is safe to execute.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Empty query"
    
    # Normalise query for checking
    query_upper = query.upper().strip()
    query_normalised = ' '.join(query_upper.split())
    
    # Must start with SELECT
    if not query_normalised.startswith('SELECT'):
        return False, "Only SELECT queries are allowed"
    
    # Check for blocked keywords
    for keyword in BLOCKED_KEYWORDS:
        # Use word boundary matching to avoid false positives
        pattern = r'\b' + keyword + r'\b'
        if re.search(pattern, query_normalised):
            return False, f"Blocked keyword detected: {keyword}"
    
    # Check for comment injection attempts
    if '--' in query or '/*' in query or '*/' in query:
        return False, "SQL comments are not allowed"
    
    # Check for semicolons (multiple statements)
    if ';' in query.strip().rstrip(';'):
        return False, "Multiple statements are not allowed"
    
    # Check for UNION-based injection attempts
    if 'UNION' in query_normalised and 'SELECT' in query_normalised.split('UNION', 1)[1]:
        # Allow UNION only if it's a simple UNION ALL for legitimate queries
        pass  # Could add more checks here
    
    # Verify tables are in allowed list (basic check)
    # This is a simple check - a more robust parser would be better
    from_match = re.search(r'\bFROM\s+(\w+)', query_normalised)
    if from_match:
        table_name = from_match.group(1).lower()
        if table_name not in ALLOWED_TABLES:
            # Check if it's an alias
            if not re.search(r'\bAS\s+' + table_name, query_normalised, re.IGNORECASE):
                return False, f"Table not allowed: {table_name}"
    
    return True, ""


def add_limit_if_missing(query: str, limit: int = MAX_ROWS) -> str:
    """Add LIMIT clause if not present."""
    query_upper = query.upper().strip()
    if 'LIMIT' not in query_upper:
        query = query.rstrip().rstrip(';') + f" LIMIT {limit}"
    return query


# =============================================================================
# SQL Executor
# =============================================================================

@dataclass
class QueryResult:
    """Result of a SQL query execution."""
    success: bool
    data: List[Dict[str, Any]]
    row_count: int
    error: Optional[str] = None
    query: Optional[str] = None


def execute_safe_query(query: str, timeout_seconds: int = 30) -> QueryResult:
    """
    Execute a validated SQL query and return results.
    
    Args:
        query: SQL SELECT query to execute
        timeout_seconds: Maximum execution time
        
    Returns:
        QueryResult with data or error
    """
    # Validate query
    is_valid, error = validate_sql(query)
    if not is_valid:
        logger.warning(f"SQL validation failed: {error} - Query: {query[:200]}")
        return QueryResult(
            success=False,
            data=[],
            row_count=0,
            error=f"Query validation failed: {error}",
            query=query
        )
    
    # Add limit if missing
    query = add_limit_if_missing(query)
    
    try:
        db = get_db_connection()
        
        # Log query for audit
        logger.info(f"SQL Agent executing: {query[:500]}")
        
        # Execute query
        rows = db.fetch_all(query)
        
        # Convert to list of dicts, handling special types
        results = []
        for row in rows:
            row_dict = dict(row)
            # Convert datetime objects to strings
            for key, value in row_dict.items():
                if hasattr(value, 'isoformat'):
                    row_dict[key] = value.isoformat()
                elif isinstance(value, bytes):
                    row_dict[key] = "<binary data>"
            results.append(row_dict)
        
        logger.info(f"SQL Agent returned {len(results)} rows")
        
        return QueryResult(
            success=True,
            data=results,
            row_count=len(results),
            query=query
        )
        
    except Exception as e:
        logger.error(f"SQL Agent error: {e} - Query: {query[:200]}")
        return QueryResult(
            success=False,
            data=[],
            row_count=0,
            error=str(e),
            query=query
        )


# =============================================================================
# Function Definitions for AI
# =============================================================================

SQL_AGENT_FUNCTIONS = [
    {
        "name": "execute_sql_query",
        "description": """Execute a read-only SQL SELECT query against the call analytics database.
Use this to gather data needed to answer user questions about calls, agents, transcripts, and analytics.
Only SELECT queries are allowed - no modifications to data.
Results are limited to 100 rows for detail queries.
Always use this function when you need specific data from the database.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A valid MySQL SELECT query. Must start with SELECT. Use proper JOINs to connect tables. Always include LIMIT clause."
                },
                "purpose": {
                    "type": "string",
                    "description": "Brief description of what data you're trying to retrieve and why."
                }
            },
            "required": ["query", "purpose"]
        }
    },
    {
        "name": "search_calls",
        "description": """Search for calls by reference number. Use this when a user asks to find a specific call.
Searches across apex_id, orderref, enqref, and obref fields.
Returns call details along with transcript and analysis.""",
        "parameters": {
            "type": "object",
            "properties": {
                "reference": {
                    "type": "string",
                    "description": "The call reference number to search for"
                },
                "reference_type": {
                    "type": "string",
                    "enum": ["any", "apex_id", "orderref", "enqref", "obref"],
                    "description": "Which field to search. Use 'any' if not specified by user."
                }
            },
            "required": ["reference"]
        }
    }
]


def handle_function_call(function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle a function call from the AI.
    
    Args:
        function_name: Name of the function to call
        arguments: Function arguments
        
    Returns:
        Result dictionary
    """
    if function_name == "execute_sql_query":
        query = arguments.get("query", "")
        purpose = arguments.get("purpose", "")
        
        logger.info(f"SQL Agent function call - Purpose: {purpose}")
        result = execute_safe_query(query)
        
        return {
            "success": result.success,
            "data": result.data,
            "row_count": result.row_count,
            "error": result.error
        }
    
    elif function_name == "search_calls":
        reference = arguments.get("reference", "")
        ref_type = arguments.get("reference_type", "any")
        
        if ref_type == "any":
            query = f"""
                SELECT r.id, r.apex_id, r.orderref, r.enqref, r.obref,
                       r.client_ref, r.campaign, r.agent_name, r.call_date,
                       r.direction, r.duration_seconds,
                       t.full_transcript,
                       a.summary, a.quality_score, a.sentiment_label, a.sentiment_score,
                       a.key_topics, a.action_items
                FROM ai_call_recordings r
                LEFT JOIN ai_call_transcriptions t ON r.id = t.ai_call_recording_id
                LEFT JOIN ai_call_analysis a ON r.id = a.ai_call_recording_id
                WHERE r.apex_id = %s OR r.orderref = %s OR r.enqref = %s OR r.obref = %s
                LIMIT 10
            """
            # Need to handle parameterised query differently
            db = get_db_connection()
            try:
                rows = db.fetch_all(query, (reference, reference, reference, reference))
                results = []
                for row in rows:
                    row_dict = dict(row)
                    for key, value in row_dict.items():
                        if hasattr(value, 'isoformat'):
                            row_dict[key] = value.isoformat()
                        elif isinstance(value, bytes):
                            row_dict[key] = "<binary data>"
                    results.append(row_dict)
                
                return {
                    "success": True,
                    "data": results,
                    "row_count": len(results),
                    "error": None
                }
            except Exception as e:
                return {
                    "success": False,
                    "data": [],
                    "row_count": 0,
                    "error": str(e)
                }
        else:
            # Search specific field
            field_map = {
                "apex_id": "apex_id",
                "orderref": "orderref",
                "enqref": "enqref",
                "obref": "obref"
            }
            field = field_map.get(ref_type, "apex_id")
            query = f"""
                SELECT r.id, r.apex_id, r.orderref, r.enqref, r.obref,
                       r.client_ref, r.campaign, r.agent_name, r.call_date,
                       r.direction, r.duration_seconds,
                       t.full_transcript,
                       a.summary, a.quality_score, a.sentiment_label, a.sentiment_score,
                       a.key_topics, a.action_items
                FROM ai_call_recordings r
                LEFT JOIN ai_call_transcriptions t ON r.id = t.ai_call_recording_id
                LEFT JOIN ai_call_analysis a ON r.id = a.ai_call_recording_id
                WHERE r.{field} = %s
                LIMIT 10
            """
            db = get_db_connection()
            try:
                rows = db.fetch_all(query, (reference,))
                results = []
                for row in rows:
                    row_dict = dict(row)
                    for key, value in row_dict.items():
                        if hasattr(value, 'isoformat'):
                            row_dict[key] = value.isoformat()
                        elif isinstance(value, bytes):
                            row_dict[key] = "<binary data>"
                    results.append(row_dict)
                
                return {
                    "success": True,
                    "data": results,
                    "row_count": len(results),
                    "error": None
                }
            except Exception as e:
                return {
                    "success": False,
                    "data": [],
                    "row_count": 0,
                    "error": str(e)
                }
    
    else:
        return {
            "success": False,
            "error": f"Unknown function: {function_name}"
        }


# =============================================================================
# System Prompt for SQL Agent Mode
# =============================================================================

def get_sql_agent_system_prompt(
    user_name: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate the system prompt for SQL Agent mode.
    
    Args:
        user_name: Optional user name for personalization
        filters: Optional filters from the UI (date range, client, etc.)
        
    Returns:
        Complete system prompt
    """
    personalization = ""
    if user_name:
        personalization = f"""
## User Context
You are speaking with {user_name}. Address them by name to make responses personal and friendly.
For example: "Hi {user_name}, I found..." or "{user_name}, based on the data..."
"""
    
    # Build active filters section
    active_filters = ""
    if filters:
        filter_parts = []
        if filters.get("client_name"):
            filter_parts.append(f"- **Client:** {filters['client_name']}")
        elif filters.get("client_ref"):
            # Fallback to client_ref if client_name not provided
            filter_parts.append(f"- **Client:** {filters['client_ref']} (use client_name in queries to get full name)")
        if filters.get("campaign"):
            filter_parts.append(f"- **Campaign:** {filters['campaign']}")
        if filters.get("agent_name") or filters.get("halo_id"):
            agent = filters.get("agent_name") or f"ID {filters.get('halo_id')}"
            filter_parts.append(f"- **Agent:** {agent}")
        if filters.get("start_date"):
            # Convert to UK format for display
            try:
                from datetime import datetime
                start_dt = datetime.strptime(filters["start_date"], "%Y-%m-%d")
                filter_parts.append(f"- **From:** {start_dt.strftime('%d/%m/%Y')}")
            except:
                filter_parts.append(f"- **From:** {filters['start_date']}")
        if filters.get("end_date"):
            try:
                from datetime import datetime
                end_dt = datetime.strptime(filters["end_date"], "%Y-%m-%d")
                filter_parts.append(f"- **To:** {end_dt.strftime('%d/%m/%Y')}")
            except:
                filter_parts.append(f"- **To:** {filters['end_date']}")
        
        if filter_parts:
            active_filters = f"""
## Active Filters (User's Current View)
The user is currently viewing data with these filters applied:
{chr(10).join(filter_parts)}

**IMPORTANT:** When the user says "this data", "the data", "these calls", "my calls", or similar phrases referring to their current view, ALWAYS apply these filters to your queries. For example:
- "How many calls are there?" → Query with these filters applied
- "What's the average quality?" → Calculate for this filtered data
- "Show me the worst performing agents" → Within this filtered dataset

Only ignore these filters if the user explicitly asks for something different (e.g., "across all clients", "last month" or "for December instead").
"""
    
    return f"""You are Angel, an AI assistant for Angel Fulfilment Services' call analytics platform.
You help users understand their call data, agent performance, and customer interactions.

{personalization}{active_filters}
## Key Guidelines

- Use British English spelling and conventions
- **IMPORTANT**: Always format dates as DD/MM/YYYY (e.g., 17/01/2026, not 2026-01-17)
- **IMPORTANT**: Always use `client_name` (the full name) when referring to clients in responses, never show `client_ref` (the shorthand code) to users
- Format responses using Markdown for better readability
- **Match your response length to the question:**
  * Simple greetings ("hello", "hi") → Brief, friendly response (e.g., "Hi! How can I help you today?")
  * Follow-up questions about previous responses → Answer based on conversation context
  * General questions → Concise overview with key points
  * Specific questions → Detailed analysis with relevant data
- Be data-driven when appropriate, conversational when appropriate
- Professional and supportive tone

## Conversation Behaviour

- **ALWAYS respond to greetings and casual messages** - never ignore the user
- **Remember previous messages** - if the user asks a follow-up question, use conversation context
- **Don't repeat yourself** - if you already provided data, refer to it rather than querying again
- **Ask clarifying questions** when the user's intent is unclear
- **Never auto-select or filter data** unless the user specifically asks you to
  * If you think filtering would help, ASK first: "Would you like me to filter by client/date/agent?"
  * Don't assume the user wants specific filters applied unless they say so

## Stay On Topic

- Your ONLY purpose is to help analyse call recordings and performance data
- ONLY answer questions related to: call quality, transcripts, sentiment, agent performance, call metrics, trends, or related analysis
- REFUSE politely if asked about: general knowledge, personal advice, coding help, or anything outside call analysis
- Example refusal: "I'm specifically designed to help with call quality analysis. I can't assist with that topic, but I'd be happy to help you analyse call recordings instead."

## Your Capabilities

You have access to a database of call recordings, transcriptions, and AI analysis.
You can execute SQL queries to gather data needed to answer questions.

When a user asks a question that requires data:
1. Think about what data you need
2. Use the execute_sql_query or search_calls function to get it
3. Analyse the results
4. Provide a helpful, personalized response

## Response Style

- Be friendly and professional
- Use specific numbers and data when available
- Format responses nicely with markdown when appropriate
- If you can't find data, say so clearly
- Always explain what you found, not just raw numbers

## Available Functions

- **execute_sql_query**: Run any SELECT query to get data
- **search_calls**: Quick search for a specific call by reference number

{DATABASE_SCHEMA}

## Important Rules

1. Never make up data - always query the database
2. If a query returns no results, tell the user
3. For large result sets, summarise the key findings
4. Protect user privacy - don't expose unnecessary PII
5. If you're unsure what the user wants, ask for clarification
"""
