# Call Analysis Configuration

Guide to configuring topics, agent actions, and performance rubrics.

## Three-Tier Configuration System

Angel Intelligence uses a three-tier configuration system:

| Tier | Structure | Storage | Purpose |
|------|-----------|---------|---------|
| **Universal** | Rigid | Database + file fallback | Standard analysis framework |
| **Client** | Dynamic | Database only | Client-specific context |
| **Campaign Type** | Dynamic | Database only | Reusable campaign goals |

### Configuration Hierarchy

When analysing a call:
1. **Universal config** provides the analysis framework (topics, actions, rubric)
2. **Client config** provides organisational context (charity info, tone guidelines)
3. **Campaign type config** provides call-specific goals (success criteria, expectations)

### API Endpoints (v2)

```bash
# Universal config
GET  /api/v2/config/universal
POST /api/v2/config/universal

# Client configs
GET  /api/v2/config/clients
GET  /api/v2/config/client/{client_ref}
POST /api/v2/config/client/{client_ref}
DELETE /api/v2/config/client/{client_ref}

# Campaign type configs
GET  /api/v2/config/campaign-types
GET  /api/v2/config/campaign-type/{campaign_type}
POST /api/v2/config/campaign-type/{campaign_type}
DELETE /api/v2/config/campaign-type/{campaign_type}

# Merged config preview
GET  /api/v2/config/merged?client_ref=ABC123&campaign_type=inbound_donation
```

---

## Universal Configuration

Universal config has a **rigid structure** - the same fields for all clients.

### File-Based (Fallback)

The main configuration file is `call_analysis_config.json` in the project root.

```json
{
  "topics": [...],
  "agent_actions": [...],
  "performance_rubric": [...]
}
```

### Database-Based (Preferred)

```bash
POST /api/v2/config/universal
```

```json
{
  "topics": ["One-off donation", "Regular giving signup", ...],
  "agent_actions": ["greeting", "verification", "explanation", ...],
  "performance_rubric": ["Empathy", "Clarity", "Listening", ...]
}
```

---

## Client Configuration

Client config has a **dynamic structure** - different clients can have different fields.

### Common Fields

```json
{
  "organisation_name": "Cancer Research UK",
  "organisation_type": "charity",
  "mission": "Beat cancer through research and awareness",
  "tone_guidelines": "Warm, empathetic, professional",
  "compliance_notes": "FCA regulated for some activities"
}
```

### Custom Fields

Clients can add any fields they need:

```json
{
  "organisation_name": "RSPCA",
  "key_campaigns": ["winter appeal", "legacy giving"],
  "prohibited_topics": ["euthanasia", "prosecution details"],
  "vulnerable_caller_protocol": "Always offer callback"
}
```

### API Example

```bash
POST /api/v2/config/client/CRUK001
```

```json
{
  "client_ref": "CRUK001",
  "config_data": {
    "organisation_name": "Cancer Research UK",
    "mission": "Beat cancer through research and awareness",
    "tone_guidelines": "Warm, empathetic, professional"
  }
}
```

---

## Campaign Type Configuration

Campaign types are **reusable templates** for different call types.

### Common Campaign Types

| Campaign Type | Description |
|--------------|-------------|
| `inbound_donation` | Incoming calls from supporters wanting to donate |
| `outbound_upgrade` | Calls to upgrade existing donations |
| `retention` | Calls to save cancelling supporters |
| `welcome_call` | New supporter welcome calls |
| `complaint` | Complaint handling calls |

### Campaign Type Structure

```json
{
  "description": "Calls to upgrade existing regular donors",
  "goals": [
    "Increase monthly donation amount",
    "Maintain positive supporter relationship"
  ],
  "success_criteria": [
    "Upgrade achieved",
    "Gift Aid confirmed if eligible",
    "Supporter remains positive"
  ],
  "key_metrics": [
    "Upgrade rate",
    "Average increase amount",
    "Cancellation from call"
  ],
  "agent_expectations": "Listen actively, acknowledge current support, present upgrade positively"
}
```

### API Example

```bash
POST /api/v2/config/campaign-type/outbound_upgrade
```

```json
{
  "campaign_type": "outbound_upgrade",
  "config_data": {
    "description": "Outbound upgrade calls",
    "goals": ["Increase donation amount", "Maintain relationship"],
    "success_criteria": ["Upgrade achieved", "Supporter positive"]
  }
}
```

---

## Topics

Topics are the main subjects or call types that the AI identifies in calls.

### Default Topics

```json
{
  "topics": [
    "One-off donation request",
    "Regular giving signup",
    "Donation upgrade/adjustment",
    "Raffle/Lottery signup",
    "Gift Aid explanation/enrolment",
    "Direct Debit instruction confirmation",
    "Payment card details capture",
    "Bank details verification",
    "Contact information update",
    "Email/postal preference capture",
    "Complaint handling",
    "Service/product enquiry",
    "Information request",
    "Callback request/missed call follow-up",
    "Renewal of commitment",
    "Cancellation request",
    "Retention attempt",
    "Deceased notification handling",
    "Data protection query",
    "Refund request",
    "Thank you/appreciation call",
    "Welcome call",
    "Verification call",
    "Warm-up/cultivation call",
    "Campaign awareness",
    "Event promotion/registration",
    "Beneficiary story sharing",
    "Vulnerability assessment",
    "Safeguarding concern",
    "Donation completion/payment processing"
  ]
}
```

### Custom Topics

Add client-specific topics:

```json
{
  "topics": [
    "Legacy enquiry",
    "Membership renewal",
    "Volunteer recruitment",
    "Corporate partnership"
  ]
}
```

### Output Format

Topics appear in analysis as:

```json
{
  "key_topics": [
    {
      "name": "Regular giving signup",
      "confidence": 0.95,
      "timestamp_start": 45.5,
      "timestamp_end": 78.2
    }
  ]
}
```

---

## Agent Actions

Actions are specific behaviours or tasks agents should perform.

### Default Actions

```json
{
  "agent_actions": [
    "Greeted supporter",
    "Verified identity/security questions",
    "Confirmed purpose of call",
    "Offered appropriate options",
    "Explained terms clearly",
    "Captured consent appropriately",
    "Summarised commitment accurately",
    "Thanked supporter",
    "Provided reference number",
    "Signposted next steps",
    "Offered contact alternatives",
    "Handled objection effectively",
    "Identified vulnerability and responded",
    "Escalated appropriately",
    "Followed script",
    "Deviated from script appropriately",
    "Maintained professional tone",
    "Demonstrated empathy",
    "Built rapport",
    "Confirmed understanding",
    "Provided Gift Aid explanation",
    "Completed data protection statement",
    "Offered complaints procedure",
    "Processed refund request",
    "Logged call notes accurately"
  ]
}
```

### Output Format

Actions appear in analysis as:

```json
{
  "agent_actions_performed": [
    {
      "action": "Greeted supporter",
      "timestamp_start": 0.0,
      "quality": 5
    },
    {
      "action": "Verified identity/security questions",
      "timestamp_start": 15.2,
      "quality": 4
    }
  ]
}
```

### Quality Scale

| Score | Description |
|-------|-------------|
| 5 | Excellent - exceeded expectations |
| 4 | Good - performed well |
| 3 | Adequate - met minimum standard |
| 2 | Poor - needs improvement |
| 1 | Failed - did not perform correctly |

---

## Performance Rubric

Criteria for scoring agent performance on a 1-10 scale.

### Default Rubric

```json
{
  "performance_rubric": [
    "Clarity of speech",
    "Tone control",
    "Active listening",
    "Empathy & rapport",
    "Confidence & authority",
    "Accurate information delivery",
    "Script/protocol adherence",
    "Payment and data protection compliance",
    "Recording of mandatory information",
    "Call structure/flow control",
    "Quality of donation ask or conversion attempt",
    "Objection handling skill",
    "Engagement effectiveness",
    "Problem solving",
    "Effective closing"
  ]
}
```

### Output Format

Scores appear in analysis as:

```json
{
  "performance_scores": {
    "Clarity of speech": 8,
    "Tone control": 9,
    "Active listening": 7,
    "Empathy & rapport": 8,
    "Confidence & authority": 7,
    "Accurate information delivery": 9,
    "Script/protocol adherence": 8,
    "Payment and data protection compliance": 10,
    "Recording of mandatory information": 9,
    "Call structure/flow control": 7,
    "Quality of donation ask or conversion attempt": 8,
    "Objection handling skill": 6,
    "Engagement effectiveness": 8,
    "Problem solving": 7,
    "Effective closing": 8
  }
}
```

### Score Scale

| Score | Description |
|-------|-------------|
| 10 | Perfect - exceptional performance |
| 8-9 | Excellent - above expectations |
| 6-7 | Good - meets expectations |
| 4-5 | Adequate - room for improvement |
| 2-3 | Below standard - training needed |
| 1 | Unacceptable - serious issue |

---

## Client-Specific Configuration

Override global configuration for specific clients via API.

### Create Client Config

```bash
curl -X POST http://localhost:8000/api/config \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "client_ref": "CLIENT001",
    "config_type": "topics",
    "config_data": [
      "Membership renewal",
      "Legacy enquiry",
      "Special appeal"
    ],
    "is_active": true
  }'
```

### Config Types

| Type | Description |
|------|-------------|
| `topics` | List of topics to detect |
| `agent_actions` | List of actions to track |
| `performance_rubric` | Performance criteria |
| `prompt` | Custom analysis prompt |
| `analysis_mode` | `audio` or `transcript` |

### Get Client Config

```bash
curl http://localhost:8000/api/config?client_ref=CLIENT001 \
  -H "Authorization: Bearer $TOKEN"
```

### Delete Client Config

```bash
curl -X DELETE http://localhost:8000/api/config/123 \
  -H "Authorization: Bearer $TOKEN"
```

---

## Database Storage

Client configurations are stored in `ai_client_configs`:

```sql
SELECT * FROM ai_client_configs WHERE client_ref = 'CLIENT001';
```

---

## How Configuration is Applied

1. Worker loads global config from `call_analysis_config.json`
2. Worker checks database for client-specific overrides
3. Client config merges with/replaces global config
4. Combined config is passed to analyzer

### Merge Logic

```python
# Load global config
config = load_config("call_analysis_config.json")

# Get client overrides
client_config = get_client_config(client_ref)

# Apply overrides
if client_config:
    if client_config.get("topics"):
        config["topics"] = client_config["topics"]
    if client_config.get("agent_actions"):
        config["agent_actions"] = client_config["agent_actions"]
    # etc.
```

---

## Best Practices

### Topics

1. **Be Specific**: "Regular giving signup" not just "Donation"
2. **Avoid Overlap**: Each topic should be distinct
3. **Keep Manageable**: 20-30 topics maximum
4. **Include Negatives**: "Cancellation request", "Complaint handling"

### Agent Actions

1. **Observable Behaviours**: Things that can be heard in call
2. **Sequential Order**: List in typical call flow order
3. **Measurable**: Actions should be binary (did/didn't do)
4. **Include Compliance**: Data protection, consent capture

### Performance Rubric

1. **Balanced Mix**: Communication, compliance, outcomes
2. **Client Priorities**: Weight what matters most
3. **Trainable Skills**: Criteria agents can improve
4. **Clear Definitions**: Document what each score means

---

## Example: Custom Client Configuration

### Age UK Configuration

```json
{
  "topics": [
    "Information request",
    "Advice line enquiry",
    "Befriending service referral",
    "Shop volunteer enquiry",
    "Donation request",
    "Legacy enquiry",
    "Complaint",
    "Safeguarding concern"
  ],
  "agent_actions": [
    "Greeted caller warmly",
    "Identified caller needs",
    "Provided accurate information",
    "Offered additional support",
    "Made appropriate referral",
    "Captured consent",
    "Completed safeguarding check",
    "Thanked caller"
  ],
  "performance_rubric": [
    "Warmth and empathy",
    "Active listening",
    "Accuracy of information",
    "Patience with caller",
    "Safeguarding awareness",
    "Referral appropriateness",
    "Call documentation"
  ]
}
```

### Save via API

```bash
curl -X POST http://localhost:8000/api/config \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "client_ref": "AGE_UK",
    "config_type": "topics",
    "config_data": ["Information request", "Advice line enquiry", ...],
    "is_active": true
  }'
```

---

## Reloading Configuration

### Global Config

1. Edit `call_analysis_config.json`
2. Restart workers (config loaded at startup)

### Client Config

- Changes take effect immediately (loaded per-call)
- No restart required

### Hot Reload (Future)

Enable automatic config reloading:

```env
ENABLE_CONFIG_HOT_RELOAD=true
CONFIG_RELOAD_INTERVAL=60
```
