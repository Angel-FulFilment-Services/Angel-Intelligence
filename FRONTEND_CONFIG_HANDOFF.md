# Frontend Handoff: Three-Tier Configuration System

## Overview

Angel Intelligence uses a **three-tier configuration system** for call analysis. This allows flexible customization at different levels:

| Tier | Structure | Purpose | Example |
|------|-----------|---------|---------|
| **Universal** | Rigid (fixed fields) | Standard analysis framework | Topics, agent actions, performance rubric |
| **Client** | Dynamic (any fields) | Client-specific context | Charity name, mission, tone guidelines |
| **Campaign Type** | Dynamic (any fields) | Reusable call templates | Goals, success criteria for "inbound_donation" |

### How It Works

When a call is analysed:
1. **Universal config** provides the default scoring framework
2. **Client config** can **override** universal arrays OR add organisational context
3. **Campaign type config** can **override** client/universal arrays OR add call-specific goals

### Override Behaviour

If a client or campaign type config includes `topics`, `agent_actions`, or `performance_rubric`, it **completely replaces** the universal values.

**Priority:** `campaign_type` > `client` > `universal`

Example:
- Universal has 40 topics
- Client config has `"topics": ["Legacy enquiry", "Animal welfare"]` → only those 2 topics used
- Campaign type has `"topics": ["Complaint handling"]` → only that 1 topic used (overrides client)

The `prompt_context` field shows how the non-array config data appears in LLM prompts.

---

## API Endpoints

All endpoints require authentication via Bearer token.

### Reading Configuration

Use **`/merged`** for all read operations — it returns all tiers with override logic applied.

| Use Case | Endpoint |
|----------|----------|
| Get universal config only | `GET /api/v2/config/merged` (no params) |
| Get client + universal | `GET /api/v2/config/merged?client_ref=CRUK001` |
| Get campaign type + universal | `GET /api/v2/config/merged?campaign_type=inbound_donation` |
| Get all three tiers merged | `GET /api/v2/config/merged?client_ref=CRUK001&campaign_type=inbound_donation` |

---

### Universal Configuration

**Rigid structure** - same fields for all clients.

#### POST `/api/v2/config/universal`

Save universal configuration.

**Request:**
```json
{
  "topics": ["Topic 1", "Topic 2"],
  "agent_actions": ["action_1", "action_2"],
  "performance_rubric": ["Criteria 1", "Criteria 2"]
}
```

**Response:** Same as GET

---

### Client Configuration

**Dynamic structure** - clients can have different fields.

#### GET `/api/v2/config/clients`

List all clients with configurations.

**Response:**
```json
{
  "clients": ["CRUK001", "RSPCA", "BHF"],
  "count": 3
}
```

#### POST `/api/v2/config/client/{client_ref}`

Save client configuration.

**Request:**
```json
{
  "client_ref": "CRUK001",
  "config_data": {
    "organisation_name": "Cancer Research UK",
    "organisation_type": "charity",
    "mission": "Beat cancer through research and awareness",
    "tone_guidelines": "Warm, empathetic, professional"
  }
}
```

**Common Fields (optional):**
- `organisation_name` - Name of the charity/organisation
- `organisation_type` - Type (e.g., "charity", "nonprofit")
- `mission` - Mission statement
- `tone_guidelines` - How agents should communicate
- `compliance_notes` - Special compliance requirements

Clients can add **any custom fields** they need.

#### DELETE `/api/v2/config/client/{client_ref}`

Deactivate a client's configuration.

**Response:**
```json
{
  "message": "Configuration for client 'CRUK001' deactivated"
}
```

---

### Campaign Type Configuration

**Dynamic structure** - reusable templates for different call types.

#### GET `/api/v2/config/campaign-types`

List all campaign types with configurations.

**Response:**
```json
{
  "campaign_types": ["inbound_donation", "outbound_upgrade", "retention", "complaint"],
  "count": 4
}
```

#### POST `/api/v2/config/campaign-type/{campaign_type}`

Save campaign type configuration.

**Request:**
```json
{
  "campaign_type": "inbound_donation",
  "config_data": {
    "description": "Incoming calls from supporters wanting to donate",
    "goals": ["Capture donation", "Gift Aid confirmation"],
    "success_criteria": ["Payment processed", "Supporter satisfied"]
  }
}
```

**Common Fields (optional):**
- `description` - What this campaign type is
- `goals` - List of goals for this type of call
- `success_criteria` - What defines success
- `key_metrics` - What to measure
- `agent_expectations` - Expectations for agent behaviour

Campaign types can add **any custom fields**.

#### DELETE `/api/v2/config/campaign-type/{campaign_type}`

Deactivate a campaign type configuration.

---

### Merged Configuration Preview

#### GET `/api/v2/config/merged?client_ref=XXX&campaign_type=YYY`

Preview how configs merge for a specific context.

**Query Parameters:**
- `client_ref` (optional) - Client reference
- `campaign_type` (optional) - Campaign type

**Response:**
```json
{
  "universal": {
    "topics": ["Donation", "Gift Aid", "...40 items total..."],
    "agent_actions": ["greeting", "verification", "..."],
    "performance_rubric": ["Empathy", "Clarity", "..."],
    "source": "database"
  },
  "client": {
    "client_ref": "CRUK001",
    "config_data": {
      "organisation_name": "Cancer Research UK",
      "mission": "Beat cancer",
      "topics": ["Legacy enquiry", "Corporate giving"]
    },
    "prompt_context": "CLIENT CONTEXT:\n- Organisation: Cancer Research UK\n- Mission: Beat cancer"
  },
  "campaign_type": {
    "campaign_type": "inbound_donation",
    "config_data": {
      "goals": ["Capture donation"]
    },
    "prompt_context": "CAMPAIGN TYPE: Inbound Donation\nGoals:\n  - Capture donation"
  },
  "prompt_context": "CLIENT CONTEXT:\n- Organisation: Cancer Research UK\n- Mission: Beat cancer\n\nCAMPAIGN TYPE: Inbound Donation\nGoals:\n  - Capture donation",
  "effective_topics": ["Legacy enquiry", "Corporate giving"],
  "effective_agent_actions": ["greeting", "verification", "...from universal..."],
  "effective_performance_rubric": ["Empathy", "Clarity", "...from universal..."]
}
```

**Note:** The `effective_*` fields show exactly what will be used in analysis after overrides are applied. In this example:
- `effective_topics` comes from **client** (overrides universal)
- `effective_agent_actions` comes from **universal** (no override)
- `effective_performance_rubric` comes from **universal** (no override)

---

## UI Recommendations

### Configuration Management Page

1. **Universal Config Tab**
   - Three editable lists: Topics, Agent Actions, Performance Rubric
   - Array editor with add/remove/reorder
   - Show "source" badge (database vs file fallback)

2. **Clients Tab**
   - List of clients with configs
   - Click to edit → dynamic form builder
   - Common fields as suggestions, allow custom fields
   - **Optional override arrays**: `topics`, `agent_actions`, `performance_rubric`
   - Preview `prompt_context` live
   - Show warning when overriding universal arrays

3. **Campaign Types Tab**
   - List of campaign types
   - Click to edit → dynamic form builder
   - Suggest common campaign types: `inbound_donation`, `outbound_upgrade`, `retention`, `complaint`, `welcome_call`
   - **Optional override arrays**: `topics`, `agent_actions`, `performance_rubric`
   - Preview `prompt_context` live

4. **Preview/Test Tab**
   - Select client + campaign type
   - Call `/api/v2/config/merged` to show combined config
   - Display the final `prompt_context` that will be used in LLM prompts
   - **Show effective arrays** - highlight which tier each came from

### Field Types

For dynamic config editors, support these field types:
- **String** - Text input
- **Text** - Multi-line textarea
- **List** - Array of strings with add/remove
- **Object** - Nested key-value pairs

### Reserved Override Fields

These field names in client/campaign_type configs have special meaning:

| Field | Effect |
|-------|--------|
| `topics` | Replaces universal topics |
| `agent_actions` | Replaces universal agent actions |
| `performance_rubric` | Replaces universal performance rubric |

All other fields are treated as context and appear in `prompt_context`.

### Suggested Campaign Types

| Campaign Type | Description |
|--------------|-------------|
| `inbound_donation` | Incoming donation calls |
| `outbound_upgrade` | Upgrade existing donations |
| `retention` | Save cancelling supporters |
| `welcome_call` | New supporter welcome |
| `complaint` | Complaint handling |
| `legacy_giving` | Legacy/bequest enquiries |
| `event_registration` | Event sign-ups |

---

## Database Schema

### ai_call_recordings

New field: `campaign_type` - used to look up campaign type config during analysis.

```sql
campaign_type VARCHAR(100) DEFAULT NULL
```

This is **separate** from `campaign` (which is the campaign name).

### ai_configs (renamed from ai_client_configs)

```sql
CREATE TABLE ai_configs (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    config_tier ENUM('universal', 'client', 'campaign_type') NOT NULL DEFAULT 'client',
    client_ref VARCHAR(100) DEFAULT NULL,
    campaign_type VARCHAR(100) DEFAULT NULL,
    config_type VARCHAR(100) NOT NULL,
    config_data JSON NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME,
    updated_at DATETIME
);
```

---

## Notes

- All responses use British English (organisation, colour, behaviour)
- The `prompt_context` field shows exactly how the config appears in LLM prompts
- Universal config has file fallback (`call_analysis_config.json`) if database is empty
- Client and campaign type configs are database-only
- Configs can be deactivated (soft delete) rather than deleted
