# AF3 HealthNav MCP Integration - Demo Script

## What We're Demonstrating

End-to-end agent evaluation on AWS Commercial WxO, showing:
1. **Excel input** with questions and expected answers
2. **Test case generation** from Excel (with LLM keyword extraction)
3. **Agent evaluation** hitting two different MCP backends:
   - **Verint MCP** (our Lambda on AWS) - for general service questions
   - **HealthNav MCP** (Alight's server) - for health plan questions
4. **Supervisor routing** - one agent routes to the correct backend based on intent

---

## Prerequisites (Already Done)

```bash
# 1. Activate environment
orchestrate env activate aws-commercial --api-key <key>

# 2. Set API key
export WO_API_KEY="<your-api-key>"    # Mac
set WO_API_KEY=<your-api-key>         :: Windows

# 3. Agents deployed: af3_alight_supervisor_agent, af3_verint_agent, af3_healthnav_agent
# 4. HealthNav MCP connection: af3_healthnav_mcp (OAuth2/Cognito) ✅
# 5. HealthNav MCP toolkit: af3_healthnav_toolkit (4 tools) ✅
```

---

## Demo Flow

### Step 1: Show the Excel Input

Open `questions.xlsx` - 3 questions with expected answers:

| # | Question | Expected Answer | Target Backend |
|---|----------|-----------------|----------------|
| 1 | Are our 1099 forms available online? | Yes, your 1099 forms are available online... | Verint MCP |
| 2 | How do I add a dependent after marriage? | You have 31 days from your marriage date... | Verint MCP |
| 3 | What does my medical plan cover? | Your medical plan covers office visits, specialist visits... | HealthNav MCP |

**Key point:** Questions 1-2 are general service questions (Verint). Question 3 is a health plan question (HealthNav).

### Step 2: Generate Test Cases from Excel

```bash
python commcloud_generate.py
```

**What happens:**
- Reads `questions.xlsx`
- Uses LLM to extract keywords from expected answers
- Outputs 3 test case JSON files to `test_data/`
- Each test case targets `af3_alight_supervisor_agent`

**Show output:** 3 test files generated (test_001.json, test_002.json, test_003.json)

```bash
ls test_data/
```

### Step 3: Run Agent Evaluation

```bash
python commcloud_eval.py --evaluate -v
```

**What happens for each test case:**

**test_001 (1099 forms) - Verint path:**
```
User: "Are our 1099 forms available online?"
  → af3_alight_supervisor_agent (detects: general service question)
    → af3_verint_agent (collaborator)
      → call_verint_studio (tool) → AWS Lambda MCP Server
        → Response: "Yes, your 1099 forms are available online..."
```

**test_003 (medical plan) - HealthNav path:**
```
User: "What does my medical plan cover?"
  → af3_alight_supervisor_agent (detects: health plan question)
    → af3_healthnav_agent (collaborator)
      → get_medical_plan (tool) → HealthNav MCP Server (Cognito auth)
        → Response: Real plan data (BCBS, HDHP, deductibles, copays...)
```

### Step 4: Show Results

**Key things to point out:**
1. **Routing works** - Supervisor correctly routes to Verint vs HealthNav based on question type
2. **HealthNav MCP returns real data** - Actual plan details: BCBS insurance, $3000 deductible, 25% coinsurance, office visits, emergency care, mental health
3. **End-to-end flow** - From Excel question → generated test case → agent evaluation → MCP server hit → real data returned
4. **Both MCP backends work** - Verint Lambda AND HealthNav Cognito-protected server

### Step 5: Show the Results Table

```bash
python commcloud_eval.py --report
```

Shows metrics: Journey Success, Text Match, Tool Calls, Response Time.

---

## Architecture

```
questions.xlsx
     │
     ▼ (generate)
test_data/*.json
     │
     ▼ (evaluate)
af3_alight_supervisor_agent
     │
     ├── General service Q → af3_verint_agent
     │                            │
     │                            ▼
     │                    af3_call_verint_studio (OpenAPI tool)
     │                            │
     │                            ▼
     │                    AWS Lambda MCP Server
     │                    (bn1jku4pg0.execute-api.us-east-1.amazonaws.com)
     │
     └── Health plan Q → af3_healthnav_agent
                              │
                              ▼
                      af3_healthnav_toolkit (MCP tools)
                      ├── get_medical_plan
                      ├── get_dental_plan
                      ├── get_vision_plan
                      └── get_patient
                              │
                              ▼
                      HealthNav MCP Server (OAuth2/Cognito)
                      (healthnav-mcp-ext-dv.ap.alight.com)
```

---

## Quick Commands Reference

```bash
# Generate test cases from Excel
python commcloud_generate.py

# Run evaluation only
python commcloud_eval.py --evaluate -v

# Run evaluation + LLM judge + report
python commcloud_eval.py --all

# Just show results
python commcloud_eval.py --report

# Limit to 1 test case (quick test)
python commcloud_eval.py --evaluate --limit 1
```

---

## Talking Points

- **Two MCP integrations** working on same WxO instance - Verint (AWS Lambda) + HealthNav (Alight Cognito)
- **Supervisor routing** decides which backend to call based on user intent
- **HealthNav returns real data** - BCBS plan, deductibles, copays, covered services
- **Fully automated pipeline** - Excel → test cases → evaluation → results
- **AF3 prefix** isolates our resources from other teams on shared instance
- **OAuth2/Cognito authentication** handled automatically by WxO connection
