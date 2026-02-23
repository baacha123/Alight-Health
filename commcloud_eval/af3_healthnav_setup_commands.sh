

# ----- STEP 1: Create the connection -----
orchestrate connections add \
  --name af3_healthnav_mcp \
  --kind oauth2

# ----- STEP 2: Configure the connection -----
orchestrate connections configure \
  --app-id af3_healthnav_mcp \
  --env draft \
  --type team \
  --kind oauth_auth_client_credentials_flow \
  --url "https://healthnav-mcp-ext-dv.ap.alight.com/mcp"

# ----- STEP 3: Set credentials (send-via body is critical!) -----
orchestrate connections set-credentials \
  --app-id af3_healthnav_mcp \
  --env draft \
  --client-id "3vvcitmdfh50p26680a9kr3qan" \
  --client-secret "kpu4v48i9mqsg77npt20t1kpm1k34268ur0iiheicjcovnm1qq5" \
  --token-url "https://healthnav-mcp-healthnav-service-dv-domain.auth.us-east-1.amazoncognito.com/oauth2/token" \
  --grant-type "client_credentials" \
  --send-via body

# ----- STEP 4: Click "Connect" in the WxO UI -----
# NOTE: After set-credentials, you MUST go to the WxO UI > Connections,
# open af3_healthnav_mcp, scroll to bottom, and click the "Connect" button.
# The CLI sets the credentials but does NOT activate the connection.


# ----- STEP 5: Create the toolkit (transport = streamable_http!) -----
# NOTE: "sse" transport causes a gateway 503 timeout error.
#        "streamable_http" works.
orchestrate toolkits add \
  --kind mcp \
  --name af3_healthnav_toolkit \
  --description "AF3 HealthNav MCP Toolkit" \
  --url "https://healthnav-mcp-ext-dv.ap.alight.com/mcp" \
  --transport streamable_http \
  --tools "*" \
  --app-id af3_healthnav_mcp

# ----- STEP 6: Deploy agents -----
orchestrate agents import -f agents/af3/af3_healthnav_agent.yaml
orchestrate agents import -f agents/af3/af3_supervisor_agent.yaml

# Tools auto-discovered from HealthNav MCP:
#   - af3_healthnav_toolkit:get_medical_plan
#   - af3_healthnav_toolkit:get_dental_plan
#   - af3_healthnav_toolkit:get_vision_plan
#   - af3_healthnav_toolkit:get_patient
#
# Lessons Learned:
#   1. --send-via body (not header) - must match working mcp-auth2_hn connection
#   2. --transport streamable_http (not sse) - sse causes gateway timeout
#   3. Must click "Connect" button in UI after CLI set-credentials
#   4. Connection shows ❌ until you click Connect, even if CLI says success
# =============================================================================
