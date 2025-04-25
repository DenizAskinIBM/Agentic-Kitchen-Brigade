import re
import time
import requests
from typing import Any, Mapping
import flask
import functions_framework

# === CONFIGURATION ===
API_KEY = "azE6dXNyXzRlZDk4MDg2LTQ0NWItM2FlZS1iOTEzLTA1MTE4NDViMmUxNzplREJzRUVib3hOaHN1NFhqNFBtWDhGMDJ6UjNQSzhtTEFEd0QyRUo3Nlh3PTovY2kr"  # Replace with your Orchestrate API key
TOKEN = None
TOKEN_EXPIRATION = 0
ORCHESTRATE_URL = "https://api.dl.watson-orchestrate.ibm.com:443/instances/20250212-1521-3150-30c9-e789cad9cae1/v1/skills/_personal_/trial-generative__latest__add_1/prompts/add_1/generation/text"

# === COMMANDS ===
ABOUT_COMMAND_ID = 1
HELP_COMMAND_ID = 2

# === TOKEN MANAGEMENT ===
def is_token_expired(expiration_time):
    return int(time.time()) > expiration_time

def get_bearer_token(api_key):
    global TOKEN, TOKEN_EXPIRATION
    if TOKEN and not is_token_expired(TOKEN_EXPIRATION):
        return TOKEN
    url = "https://iam.platform.saas.ibm.com/siusermgr/api/1.0/apikeys/token"
    payload = {"apikey": api_key}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        TOKEN = data.get("token")
        TOKEN_EXPIRATION = int(time.time()) + data.get("expires_in", 3600)
        return TOKEN
    else:
        print("Token request failed:", response.text)
        return None

# === ORCHESTRATE CALL ===
def invoke_orchestrate_skill():
    token = get_bearer_token(API_KEY)
    if not token:
        return "Unable to authenticate with Watson Orchestrate."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    payload = {
        "n1": "2",
        "n2": "3",
        "output1": " "
    }

    response = requests.post(ORCHESTRATE_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json().get("generated_text", "No output from skill.")
    else:
        return f"Orchestrate call failed: {response.status_code}"

# === MAIN ENTRY POINT ===
@functions_framework.http
def avatar_app(req: flask.Request) -> Mapping[str, Any]:
    event = req.get_json(silent=True)
    if not event:
        return {"text": "No event received."}

    event_type = event.get("type", "UNKNOWN")
    message_text = event.get("message", {}).get("text", "").strip()
    user = event.get("message", {}).get("sender", {})
    user_name = user.get("displayName", "Unknown")

    if event_type == "ADDED_TO_SPACE":
        return {"text": f"Hello {user_name}, thanks for adding me!"}

    if "appCommandMetadata" in event:
        return handle_app_commands(event)

    if event_type == "MESSAGE":
        return handle_regular_message(event)

    return {"text": "Unrecognized event type."}

# === MESSAGE HANDLER ===
def handle_regular_message(event: Mapping[str, Any]) -> Mapping[str, Any]:
    user = event.get("message", {}).get("sender", {})
    message_text = event.get("message", {}).get("text", "").strip().lower()
    user_name = user.get("displayName", "Unknown User")

    ticket_numbers = extract_ticket_number(message_text)
    if ticket_numbers:
        ticket_number = ticket_numbers[0]
        orchestrate_output = invoke_orchestrate_skill() or "No output from Watson Orchestrate."

        return {
            "text": f"Ticket {ticket_number}: {orchestrate_output}",
            "cardsV2": [{
                "cardId": "ticketWithOrchestrate",
                "card": {
                    "name": "Ticket + Orchestrate Summary",
                    "header": {
                        "title": f"Ticket {ticket_number} - Summary for {user_name}"
                    },
                    "sections": [{
                        "widgets": [
                            {"textParagraph": {"text": f"*Ticket ID:* {ticket_number}"}},
                            {"textParagraph": {"text": f"*Watson Orchestrate Output:* {orchestrate_output}"}}
                        ]
                    }]
                }
            }]
        }

    return {"text": "No valid ticket number found. Use format A1234 or B5678."}

# === HELPERS ===
def extract_ticket_number(sentence: str):
    return re.findall(r'\b[a-bA-B]\d{4}\b', sentence)

def handle_app_commands(event: Mapping[str, Any]) -> Mapping[str, Any]:
    cmd_id = event["appCommandMetadata"]["appCommandId"]
    user = event.get("user", {}).get("displayName", "User")
    if cmd_id == ABOUT_COMMAND_ID:
        return {"text": f"Avatar Bot is active. Hello, {user}!"}
    elif cmd_id == HELP_COMMAND_ID:
        return {"text": "Try typing a ticket like 'B0123' to get a summary with Watson Orchestrate help."}
    return {"text": "Unknown command received."}
