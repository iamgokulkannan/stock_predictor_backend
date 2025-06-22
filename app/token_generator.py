import requests
import json
from auth_config import AUTH_CONFIG
import time
import os

TOKEN_FILE = "access_token.txt"


def save_token(token):
    with open(TOKEN_FILE, "w") as f:
        f.write(token)


def load_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            return f.read().strip()
    return None


def generate_token():
    print("[INFO] Generating Fyers token...")
    headers = {"Content-Type": "application/json"}
    payload = {
        "fy_id": AUTH_CONFIG["FYERS_ID"],
        "password": AUTH_CONFIG["FYERS_PASSWORD"],
        "app_id": AUTH_CONFIG["APP_ID"],
        "pin": AUTH_CONFIG["PIN"],
        "redirect_uri": AUTH_CONFIG["REDIRECT_URI"]
    }

    # Hit Fyers API login endpoint (headless browser or official automation)
    response = requests.post(
        "https://api.fyers.in/api/v2/generate-authcode?client_id=YOUR_APP_ID&redirect_uri=YOUR_CALLBACK_URL&response_type=code&state=sample", json=payload, headers=headers)
    if response.status_code == 200 and 'auth_code' in response.json():
        auth_code = response.json()['auth_code']
        print("[INFO] Auth Code:", auth_code)

        # Exchange auth code for access token
        token_payload = {
            "grant_type": "authorization_code",
            "app_id": AUTH_CONFIG["APP_ID"],
            "secret_id": AUTH_CONFIG["APP_SECRET"],
            "code": auth_code,
        }
        token_response = requests.post(
            "https://api.fyers.in/api/v2/token", json=token_payload)
        if token_response.status_code == 200 and 'access_token' in token_response.json():
            token = token_response.json()["access_token"]
            save_token(token)
            print("[SUCCESS] Token generated and saved!")
            return token
        else:
            raise Exception("Token exchange failed: " + token_response.text)
    else:
        raise Exception("Auth code failed: " + response.text)
