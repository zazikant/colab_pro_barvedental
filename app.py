import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
import requests
import json
import time

from flask_ngrok import run_with_ngrok
from flask import Flask, request

from functions import draft_email

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)

# Initialize the Flask app
flask_app = Flask(__name__)
run_with_ngrok(flask_app)
handler = SlackRequestHandler(app)

# Initialize variables to store the result of draft_email
cached_response = None
cached_text = None

def get_bot_user_id():
    try:
        # Initialize the Slack client with your bot token
        slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
        response = slack_client.auth_test()
        return response["user_id"]
    except SlackApiError as e:
        print(f"Error: {e}")


#manually do a GET Request.. it takes time with high resource (RAM).. http://7908-34-91-45-139.ngrok-free.app/get_response?text=@GEM%20about%204D%20BIM%20shashi@123
@flask_app.route("/get_response", methods=["GET"])
def get_response():
    global cached_response, cached_text

    text = request.args.get("text", "")

    # Check if the text has changed
    if text != cached_text:
        # If the text has changed, call draft_email and update the cache
        email, response = draft_email(text)
        cached_text = text
        cached_response = {"email": email, "response": response}

    return cached_response

@app.event("app_mention")
def handle_mentions(body, say):
    global cached_response, cached_text

    text = body["event"]["text"]
    mention = f"<@{SLACK_BOT_USER_ID}>"
    text = text.replace(mention, "").strip()
    say("Sure, I'll get right on that!")

    # Update the cache with the latest text
    cached_text = text
    # Reset the cached response to None, so that it gets recalculated in the next GET request
    cached_response = None

    # No need to return anything here, as this is an event handler

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

if __name__ == "__main__":
    flask_app.run()