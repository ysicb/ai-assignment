from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from transformers import pipeline
import mysql.connector
from bs4 import BeautifulSoup
import json
import html
import re
import openai
import os
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
labels = ["successful", "unsuccessful"]

def get_connection():
    return mysql.connector.connect(
        host="pcz218dbl23",
        user="prakashd",
        password="TLzWqu8Kyp",
        database="omni_qa_db"
    )

class MessageEntry(BaseModel):
    Sender: str
    Message: str
    Timestamp: str

class ConversationResponse(BaseModel):
    ConversationId: int
    Classification: str
    ConfidenceScore: float
    Reason: str
    Conversation: List[MessageEntry]

def clean_text(text):
    if text is None:
        return ""
    text = str(text).strip()
    try:
        if text.startswith("{") or text.startswith("["):
            parsed_text = json.loads(text)
            if isinstance(parsed_text, list):
                return " ".join(item.get("text", "") for item in parsed_text if isinstance(item, dict))
            elif isinstance(parsed_text, dict):
                return " ".join(parsed_text.values())
    except json.JSONDecodeError:
        pass
    if "<" in text and ">" in text:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def classify_conversation_with_openai(conversation_text):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""
    You are an AI assistant trained to evaluate chatbot conversations. Your task is to classify a conversation as either 'successful' or 'unsuccessful' based on the quality of the bot’s responses. The primary focus is on bot's responses than user behaviour. User behavior (such as exiting the chat early or not responding) does not determine failure focus on what the bot replies.

    **Success Criteria:**
    - The bot provides a relevant and satisfactory response to the user's query.
    - The conversation reaches a resolution where the user's needs are met, and if request not fulfilled due to external reasons (not because of bot's misunderstanding) and bot communicates this, then it is successful(eg: while scheduling a tour if a slot is unavailable then the bot specifies as unavailable hence successful). 
    - If the bot does not have an exact answer, it provides a useful alternative(eg: if bot cannot provide a exact location, it will give link of the map).

    **Failure Indicators:**
    - The bot provides fallback phrases(eg i am virtual assistant i don't understand, can you rephrase,I'm sorry, I'm having trouble hearing you, etc), irrelevant ,incomplete or unhelpful response.
    - The user expresses dissatisfaction or repeats/reframes the question.
    - If the user asks to talk to a human or leasing agent or bot suggests to talk to a leasing agent.
    
    {conversation_text}

    **Response Format (JSON Only):**
    {{
        "classification": "successful" or "unsuccessful",
        "reason": "Explain why the conversation was classified as successful or unsuccessful",
        "confidence_score": float (between 0 and 1)
    }}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI that classifies chatbot conversations."},
            {"role": "user", "content": prompt}
        ],
        temperature = 0.0,
        response_format={"type": "json_object"}
    )
    result_json = response.choices[0].message.content
    parsed_result = json.loads(result_json)
    return (
        parsed_result["classification"],
        parsed_result["reason"],
        round(parsed_result["confidence_score"], 2)
    )

def fetch_and_classify_conversation(conversation_id: int):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(f"SELECT message, conversationincomingtime FROM conversationincoming WHERE conversationId = {conversation_id}")
    incoming = cursor.fetchall()
    cursor.execute(f"SELECT message, conversationoutgoingtime FROM conversationoutgoing WHERE conversationId = {conversation_id}")
    outgoing = cursor.fetchall()
    cursor.close()
    connection.close()
    if not incoming and not outgoing:
        return None, "Conversation ID not found"
    structured_conversation = []
    for message, timestamp in incoming:
        structured_conversation.append({"Sender": "User", "Message": clean_text(message), "Timestamp": str(timestamp)})
    for message, timestamp in outgoing:
        structured_conversation.append({"Sender": "Bot", "Message": clean_text(message), "Timestamp": str(timestamp)})
    structured_conversation.sort(key=lambda x: x["Timestamp"])
    conversation_text = " ".join([f"{msg['Sender']}: {msg['Message']}" for msg in structured_conversation])
    classification, reason, confidence_score = classify_conversation_with_openai(conversation_text)
    return classification, reason, confidence_score, structured_conversation

@app.get("/conversation/{conversation_id}", response_model=ConversationResponse)
def get_conversation_classification(conversation_id: int):
    try:
        classification, reason, confidence_score, conversation = fetch_and_classify_conversation(conversation_id)
        if classification is None:
            raise HTTPException(status_code=404, detail="Conversation ID not found")
        return {
            "ConversationId": conversation_id,
            "Classification": classification,
            "ConfidenceScore": confidence_score,
            "Reason": reason,
            "Conversation": conversation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
