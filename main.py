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
# FastAPI app instance
app = FastAPI()

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load BART model for classification
#classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
labels = ["successful", "unsuccessful"]

# Database connection function
def get_connection():
    return mysql.connector.connect(
        host="pcz218dbl23",
        user="prakashd",
        password="TLzWqu8Kyp",
        database="omni_qa_db"
    )

# Define a Pydantic model for structured response
class MessageEntry(BaseModel):
    Sender: str
    Message: str
    Timestamp: str

class ConversationResponse(BaseModel):
    ConversationId: int
    Classification: str
    ConfidenceScore: float
    Conversation: List[MessageEntry]

def clean_text(text):
    """Convert JSON or HTML text into readable format."""
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

    client = openai.OpenAI(api_key=OPENAI_API_KEY)  # Create OpenAI client

    prompt = f"""
    Given the following chatbot conversation, classify it as either 'successful' or 'unsuccessful' based on satisfactorily answering the user's queries.
    
    Conversation:
    {conversation_text}
    
    Respond only with JSON:
    {{
        "classification": "successful" or "unsuccessful",
        "confidence_score": float (between 0 and 1)
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI that classifies chatbot conversations."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}  
    )

    result_json = response.choices[0].message.content  # Extract response content
    parsed_result = json.loads(result_json)  # Convert to dictionary

    return parsed_result["classification"], round(parsed_result["confidence_score"], 2)

def fetch_and_classify_conversation(conversation_id: int):
    connection = get_connection()
    cursor = connection.cursor()

    # Fetch incoming messages (User Messages)
    cursor.execute(f"SELECT message, conversationincomingtime FROM conversationincoming WHERE conversationId = {conversation_id}")
    incoming = cursor.fetchall()

    # Fetch outgoing messages (Bot Messages)
    cursor.execute(f"SELECT message, conversationoutgoingtime FROM conversationoutgoing WHERE conversationId = {conversation_id}")
    outgoing = cursor.fetchall()

    cursor.close()
    connection.close()

    if not incoming and not outgoing:
        return None, "Conversation ID not found"

    # Create a structured conversation list
    structured_conversation = []
    for message, timestamp in incoming:
        structured_conversation.append({"Sender": "User", "Message": clean_text(message), "Timestamp": str(timestamp)})
    for message, timestamp in outgoing:
        structured_conversation.append({"Sender": "Bot", "Message": clean_text(message), "Timestamp": str(timestamp)})
    
    # Sort the messages by timestamp
    structured_conversation.sort(key=lambda x: x["Timestamp"])

    # Format the text for classification
    conversation_text = " ".join([f"{msg['Sender']}: {msg['Message']}" for msg in structured_conversation])

    # BART
    # result = classifier(conversation_text, labels)
    # classification = result["labels"][0]  # Get the most likely label
    # confidence_score = round(result["scores"][0], 2)  # Get confidence score rounded to 2 decimal places
    # GPT
    classification, confidence_score = classify_conversation_with_openai(conversation_text)
    
    return classification, confidence_score, structured_conversation

@app.get("/conversation/{conversation_id}", 
         response_model=ConversationResponse,
         summary="Get conversation classification",
         description="Retrieve the classification result (successful or unsuccessful) and the conversation text for a given conversation ID.")
def get_conversation_classification(conversation_id: int):
    try:
        classification, confidence_score, conversation = fetch_and_classify_conversation(conversation_id)
        
        if classification is None:
            raise HTTPException(status_code=404, detail="Conversation ID not found")
        
        return {
            "ConversationId": conversation_id,
            "Classification": classification,
            "ConfidenceScore": confidence_score,
            "Conversation": conversation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))