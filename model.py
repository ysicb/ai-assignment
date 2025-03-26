import pandas as pd
import mysql.connector
import openai
import json
import os
import re
import html
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Database connection function
def get_connection():
    return mysql.connector.connect(
        host="pcz218dbl23",
        user="prakashd",
        password="TLzWqu8Kyp",
        database="omni_qa_db"
    )

# Text cleaning function
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

# Load test data (conversation IDs and actual labels)
labels_df = pd.read_csv("test_set.csv")

# Fetch conversation texts
def fetch_conversations(conversation_ids):
    connection = get_connection()
    cursor = connection.cursor()
    conversations = {}

    for conv_id in conversation_ids:
        cursor.execute(f"SELECT message FROM conversationincoming WHERE conversationId = {conv_id}")
        incoming = [clean_text(row[0]) for row in cursor.fetchall()]

        cursor.execute(f"SELECT message FROM conversationoutgoing WHERE conversationId = {conv_id}")
        outgoing = [clean_text(row[0]) for row in cursor.fetchall()]

        if incoming or outgoing:
            combined_text = " ".join([f"User: {msg}" for msg in incoming] + [f"Bot: {msg}" for msg in outgoing])
            conversations[conv_id] = combined_text
    
    cursor.close()
    connection.close()
    return conversations

# Batch classify conversations using OpenAI
def classify_conversations(conversations):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    prompt =  """
    You are an AI assistant trained to evaluate chatbot conversations. Your task is to classify a conversation as either 'successful' or 'unsuccessful' based on the quality of the bot’s responses. The primary focus is on bot's responses than user behaviour. User behavior (such as exiting the chat early or not responding) does not determine failure focus on what the bot replies.

    **Success Criteria:**
    - The bot provides a relevant and satisfactory response to the user's query.
    - The conversation reaches a resolution where the user's needs are met, and if request not fulfilled due to external reasons (not because of bot's misunderstanding) and bot communicates this, then it is successful(eg: while scheduling a tour if a slot is unavailable then the bot specifies as unavailable hence successful). 
    - If the bot does not have an exact answer, it provides a useful alternative(eg: if bot cannot provide a exact location, it will give link of the map).

    **Failure Indicators:**
    - The bot provides fallback phrases(eg i am virtual assistant i don't understand, can you rephrase, etc), irrelevant ,incomplete or unhelpful response.
    - The user expresses dissatisfaction or repeats/reframes the question.
    - If the user asks to talk to a human or leasing agent or bot suggests to talk to a leasing agent.
    
    Conversations:

    Response format: JSON
    {
        "conversation_id": {
            "classification": "successful" or "unsuccessful",
            "confidence_score": float (between 0 and 1),
            "reason": "Brief explanation for classification"
    },
        ...
    }
    """
    

    for conv_id, text in conversations.items():
        prompt += f"\nConversation ID: {conv_id}\n{text}\n"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You classify chatbot conversations. that I am providing " },
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    result_json = response.choices[0].message.content
    
    return json.loads(result_json)


# Main execution
conversation_ids = labels_df["conversationid"].tolist()
conversations = fetch_conversations(conversation_ids)
predictions = classify_conversations(conversations)

# Store predicted labels in DataFrame
labels_df["predicted_label"] = labels_df["conversationid"].map(lambda conv_id: 1 if predictions.get(str(conv_id), {}).get("classification") == "successful" else 0)

# Compute evaluation metrics
y_true = labels_df["label"]
y_pred = labels_df["predicted_label"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Print actual vs predicted labels
print(labels_df[["conversationid", "label", "predicted_label"]].tail(60))

#Print evaluation results
print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

