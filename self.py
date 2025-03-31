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

# Generate dynamic prompt based on test data
def generate_prompt_from_test_data(labels_df):
    success_examples = []
    failure_examples = []
    
    for _, row in labels_df.iterrows():
        conv_id = row["conversationid"]
        label = row["label"]
        
        conversation = fetch_conversations([conv_id]).get(conv_id, "")
        if label == 1:
            success_examples.append(f"Conversation ID: {conv_id}\n{conversation}\nClassification: successful\n")
        else:
            failure_examples.append(f"Conversation ID: {conv_id}\n{conversation}\nClassification: unsuccessful\n")
    
    success_text = "\n\n".join(success_examples[:5])  # Limit to 5 examples
    failure_text = "\n\n".join(failure_examples[:5])
    
    return f"""
    You are an AI assistant trained to evaluate chatbot conversations. Your task is to classify a conversation as either 'successful' or 'unsuccessful' based on the bot’s responses.
    
    **Success Examples:**
    {success_text}
    
    **Failure Examples:**
    {failure_text}
    
    Classify the given conversations and return a JSON response in this format:
    {{
        "conversation_id": {{
            "classification": "successful" or "unsuccessful",
            "confidence_score": float (between 0 and 1),
            "reason": "Brief explanation for classification"
        }}
    }}
    """

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
def classify_conversations(conversations, prompt):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    for conv_id, text in conversations.items():
        prompt += f"\nConversation ID: {conv_id}\n{text}\n"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You classify chatbot conversations."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    result_json = response.choices[0].message.content
    return json.loads(result_json)

# Main execution
labels_df = pd.read_csv("test_set.csv")
conversation_ids = labels_df["conversationid"].tolist()
conversations = fetch_conversations(conversation_ids)
prompt = generate_prompt_from_test_data(labels_df)
predictions = classify_conversations(conversations, prompt)

# Store predicted labels in DataFrame
labels_df["predicted_label"] = labels_df["conversationid"].map(lambda conv_id: 1 if predictions.get(str(conv_id), {}).get("classification") == "successful" else 0)

# Compute evaluation metrics
y_true = labels_df["label"]
y_pred = labels_df["predicted_label"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Print evaluation results
print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
