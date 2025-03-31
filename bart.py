from transformers import pipeline
import pandas as pd

# Load preprocessed data
df1 = pd.read_pickle("grouped_conversations.pkl")
df= df1.head(5000)
# Convert structured conversations into text format
def format_conversation(conversation):
    return " ".join([f"{timestamp} - {sender}: {message}" for sender, message, timestamp in conversation])

df["Text"] = df["Conversation"].apply(format_conversation)

# Load BART model with batch processing enabled
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)  # Keep on CPU

# Define classification labels
labels = ["successful", "unsuccessful"]

# Process in batches
batch_size = 16  # Adjust based on memory
texts = df["Text"].tolist()

results = classifier(texts, labels, batch_size=batch_size)
df["Predicted_Label"] = [res["labels"][0] for res in results]

# Save classified conversations
df.to_pickle("classified_conversations.pkl")

print("Classification complete. Saved as 'classified_conversations.pkl'")