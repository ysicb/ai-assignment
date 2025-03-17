from database import get_connection
import pandas as pd

def fetch_conversations():
    connection = get_connection()
    cursor = connection.cursor()

    # Fetch incoming messages (User Messages)
    cursor.execute("SELECT conversationId, message, conversationincomingtime, 'User' as Sender FROM conversationincoming")
    incoming = cursor.fetchall()
    
    # Fetch outgoing messages (Bot Messages)
    cursor.execute("SELECT conversationId, message, conversationoutgoingtime, 'Bot' as Sender FROM conversationoutgoing")
    outgoing = cursor.fetchall()

    cursor.close()
    connection.close()

    # Convert to DataFrames
    df_incoming = pd.DataFrame(incoming, columns=["ConversationId", "Message", "Timestamp", "Sender"])
    df_outgoing = pd.DataFrame(outgoing, columns=["ConversationId", "Message", "Timestamp", "Sender"])
   

    # Convert timestamps to datetime for sorting
    df_incoming["Timestamp"] = pd.to_datetime(df_incoming["Timestamp"])
    df_outgoing["Timestamp"] = pd.to_datetime(df_outgoing["Timestamp"])

    # Combine both user and bot messages into a single dataframe
    df_combined = pd.concat([df_incoming, df_outgoing], ignore_index=True)

    # Sort messages by conversation ID and timestamp
    df_combined.sort_values(by=["ConversationId", "Timestamp"], inplace=True)

    # Group all messages per conversation
    grouped_conversations = df_combined.groupby("ConversationId").apply(lambda x: list(zip(x["Sender"], x["Message"], x["Timestamp"]))).reset_index(name="Conversation")

    # Save the structured data for later use
    grouped_conversations.to_pickle("grouped_conversations.pkl")

    return grouped_conversations

# Run the function
df = fetch_conversations()

