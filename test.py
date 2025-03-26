
# import pandas as pd

# # Load the DataFrame from the pickle file
# df = pd.read_pickle("classified_conversations.pkl")

# # Display basic info about the DataFrame
# print(df.info())  # Shows column names, data types, and non-null counts

# # Display the first 100 conversations
# for conversation_id in range(100):
#     conversation_rows = df[df["ConversationId"] == conversation_id]
    
#     if not conversation_rows.empty:  # Check if conversation exists
#         print(f"\nConversation ID: {conversation_id}\n")

#         for _, row in conversation_rows.iterrows():
#             conversation = row["Conversation"]  # Extract the structured conversation
#             predicted_label = row["Predicted_Label"]  # Get classification result

#             for sender, message, timestamp in conversation:
#                 print(f"{timestamp} - {sender}: {message}")

#             print(f"\nPredicted Label: {predicted_label}\n" + "-"*50)

import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv("filtered_conversations.csv")

# Print conversations in a structured format
for conversation_id in df["ConversationId"].unique()[40651:40700]:  # First 100 conversations
    conversation_rows = df[df["ConversationId"] == conversation_id]
    
    print(f"\nConversation ID: {conversation_id}\n")
    
    for _, row in conversation_rows.iterrows():
        print(f"{row['Timestamp']} - {row['Sender']}: {row['Message']}")
    
    print("-" * 80)  # Separator for clarity








# # temporary

# import pymysql

# # MySQL Database Configuration
# DB_CONFIG = {
#     "host": "pcz218dbl23",  # Remote Server
#     "user": "prakashd",
#     "password": "TLzWqu8Kyp",
#     "database": "omni_qa_db",
# }

# # SQL Query to Create the Table
# CREATE_TABLE_QUERY = """
# CREATE TABLE IF NOT EXISTS conversationreview (
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     ConversationId VARCHAR(255) NOT NULL,
#     Status VARCHAR(50) NOT NULL,
#     ConfidenceScore FLOAT NOT NULL,
#     CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );
# """

# def create_conversationreview_table():
#     """Connect to MySQL and create ConversationReview table."""
#     try:
#         connection = pymysql.connect(**DB_CONFIG)
#         cursor = connection.cursor()
#         cursor.execute(CREATE_TABLE_QUERY)
#         connection.commit()
        
#         print("ConversationReview table created successfully (or already exists).")

#         cursor.close()
#         connection.close()
#     except Exception as e:
#         print("Error creating table:", str(e))

# if __name__ == "__main__":
#     create_conversationreview_table()

# import mysql.connector

# def check_database():
#     connection = mysql.connector.connect(
#         host="pcz218dbl23",
#         user="prakashd",
#         password="TLzWqu8Kyp",
#         database="omni_qa_db"
#     )
#     cursor = connection.cursor()

#     cursor.execute("DESCRIBE conversationreview;")
#     columns = cursor.fetchall()
#     print("\n📌 Columns in 'conversationreview':")
#     for column in columns:
#         print(f"- {column[0]} ({column[1]})")



#     cursor.close()
#     connection.close()

# if __name__ == "__main__":
#     check_database()
