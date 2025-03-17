# Chatbot Conversation Classifier
## Overview
This FastAPI application classifies chatbot conversations as successful or unsuccessful using OpenAI's GPT-4o. It retrieves conversation data from a MySQL database, processes messages, and determines classification with a confidence score.

### Features
Retrieves user and bot messages from MySQL.
Cleans and formats messages (removing JSON, HTML).
Uses OpenAI GPT-4o for classification.
Provides structured responses via FastAPI.

###Set Up:
**Clone the repository:**
git clone <repo_url>
cd <repo_folder>

**Create a virtual environment:**
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

**Install dependencies:**
pip install -r requirements.txt

**Create a .env file and add:**
OPENAI_API_KEY=your_openai_api_key

**Running the API**
Start the FastAPI application:
uvicorn main:app --reload
