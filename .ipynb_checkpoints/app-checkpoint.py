from flask import Flask, render_template, request, jsonify, redirect, url_for
from llama_index.legacy import SimpleDirectoryReader, StorageContext, ServiceContext
from llama_index.legacy.indices.vector_store import VectorStoreIndex
from llama_iris import IRISVectorStore
from llama_index.legacy.llms import OpenAI
from openai import OpenAI as OpenAIClient
import os
from dotenv import load_dotenv

# Load environment variables and set up OpenAI API key
load_dotenv(override=True)

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = '*********'

# Initialize Flask app
app = Flask(__name__)

# Mock user data (you can replace this with a database later)
user_data = {
    "name": "Kaden",
    "next_of_kin_name": "Ravi",
    "next_of_kin_contact": "81617843",
    "next_of_kin_address": "123 Main St, City, Country"
}

# Load documents and set up llama_index
documents = SimpleDirectoryReader("/home/jwongmun/Healthhack_2025/Healthhack_2025/data/aunty_ling").load_data()

# Set up the vector store
username = 'demo'
password = 'demo'
hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
port = '1972'
namespace = 'USER'
CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

vector_store = IRISVectorStore.from_params(
    connection_string=CONNECTION_STRING,
    table_name="paul_graham_essay",
    embed_dim=1536,  # OpenAI embedding dimension
)

# Initialize the LLM for llama_index
llm = OpenAI()

# Set up the service context
service_context = ServiceContext.from_defaults(
    llm=llm,  # Pass the LLM here
    embed_model="default"  # Use the default embedding model (OpenAI embeddings)
)

# Create the vector store index
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    service_context=service_context,  # Pass the service context here
    show_progress=True,
)

# Create the query engine
query_engine = index.as_query_engine()

# Initialize the OpenAI client
client = OpenAIClient()

# Define the Elderly Health Assistant prompt
system_prompt = """
You are an **Elderly Health Assistant**, a friendly and knowledgeable AI designed to help elderly individuals manage their health, well-being, and daily life. Your goal is to provide clear, empathetic, and actionable advice while maintaining a warm and supportive tone. Follow these guidelines:

1. **Tone and Style**:
   - Use simple, easy-to-understand language.
   - Be patient, empathetic, and supportive.
   - Avoid medical jargon unless explained clearly.

2. **Scope of Assistance**:
   - Provide general health tips and reminders (e.g., hydration, medication, exercise).
   - Offer guidance on managing chronic conditions (e.g., diabetes, arthritis, hypertension).
   - Suggest mental health and emotional well-being strategies (e.g., staying socially active, reducing stress).
   - Recommend lifestyle adjustments for better mobility, nutrition, and sleep.
   - Answer questions about common age-related health concerns.

3. **Safety and Boundaries**:
   - Always remind users to consult their doctor or healthcare provider for personalized medical advice.
   - Do not diagnose conditions or prescribe treatments.
   - Encourage users to seek emergency help if they describe symptoms of serious conditions (e.g., chest pain, severe dizziness).

4. **Examples of Questions You Can Answer**:
   - "What are some gentle exercises I can do to improve my mobility?"
   - "How can I remember to take my medications on time?"
   - "What foods should I eat to manage my blood sugar levels?"
   - "How can I improve my sleep quality?"
   - "What are some ways to stay socially active if I live alone?"

5. **Emergency Reminder**:
   - If the user describes symptoms like chest pain, difficulty breathing, or sudden confusion, respond with: "This sounds serious. Please contact your doctor or seek emergency medical help immediately."
"""

# Home page
@app.route('/')
def index():
    return render_template('index.html', user=user_data)

# Chatbot page
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html', user=user_data)

# Settings page
@app.route('/settings', methods=['GET', 'POST'])
def settings():
    global user_data
    if request.method == 'POST':
        # Update user data
        user_data["name"] = request.form.get("name")
        user_data["next_of_kin_name"] = request.form.get("next_of_kin_name")
        user_data["next_of_kin_contact"] = request.form.get("next_of_kin_contact")
        user_data["next_of_kin_address"] = request.form.get("next_of_kin_address")
        return redirect(url_for('index'))
    return render_template('settings.html', user=user_data)

# Upcoming Appointments page
@app.route('/appointments')
def appointments():
    return render_template('appointments.html', user=user_data)

# Chatbot API endpoint
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    bot_response = handle_chat(user_message)
    return jsonify({'response': bot_response})

# Chatbot logic
def handle_chat(message):
    message = message.lower()

    # Conditional reply for appointment-related questions
    if "next appointment" in message or "when is my appointment" in message:
        return "Your next appointment is on 31st March 2025 at 10:00 AM with the Physio Therapist at NUH West Wing #05-11."

    # Step 1: Retrieve relevant documents using llama_index
    retrieved_docs = query_engine.query(message)

    # Step 2: Pass the retrieved context to the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4",  # Use "gpt-4" or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": system_prompt},  # Set the LLM's role
            {"role": "user", "content": f"Context: {retrieved_docs}\n\nQuestion: {message}"}
        ],
    )

    # Return the final response from OpenAI
    return response.choices[0].message.content

if __name__ == '__main__':
    app.run(debug=True)