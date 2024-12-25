from flask import Flask, request, jsonify, render_template
from langchain_aws import BedrockLLM  # Updated import
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import boto3

# Flask app setup
app = Flask(__name__)

# Bedrock client setup
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="ap-south-1",
)

# Bedrock model setup
model_id = "mistral.mistral-7b-instruct-v0:2"
llm = BedrockLLM(  # Updated class name
    model_id=model_id,
    client=bedrock_client,
    model_kwargs={"temperature": 0.9}
)

# Chatbot function
def my_chatbot(language, user_text):
    prompt = PromptTemplate(
        input_variables=["language", "user_text"],
        template="You are a chatbot. You are in {language}.\n\n{user_text}"
    )
    bedrock_chain = LLMChain(llm=llm, prompt=prompt)
    response = bedrock_chain({'language': language, 'user_text': user_text})
    return response['text']

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    language = data.get('language')
    user_text = data.get('user_text')

    if not language or not user_text:
        return jsonify({"error": "Both language and user_text are required"}), 400

    try:
        response_text = my_chatbot(language, user_text)
        return jsonify({"response": response_text}) 
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start the server
if __name__ == '__main__':
    app.run(debug=True)
