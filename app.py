

from flask import Flask, render_template, request, jsonify
from aitoolsnew.text2sql import (text2sql_memory, execute_sql_memory, 
                                 sqlresult2text, sql_agent, freechat_memory)
from langchain.memory import ConversationBufferMemory

import json
import os
app = Flask(__name__)

@app.route('/process_question', methods=['POST'])
def process_question():
    memory = ConversationBufferMemory(return_messages=True)
    
    # Extract data from request
    data = request.json
    model_name = data.get('model_name', 'gpt3')  # Default to 'gpt3' if not provided
    db_name = data.get('db_name', 'Chinook')     # Default to 'Chinook' if not provided
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    if question.startswith("@"):
        question = question[1:]
        sql_query = text2sql_memory(memory, model_name, db_name, question)
        sql_result = execute_sql_memory(sql_query, db_name, memory)
        response = sqlresult2text(model_name, db_name, question, sql_query, sql_result)
        return jsonify({
            "sql_query": sql_query,
            "sql_result": sql_result,
            "result_description": f"{response}"[9:-1]
        })

    elif question.startswith("#"):
        response = sql_agent(question)
        return jsonify({"response": response})

    else:
        response = freechat_memory(memory, model_name, question)
        return jsonify({"response": response})

@app.route('/')
@app.route('/home')
def index():
    return render_template("index.html")


@app.route('/chat')
def chat():
    return render_template("chat.html")

@app.route('/signup')
def signup():
    return render_template("signup.html")

@app.route('/api/generate_response', methods=['POST'])
def generate_response():
    user_message = request.json['message']

    # Call your API or perform any desired processing
    response_message = f"Received: {user_message}. This is a generated response."

    return jsonify({'message': response_message})


@app.route('/api/set_model', methods=['POST'])
def set_model():
    global selected_model
    selected_model = request.json['model']
    return jsonify({'message': f'Selected model set to {selected_model}'})


@app.route('/login')
def login():
    return render_template("login.html")



if __name__ == '__main__':
    app.run(debug=True)


