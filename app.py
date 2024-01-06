from ansi2html import Ansi2HTMLConverter

from flask import Flask, render_template, request, jsonify
from text2sql import (text2sql_memory, execute_sql_memory, 
                                 sqlresult2text, sql_agent, freechat_memory)
from langchain.memory import ConversationBufferMemory

import json
import os
app = Flask(__name__)
memory=ConversationBufferMemory()
def convert_newlines_to_html(text):
    text=f"{text}"
    return text.replace("\n", "<br>")

@app.route('/process_question', methods=['POST'])
def process_question():
    # memory = ConversationBufferMemory(return_messages=True)
    
    # Extract data from request
    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    if question.startswith("@"):
        question=question[1:]#remove @
        sqlfromtext = text2sql_memory(memory, "gpt3", "Chinook", question)
        print("AI response:", sqlfromtext)
        sql_result = execute_sql_memory(sqlfromtext, "Chinook", memory)
        print("SQL result:", sql_result)
        result_description = sqlresult2text("gpt3", "Chinook", question, sqlfromtext, sql_result)
        # print("AI response:", result_description)
        sqlfromtext = convert_newlines_to_html(sqlfromtext)
        sql_result = convert_newlines_to_html(sql_result)
        return jsonify({
            "Query": sqlfromtext,
            "Result": sql_result,
            "Description": f"{result_description}"[9:-1]
        })

    elif question.startswith("#"):
        question=question[1:]#remove #
        response = sql_agent(question,"Chinook") 
        conv = Ansi2HTMLConverter()
        response = conv.convert(response)
        return jsonify({
            "Query": response,
            "Result": "null",
            "Description": "null"
        })

    else:
        response = freechat_memory(memory, "gpt3", question)
        response = convert_newlines_to_html(response)
        return jsonify({
            "Query": response,
            "Result": "null",
            "Description": "null"
        })

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

# @app.route('/api/generate_response', methods=['POST'])
# def generate_response():
#     user_message = request.json['message']

#     # Call your API or perform any desired processing
#     response_message = f"Received: {user_message}. This is a generated response."

#     return jsonify({'message': response_message})


# @app.route('/api/set_model', methods=['POST'])
# def set_model():
#     global selected_model
#     selected_model = request.json['model']
#     return jsonify({'message': f'Selected model set to {selected_model}'})


@app.route('/login')
def login():
    return render_template("login.html")



if __name__ == '__main__':
    app.run(debug=True, port=8000)


