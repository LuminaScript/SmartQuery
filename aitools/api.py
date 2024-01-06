from flask import Flask, request, jsonify
from text2sql import text2sql, execute_sql,sqlresult2text, text2sql_end2end, sql_agent, sql_explaination, text2sql_memory, execute_sql_memory, freechat_memory
from langchain.memory import ConversationBufferMemory
import json
import os


# Assuming all your provided functions and necessary imports are defined here
# ...
SQL_FAIL_MESSAGE = "SQL_ERROR"
def read_api_key(file_path):
    '''read the api key from the file
    :param file_path: the path of the file
    '''
    with open(file_path, 'r') as file:
        return file.read().strip()

REPLICATE_API_TOKE = read_api_key('../API_Key/REPLICATE_API_TOKEN.txt')
OPENAI_API_KEY = read_api_key('../API_Key/OPENAI_API_KEY.txt')

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKE
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


app = Flask(__name__)

memory=ConversationBufferMemory()
@app.route('/text2sql', methods=['POST'])
def handle_text2sql():
    data = request.get_json()
    model_name = data['model_name']
    db_name = data['db_name']
    question = data['question']
    response = text2sql(model_name, db_name, question)
    return jsonify(response)

@app.route('/execute_sql', methods=['POST'])
def handle_execute_sql():
    data = request.get_json()
    query = data['query']
    db_name = data['db_name']
    result = execute_sql(query, db_name)
    return jsonify({"result": result})

@app.route('/sqlresult2text', methods=['POST'])
def handle_sqlresult2text():
    data = request.get_json()
    model_name = data['model_name']
    db_name = data['db_name']
    question = data['question']
    sql_query = data['sql_query']
    sql_result = data['sql_result']
    response = sqlresult2text(model_name, db_name, question, sql_query, sql_result)
    return jsonify(response)

@app.route('/text2sql_end2end', methods=['POST'])
def handle_text2sql_end2end():
    data = request.get_json()
    model_name = data['model_name']
    db_name = data['db_name']
    question = data['question']
    response = text2sql_end2end(model_name, db_name, question)
    return jsonify(response)

@app.route('/sql_agent', methods=['POST'])
def handle_sql_agent():
    data = request.get_json()
    question = data['question']
    db_name = data.get('db_name', "Chinook")
    response = sql_agent(question, db_name)
    return jsonify({"response": response})

@app.route('/sql_explanation', methods=['POST'])
def handle_sql_explanation():
    data = request.get_json()
    model_name = data['model_name']
    db_name = data['db_name']
    question = data['question']
    sql_query = data['sql_query']
    sql_result = data['sql_result']
    response = sql_explaination(model_name, db_name, question, sql_query, sql_result)
    return jsonify(response)

@app.route('/text2sql_memory', methods=['POST'])
def handle_text2sql_memory():
    data = request.get_json()
    model_name = data['model_name']
    db_name = data['db_name']
    question = data['question']
    response = text2sql_memory(memory, model_name, db_name, question)
    return jsonify(response)

@app.route('/execute_sql_memory', methods=['POST'])
def handle_execute_sql_memory():
    data = request.get_json()
    query = data['query']
    db_name = data['db_name']
    result = execute_sql_memory(query, db_name, memory)
    return jsonify({"result": result})

@app.route('/freechat_memory', methods=['POST'])
def handle_freechat_memory():
    data = request.get_json()
    model_name = data['model_name']
    user_input = data['user_input']
    response = freechat_memory(memory, model_name, user_input)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
