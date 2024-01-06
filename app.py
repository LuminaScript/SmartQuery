from flask import Flask, render_template, request, jsonify
import os
import io
from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Replicate
from contextlib import redirect_stdout

# Chain to query with memory
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory #use sliding window memory to store the k latest
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableLambda

from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.llms.openai import OpenAI
app = Flask(__name__)
memory=ConversationBufferMemory()
# Global variable to store the selected model (initialize with a default model)


SQL_FAIL_MESSAGE = "SQL_ERROR"
def read_api_key(file_path):
    '''read the api key from the file
    :param file_path: the path of the file
    '''
    with open(file_path, 'r') as file:
        return file.read().strip()

REPLICATE_API_TOKE = read_api_key('API_Key/REPLICATE_API_TOKEN.txt')
OPENAI_API_KEY = read_api_key('API_Key/OPENAI_API_KEY.txt')

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKE
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def model_select(model_name):
    if model_name == "llama2_chat":
        return ChatOllama(model="llama2:13b-chat")
    elif model_name == "llama2_code":
        return ChatOllama(model="codellama:7b-instruct")
    elif model_name == "gpt4":
        return ChatOpenAI(model='gpt-4-0613')
    elif model_name == "gpt3":
        return ChatOpenAI(model='gpt-3.5-turbo-1106')
    else:
        return ChatOpenAI(model='gpt-3.5-turbo-1106')

def init(model_name,db_name):
    model = model_select(model_name)

    # Replicate API
    replicate_id = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
    llama2_chat_replicate = Replicate(
    model=replicate_id, model_kwargs={"temperature": 0.00, "max_length": 500, "top_p": 1}
    )
    # Database

    db = SQLDatabase.from_uri(f"sqlite:///./{db_name}.db", sample_rows_in_table_info=0)
    return model,db

def text2sql(model_name,db_name,question):
    model,db = init(model_name,db_name)
    # Using Closure desgin pattern to pass the db to the model
    def get_schema(_):
        return db.get_table_info()
    template = """Based on the table schema below, write a SQLite query that would answer the user's question. Only output the SQL query:
    {schema}

    Question: {question}
    SQL Query:"""
    prompt = ChatPromptTemplate.from_template(template)
    sql_response = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | model.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )
    return sql_response.invoke({"question": question})  

def execute_sql(query,db_name):
    db = SQLDatabase.from_uri(f"sqlite:///./{db_name}.db", sample_rows_in_table_info=0)
    update_action_list = ['UPDATE','ADD','DELETE','DROP','MODIFY','INSERT']
    try:
        if any(item in query for item in update_action_list)==False:# no update actions
            result = db.run(query)
            if result:
                return result
            else:
                return "No results found."
        else: return 'Finished' #update actions return no result but "Finished"
    except Exception as e:
        error_message = str(e)
        print(SQL_FAIL_MESSAGE,error_message)
        return SQL_FAIL_MESSAGE

def sqlresult2text(model_name,db_name,question,sql_query,sql_result):
    # Using Closure desgin pattern to pass the db to the model
    model,db = init(model_name,db_name)
    def get_schema(_):
        return db.get_table_info()
    ## To natural language
    
    template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
    {schema}
    Question: {question}
    SQL Query: {query}
    SQL Response: {response}"""


    prompt_response = ChatPromptTemplate.from_template(template)


    text_response = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt_response
        | model
    )

    # execute the model 
    return   text_response.invoke({"question": question,"query":sql_query,"response":sql_result})


def text2sql_end2end(model_name,db_name,question):
    model,db = init(model_name,db_name)
    # Prompts
    # 
    def get_schema(_):
        return db.get_table_info()

    def run_query(query):
        print("running query\n", query)
        try:
            result = db.run(query)
            if result:
                print("successfully run query")
                return result
            else:
                return "No results found."
        except Exception as e:
            error_message = str(e)
            print(SQL_FAIL_MESSAGE)
            return SQL_FAIL_MESSAGE

    template = """Based on the table schema below, write a SQLite query that would answer the user's question. Only output the SQL query:  
    {schema}

    Question: {question}
    SQL Query:"""
    prompt = ChatPromptTemplate.from_template(template)

    sql_response = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | model.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )

    ## To natural language
    template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
    {schema}

    Question: {question}
    SQL Query: {query}
    SQL Response: {response}"""


    prompt_response = ChatPromptTemplate.from_template(template)


    full_chain = (
        RunnablePassthrough.assign(query=sql_response).assign(
            schema=get_schema,
            response=lambda x: run_query(x["query"]),
        )
        | prompt_response
        | model
    )

    # execute the model 
    return full_chain.invoke({"question": question})

def sql_agent(question,db_name="Chinook"):
    db=SQLDatabase.from_uri(f"sqlite:///./{db_name}.db")
    toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model='gpt-3.5-turbo-1106',temperature=0))
    agent_executor = create_sql_agent(
    llm=ChatOpenAI(model='gpt-3.5-turbo-1106',temperature=0),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
    buffer = io.StringIO()

    # Redirect stdout to the buffer
    with redirect_stdout(buffer):
        agent_executor.run(question)

    output = buffer.getvalue()
    return output
 


def sql_explaination(model_name,db_name,question,sql_query,sql_result):
    # Using Closure desgin pattern to pass the db to the model
    model,db = init(model_name,db_name)
    def get_schema(_):
        return db.get_table_info()
    ## To natural language
    
    template = """Based on the table schema below, question, sql query, and sql response, explain the sql query step by step:
    {schema}
    Question: {question}
    SQL Query: {query}
    SQL Response: {response}"""


    prompt_response = ChatPromptTemplate.from_template(template)


    text_response = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt_response
        | model
    )

    # execute the model 
    return   text_response.invoke({"question": question,"query":sql_query,"response":sql_result})

def text2sql_memory(memory,model_name,db_name,question):
    model,db = init(model_name,db_name)
    # Using Closure desgin pattern to pass the db to the model and response to memory
    def save(input_output):
        output = {"output": input_output.pop("output")}
        memory.save_context(input_output, output)
        return output["output"]
    def get_schema(_):
        return db.get_table_info()
    template = """Based on the table schema below, write a SQLite query that would answer the user's question. Only output the SQL query:
    {schema}

    Question: {question}
    SQL Query:"""
    prompt = ChatPromptTemplate.from_template(template)
    sql_response = (
        RunnablePassthrough.assign(schema=get_schema,
        history=RunnableLambda(lambda x: memory.load_memory_variables(x)["history"]))
        | prompt
        | model.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )
    sql_response_memory = RunnablePassthrough.assign(output=sql_response) | save
    return sql_response_memory.invoke({"question": question}) 


def execute_sql_memory(query,db_name,memory):
    db = SQLDatabase.from_uri(f"sqlite:///./{db_name}.db", sample_rows_in_table_info=0)
    try:
        result = db.run(query)
        if result:
            output = f"sql result is: {result}"
            memory.chat_memory.add_ai_message(output)
            return result
        else:
            return "No results found."
    except Exception as e:
        error_message = str(e)
        print(SQL_FAIL_MESSAGE,error_message)
        return SQL_FAIL_MESSAGE

def freechat_memory(memory,model_name,user_input):
    model = model_select(model_name)
    template = """Your name is EduSmartQuery Bot. You are a chatbot mentor which is good at sql and willing to educate others. You are chatting with a student who is learning sql. 

    Previous conversation:
    {history}

    New human question: {question}
    Response:"""
    prompt = PromptTemplate.from_template(template)
    # Notice that we need to align the `memory_key`
    conversation = LLMChain(
        llm=model,
        prompt=prompt,
        verbose=False,
        memory=memory
    )
    
    return conversation({"question": user_input})['text']

@app.route('/text2sql', methods=['POST'])
def handle_text2sql():
    data = request.get_json()
    model_name = data['model_name']
    db_name = data['db_name']
    question = data['question']
    sql_query = text2sql(model_name, db_name, question)
    response = {
        "query": sql_query,
        "message": "SQL query generated successfully"
    }

    # Log the response for debugging
    print("response: ")
    print(response)

    # Return the response as JSON
    return jsonify(response)
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

