# Text2SQL

## Tool Overview

Text2SQL is an innovative tool developed on langchain for text to sql problems. 


### Included:
text to sql functions. (text2sql.py)
text to sql apis. (api.py)
playground for step by step testing. (Text2SQL.ipynb)
3 database and 4 examples for testing. 

### TechStack
langchain, flask, sqlite
supported models: llama2(local model) , Gpt3.5/4
supported databases: Chinook, nba_roster, sakila


### Environment Setup
1.	llama2 download
https://python.langchain.com/docs/integrations/llms/ollama

2.	Terminal execute: pip3 install openai
3.	Terminal execute: pip3 install langchain
4.	Download and unzip API_key.zip. Put folder API_key and codes(Text2SQL) in the same path.

See an Application built on it:
https://github.com/LuminaScript/SmartQuery

### Quick Start：
test functions:
comment the examples in text2sql.py and run python text2sql.py

test apis:
run python api.py and test using Postman.
