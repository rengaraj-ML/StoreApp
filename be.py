from example import example
from datetime import datetime
import pandas as pd 
# agent will directly create query and run the query in DB
from langchain.agents import create_sql_agent 
# Simple chain to create the SQL statements, it doesn't execute the query
from langchain.chains import create_sql_query_chain 
# to execute the query
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool 
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.vectorstores import Chroma
from langchain.prompts import SemanticSimilarityExampleSelector
# Prompt input for MYSQL
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
# Create the prompt template for creating the prompt for mysqlprompt
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
# to create the tools to be used by agent
from langchain.agents import Tool

# create the agent prompts
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Huggingface embeddings using Langchain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Load Env parameters
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, text, URL


def config():
    load_dotenv()  # load env parameters
    llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")   # create LLM
    #llm = OpenAI(temperature=0.5)   # create LLM
    return llm

# Setting up URL parameter to connect to MySQL Database
def get_db_chain(question):
    db_user="root"
    db_password="root"
    db_host="localhost"
    db_name="retail"
    
    # create LLM
    llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")
    # Initialize SQL DB using Langchain
    db = SQLDatabase.from_uri(f"mysql://{db_user}:{db_password}@{db_host}/{db_name}")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    # create the list with only values and ready to be vectorized 
    to_vectorize = [" ".join(example.values()) for example in example] # use join to convert Dict to list
    # Setup the Chroma database and vectorize
    vectorstore = Chroma.from_texts(to_vectorize, embedding=embeddings, metadatas=example)
    # Based on the user question, convert them to vector and take the similar looking vectors from Chroma DB
    example_selector = SemanticSimilarityExampleSelector(
                        vectorstore = vectorstore,
                        k=2)
    example_prompt = PromptTemplate(
                        input_variables=["Question", "SQLQuery", "SQLResult", "Answer",],
                        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\n?Answer: {Answer}",)
    few_shot_prompt = FewShotPromptTemplate( 
                        example_selector=example_selector,                # Hey LLM, if you dont know refer the examples giving in vector DB
                        example_prompt=example_prompt,                    # This is the Prompt template we have created        
                        prefix=_mysql_prompt,                             # This is prefix of the prompt
                        suffix=PROMPT_SUFFIX,                             # This is suffix of the prompt
                        input_variables=["input", "table_info", "top_k"], # variables used in forming the prompt to LLM
                        )
    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    response = chain.invoke(question)
    return response
    # Call the LLM with the question and the fewshotprompt
    # write_query = create_sql_query_chain(llm=llm,db=db, prompt=few_shot_prompt)
    #print(write_query)
    # Execute the Query using QuerySQLDataBaseTool 
    #execute_query = QuerySQLDataBaseTool(db=db)
    # Chain to combine write SQL and Execute SQL 
    #chain = write_query | execute_query | llm
    #response = chain.invoke("Question")
def get_store_address(store):
    url_object = URL.create(
    "mysql",
    username="root",
    password="root",  # plain (unescaped) text
    host="localhost",
    database="retail",
    )
    engine = create_engine(url_object)
    #connect to engine
    connection = engine.connect()
    sql_query = "SELECT STORE_NUMBER, STORE_ADDRESS FROM STORES WHERE STORE_NUMBER = " + store
    df = pd.read_sql(sql_query, con=engine)
    response = df.to_string()
    return response
def outreach_sms_message(outreach_input):
        # create LLM
    llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo", verbose=True)
    prompt = ChatPromptTemplate.from_template("You are a expert in writing a text message for appointment setup with less than 35 words." 
                                              "With {outreach_input}, generate a text message for appointment to be sent to customer")
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    response = chain.invoke({"outreach_input": outreach_input})
    return response

#if __name__ == "__main__":
#    chain = get_db_chain()
#    print(chain.run("List of all sales transactions for Trevor Nelson in June 2020"))
# Setting up URL parameter to connect to MySQL Database
