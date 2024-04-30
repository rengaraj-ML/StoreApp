from langchain.tools import DuckDuckGoSearchRun
from langchain.agents.tools import tool
from langchain import OpenAI
from langchain.agents import Tool, load_tools, initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# Streamlit UI Callback
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMMathChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, URL
from htmlTemplates import css, bot_template, user_template

import openai
import os
import time

from be import config, get_db_chain, outreach_sms_message, get_store_address
from PIL import Image
def conversation_agent(question):
    llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")   # create LLM
    # search = DuckDuckGoSearchRun()
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)
    tools = [
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful when you need to answer questions with math"
        )
    ]
    #########   CREATING ALL THE TOOLS FOR THE AGENT TO USE #####################
    # Create the final SMS message
    outreach_sms_message_tool = Tool(
        name="Outreach SMS Message",
        func=outreach_sms_message,
        description="Create an outreach SMS message for the customer. Pass both user input and the Store Address as ONE SINGLE INPUT STRING.  Use this Tool only to create an outreach SMS or Text message.  At the end always include the Store Address for appointment confirmation messages"
    )
    #tools.append(outreach_sms_message_tool)
    # Creating a Query Tool - to generate SQL statements and query the database
    get_db_chain_tool = Tool(
        name='Query Generation Tool',
        func=get_db_chain,
        description="ONLY use this tool for query generation and to fetch any information.  Use this to Create MYSQL Query with the ORIGINAL User Input question to pull customer, store, product information. MySQL database connections are established within this tool.  Use this tool first"
                "During SQL Query Generation, make sure the SELECT list of columns are in GROUP BY clause"
                "Use this to get the store address from the database"
    )
    # create the tool for finding the store details
    get_store_address_tool = Tool(
        name="Get Store Address",
        func=get_store_address,
        description="Use this tool with store number to get the store address.  INPUT to this tool is Store number.  Do not use this tool if you don't have Store number as input"
    ) 
    #tools.append(get_db_chain_tool)
    # List all the tools for the agent to use 
    tools = [get_db_chain_tool, get_store_address_tool, outreach_sms_message_tool]
    conversational_agent = initialize_agent(
        agent="conversational-react-description",
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=10,
        memory=st.session_state.memory,
        handle_parsing_errors=True
    )
    response = conversational_agent.invoke(question)
    return response 

def main():
    img = Image.open('assist_logo.jpg')
   # user_avatar = Image.open('renga_profile.jpg')
   # ai_avatar = Image.open('Designer.png')
    load_dotenv()  # load env parameters
    st.set_page_config(page_title="Assist", page_icon=img)
    st.write(css, unsafe_allow_html=True)
    # Logo and image next to each other with a space column separating them out for rendering in small devices
    st.title(':blue[Assist] Store Associates')
    with st.sidebar:
        st.image('assist_logo.jpg', width=120)
        st.sidebar.header("Assist App for Store Associates")
        st.write("Assist store associates to get information on Customers, Stores, Product, Sales Analytics, Inventory Management and help with customer outreach")
        st.write(" ")
        st.write("Tasks I can help with:")
        st.write("a. Extract Data/info")
        st.write("b. Outreach message ")
        st.write("c. Send Text to Customers")
        st.write("d. Search websites and look up Product prices & other info")
        st.write("e. Generate charts for greater visualization")

    if "chat_history" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")
    # ini chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_question = st.chat_input("Type your question")
    if user_question:
            
            assistant_response = conversation_agent(user_question)
            st.session_state.chat_history = assistant_response['chat_history']
            chistory = assistant_response["chat_history"]
            # Process the chat history 
            messages = chistory.split("Human: ")
            l = len(messages)       
            # Step 3: Print the output
            for i in range(1, l):
                response = messages[i].strip()
                response = response.split("AI:")
            # Print the Human message from history
                with st.chat_message("human"):
                    st.markdown(response[0])
            # Print the AI message from history
                with st.chat_message("ai"):
                    st.markdown(response[1])
            #Print the last question from user
            with st.chat_message("human"):
               st.markdown(user_question)
            #st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
            # Print the last answer from user
            assistant_response_output = assistant_response["output"]
            with st.chat_message("ai"):
                st.write(assistant_response_output)
            #st.write(bot_template.replace("{{MSG}}", assistant_response_output), unsafe_allow_html=True)


if __name__== '__main__':
    main()

