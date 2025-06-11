import os
import streamlit as st

from constants import groq_api_key  

# LangChain Imports
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# Set Groq API Key
os.environ["GROQ_API_KEY"] = groq_api_key

# Streamlit UI
st.title('Celebrity Search Results')
input_text = st.text_input("Search the topic you want")

# Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born?"
)

third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events that happened around {dob} in the world."
)

# Memory Buffers
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

# Initialize Groq LLM
llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama3-70b-8192",  # Supported and powerful
    temperature=0.8
)


# Chains
chain = LLMChain(llm=llm, prompt=first_input_prompt, output_key='person', memory=person_memory, verbose=True)
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, output_key='dob', memory=dob_memory, verbose=True)
chain3 = LLMChain(llm=llm, prompt=third_input_prompt, output_key='description', memory=descr_memory, verbose=True)

# Parent Chain
parent_chain = SequentialChain(
    chains=[chain, chain2, chain3],
    input_variables=['name'],
    output_variables=['person', 'dob', 'description'],
    verbose=True
)

# Execute if user provides input
if input_text:
    result = parent_chain({'name': input_text})

    st.subheader("Results")
    st.write(result)

    with st.expander('Person Info'):
        st.info(person_memory.buffer)

    with st.expander('Birth Info'):
        st.info(dob_memory.buffer)

    with st.expander('Major Events'):
        st.info(descr_memory.buffer)
