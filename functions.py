from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
import openai
import pprint
import json
import pandas as pd
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import re

import requests
import csv

import matplotlib.pyplot as plt
import io

load_dotenv(find_dotenv())

load_dotenv()

from dotenv import find_dotenv, load_dotenv

import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

from langchain.document_loaders import PyPDFLoader

from dotenv import load_dotenv

import os
import openai
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.memory import ConversationSummaryBufferMemory

from langchain.chains.summarize import load_summarize_chain

from langchain.document_loaders import DirectoryLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Laden Sie die Umgebungsvariablen aus der .env-Datei
load_dotenv()

from langchain.prompts import ChatPromptTemplate
from langchain import PromptTemplate, LLMChain

def parser(text):
 
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7)

    context = text.strip()

    email_schema = ResponseSchema(
        name="email_parser",
        description="extract the email id from the text. If required, strip and correct it in format like sample@xyz.com. Only provide these words. If no email id is present, return null@null.com",
    )
    subject_schema = ResponseSchema(
        name="content", description="Just extract the content removing email ids. Do not add any interpretation."
    )

    response_schemas = [email_schema, subject_schema]

    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()

    template = """
    Interprete the text and evaluate the text.
    email_parser: extract the email id from the text. Only provide these words. If no email id is present, return null@null.com. Use 1 line.
    content: Just extract the content removing email ids. Do not add any interpretation.

    text: {context}

    Just return the JSON, do not add ANYTHING, NO INTERPRETATION!
    {format_instructions}:"""

    #imprtant to have the format instructions in the template represented as {format_instructions}:"""

    #very important to note that the format instructions is the json format that consists of the output key and value pair. It could be multiple key value pairs. All the context with input variables should be written above that in the template.

    prompt  = PromptTemplate(
        input_variables=["context", "format_instructions"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt, output_key= "testi")
    response = chain.run({"context": context, "format_instructions": format_instructions})

    output_dict = parser.parse(response)
    return output_dict

def draft_email(user_input):
    
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7)
    
    #--- convert to UTF8
    
        # Function to convert a file to UTF-8 encoding
    def convert_to_utf8(input_file_path, output_file_path):
        with open(input_file_path, "r", encoding="CURRENT_ENCODING") as infile:
            content = infile.read()
        with open(output_file_path, "w", encoding="utf-8") as outfile:
            outfile.write(content)

    # Directory containing CSV files
    directory_path = "./shashi"

    # Iterate through CSV files in the directory and convert them to UTF-8
    import os
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".csv"):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(root, f"{file.split('.')[0]}_utf8.csv")
                convert_to_utf8(input_file_path, output_file_path)

    # Now, load the documents using the DirectoryLoader with the modified CSV files
    loader = DirectoryLoader(
        directory_path, glob="**/*_utf8.csv", loader_cls=CSVLoader, show_progress=True
    )
    docs = loader.load() 
        
    #---  

    # loader = DirectoryLoader(
    #     "./shashi", glob="**/*.csv", loader_cls=CSVLoader, show_progress=True
    # )
    # docs = loader.load()

    #textsplitter-----------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=2,
    )

    docs = text_splitter.split_documents(docs)
    # print(docs[3].page_content)
    #-----------------

    from langchain.embeddings import OpenAIEmbeddings
    openai_embeddings = OpenAIEmbeddings()

    from langchain.vectorstores.faiss import FAISS
    import pickle

    #Very important - db below is used for similarity search and not been used by agents in tools

    db = FAISS.from_documents(docs, openai_embeddings)
    
    import pickle

    with open("db.pkl", "wb") as f:
        pickle.dump(db, f)
        
    with open("db.pkl", "rb") as f:
        db = pickle.load(f)
        

    parser_output = parser(user_input)
    
    email = parser_output["email_parser"]
    
    content = parser_output["content"]
    
    docs = db.similarity_search(content, k=4)
    
    #----------------
   
    prompt_template = """You are a helpful assistant for our dental clinic.for any answer that you do no know, strictly say "DON'T KNOW" without adding any additional context or explaination. Answer all questions factually only.

    {context}

    Question: {question}
    Answer here:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    from langchain.chains import RetrievalQA

    chain_type_kwargs = {"prompt": PROMPT}

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
    )

    query = "list down all treatments that you offer"
    response = qa.run(query)

    #----------------


    # template = """
    # you are a pediatric dentist and you are writing a key features serial wise for following information: 

    # text: {context}
    # """    
    # map_prompt = """
    # Write a concise summary of the following:
    # "{text}"
    # CONCISE SUMMARY:
    # """
    
    # map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    
    # combine_prompt = """
    # You are a summarisation expert. Focus on maintaining a coherent flow and using proper grammar and language. Write a detailed summary of the following text:
    # "{text}"
    # SUMMARY:
    # """
    
    # combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    
    # summary_chain = load_summarize_chain(llm=llm,
    #                                  chain_type='map_reduce',
    #                                  map_prompt=map_prompt_template,
    #                                  combine_prompt=combine_prompt_template)

    # response = summary_chain.run({"input_documents": docs})

    return email, response