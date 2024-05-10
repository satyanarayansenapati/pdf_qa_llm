import streamlit as st
import os

from LLM_QA_PDF import logger
from LLM_QA_PDF.component.main import load_pdf


def user_response(pdf_file_name, user_input):
    logger.info(f'user selected pdf file : {pdf_file_name}')
    logger.info(f'user input : {user_input}')
    a = load_pdf(user_input, pdf_file_name)
    st.write(f'{a}')
    #st.write(f'user input : {user_input}')

st.title("PDF Question Answer")
st.markdown("Upload and select a pdf file to query")

#creating directory
pdf_file = 'pdf_directory' 
os.makedirs(pdf_file, exist_ok=True)

#uploading pdf file
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

#saving the file
if uploaded_file is not None:
    with open(os.path.join(pdf_file, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully")


#loading all the pdf files in the directory
pdf_files = os.listdir(pdf_file)

#selecting the pdf file

pdf_file_name = st.selectbox("Select a PDF file", pdf_files)

bt1 = st.button("Continue")
if pdf_file_name is not None and bt1:
    st.write(f'Selected PDF file is {pdf_file_name}')
    logger.info(f'Selected PDF file is {pdf_file_name}')


user_input = st.text_input("Enter your question")


if user_input and pdf_file_name:
        
    logger.info(f'user input is {user_input}')
    try:
        user_response(pdf_file_name, user_input)
        
    except Exception as e:
        logger.error(f'{e}')
        st.error("Error occured !!!")

