from LLM_QA_PDF import logger
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

'''Part 1: pass the path and name of the PDF file'''


# necessary import
from langchain_community.document_loaders import PyPDFLoader


query = "Enter your query"

def load_pdf(query:str,path_of_file = 'pdf_directory/short_story.pdf'):

    logger.info('-----------Part 1: pass the path and name of the PDF file: BEGINS-----------')
    #creating an instance of it
    pdf_file_path = 'pdf_directory/'+ path_of_file
    loader = PyPDFLoader(pdf_file_path)
    logger.info(f"PDF file from {pdf_file_path} has been loaded")

    #loading the data
    documents = loader.load()

    '''-----------------------Document processing------------------------'''
    logger.info('Document processing started')
    # splitting the pdf into chunks
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    #creating an instance
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    #splitting the data
    texts = text_splitter.split_documents(documents)
    logger.info(f'Document splited into chunks')
    logger.info('Document processing finished')
    logger.info('-----------Part 1: pass the path and name of the PDF file: ENDS -----------')
      

    
    '''Part 2: pass the path and name of the PDF file'''

    logger.info('-----------Part 2: pass the path and name of the PDF file: BEGINS-----------')

    '''----------------------- Embedding------------------------'''
    logger.info('embedding started') 


    #creating an instance
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings':True},
        google_api_key = os.environ.get('google_api_key')
    )
    logger.info('API Key for embeddings is fetched')  
    

    #creating db
    db = FAISS.from_documents(texts, embeddings)

    logger.info('embedding finished')

    # creating a retriever for querying answers from the db
    retriever = db.as_retriever(search_type="similarity",search_kwargs={"k":2})

    logger.info('retriever created')

    '''-----------------------LLM------------------------'''
    logger.info('LLM initialization started')

    #importing
    
    #creating an instance
    llm = GoogleGenerativeAI(model="models/text-bison-001",google_api_key = os.environ.get('GOOGLE_API_KEY'))

    logger.info('LLM initialization finished')

    


    '''------------------------Prompts------------------------'''
    logger.info('Phase : Prompt, status : Begin')
    # prompt
    

    prompt_template="""
    Use the following piece of context to answer the question asked.
    Please try to provide the answer only based on the context.
    If you don't find the answer from the context, then say "The answer isn't present in the text".

    {context}
    Question:{question}"""

    prompt = PromptTemplate(
        input_variables=["context","question"],
        template=prompt_template
    )

    logger.info('Phase : Prompt, status : End')

    '''------------------------Q & A------------------------'''
    logger.info('Retriever + LLM Creation')



    retrievalQA=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt":prompt}
    )

    logger.info('Retriever + LLM Creation finished')

    

    logger.info('-----------Part 3: Q & A: BEGINS-----------')

    # Q & A
    logger.info('Q & A started')
    
    
    logger.info('Invoking retriever + LLM instance')
    # Call the QA chain with our query.
    result = retrievalQA.invoke({"query": query})
    logger.info('output delivered successfully')


    return result['result']