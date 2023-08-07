from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain,RetrievalQA, ConversationalRetrievalChain
import streamlit as st
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
#https://python.langchain.com/docs/use_cases/question_answering/how_to/local_retrieval_qa

def LlamaBot():
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    n_gpu_layers = 1  # Metal set to 1 is enough.
    n_batch = 256  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.


    #load the embeddings from vector store
    st.write('Welcome to PDF bot')
    #for metal gpu use mps else cpu
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = LlamaCpp(
        model_path="./models/llama-2-7b-chat.ggmlv3.q4_K_M.bin",
        input={"temperature": 0.75, "max_length": 2000, "top_p": 1},
        #n_gpu_layers=n_gpu_layers,
        #n_batch=n_batch,
        n_ctx=2048,
        f16_kv=True,
        callback_manager=callback_manager,
        verbose=True)

    #prompt = PromptTemplate(template=custom_prompt_template,input_variables=['context','query'])

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    retriever= db.as_retriever(search_kwargs={"k": 3, "search_type": "similarity"})
    #memory is a mandatory parameter
    with st.spinner("loading model"):
        chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff", retriever=retriever,return_source_documents = False, verbose = False, memory=memory)
                                           #chain_type_kwargs={"prompt": prompt},
    st.write("model loaded")
    #chain = LLMChain(llm=llm, prompt=prompt)

    question = st.text_input("Ask questions about related your upload pdf file")
    #question = "What is a funnel transformer ?"
    #st.write(question)

    with st.spinner("triggering model with the question"):
        response = chain(({"question":question }))
        st.write(response["answer"])
        #st.write(response["source_documents"])
if __name__ == '__main__':
    LlamaBot()




