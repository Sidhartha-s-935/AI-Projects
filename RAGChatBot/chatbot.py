from langchain_ollama import ChatOllama
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

document_path = "test.txt"  
loader = TextLoader(document_path)

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = loader.load()
chunks = text_splitter.split_documents(documents)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  
vector_store = FAISS.from_documents(chunks, embedding_model)
llm = ChatOllama(model="llama3.2:1b", temperature=0, num_predict=512)
retriever = vector_store.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
)
prompt_template = "Context: {context} Question: {question} Answer:"
prompt = PromptTemplate(input_variables=["context", "question"],template = prompt_template)
def chatbot_conversation():
    conversation_history = ""
    while True:
        user_input = input("Speak: ")
        relevant_docs = retriever.get_relevant_documents(conversation_history)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        response = qa_chain({"context": context, "query": user_input})
        print(f"Listen: {response['result']}")
chatbot_conversation()

