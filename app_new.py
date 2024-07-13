# %%
import os
from dotenv import load_dotenv

#%%
load_dotenv('./.env')
os.environ.get('PINECONE_ENV')

# %%
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import gradio as gr

#%%
from langchain.schema import(
    AIMessage,
    HumanMessage,
    SystemMessage
)

#%%
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Carregando {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Carregando {file}')
        loader = Docx2txtLoader(file)
    else:
        print('Formato não suportado!')
        return None

    data = loader.load()
    return data

# Wikipedia Loader
def load_from_wikipedia(query, lang='pt', load_max_docs=2):
    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data



#%%
def chunk_data(data, chunk_size=1000):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks

# %%
data = load_document('docs/CLT.pdf')
chunks = chunk_data(data)


#%%
embeddings = OpenAIEmbeddings()


#%%
vector = embeddings.embed_query(chunks[0].page_content)

#%%
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key="xxxxxx")


#%%
index_name = "linuxtips"


#%%
pc.create_index(
    name=index_name,
    dimension=1536, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)


#%%
vector_store = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)


llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, max_tokens=1024)

template="""Assistente é uma IA jurídica que tira dúvidas.
    Assistente elabora repostas simplificadas, com base no contexto fornecido.
    Assistente fornece referências extraídas do contexto abaixo. Não gere links ou referência adicionais.
    Ao final da resposta exiba no formato de lista as referências extraídas.
    Caso não consiga encontrar no contexto abaixo ou caso a pergunta não esteja relacionada do contexto jurídico, 
    diga apenas 'Eu não sei!'

    Pergunta: {query}

    Contexto: {context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["query", "context"]
)

#%%

def search(query):
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    resp = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
    #resp = LLMChain(prompt=prompt, llm=llm)
    return resp.run(query=query, context=retriever)

with gr.Blocks(title="IA jurídica", theme=gr.themes.Soft()) as ui:
    gr.Markdown("# Sou uma IA que tem a CLT como base de conhecimento")
    query = gr.Textbox(label='Faça a sua pergunta:', placeholder="EX: como funcionam as férias do trabalhador?")
    text_output = gr.Textbox(label="Resposta")
    btn = gr.Button("Perguntar")
    btn.click(fn=search, inputs=query, outputs=[text_output])
ui.launch(debug=True, share=True)
# %%
