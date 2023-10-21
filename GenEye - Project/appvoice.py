import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader,JSONLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.prompts import PromptTemplate
from datasets import load_dataset
from gtts import gTTS
import playsound
import tempfile
import os

# Add voice recognition support
import speech_recognition as sr

# load the pdf files from the path
loader = DirectoryLoader('data/',glob="*.pdf",loader_cls=PyPDFLoader)
documents = loader.load()

#split text into chunks
text_splitter  = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

#create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device':"cpu"})

#vectorstore
vector_store = FAISS.from_documents(text_chunks,embeddings)

custom_prompt_template = """[INST] <<SYS>>

You are a Mental Health  assistant, you will use the provided context to answer user questions.

Read the given context before answering questions and think step by step. If you can not answer a user question based on 

the provided context, inform the user. Do not use any other information for answering user. try to minimize the answer dont brief 

<</SYS>>

{context}

{question}

[/INST] """


prompt = PromptTemplate(template=custom_prompt_template,input_variables=['context','question'])



#create llm
llm = CTransformers(model="D:\python-projects\llama-2-7b-chat.ggmlv3.q8_0.bin",model_type="llama",streaming=True, 
                    callbacks=[StreamingStdOutCallbackHandler()],
                    config={'max_new_tokens':256,'temperature':0.6,'context_length':-1})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k":2}),
                                              memory=memory,
                                              combine_docs_chain_kwargs={'prompt':prompt})

st.title("Mental Health Chatbot 🧑🏽‍⚕️")
def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]


def simulate_button_click(user_input):
    """
    Simulate a button click by directly calling conversation_chat with the given user_input its actually used for speech recogonition and text to speech generation.
    """
    output = conversation_chat(user_input)

    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)

    # Convert the response to speech and play it
    tts = gTTS(output, lang='en')
    # Specify the path for the output audio file
    output_audio_path = "D:\\python-projects\\output_audio.mp3"
    
    # Remove the existing output audio file if it exists
    if os.path.exists(output_audio_path):
        os.remove(output_audio_path)

    # Save the new audio content to the output audio file
    tts.save(output_audio_path)
    
    # Play the new audio file
    playsound.playsound(output_audio_path, True)
    # temp_file = tempfile.NamedTemporaryFile(delete=False)
    # tts.save(temp_file.name)
    # st.audio(temp_file.name, format='audio/mp3')
    # playsound.playsound("D:\\python-projects\\output_audio.mp3", True)

    # # Remove the temporary file
    # temp_file.close()  # Close the temporary file
    # os.unlink(temp_file.name)  # Delete the temporary file

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        voice_recognition = st.checkbox("Start Voice Recognition")
        if voice_recognition:
            st.write("Listening... Speak your question")
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                audio = recognizer.listen(source)
            try:
                user_input = recognizer.recognize_google(audio)
                st.text_input("You said:", user_input)

                # Automatically submit the user's question after voice input
                simulate_button_click(user_input)

            except sr.UnknownValueError:
                st.write("Sorry, could not understand audio.")
            except sr.RequestError as e:
                st.write(f"Could not request results from Google Web Speech API service; {e}")
            

        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Mental Health", key='input')
            submit_button = st.form_submit_button(label='Submit')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# Rest of the code remains the same

# Initialize session state
initialize_session_state()
# Display chat history
display_chat_history()

