import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. SET CACHE PATHS BEFORE IMPORTING AI LIBRARIES
os.environ["HF_HOME"] = "/home/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/huggingface_cache"
os.makedirs("/home/huggingface_cache", exist_ok=True)

# --- UI Configuration ---
st.set_page_config(
    page_title="AI Underwriter Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


def analyze_risk(parsed_guidlines, parsed_application_form):
    prompt = f"""<guidelines>{parsed_guidlines}</guidelines> <application_form>{parsed_application_form}</application_form>."""

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided an insurance underwriting guidelines (delimited with XML tags) <guidelines></guidelines> and an insurance application form (delimited with XML tags) <application_form></application_form>. Can you provide some information about the relevant risks on the application following the guidelines ?."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    relevant_risks = []
    for choice in response.choices:
        relevant_risks.append(choice.message.content)

    return relevant_risks


@st.cache_data
@st.cache_resource
def extract_data(file, file_name):
    file_bytes = file.getvalue()
    file_path = f"temp_{file_name}.pdf"
    
    with open(file_path, "wb") as temp_file:
        temp_file.write(file_bytes)

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    hf_token = os.environ.get("HF_TOKEN")

    if file_name == "underwriting_guidlines":
        # 1. Use the lightweight, math-based splitter (No AI download required)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200, length_function=len)
        guidelines_docs = text_splitter.split_documents(pages)
        
        # 2. Azure now only needs to download the Summarizer
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", token=hf_token)
        
        summarized_texts = []
        # 3. Iterate cleanly through the LangChain document objects
        for doc in guidelines_docs:
            summarized_chunk = summarizer(doc.page_content, max_length=200, min_length=50, do_sample=False)
            summarized_texts.append(summarized_chunk[0]["summary_text"])
            
        text_content = "\n\n".join(summarized_texts)
    
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50, length_function=len)
        docs = text_splitter.split_documents(pages)
        text_content = "\n\n".join([doc.page_content for doc in docs])

    if os.path.exists(file_path):
        os.remove(file_path)

    return text_content


def main():
    st.markdown("<h1 style='text-align: center;'>Insurance Risk Analysis Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6B7280;'>Upload the underwriting guidelines and application form to instantly generate a risk profile.</p>", unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Guidelines")
        uploaded_underwriting_guidelines = st.file_uploader('Upload Underwriting Guidelines', type="pdf", help='Must be a standard PDF document')
        
    with col2:
        st.subheader("📝 Application")
        uploaded_application_form = st.file_uploader('Upload Application Form', type="pdf", help='Must be a standard PDF document')
        
    st.write("") 

    _, col_btn, _ = st.columns([1, 1, 1])
    with col_btn:
        analyze_btn = st.button('Analyze Risk Profile', use_container_width=True, type="primary")

    st.divider()
        
    if analyze_btn:
        if uploaded_underwriting_guidelines and uploaded_application_form:       
            with st.status("Initializing AI Models (This takes 1-2 minutes the very first time)...", expanded=True) as status:
                
                st.write("Extracting and summarizing data using local ML models...")
                parsed_guidelines = extract_data(uploaded_underwriting_guidelines, "underwriting_guidlines")
                parsed_application_form = extract_data(uploaded_application_form, "application_form")
                
                st.write("Cross-referencing risks with OpenAI...")
                relevant_risks = analyze_risk(parsed_guidelines, parsed_application_form)
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                
            st.success("Risk factors identified successfully.")
            st.subheader("Key Findings")
            
            for i, risk in enumerate(relevant_risks):
                with st.expander(f"Risk Factor {i+1}", expanded=True):
                    st.info(risk)
        else:
            st.warning("⚠️ Please upload both the guidelines and the application form to proceed.")

if __name__ == '__main__':
    main()
