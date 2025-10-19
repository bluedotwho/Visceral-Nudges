import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# -----------------------------
# Load Hugging Face token from Streamlit secrets
# -----------------------------
HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

if not HF_TOKEN:
    st.error("HUGGINGFACEHUB_API_TOKEN not found in Streamlit secrets")
    st.stop()


llm_endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
)


model = ChatHuggingFace(llm=llm_endpoint)

# Define the Pydantic model
class EmotionOutput(BaseModel):
    emotion: str = Field(description="One of ['Anger', 'Sad', 'Happy', 'Love']")
    confidence: float = Field(description="Confidence score between 0 and 1")

# Create the parser
parser = PydanticOutputParser(pydantic_object=EmotionOutput)

template = """
You are an AI model for sentiment analysis. 
Classify the input text into one of the emotions: Anger, Sad, Happy, Love.
Also provide a confidence score (0 to 1) with key "confidence".

Text: {text}

Respond in strict JSON format compatible with the schema.
"""

prompt = PromptTemplate(template=template, input_variables=["text"])

chain = prompt | model | parser 

# 
# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Emotion Analysis App ❤️")
st.write("Type a text below and the AI will classify its emotion.")

user_input = st.text_input("Enter your text here:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        with st.spinner("Analyzing..."):
            try:
                result = chain.invoke({"text": user_input})
                st.success("Analysis Complete!")
                st.json(result.dict())  # Display structured JSON
                st.write(f"**Emotion:** {result.emotion}")
                st.write(f"**Confidence:** {result.confidence:.2f}")
            except Exception as e:
                st.error(f"Error: {e}")
