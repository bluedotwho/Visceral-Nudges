import streamlit as st
import requests
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# ================================
# Hugging Face + Model Setup
# ================================
HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

llm_endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
)
model = ChatHuggingFace(llm=llm_endpoint)

class EmotionOutput(BaseModel):
    emotion: str = Field(description="One of ['Anger', 'Sad', 'Happy', 'Love', 'Neutral']")
    confidence: float = Field(description="Confidence score between 0 and 1")

parser = PydanticOutputParser(pydantic_object=EmotionOutput)

prompt = PromptTemplate(
    template="""
    You are an AI model for sentiment analysis. 
    Classify the input text into one of the emotions: Anger, Sad, Happy, Love, Neutral.
    Also provide a confidence score (0 to 1) with key "confidence".

    Text: {text}

    Respond in strict JSON format compatible with the schema.
    """,
    input_variables=["text"],
)
chain = prompt | model | parser

# ================================
# Streamlit UI
# ================================
st.title("Emotion Detector 💫 (Connected to ESP32)")
st.write("Type text and send the detected emotion to Adafruit IO → ESP32.")

user_input = st.text_input("Enter your text:")

# ================================
# Adafruit IO Setup
# ================================
AIO_USERNAME = "bluedotwho"
AIO_KEY = st.secrets["ADAFRUIT_IO_KEY"]  # Store your aio key in Streamlit secrets!
feed_url = f"https://io.adafruit.com/api/v2/{AIO_USERNAME}/feeds/emotion/data"

# ================================
# Main Logic
# ================================
if st.button("Analyze and Send"):
    if not user_input.strip():
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing..."):
            try:
                result = chain.invoke({"text": user_input})
                emotion = result.emotion
                conf = result.confidence

                st.success("✅ Analysis Complete!")
                st.write(f"**Emotion:** {emotion}")
                st.write(f"**Confidence:** {conf:.2f}")

                # Send to Adafruit IO feed
                headers = {"X-AIO-Key": AIO_KEY, "Content-Type": "application/json"}
                payload = {"value": emotion}

                res = requests.post(feed_url, json=payload, headers=headers)

                if res.status_code == 200 or res.status_code == 201:
                    st.success(f"🚀 Sent '{emotion}' to Adafruit IO successfully!")
                else:
                    st.error(f"⚠️ Failed to send data to Adafruit IO. Status: {res.status_code}")
                    st.text(res.text)

            except Exception as e:
                st.error(f"Error: {e}")
