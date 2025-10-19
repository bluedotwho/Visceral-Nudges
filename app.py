import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import requests

# -----------------------------
# Load secrets
# -----------------------------
HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
AIO_USERNAME = st.secrets["AIO_USERNAME"]
AIO_KEY = st.secrets["AIO_KEY"]
AIO_FEED_NAME = st.secrets["AIO_FEED_NAME"]

# -----------------------------
# Validate secrets
# -----------------------------
if not HF_TOKEN or not AIO_USERNAME or not AIO_KEY:
    st.error("One or more required secrets are missing. Please check your Streamlit secrets.")
    st.stop()

# -----------------------------
# Configure Hugging Face endpoint
# -----------------------------
llm_endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
)

model = ChatHuggingFace(llm=llm_endpoint)

# -----------------------------
# Define structured output
# -----------------------------
class EmotionOutput(BaseModel):
    emotion: str = Field(description="One of ['Anger', 'Sad', 'Happy', 'Love', 'Neutral']")
    confidence: float = Field(description="Confidence score between 0 and 1")

parser = PydanticOutputParser(pydantic_object=EmotionOutput)

template = """
You are an AI model for sentiment analysis. 
Classify the input text into one of the emotions: Anger, Sad, Happy, Love, Neutral.
Also provide a confidence score (0 to 1) with key "confidence".

Text: {text}

Respond in strict JSON format compatible with the schema.
"""

prompt = PromptTemplate(template=template, input_variables=["text"])
chain = prompt | model | parser

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("💬 Emotion Detector + ESP32 Bridge via Adafruit IO ☁️")
st.write("Type a sentence, and this app will analyze its emotion and send it to your ESP32 through Adafruit IO.")

user_input = st.text_input("Enter your text here:")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text!")
    else:
        with st.spinner("Analyzing emotion..."):
            try:
                # Run emotion detection
                result = chain.invoke({"text": user_input})
                st.success("✅ Analysis Complete!")

                st.write(f"**Emotion:** {result.emotion}")
                st.write(f"**Confidence:** {result.confidence:.2f}")

                # -----------------------------
                # ☁️ Send Emotion to Adafruit IO
                # -----------------------------
                try:
                    headers = {"X-AIO-Key": AIO_KEY, "Content-Type": "application/json"}
                    url = f"https://io.adafruit.com/api/v2/{AIO_USERNAME}/feeds/{AIO_FEED_NAME}/data"
                    data = {"value": result.emotion}

                    response = requests.post(url, headers=headers, json=data)

                    if response.status_code in [200, 201]:
                        st.success(f"🚀 Sent '{result.emotion}' to Adafruit IO successfully!")
                    else:
                        st.warning(f"⚠️ Adafruit IO responded with {response.status_code}: {response.text}")

                except Exception as e:
                    st.error(f"Error sending to Adafruit IO: {e}")

            except Exception as e:
                st.error(f"Error analyzing text: {e}")
