import os
import getpass
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import base64
import os
from google import genai
from google.genai import types
import pypdfium2 as pdfium
import io 
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

with open('config.json', 'r') as f:
    config = json.load(f)

with open('prompt_extractor.txt', 'r') as f:
    prompt_for_extracting = f.read()

os.environ["GOOGLE_API_KEY"] = config.get("api_key", "")

embedding_model = config.get("embedding_model", "models/gemini-embedding-001")
embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

prompt_for_extracting = prompt_for_extracting

def b64_encode_image(image, format="PNG"):
    buf = io.BytesIO()
    image.save(buf, format=format)
    img_bytes = buf.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

def generate(image, prompt=prompt_for_extracting):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = config.get("text_extraction_model", "gemini-2.5-flash-lite")
    encoded_image = b64_encode_image(image)
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=base64.b64decode(
                        encoded_image
                    ),
                ),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=config.get("temperature", 0.2))

    extracted_data = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,)
    return extracted_data 

final_text = ""
pdf_path = config.get("pdf_path", "./")

for i in os.listdir(pdf_path):
    if i.endswith('.pdf'):
        pdf = pdfium.PdfDocument(i)
        for j in range(len(pdf)):
            page = pdf[j]
            image = page.render(scale=4).to_pil()
            data = generate(image=image, prompt=prompt_for_extracting)
            final_text += data.text
        #time.sleep(20)

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1948,
        chunk_overlap=100,
        length_function=len,
    )
texts = text_splitter.split_text(final_text)

db_directory = config.get("db_directory", "chroma_langchain_db")

#metadatas = [{"source": f"chunk_{i}"} for i in range(len(texts))]

vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    persist_directory=db_directory)