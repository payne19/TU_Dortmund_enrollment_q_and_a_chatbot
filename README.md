---

# 🎓 TU Dortmund Enrollment Chatbot

A **Streamlit-based chatbot** designed to handle **Q\&A queries related to TU Dortmund enrollment issues**.
Built using a **Retrieval-Augmented Generation (RAG) pipeline** with **Gemini models** for embeddings and summarization.

---

## ✨ Features

* 🧠 **RAG Approach** – Accurate answers from enrollment-related documents.
* 🔎 **Gemini Embeddings (001)** – For semantic search and retrieval.
* 📑 **PDF Support** – Extracts information from university documents.
* 📝 **Gemini 2.5 Flash Lite** – For query summarization and response generation.
* 🗂 **ChromaDB as Vector Store** – Stores embeddings efficiently.
* 🎨 **Streamlit UI** – Clean, intuitive interface for students.
* 🌍 **Deployment** – Works locally or via a free hosted version.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/payne19/TU_Dortmund_enrollment_q-a_chatbot.git
cd TU_Dortmund_enrollment_q-a_chatbot
```

### 2. Unzip the ChromaDB

```bash
unzip chroma_langchain_db.zip -d chroma_langchain_db
```

### 3. Install the requirements file

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Edit `config.json` and add your Gemini API key:

```json
{
  "GEMINI_API_KEY": "your_api_key_here"
}
```

### 5. Run the App Locally

```bash
streamlit run tu_dortmund_chat.py
```

### 6. Try the Deployed Version

If you don’t want to set up locally, use the free hosted version:
👉 [Chat with the Bot](https://chat.com)

---

## 🏗 Tech Stack

* **Frontend:** Streamlit
* **LLM & Embeddings:** Gemini API
* **Vector DB:** ChromaDB
* **Database:** SQLite
* **RAG Framework:** Custom pipeline

---

## 📖 Example Use Cases

* What documents are required for TU Dortmund enrollment?
* How do I get my student ID card?
* Deadlines for enrollment and re-registration.
* Fees and payment methods.

---

---

## 📜 License

MIT License – Free to use and modify.

---
