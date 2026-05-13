# Medical Q&A Chatbot

I built this project as part of my AI engineering internship work. 
The idea was simple — instead of an LLM just making up medical answers, 
what if it only answered from real medical documents? That's what RAG does.

## What it does
You type a medical question, it searches through 11,274 NIH medical documents 
and gives you an answer based only on what's actually in those documents. 
No hallucination.

## How I built it
I used LangChain to connect everything together. The documents get converted 
into vectors using HuggingFace embeddings and stored in a FAISS database. 
When you ask a question, it finds the most relevant chunks and passes them 
to LLaMA 3.3 70B (via Groq) to generate the answer.

## Tech used
- Python
- LangChain
- FAISS (vector database)
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Groq API (LLaMA 3.3 70B)
- Streamlit
- MedQuAD dataset (NIH)

## How to run it

Clone the repo
git clone https://github.com/Sameera674/Medical-RAG-Chatbot.git

Install packages
pip install -r requirements.txt

Add your Groq API key in a .env file
GROQ_API_KEY=your-key-here

Build the database
python Ingest.py

Run the app
streamlit run app.py

## Questions you can ask
- What causes asthma?
- What are symptoms of diabetes?
- How is depression treated?
- What is arthritis?

## What I learned
Honestly the hardest part was getting all the LangChain versions to work 
together. Once that was sorted, connecting the retriever to the LLM was 
straightforward. FAISS was much easier to set up than ChromaDB on Windows.