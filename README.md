## banking_chatbot
App is using **meta/llama-3.1-8b-instruct** (via NIM) to retrieve information from the **bank.pdf** containing mortgage rules, add it to the knowledge base, and use it in the chatbot to provide you with the best possible answer.

Nvidia NIM credits are needed to run the code. Signup at https://build.nvidia.com/explore/discover to get 1000 credits.

`Answers` folder provides examples of chatbot answers.

1. Install
`pip install streamlit  pypdf langchain faiss-cpu python-dotenv langchain-nvidia-ai-endpoints -U langchain-community transformers sentence-transformers.`

2. Run
`streamlit run bank.py`
