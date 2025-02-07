from src.helper import *
import streamlit


def main():
  streamlit.title("Information Retrival from URL - RAG MODEL")
  url = streamlit.text_input("Enter URL")
  btn = streamlit.button("Submit")
  if btn:
    docs = url_call(url)
    db = db_store(docs)
    query = streamlit.text_input("Enter your question")
    q_btn = streamlit.button("Answer")
    if q_btn: 
      result = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
      streamlit.write(result.invoke(query))


if __name__ == "__main__":
  main()


