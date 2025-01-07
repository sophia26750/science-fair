import streamlit as st 

st.write("This is the main page of the website!")

code = '''def hello():
    print("Hello, dad!!")'''
st.code(code, language="python") 