import streamlit as st
import pandas as pd
from preprocessor import evaluate

def write_to_file(filepath, data):
    with open(filepath, 'w') as f:
        f.write(data)

st.title('Subjective Answer Evaluation System')
question = st.file_uploader('Choose Question File',type=['txt'])
model = st.file_uploader('Choose Model Answer File',type=['txt'])
answers = st.file_uploader('Choose Answer Files',type=['txt'],accept_multiple_files=True)
button = st.button('Evaluate')
if button :
    if question is not None and model is not None and answers is not None:
        write_to_file('Data\\question.txt', question.getvalue().decode('utf-8'))
        write_to_file('Data\\model.txt', model.getvalue().decode('utf-8'))
        for i, answer in enumerate(answers):
            write_to_file(f'Data\\answer{i+1}.txt', answer.getvalue().decode('utf-8'))
        evaluate(len(answers))
        df = pd.read_csv('Data\\dataset.csv')
        st.write(df.head())
        st.download_button("Download CSV",data=df.to_csv().encode('utf-8'),file_name = 'marks.csv',mime='text/csv')
    else:
        st.error('Please upload all the files')