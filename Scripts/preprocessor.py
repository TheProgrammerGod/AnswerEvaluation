from keybert._model import KeyBERT
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import nltk
import re
import string
import torch

# rake = Rake()
# tensroflow hub module for Universal sentence Encoder 
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
#@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

embed = hub.load(module_url)

def get_features(texts):
    if type(texts) is str:
        texts = [texts]
    #tf.run([tf.global_variables_initializer(), tf.tables_initializer()])
    return embed(tf.constant(texts))

#read the question and answer data from the file
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

def getMarks(question):
    s = question[(question.rindex('[')+1):(question.rindex(']'))]
    return float(s)

def listToString(list):
    str=" "
    return(str.join(list))

def preprocess(text):
    words = word_tokenize(text)
    words = [word for word in words if word not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    filtered_text = [w for w in words if not w in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in filtered_text]
    return listToString(words)

def keywordsExtract(text):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text)
    keywords = [k[0] for k in keywords]
    return keywords

# def cosine_similarity(model, answer):
#     rvector = set(model).union(set(answer))
#     l1=[]
#     l2=[]
#     for w in rvector: 
#         if w in set(model): l1.append(1) 
#         else: l1.append(0) 
#         if w in set(answer): l2.append(1) 
#         else: l2.append(0)

#     c = 0
#     for i in range(len(rvector)): 
#         c+= l1[i]*l2[i]

#     cosine = c / float((sum(l1)*sum(l2))**0.5)
#     return cosine 

def cosine_similarity(v1, v2):
    # a = np.array(v1)
    # b = np.array(v2)
    # mag1 = np.linalg.norm(a)
    # mag2 = np.linalg.norm(b)
    # if (not mag1) or (not mag2):
    #     return 0
    #return np.dot(a, b) / (mag1 * mag2)
    #cosi = torch.nn.CosineSimilarity(dim=0)
    #return cosi(v1, v2)
    #return tf.keras.losses.cosine_similarity(v1,v2 )
    # tf.convert_to_tensor(v1)
    # tf.convert_to_tensor(v2)
    # print(type(v1))
    # cos = tf.nn.CosineSimilarity(dim=1)
    # return cos(v1,v2)
    t1 = tf.nn.l2_normalize(v1, axis=1)
    t2 = tf.nn.l2_normalize(v2, axis=1)
    cosine = tf.reduce_sum(tf.multiply(t1, t2), axis=1)
    clip_cosine = tf.clip_by_value(cosine, -1.0, 1.0)
    scores = 1.0 - tf.acos(clip_cosine) / np.pi
    return scores

def test_similarity(text1, text2):
    vec1 = get_features(text1)
    vec2 = get_features(text2)
    #print(vec1.shape)
    # tens_1 = torch.tensor(vec1)
    # tens_2 = torch.tensor(vec2)
    return cosine_similarity(vec1, vec2)

def keyword_Matching(model, answer):
    score = 0
    numKeywords = len(model)
    for item in model:
        if item in answer:
            score += 1
    score = score / numKeywords
    return score

if __name__ == '__main__':
    question = read_data('Data\\Q1\\question.txt')[0]
    marks = getMarks(question)
    question = question[:(question.rindex('['))]
    model = read_data('Data\\Q1\\model.txt')[0].lower()
    model_keywords = keywordsExtract(model)
    #print(model_keywords)
    model = preprocess(model)
    file = open("Data\\Q1\\dataset.csv","a")
    for i in range(1,4):
        answer = read_data('Data\\Q1\\answer'+str(i)+'.txt')[0].lower()
        #file.write("\n" + question + "," + answer + "," + marks + ",")
        answer_keywords = keywordsExtract(answer)
        answer = preprocess(answer)
        keyword_score = keyword_Matching(model_keywords, answer_keywords) * 0.2 * marks
        qst_score = test_similarity(model, answer) * 0.6 * marks
        