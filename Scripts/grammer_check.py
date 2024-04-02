import requests
from dotenv import load_dotenv
import os
load_dotenv()
sentence = input()
base_url = "https://api.textgears.com/grammar?key="+os.environ['GRAMMAR_API_KEY']+"&language=en-GB&ai=true"
# sentence = sentence.replace(".","!")
sentence = sentence.replace(" ","+")
r = requests.get(base_url+"&text="+sentence)
print(r.json())