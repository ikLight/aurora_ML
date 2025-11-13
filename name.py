from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv('./.gitignore/.env')


##-----------------------------------------------------------------------------##
class Name:

    def __init__(self):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        #--------------------------------------#

    def extract_name(self, question, username_list):
        prompt = f"""This question pertains to a particular username. 
                     Question - {question}
                     List of usernames - {username_list}
                     Based on this info, output only the username that is being referenced in the question. Do not add any other text.
                  """
        response = self._client.chat.completions.create(model="gpt-5-nano-2025-08-07", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content

##-----------------------------------------------------------------------------##

## TEST
# username_list = ['Sophia Al-Farsi', 'Fatima El-Tahir', 'Armand Dupont',
#        'Hans Müller', 'Layla Kawaguchi', 'Amina Van Den Berg',
#        'Vikram Desai', "Lily O'Sullivan", 'Lorenzo Cavalli',
#        'Thiago Monteiro']
# question = "What are Amira’s favorite restaurants?"
# username = Name().extract_name(question, username_list)
# print(username)
    

        