from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os
from data_extraction import Data
from name import Name
from vector_store import VectorStore
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re

load_dotenv('./.gitignore/.env')


##-----------------------------------------------------------------------------##

class Agent:

    def __init__(self, max_workers=4, use_vector_search=True, embedding_dim=512):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._data_extractor = Data()
        self._name_extractor = Name()
        self._data = None
        self._username_list = []
        self._user_data_cache = {}
        self._max_workers = max_workers
        self._use_vector_search = use_vector_search
        self._vector_store = VectorStore(
            max_workers=max_workers, 
            embedding_dim=embedding_dim
        ) if use_vector_search else None
        self._prompts = self._load_prompts()

    #--------------------------------------#

    def _load_prompts(self):
        """Load prompt templates from prompt_library.json"""
        try:
            with open('prompt_library.json', 'r') as f:
                return json.load(f)
        except:
            return {}

    #--------------------------------------#

    def _classify_question_type(self, question):
        """Classify question type to use appropriate prompt template"""
        question_lower = question.lower()
        
        # Preference/pattern questions
        if any(word in question_lower for word in ['favorite', 'prefer', 'like', 'usually', 'often', 'habit']):
            return 'preference_analysis'
        
        # Factual extraction questions
        if any(word in question_lower for word in ['when', 'where', 'what time', 'which', 'list', 'name']):
            return 'factual_extraction'
        
        return 'standard'

    #--------------------------------------#

    def _process_user_data(self, username_chunk):
        """Process a chunk of users in parallel - worker function"""
        chunk_cache = {}
        for username in username_chunk:
            user_df = self._data[self._data['user_name'] == username]
            # Pre-format context using vectorized operations
            messages = (user_df['timestamp'].astype(str) + '] ' + user_df['message']).tolist()
            chunk_cache[username] = messages
        return chunk_cache

    #--------------------------------------#

    def load_data(self, url):
        """Load data from the API endpoint and build vector index in parallel"""
        self._data = self._data_extractor.get_data(url)
        if self._data is not None:
            self._username_list = self._data['user_name'].unique().tolist()
            
            # Build vector index if enabled
            if self._use_vector_search:
                self._vector_store.build_index(self._data)
            
            # Split usernames into chunks for parallel processing (fallback cache)
            num_users = len(self._username_list)
            chunk_size = max(1, num_users // self._max_workers)
            username_chunks = [
                self._username_list[i:i + chunk_size] 
                for i in range(0, num_users, chunk_size)
            ]
            
            # Process chunks in parallel
            self._user_data_cache = {}
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    executor.submit(self._process_user_data, chunk): chunk 
                    for chunk in username_chunks
                }
                
                for future in as_completed(futures):
                    chunk_cache = future.result()
                    self._user_data_cache.update(chunk_cache)
        
        return self._data

    #--------------------------------------#

    def _prepare_context(self, username):
        """Prepare context from cached user data"""
        messages = self._user_data_cache.get(username)
        if messages is None or len(messages) == 0:
            return None
        
        context = f"User: {username}\n\nMessages:\n"
        context += "\n".join(f"- [{msg}" for msg in messages)
        
        return context

    #--------------------------------------#

    def _prepare_contexts_parallel(self, usernames):
        """Prepare contexts for multiple users in parallel"""
        contexts = {}
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_user = {
                executor.submit(self._prepare_context, username): username 
                for username in usernames
            }
            
            for future in as_completed(future_to_user):
                username = future_to_user[future]
                context = future.result()
                if context:
                    contexts[username] = context
        
        return contexts
    
    
    #--------------------------------------#
    
    def answer_question(self, question, top_k=20, use_reasoning=True):
        """Enhanced version with better prompts, reasoning, and structured output"""
        if self._data is None:
            return "Error: No data loaded. Please call load_data() first."
        
        # Extract username from question
        username = self._name_extractor.extract_name(question, self._username_list)
        
        # Use vector search if enabled
        if self._use_vector_search:
            # Search for relevant messages for this user
            relevant_results = self._vector_store.search(
                query=question, 
                username=username, 
                top_k=top_k
            )
            
            if not relevant_results:
                return f"No relevant data found for {username}."
            
            # Build enhanced context with relevance scores
            context_messages = []
            for i, result in enumerate(relevant_results, 1):
                timestamp = result['metadata']['timestamp']
                message = result['message']
                score = result['score']
                context_messages.append(f"{i}. [{timestamp}] {message} (relevance: {score:.2f})")
            
            context = "\n".join(context_messages)
        else:
            # Fallback to cached data
            context = self._prepare_context(username)
            if context is None:
                return f"No data found for {username}."
        
        # Classify question type and select appropriate prompt template
        question_type = self._classify_question_type(question)
        prompt_template = self._prompts.get('question_answering', {}).get(question_type, '')
        
        if prompt_template:
            prompt = prompt_template.format(
                username=username,
                context=context,
                question=question
            )
        else:
            # Fallback to enhanced default prompt
            prompt = f"""User: {username}

                        Messages (most relevant first):
                        {context}

                        Question: {question}

                        Answer briefly and directly. Only state facts from the messages."""
        
        messages = [{"role": "user", "content": prompt}]
        
        # Add reasoning step if enabled
        if use_reasoning:
            messages.insert(0, {
                "role": "system",
                "content": self._prompts.get('system_prompts', {}).get('analytical', 
                    "Provide concise, factual answers based on the data. Be brief.")
            })
        
        response = self._client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=messages
        )
        
        return response.choices[0].message.content
    
    #--------------------------------------#
    
    def answer_question_with_verification(self, question, top_k=20):
        """Answer with self-verification step to reduce hallucinations"""
        if self._data is None:
            return "Error: No data loaded. Please call load_data() first."
        
        # Get initial answer
        initial_answer = self.answer_question_fast(question, top_k=top_k, use_reasoning=True)
        
        # Extract username and context for verification
        username = self._name_extractor.extract_name(question, self._username_list)
        relevant_results = self._vector_store.search(query=question, username=username, top_k=top_k)
        
        context_summary = "\n".join([
            f"- {r['message']}" for r in relevant_results[:5]  # Top 5 for verification
        ])
        
        # Verification prompt
        verification_prompt = f"""Question: {question}

                                Top messages:
                                {context_summary}

                                Initial answer: {initial_answer}

                                Is this answer accurate and supported by the messages? If yes, return it as-is. If no, provide a brief corrected answer.

                                Answer:"""
        
        response = self._client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[{"role": "user", "content": verification_prompt}]
        )
        
        return response.choices[0].message.content
    

##-----------------------------------------------------------------------------##


## TEST
# API_ROUTE = "https://november7-730026606190.europe-west1.run.app/messages/"
# agent = Agent()
# agent.load_data(API_ROUTE)

# question = "What are Sophia's travel preferences?"
# answer = agent.answer_question(question)
# print(answer)
