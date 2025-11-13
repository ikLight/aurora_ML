from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import faiss

load_dotenv('./.gitignore/.env')


##-----------------------------------------------------------------------------##

class VectorStore:

    def __init__(self, max_workers=4, embedding_dim=512, embedding_model="text-embedding-3-small"):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._index = None
        self._messages = []
        self._metadata = []
        self._max_workers = max_workers
        self._embedding_dim = embedding_dim
        self._embedding_model = embedding_model
        self._username_to_ids = {}  

    #--------------------------------------#

    def _get_embedding(self, text):
        """Generate embedding for a single text"""
        response = self._client.embeddings.create(
            model=self._embedding_model,
            input=text,
            dimensions=self._embedding_dim
        )
        return response.data[0].embedding

    #--------------------------------------#

    def _get_embeddings_batch(self, texts):
        """Generate embeddings for multiple texts in one API call"""
        response = self._client.embeddings.create(
            model=self._embedding_model,
            input=texts,
            dimensions=self._embedding_dim
        )
        return [item.embedding for item in response.data]

    #--------------------------------------#

    def _process_batch_embeddings(self, batch_data):
        """Process a batch of messages and generate embeddings in bulk"""
        # Extract all messages from the batch
        messages = [row['message'] for idx, row in batch_data]
        
        # Get embeddings in bulk (up to 2048 texts per API call)
        batch_size = 2048
        all_embeddings = []
        
        for i in range(0, len(messages), batch_size):
            chunk = messages[i:i + batch_size]
            embeddings = self._get_embeddings_batch(chunk)
            all_embeddings.extend(embeddings)
        
        # Build results
        batch_results = []
        for i, (idx, row) in enumerate(batch_data):
            batch_results.append({
                'embedding': all_embeddings[i],
                'message': row['message'],
                'metadata': {
                    'user_name': row['user_name'],
                    'timestamp': str(row['timestamp']),
                    'user_id': row['user_id'],
                    'id': row['id']
                }
            })
        return batch_results

    #--------------------------------------#

    def build_index(self, dataframe):
        """Build FAISS vector index from dataframe in parallel with batched embeddings"""
        print("Building FAISS vector index...")
        
        # Use larger batches since we're now batching embedding requests
        # Each worker will process ~500-1000 messages with batched API calls
        batch_size = max(500, len(dataframe) // self._max_workers)
        batches = []
        
        for i in range(0, len(dataframe), batch_size):
            batch = [(idx, row) for idx, row in dataframe.iloc[i:i + batch_size].iterrows()]
            batches.append(batch)
        
        print(f"Processing {len(dataframe)} messages in {len(batches)} batches with {self._max_workers} workers...")
        
        # Process batches in parallel
        all_results = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(self._process_batch_embeddings, batch): i 
                for i, batch in enumerate(batches)
            }
            
            total_batches = len(batches)
            for completed, future in enumerate(as_completed(futures), 1):
                batch_results = future.result()
                all_results.extend(batch_results)
                print(f"Completed {completed}/{total_batches} batches ({len(all_results)} messages processed)")
        
        # Store results and build FAISS index
        embeddings_list = []
        for idx, result in enumerate(all_results):
            embeddings_list.append(result['embedding'])
            self._messages.append(result['message'])
            self._metadata.append(result['metadata'])
            
            # Build username -> indices mapping
            username = result['metadata']['user_name']
            if username not in self._username_to_ids:
                self._username_to_ids[username] = []
            self._username_to_ids[username].append(idx)
        
        # Convert to numpy array and normalize for cosine similarity
        embeddings_array = np.array(embeddings_list, dtype='float32')
        faiss.normalize_L2(embeddings_array)  # Normalize for cosine similarity
        
        # Create FAISS index (using IndexFlatIP for inner product = cosine similarity with normalized vectors)
        self._index = faiss.IndexFlatIP(self._embedding_dim)
        self._index.add(embeddings_array)
        
        print(f"FAISS index built with {self._index.ntotal} entries")

    #--------------------------------------#

    def search(self, query, username=None, top_k=5):
        """Search for relevant messages using FAISS"""
        if self._index is None or self._index.ntotal == 0:
            return []
        
        # Get query embedding and normalize
        query_embedding = np.array([self._get_embedding(query)], dtype='float32')
        faiss.normalize_L2(query_embedding)
        
        # Filter by username if provided
        if username:
            user_indices = self._username_to_ids.get(username, [])
            if not user_indices:
                return []
            
            # Create a subset index for this user
            user_embeddings = np.array([
                self._index.reconstruct(int(idx)) for idx in user_indices
            ], dtype='float32')
            
            # Create temporary index and search
            k = min(top_k, len(user_indices))
            temp_index = faiss.IndexFlatIP(self._embedding_dim)
            temp_index.add(user_embeddings)
            distances, indices = temp_index.search(query_embedding, k)
            
            # Map back to original indices and build results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(user_indices):
                    original_idx = user_indices[idx]
                    results.append({
                        'message': self._messages[original_idx],
                        'metadata': self._metadata[original_idx],
                        'score': float(distances[0][i])
                    })
        else:
            # Search across all messages
            k = min(top_k, self._index.ntotal)
            distances, indices = self._index.search(query_embedding, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                results.append({
                    'message': self._messages[idx],
                    'metadata': self._metadata[idx],
                    'score': float(distances[0][i])
                })
        
        return results

##-----------------------------------------------------------------------------##
