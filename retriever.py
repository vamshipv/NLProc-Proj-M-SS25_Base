import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json


class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")


    def split_text(self,text,split_len):
        chunks = [text[i:i+split_len] for i in range(0, len(text), split_len)]
        return chunks
    
    def add_documents(self,path,split_len):
        with open(path, 'r') as file:
            text = file.read()
        self.splitted_text=self.split_text(text,split_len)
        embeddings = self.model.encode(self.splitted_text,)
        print(embeddings)
        self.index = faiss.IndexFlatL2(embeddings[0].shape[0])
        self.index.add(np.array(embeddings))

    def query(self,query_text,k):
        query_embedding = self.model.encode([query_text])
        D, I = self.index.search(np.array(query_embedding), k)
        self.retrieved_chunks = [self.splitted_text[i] for i in I[0]]
        return self.retrieved_chunks
    
    def save(self, index_filename,splittext_filename):
        faiss.write_index(self.index, index_filename)

        with open(splittext_filename, 'w') as f:
            json.dump(self.splitted_text, f)
    
    def load(self, index_filename,splittext_filename):
        self.index = faiss.read_index(index_filename)
        with open(splittext_filename, 'r') as f:
            self.splitted_text = json.load(f)



def main():
    retriever = Retriever()
    document_file = "/Users/dechammacg/Documents/NLPPro/NLProc-Proj-M-SS25/class_1_example/winnie_the_pooh.txt"
    neighbour_size = 2
    split_len = 100

    # Extract base name without extension
    base_name = os.path.splitext(os.path.basename(document_file))[0]

    # Construct dynamic filenames
    index_file = f"{base_name}_faiss.index"
    subtext_file = f"{base_name}_subtexts.json"

    #Load the index and subtexts file if they already exist
    if os.path.exists(index_file) and os.path.exists(subtext_file):
        # different chunk size
        # if chunks
        print("Loading saved index and subtexts...")
        retriever.load(index_file,subtext_file)
    else:
        print("Index not found — creating a new one...")
        retriever.add_documents(document_file,split_len)
        retriever.save(index_file,subtext_file)

    # Query the retriever
    query_text = input("Enter your input query: ")
    results = retriever.query(query_text, neighbour_size)
    print(results)
    
if __name__ == "__main__":
    main()
