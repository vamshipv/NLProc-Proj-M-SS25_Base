from sentence_transformers import SentenceTransformer
import faiss
import numpy as np



def add_document(book, query):
    with open(book+".txt", 'r') as file:
    # Read the entire content of the file into a string
        text = file.read()

    chunks = [text[i:i+200] for i in range(0, len(text), 200)]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    print("Check point 1")
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))
    print("Check point 2")
    # Search, this should be dynamic
    # query = "Who is always sad?"
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)
    retrieved_chunks = [chunks[i] for i in I[0]]
    return retrieved_chunks


book1 = "textSample"
book2 = "winnie_the_pooh"
print("Choose the following documents for the search query:")
print("Book 1",book1)
print("Book 2",book2)
choice = input("Enter 1 or 2: ")
choiceQuery = input("Enter your input query: ")
if choice == '1' and choiceQuery != "":
    chunk = add_document(book1, choiceQuery)
    print(chunk)
elif choice == '2' and choiceQuery != "":
    chunk = add_document(book2, choiceQuery)
    print(chunk)
else:
    print("Your choice selection for the book and query are wrong. Please try again")
