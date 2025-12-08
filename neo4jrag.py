import requests
from neo4j import GraphDatabase
from langchain_google_vertexai import ChatVertexAI
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
import os

load_dotenv()  

remote_pdf_url = "https://arxiv.org/pdf/1709.00666.pdf"
pdf_filename = "ch02-downloaded.pdf"

response = requests.get(remote_pdf_url)

if response.status_code == 200:
    with open(pdf_filename, "wb") as pdf_file:
        pdf_file.write(response.content)
else:
    print("Failed to download the PDF. Status code:", response.status_code)



import pdfplumber

text = ""
remote_pdf_url = "https://arxiv.org/pdf/1709.00666.pdf"
pdf_filename = "ch02-downloaded.pdf"
with pdfplumber.open(pdf_filename) as pdf:
    for page in pdf.pages:
        text += page.extract_text()

text

from utils import chunk_text

chunks = chunk_text(text, 500, 40)
print(len(chunks))
print(chunks[0])
print (chunks)


def embed(texts: list[str]):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedder.embed_documents(texts)

embeddings = embed(chunks)


# driver to connect to neo4j
driver = GraphDatabase.driver(
    os.environ.get("NEO4J_URI"),
    auth=(os.environ.get("NEO4J_USERNAME"), os.environ.get("NEO4J_PASSWORD")),
    notifications_min_severity="OFF"
)


#Create an index in neo4j:
# this creates an index named pdf for all the nodes called c labled Chunk
# and each node has an attribute called embeddings.
# this Chunk and embeddings is different then the chunks and embeddings
# varibale I have
driver.execute_query("""CREATE VECTOR INDEX pdf IF NOT EXISTS
FOR (c:Chunk)
ON c.embedding""")




# Now inject the data

cypher_query = '''
WITH $chunks as chunks, range(0, size($chunks)) AS index
UNWIND index AS i
WITH i, chunks[i] AS chunk, $embeddings[i] AS embedding
MERGE (c:Chunk {index: i})
SET c.text = chunk, c.embedding = embedding
'''

driver.execute_query(cypher_query, chunks=chunks, embeddings=embeddings)

records, _, _ = driver.execute_query("MATCH (c:Chunk) WHERE c.index = 0 RETURN c.embedding, c.text")

print(records[0]["c.text"][0:30])
print(records[0]["c.embedding"][0:3])

################
#Embed qurstion#
################

question = "At what time was Einstein really interested in experimental works?"
question_embedding = embed([question])[0]

###################
#Query vector index
###################

query = '''
CALL db.index.vector.queryNodes('pdf', $k, $question_embedding) YIELD node AS hits, score
RETURN hits.text AS text, score, hits.index AS index
'''
similar_records, _, _ = driver.execute_query(query, question_embedding=question_embedding, k=4)

for record in similar_records:
    print(record["text"])
    print(record["score"], record["index"])
    print("======")


#############
#LLM#########
#############

system_message = "You're en Einstein expert, but can only use the provided documents to respond to the questions."

user_message = f"""
Use the following documents to answer the question that will follow:
{[doc["text"] for doc in similar_records]}

---

The question to answer using information only from the above documents: {question}
"""
 
print("Question:", question)

from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(
    model="gemini-2.0-flash",
    temperature=1.0,
)

stream = llm.stream([
    ("system", system_message),
    ("user", user_message),
])

for chunk in stream:
    print(chunk.content, end="")


#####################################
#Hybrid search#######################
#####################################

# This will create an index FOR each node c:Chunk meaning each node which has lable Chunk
# on EACH c.text property or attribute
# this will populate an index which is a datastructure with c:Chunk and its feild c.text
try :
    driver.execute_query(f"CREATE FULLTEXT INDEX ftPdfChunk FOR (c:Chunk) ON EACH [c.text]")
except:
    print("Fulltext Index already exists")


hybrid_query = '''
CALL {
    // vector index
    CALL db.index.vector.queryNodes('pdf', $k, $question_embedding) YIELD node, score
    WITH collect({node:node, score:score}) AS nodes, max(score) AS max
    UNWIND nodes AS n
    // We use 0 as min
    RETURN n.node AS node, (n.score / max) AS score
    UNION
    // keyword index
    CALL db.index.fulltext.queryNodes('ftPdfChunk', $question, {limit: $k})
    YIELD node, score
    WITH collect({node:node, score:score}) AS nodes, max(score) AS max
    UNWIND nodes AS n
    // We use 0 as min
    RETURN n.node AS node, (n.score / max) AS score
}
// dedup
WITH node, max(score) AS score ORDER BY score DESC LIMIT $k
RETURN node, score
'''

similar_hybrid_records, _, _ = driver.execute_query(hybrid_query, question_embedding=question_embedding, question=question, k=4)

for record in similar_hybrid_records:
    print(record["node"]["text"])
    print(record["score"], record["node"]["system_message = "You're en Einstein expert, but can only use the provided documents to respond to the questions."

user_message = f"""
Use the following documents to answer the question that will follow:
{[doc['node']["text"] for doc in similar_hybrid_records]}

---

The question to answer using information only from the above documents: {question}
"""
 
print("Question:", question)

from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(
    model="gemini-2.0-flash",
    temperature=1.0,
)

stream = llm.stream([
    ("system", system_message),
    ("user", user_message),
])

for chunk in stream:
    print(chunk.content, end="")
index"])
    print("======")
##########################
##LLM with hybrid search##
##########################


system_message = "You're en Einstein expert, but can only use the provided documents to respond to the questions."

user_message = f"""
Use the following documents to answer the question that will follow:
{[doc['node']["text"] for doc in similar_hybrid_records]}

---

The question to answer using information only from the above documents: {question}
"""
 
print("Question:", question)

from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(
    model="gemini-2.0-flash",
    temperature=1.0,
)

stream = llm.stream([
    ("system", system_message),
    ("user", user_message),
])

for chunk in stream:
    print(chunk.content, end="")
