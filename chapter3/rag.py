import re
from typing import List

import pdfplumber
import requests
import tiktoken
from neo4j import GraphDatabase

driver = GraphDatabase.driver("neo4j://127.0.0.1:7687",
    auth=("neo4j", "qawsedRF123"),
    notifications_min_severity="OFF"
)

remote_pdf_url = "https://arxiv.org/pdf/1709.00666.pdf"
pdf_filename = "ch03-downloaded.pdf"

response = requests.get(remote_pdf_url)

if response.status_code == 200:
    with open(pdf_filename, "wb") as pdf_file:
        pdf_file.write(response.content)
else:
    print("Failed to download the PDF. Status code:", response.status_code)



text = ""

with pdfplumber.open(pdf_filename) as pdf:
    for page in pdf.pages:
        text += page.extract_text()

#########
#tokenize
#########

def split_text_by_titles(text):
    # A regular expression pattern for titles that
    # match lines starting with one or more digits, an optional uppercase letter,
    # followed by a dot, a space, and then up to 50 characters
    title_pattern = re.compile(r"(\n\d+[A-Z]?\. {1,3}.{0,60}\n)", re.DOTALL)
    titles = title_pattern.findall(text)
    # Split the text at these titles
    sections = re.split(title_pattern, text)
    sections_with_titles = []
    # Append the first section
    sections_with_titles.append(sections[0])
    # Iterate over the rest of sections
    for i in range(1, len(titles) + 1):
        section_text = sections[i * 2 - 1].strip() + "\n" + sections[i * 2].strip()
        sections_with_titles.append(section_text)
    return sections_with_titles

sections = split_text_by_titles(text)
print(f"Number of sections: {len(sections)}")

def num_tokens_from_string(string: str, model: str = "gpt-4") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens

for s in sections:
    print(num_tokens_from_string(s))
    # 154, 254, 4186, 570, 2703, 1441, 194, 600
#######
#
#######

def chunk_text(text, chunk_size, overlap, split_on_whitespace_only=True):
    chunks = []
    index = 0

    while index < len(text):
        if split_on_whitespace_only:
            prev_whitespace = 0
            left_index = index - overlap
            while left_index >= 0:
                if text[left_index] == " ":
                    prev_whitespace = left_index
                    break
                left_index -= 1
            next_whitespace = text.find(" ", index + chunk_size)
            if next_whitespace == -1:
                next_whitespace = len(text)
            chunk = text[prev_whitespace:next_whitespace].strip()
            chunks.append(chunk)
            index = next_whitespace + 1
        else:
            start = max(0, index - overlap + 1)
            end = min(index + chunk_size + overlap, len(text))
            chunk = text[start:end].strip()
            chunks.append(chunk)
            index += chunk_size

    return chunks


parent_chunks = []
for s in sections:
    parent_chunks.extend(chunk_text(s, 2000, 40))

#####################
#importing  in neo4j#
#####################

cypher_import_query = """
MERGE (pdf:PDF {id:$pdf_id})
MERGE (p:Parent {id:$pdf_id + '-' + $id})
SET p.text = $parent
MERGE (pdf)-[:HAS_PARENT]->(p)
WITH p, $children AS children, $embeddings as embeddings
UNWIND range(0, size(children) - 1) AS child_index
MERGE (c:Child {id: $pdf_id + '-' + $id + '-' + toString(child_index)})
SET c.text = children[child_index], c.embedding = embeddings[child_index]
MERGE (p)-[:HAS_CHILD]->(c);
"""


for i, chunk in enumerate(parent_chunks):
    child_chunks = chunk_text(chunk, 500, 20)
    embeddings = embed(child_chunks)
    # Add to neo4j
    driver.execute_query(
        cypher_import_query,
        id=str(i),
        pdf_id="1709.00666",
        parent=chunk,
        children=child_chunks,
        embeddings=embeddings,
    )

#some queries to check the new graph db:

# to check he graph:
MATCH p = (pdf:PDF)-[:HAS_PARENT]->(parent)-[:HAS_CHILD]->(child)
RETURN p;

MATCH (pdf:PDF)-[:HAS_PARENT]->(p:Parent)
RETURN p.id, p.text

MATCH (p:Parent)-[:HAS_CHILD]->(c:Child)
RETURN c.id, c.text, c.embedding
ORDER BY c.id

MATCH (pdf:PDF)-[:HAS_PARENT]->(p:Parent)-[:HAS_CHILD]->(c:Child)
RETURN pdf.id AS pdf,
       p.id AS parent,
       p.text AS parentText,
       c.id AS child,
       c.text AS child_text
ORDER BY parent, child;

# to check he graph:
MATCH p = (pdf:PDF)-[:HAS_PARENT]->(parent)-[:HAS_CHILD]->(child)
RETURN p;

#################################
#creating a vector index on child
# and querying the question######
##################################

driver.execute_query("""CREATE VECTOR INDEX parent IF NOT EXISTS
FOR (c:Child)
ON c.embedding""")


retrieval_query = """
CALL db.index.vector.queryNodes($index_name, $k * 4, $question_embedding)
YIELD node, score
MATCH (node)<-[:HAS_CHILD]-(parent)
WITH parent, max(score) AS score
RETURN parent.text AS text, score
ORDER BY score DESC
LIMIT toInteger($k)
"""

def parent_retrieval(question: str, k: int = 4) -> List[str]:
    question_embedding = embed([question])[0]

    similar_records, _, _ = driver.execute_query(
        retrieval_query,
        question_embedding=question_embedding,
        k=k,
        index_name="parent",
    )

    return [record["text"] for record in similar_records]

documents = parent_retrieval(
    "Who was the Einsten's collaborator on sound reproduction system?"
)

#######################
#back step prompting###
#######################

stepback_system_message = """
You are an expert at world knowledge. Your task is to step back
and paraphrase a question to a more generic step-back question, which
is easier to answer. Here are a few examples

"input": "Could the members of The Police perform lawful arrests?"
"output": "what can the members of The Police do?"

"input": "Jan Sindel’s was born in what country?"
"output": "what is Jan Sindel’s personal history?"
"""
def chat(messages):
    llm = ChatVertexAI(model="gemini-2.0-flash")
    return llm.invoke(messages)

def generate_stepback(question: str):
    user_message = f"""{question}"""
    step_back_question = chat(
        messages=[
            {"role": "system", "content": stepback_system_message},
            {"role": "user", "content": user_message},
        ]
    )
    return step_back_question

question = "Who was the Einsten's collaborator on sound reproduction system?"
step_back_question = generate_stepback(question)
print(f"Stepback results: {step_back_question}")

answer_system_message = "You're en Einstein expert, but can only use the provided documents to respond to the questions."


def generate_answer(question: str, documents: List[str]) -> str:
    user_message = f"""
    Use the following documents to answer the question that will follow:
    {documents}

    ---

    The question to answer using information only from the above documents: {question}
    """
    result = chat(
        messages=[
            {"role": "system", "content": answer_system_message},
            {"role": "user", "content": user_message},
        ]
    )
    print("Response:", result)

def rag_pipeline(question: str) -> str:
    stepback_prompt = generate_stepback(question)
    print(f"Stepback prompt: {stepback_prompt}")
    documents = parent_retrieval(stepback_prompt.content)
    print (documents)
    answer = generate_answer(question, documents)
    return answer

rag_pipeline("Who was the Einsten's collaborator on sound reproduction system?")

rag_pipeline("When was Einstein granted the patent for his blouse design?")
