import os

import tiktoken
from neo4j import GraphDatabase
from langchain_google_vertexai import ChatVertexAI
from dotenv import load_dotenv

load_dotenv()  

neo4j_driver = GraphDatabase.driver(
    os.environ.get("NEO4J_URI"),
    auth=(os.environ.get("NEO4J_USERNAME"), os.environ.get("NEO4J_PASSWORD")),
    notifications_min_severity="OFF"
)

# open_ai_client = OpenAI(
#     api_key=os.environ.get("OPENAI_API_KEY"),
# )


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


def num_tokens_from_string(string: str, model: str = "gpt-4") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def embed(texts):
    # response = open_ai_client.embeddings.create(
    #     input=texts,
    #     model=model,
    # )
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedder.embed_documents(texts)


# def embed(texts, model="text-embedding-3-small"):
#     response = open_ai_client.embeddings.create(
#         input=texts,
#         model=model,
#     )
#     return list(map(lambda n: n.embedding, response.data))

def chat(messages):
    llm = ChatVertexAI(model="gemini-2.0-flash")
    return llm.invoke(messages)

def chat(messages, model="gpt-4o", temperature=0, config={}):
    response = open_ai_client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        **config,
    )
    return response.choices[0].message.content

def tool_choice(messages,temperature=0, tools=[], config={}):
    llm = ChatVertexAI(model="gemini-2.0-flash")
    return llm.invoke(messages, temperature, tools=tools).message.tool_calls

def tool_choice(messages, model="gpt-4o", temperature=0, tools=[], config={}):
    response = open_ai_client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        tools=tools or None,
        **config,
    )
    return response.choices[0].message.tool_calls



query_update_prompt = """
    You are an expert at updating questions to make the them ask for one thing only, more atomic, specific and easier to find the answer for.
    You do this by filling in missing information in the question, with the extra information provided to you in previous answers. 
    
    You respond with the updated question that has all information in it.
    Only edit the question if needed. If the original question already is atomic, specific and easy to answer, you keep the original.
    Do not ask for more information than the original question. Only rephrase the question to make it more complete.
    
    JSON template to use:
    {
        "question": "question1"
    }
"""
def query_update(input: str, answers: list[any]) -> str: 
    messages = [
        {"role": "system", "content": query_update_prompt},
        *answers,
        {"role": "user", "content": f"The user question to rewrite: '{input}'"},
    ]
    config = {"response_format": {"type": "json_object"}}
    output = chat(messages, model = "gpt-4o", config=config, )
    try:
        return json.loads(output)["question"]
    except json.JSONDecodeError:
        print("Error decoding JSON")
    return []