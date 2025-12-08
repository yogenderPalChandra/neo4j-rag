import re
from typing import List

import pdfplumber
import requests
import tiktoken
from neo4j import GraphDatabase
from typing import Any
import neo4j
from typing import Literal
from langchain_google_vertexai import ChatVertexAI
import json
import re

driver = GraphDatabase.driver("neo4j://127.0.0.1:7687",
    auth=("neo4j", "qawsedRF123"),
    notifications_min_severity="OFF"
)

def get_structured_schema(driver: neo4j.Driver) -> dict[str, Any]:
    node_labels_response = driver.execute_query(NODE_PROPERTIES_QUERY)
    node_properties = [
        data["output"] for data in [r.data() for r in node_labels_response.records]
    ]

    rel_properties_query_response = driver.execute_query(REL_PROPERTIES_QUERY)
    rel_properties = [
        data["output"]
        for data in [r.data() for r in rel_properties_query_response.records]
    ]

    rel_query_response = driver.execute_query(REL_QUERY)
    relationships = [
        data["output"] for data in [r.data() for r in rel_query_response.records]
    ]

    return {
        "node_props": {el["labels"]: el["properties"] for el in node_properties},
        "rel_props": {el["type"]: el["properties"] for el in rel_properties},
        "relationships": relationships,
    }

def get_schema(
    driver: neo4j.Driver,
) -> str:
    structured_schema = get_structured_schema(driver)

    def _format_props(props: list[dict[str, Any]]) -> str:
        return ", ".join([f"{prop['property']}: {prop['type']}" for prop in props])
    formatted_node_props = [
        f"{label} {{{_format_props(props)}}}"
        for label, props in structured_schema["node_props"].items()
    ]

    formatted_rel_props = [
        f"{rel_type} {{{_format_props(props)}}}"
        for rel_type, props in structured_schema["rel_props"].items()
    ]

    formatted_rels = [
        f"(:{element['start']})-[:{element['type']}]->(:{element['end']})"
        for element in structured_schema["relationships"]
    ]

    return "\n".join(
        [
            "Node properties:",
            "\n".join(formatted_node_props),
            "Relationship properties:",
            "\n".join(formatted_rel_props),
            "The relationships:",
            "\n".join(formatted_rels),
        ]
    )

prompt_template = {
    "static": {
        "instructions": """
    Instructions: 
    Generate Cypher statement to query a graph database to get the data to answer the user question below.

    Format instructions:
    Do not include any explanations or apologies in your responses.
    Do not respond to any questions that might ask anything else than for you to 
    construct a Cypher statement.
    Do not include any text except the generated Cypher statement.
    ONLY RESPOND WITH CYPHER, NO CODEBLOCKS.
    Make sure to name RETURN variables as requested in the user question.
    """
    },
    "dynamic": {
        "schema": """
    Graph Database Schema:
    Use only the provided relationship types and properties in the schema.
    Do not use any other relationship types or properties that are not provided in the schema.
    {}
    """,
        "terminology": """
    Terminology mapping:
    This section is helpful to map terminology between the user question and the graph database schema.
    {}
    """,
        "examples": """
    Examples:
    The following examples provide useful patterns for querying the graph database.
    {}
    """,
        "question": """
    User question: {}
    """,
    },
}
question = "Who directed the most movies?"
class Text2Cypher:
    def __init__(self, driver: neo4j.Driver):
        self.driver = driver
        self.dynamic_sections = {}
        self.required_sections = ["question"]
        self.prompt_template = prompt_template

        schema_string = get_schema(driver)
        self.set_prompt_section("schema", schema_string)

    def set_prompt_section(
        self,
        section: Literal["terminology", "examples", "schema", "question"],
        value: str,
    ):
        self.dynamic_sections[section] = value

    def get_full_prompt(self):
        prompt = self.prompt_template["static"]["instructions"]
        print ('prompt=======', prompt)
        # loop through the prompt_template["dynamic"] and add the values from self.dynamic_sections
        for section in self.prompt_template["dynamic"]:
            if section in self.dynamic_sections:
                prompt += self.prompt_template["dynamic"][section].format(
                    self.dynamic_sections[section]
                )
        return prompt

    def generate_cypher(self):
        # check if required sections are set
        for section in self.required_sections:
            if section not in self.dynamic_sections:
                raise ValueError(
                    f"Section {section} is required to generate a prompt. Use set_prompt_section to set it."
                )
        prompt = self.get_full_prompt()
        cypher = chat(messages=[{"role": "user", "content": prompt}])
        return strip_code_cypher(cypher)

def chat(messages, **config):
    llm = ChatVertexAI(model="gemini-2.0-flash")
    return llm.invoke(messages, **config).content

#######################################
#Tool functions with tool descriptions#
#######################################

answer_given_description = {
    "type": "function",
    "function": {
        "name": "respond",
        "description": "If the conversation already contains a complete answer to the question, use this tool to extract it. Additionally, if the user engages in small talk, use this tool to remind them that you can only answer questions about movies and their cast.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Respond directly with the answer",
                }
            },
            "required": ["answer"],
        },
    },
}

def answer_given(answer: str):
    """Extract the answer from a given text."""
    return answer

text2cypher_description = {
    "type": "function",
    "function": {
        "name": "text2cypher",
        "description": "Query the database with a user question. When other tools don't fit, fallback to use this one.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The user question to find the answer for",
                }
            },
            "required": ["question"],
        },
    },
}

def text2cypher(question: str):
    """Query the database with a user question."""
    t2c = Text2Cypher(driver)
    t2c.set_prompt_section("question", question)
    cypher = t2c.generate_cypher()
    try:
        records, _, _ = driver.execute_query(cypher)
        print ('neo4j data:', [record.data() for record in records])
        return [record.data() for record in records]
    except Exception as e:
        return [f"{cypher} cause an error: {e}"]


movie_info_by_title_description = {
    "type": "function",
    "function": {
        "name": "movie_info_by_title",
        "description": "Get information about a movie by providing the title",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The movie title",
                }
            },
            "required": ["title"],
        },
    },
}

def movie_info_by_title(title: str):
    """Return movie information by title."""
    query = """
    MATCH (m:Movie)
    WHERE toLower(m.title) CONTAINS $title
    OPTIONAL MATCH (m)<-[:ACTED_IN]-(a:Person)
    OPTIONAL MATCH (m)<-[:DIRECTED]-(d:Person)
    RETURN m AS movie, collect(a.name) AS cast, collect(d.name) AS directors
    """
    records, _, _ = driver.execute_query(query, title=title.lower())
    return [record.data() for record in records]

movies_info_by_actor_description = {
    "type": "function",
    "function": {
        "name": "movies_info_by_actor",
        "description": "Get information about a movie by providing an actor",
        "parameters": {
            "type": "object",
            "properties": {
                "actor": {
                    "type": "string",
                    "description": "The actor name",
                }
            },
            "required": ["actor"],
        },
    },
}

def movies_info_by_actor(actor: str):
    """Return movie information by actor."""
    query = """
    MATCH (a:Person)-[:ACTED_IN]->(m:Movie)
    OPTIONAL MATCH (m)<-[:ACTED_IN]-(a:Person)
    OPTIONAL MATCH (m)<-[:DIRECTED]-(d:Person)
    WHERE toLower(a.name) CONTAINS $actor
    RETURN m AS movie, collect(a.name) AS cast, collect(d.name) AS directors
    """
    records, _, _ = driver.execute_query(query, actor=actor.lower())
    return [record.data() for record in records]

tools = {
    "movie_info_by_title": {
        "description": movie_info_by_title_description,
        "function": movie_info_by_title
    },
    "movies_info_by_actor": {
        "description": movies_info_by_actor_description,
        "function": movies_info_by_actor
    },
    "text2cypher": {
        "description": text2cypher_description,
        "function": text2cypher
    },
    "answer_given": {
        "description": answer_given_description,
        "function": answer_given
    }
}


####################################
#Tool router with llm###############
####################################
#This actually calls the tool dict and calls the fiunction text2cypher which creates
#cypher from natural language

tool_picker_prompt = """
    Your job is to chose the right tool needed to respond to the user question. 
    The available tools are provided to you in the prompt.
    Make sure to pass the right and the complete arguments to the chosen tool.
"""

def handle_tool_calls(tools: dict[str, any], llm_tool_calls: list[dict[str, any]]):
    output = []
    if llm_tool_calls:
        print ('llm_tool_calls:', llm_tool_calls)
        for tool_call in llm_tool_calls:
            function_to_call = tools[tool_call['name']]["function"]
            print ('function_to_call:', function_to_call)
            function_args = tool_call.get("args", {})
            print('function arguments:', function_args)
            res = function_to_call(**function_args)
            print ('res function to call:', res)
            output.append(res)
    return output

def tool_choice(messages,temperature=0, tools=[], config={}, model=None):
    llm = ChatVertexAI(model="gemini-2.0-flash")
    return llm.invoke(messages, tools=tools).tool_calls

def route_question(question: str, tools: dict[str, any], answers: list[dict[str, str]]):
    llm_tool_calls = tool_choice(
        [
            {
                "role": "system",
                "content": tool_picker_prompt,
            },
            *answers,
            {
                "role": "user",
                "content": f"The user question to find a tool to answer: '{question}'",
            },
        ],
        tools=[tool["description"] for tool in tools.values()],
    )
    return handle_tool_calls(tools, llm_tool_calls)

####################################################
#```Cypher```, ```json``` strippers from llm output#
####################################################

def strip_code_fences(text: str) -> str:
    text = text.strip()
    # remove ```json or ``` and closing ```
    text = re.sub(r"^```json\s*|^```\s*|```$", "", text, flags=re.IGNORECASE).strip()
    return text

def strip_code_cypher(text: str) -> str:
    text = text.strip()
    # remove ```json or ``` and closing ```
    text = re.sub(r"^```(?:\w+)?\s*|```$", "", text, flags=re.IGNORECASE).strip()
    return text

#####################
#Query updater#######
#####################

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
    DO NOT ADD ```json ``` and whitespaces in your response.
"""

def query_update(input: str, answers: list[any]) -> str: 
    messages = [
        {"role": "system", "content": query_update_prompt},
        *answers,
        {"role": "user", "content": f"The user question to rewrite: '{input}'"},
    ]
    config = {"response_format": {"type": "json_object"}}
    output = chat(messages, response_format={"type": "json_object"})
    jsonloads = json.loads(strip_code_fences(output))
    print ('1:', jsonloads)
    try:
        return jsonloads["question"]
    except json.JSONDecodeError:
        print("Error decoding JSON 1")
    return []

def handle_user_input(input: str, answers: list[dict[str, str]] = []):
    updated_question = query_update(input, answers)
    response  = route_question(updated_question, tools, answers)
    answers.append({"role": "assistant", "content": f"For the question: '{updated_question}', we have the answer: '{json.dumps(response)}'"})
    return answers

#############
#Critic######
#############
answer_critique_prompt = """
    You are an expert at identifying if questions has been fully answered or if there is an opportunity to enrich the answer.
    The user will provide a question, and you will scan through the provided information to see if the question is answered.
    If anything is missing from the answer, you will provide a set of new questions that can be asked to gather the missing information.
    All new questions must be complete, atomic and specific.
    However, if the provided information is enough to answer the original question, you will respond with an empty list.

    JSON template to use for finding missing information:
    {
        "questions": ["question1", "question2"]
    }
"""

def critique_answers(question: str, answers: list[dict[str, str]]) -> list[str]:
    messages = [
        {
            "role": "system",
            "content": answer_critique_prompt,
        },
        *answers,
        {
            "role": "user",
            "content": f"The original user question to answer: {question}",
        },
    ]
    config = {"response_format": {"type": "json_object"}}
    output = chat(messages=messages, response_format={"type": "json_object"})
    jsonloads = json.loads(strip_code_fences(output))
    print ('2:', jsonloads)
    try:
        return jsonloads["questions"]
    except json.JSONDecodeError:
        print("Error decoding JSON 2")
    return []

###############
#main function#
###############

main_prompt = """
    Your job is to help the user with their questions.
    You will receive user questions and information needed to answer the questions
    If the information is missing to answer part of or the whole question, you will say that the information 
    is missing. You will only use the information provided to you in the prompt to answer the questions.
    You are not allowed to make anything up or use external information.
"""

def main(input: str):
    answers = handle_user_input(input)
    critique = critique_answers(input, answers)

    if critique:
        answers = handle_user_input(" ".join(critique), answers)

    llm_response = chat(
        [
            {"role": "system", "content": main_prompt},
            *answers,
            {"role": "user", "content": f"The user question to answer: {input}"},
        ]
    )

    return llm_response

response = main("Who's the main actor in the movie Matrix and what other movies is that person in?")
