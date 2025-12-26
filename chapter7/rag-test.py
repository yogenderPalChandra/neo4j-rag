import os
from pydantic import BaseModel, Field
from typing import Optional, List
import json
from langchain_google_vertexai import ChatVertexAI
from vertexai.generative_models import GenerativeModel
import re
from neo4j import GraphDatabase
import json
import requests
from tqdm import tqdm
from typing import Dict
from langchain.embeddings import HuggingFaceEmbeddings
import re
from typing import List


driver = GraphDatabase.driver("neo4j://127.0.0.1:7687",
    auth=("neo4j", "qawsedRF123"),
    notifications_min_severity="OFF"
)
graph_id = "book_002"
import_nodes_query = """
MERGE (b:Book {id: $book_id, graph_id: $graph_id})
MERGE (b)-[:HAS_CHUNK]->(c:__Chunk__ {id: $chunk_id, graph_id: $graph_id})
SET c.text = $text
WITH c
UNWIND $data AS row
MERGE (n:__Entity__ {name: row.entity_name, graph_id: $graph_id})
SET n.entity_type = row.entity_type,
    n.description = coalesce(n.description, []) + [row.entity_description]
MERGE (n)<-[:MENTIONS]-(c)
"""


#This will find an entity with lable __Entity__ {name: Mother Nature
#if its present then it will find it, and then it will find another
# __Entity__{name: <target_entity>} this target_entity is something
#which is provided by relationship in nodes, relationship =extract_entities()
#
import_relationships_query = """
UNWIND $data AS row
MERGE (s:__Entity__ {name: row.source_entity, graph_id: $graph_id})
MERGE (t:__Entity__ {name: row.target_entity, graph_id: $graph_id})
CREATE (s)-[r:RELATIONSHIP {description: row.relationship_description, strength: row.relationship_strength}]->(t)
"""

url = "https://gutenberg.org/cache/epub/678/pg678.txt"
response = requests.get(url)

def chat(messages, response_format=None):
    llm = ChatVertexAI(model="gemini-2.0-flash")
    return llm.invoke(messages, response_format=response_format)

# def num_tokens_from_string(string: str, model: str = "gpt-4") -> int:
#     """Returns the number of tokens in a text string."""
#     encoding = tiktoken.encoding_for_model(model)
#     num_tokens = len(encoding.encode(string))
#     return num_tokens



def chunk_into_chapters(text: str) -> List[str]:
    # 1. Remove front matter using the asterisk divider
    if "* * * * *" in text:
        text = text.split("* * * * *", 1)[1]

    text = text.strip()

    # 2. Split on CHAPTER markers (handles glued chapters)
    chapters = re.split(
        r"(?=CHAPTER\s+[IVXLCDM]+\s*—)",
        text,
        flags=re.IGNORECASE,
    )

    # 3. Clean and return valid chapters only
    return [chapter.strip() for chapter in chapters if chapter.strip()]

chapters = chunk_into_chapters(response.text)


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

chunked_chapters = [chunk_text(chapter, 1000, 40) for chapter in chapters]

####################
#Entity Extraction##
####################

GRAPH_EXTRACTION_PROMPT = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:
Entity_types: ORGANIZATION,PERSON
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
("entity"{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}ORGANIZATION{tuple_delimiter}The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday)
{record_delimiter}
("entity"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}PERSON{tuple_delimiter}Martin Smith is the chair of the Central Institution)
{record_delimiter}
("entity"{tuple_delimiter}MARKET STRATEGY COMMITTEE{tuple_delimiter}ORGANIZATION{tuple_delimiter}The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply)
{record_delimiter}
("relationship"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}Martin Smith is the Chair of the Central Institution and will answer questions at a press conference{tuple_delimiter}9)
{completion_delimiter}

######################
Example 2:
Entity_types: ORGANIZATION
Text:
TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
######################
Output:
("entity"{tuple_delimiter}TECHGLOBAL{tuple_delimiter}ORGANIZATION{tuple_delimiter}TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones)
{record_delimiter}
("entity"{tuple_delimiter}VISION HOLDINGS{tuple_delimiter}ORGANIZATION{tuple_delimiter}Vision Holdings is a firm that previously owned TechGlobal)
{record_delimiter}
("relationship"{tuple_delimiter}TECHGLOBAL{tuple_delimiter}VISION HOLDINGS{tuple_delimiter}Vision Holdings formerly owned TechGlobal from 2014 until present{tuple_delimiter}5)
{completion_delimiter}

######################
Example 3:
Entity_types: ORGANIZATION,GEO,PERSON
Text:
Five Aurelians jailed for 8 years in Firuzabad and widely regarded as hostages are on their way home to Aurelia.

The swap orchestrated by Quintara was finalized when $8bn of Firuzi funds were transferred to financial institutions in Krohaara, the capital of Quintara.

The exchange initiated in Firuzabad's capital, Tiruzia, led to the four men and one woman, who are also Firuzi nationals, boarding a chartered flight to Krohaara.

They were welcomed by senior Aurelian officials and are now on their way to Aurelia's capital, Cashion.

The Aurelians include 39-year-old businessman Samuel Namara, who has been held in Tiruzia's Alhamia Prison, as well as journalist Durke Bataglani, 59, and environmentalist Meggie Tazbah, 53, who also holds Bratinas nationality.
######################
Output:
("entity"{tuple_delimiter}FIRUZABAD{tuple_delimiter}GEO{tuple_delimiter}Firuzabad held Aurelians as hostages)
{record_delimiter}
("entity"{tuple_delimiter}AURELIA{tuple_delimiter}GEO{tuple_delimiter}Country seeking to release hostages)
{record_delimiter}
("entity"{tuple_delimiter}QUINTARA{tuple_delimiter}GEO{tuple_delimiter}Country that negotiated a swap of money in exchange for hostages)
{record_delimiter}
{record_delimiter}
("entity"{tuple_delimiter}TIRUZIA{tuple_delimiter}GEO{tuple_delimiter}Capital of Firuzabad where the Aurelians were being held)
{record_delimiter}
("entity"{tuple_delimiter}KROHAARA{tuple_delimiter}GEO{tuple_delimiter}Capital city in Quintara)
{record_delimiter}
("entity"{tuple_delimiter}CASHION{tuple_delimiter}GEO{tuple_delimiter}Capital city in Aurelia)
{record_delimiter}
("entity"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}PERSON{tuple_delimiter}Aurelian who spent time in Tiruzia's Alhamia Prison)
{record_delimiter}
("entity"{tuple_delimiter}ALHAMIA PRISON{tuple_delimiter}GEO{tuple_delimiter}Prison in Tiruzia)
{record_delimiter}
("entity"{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}PERSON{tuple_delimiter}Aurelian journalist who was held hostage)
{record_delimiter}
("entity"{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}PERSON{tuple_delimiter}Bratinas national and environmentalist who was held hostage)
{record_delimiter}
("relationship"{tuple_delimiter}FIRUZABAD{tuple_delimiter}AURELIA{tuple_delimiter}Firuzabad negotiated a hostage exchange with Aurelia{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}QUINTARA{tuple_delimiter}AURELIA{tuple_delimiter}Quintara brokered the hostage exchange between Firuzabad and Aurelia{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}QUINTARA{tuple_delimiter}FIRUZABAD{tuple_delimiter}Quintara brokered the hostage exchange between Firuzabad and Aurelia{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}ALHAMIA PRISON{tuple_delimiter}Samuel Namara was a prisoner at Alhamia prison{tuple_delimiter}8)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}Samuel Namara and Meggie Tazbah were exchanged in the same hostage release{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}Samuel Namara and Durke Bataglani were exchanged in the same hostage release{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}Meggie Tazbah and Durke Bataglani were exchanged in the same hostage release{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}FIRUZABAD{tuple_delimiter}Samuel Namara was a hostage in Firuzabad{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}FIRUZABAD{tuple_delimiter}Meggie Tazbah was a hostage in Firuzabad{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}FIRUZABAD{tuple_delimiter}Durke Bataglani was a hostage in Firuzabad{tuple_delimiter}2)
{completion_delimiter}

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""
def create_extraction_prompt(entity_types, input_text, tuple_delimiter=";"):
    prompt = GRAPH_EXTRACTION_PROMPT.format(
        entity_types=entity_types,
        input_text=input_text,
        tuple_delimiter=tuple_delimiter,
        record_delimiter="|",
        completion_delimiter="\n\n",
    )
    return prompt

ENTITY_TYPES = [
    "PERSON",
    "ROLE_OR_PROFESSION",
    "ANIMAL",
    "OBJECT",
    "LOCATION",
    "DWELLING",
    "SYMBOLIC_ENTITY",
    "ABSTRACT_CONCEPT",
    "CONDITION",
    "EVENT"
]
def parse_extraction_output(output_str, record_delimiter=None, tuple_delimiter=None):

    # Remove the completion delimiter if present.
    completion_marker = "{completion_delimiter}"
    if completion_marker in output_str:
        output_str = output_str.content.replace(completion_marker, "")
    output_str = output_str.content.strip()

    # Determine the record delimiter if not provided.
    if record_delimiter is None:
        if "{record_delimiter}" in output_str:
            record_delimiter = "{record_delimiter}"
        elif "|" in output_str:
            record_delimiter = "|"
        else:
            # Fallback: split on newlines
            record_delimiter = "\n"

    # Determine the tuple delimiter if not provided.
    if tuple_delimiter is None:
        if "{tuple_delimiter}" in output_str:
            tuple_delimiter = "{tuple_delimiter}"
        elif ";" in output_str:
            tuple_delimiter = ";"
        else:
            tuple_delimiter = "\t"

    # Split the output into individual record strings.
    raw_records = [r.strip() for r in output_str.split(record_delimiter)]

    parsed_records = []
    for rec in raw_records:
        if not rec:
            continue  # skip empty strings

        # Remove leading/trailing parentheses if present.
        if rec.startswith("(") and rec.endswith(")"):
            rec = rec[1:-1]
        rec = rec.strip()

        # Split the record into tokens using the tuple delimiter.
        tokens = [token.strip() for token in rec.split(tuple_delimiter)]
        if not tokens:
            continue

        # The first token should be either "entity" or "relationship".
        rec_type = tokens[0].strip(' "\'').lower()

        if rec_type == "entity":
            if len(tokens) != 4:
                # Optionally log or raise an error for malformed records.
                continue
            record = {
                "record_type": "entity",
                "entity_name": tokens[1],
                "entity_type": tokens[2],
                "entity_description": tokens[3]
            }
            parsed_records.append(record)
        elif rec_type == "relationship":
            if len(tokens) != 5:
                continue
            # Attempt to convert relationship_strength to a number.
            try:
                strength = float(tokens[4])
                # Convert to int if it has no fractional part.
                if strength.is_integer():
                    strength = int(strength)
            except ValueError:
                strength = tokens[4]
            record = {
                "record_type": "relationship",
                "source_entity": tokens[1],
                "target_entity": tokens[2],
                "relationship_description": tokens[3],
                "relationship_strength": strength
            }
            parsed_records.append(record)
        else:
            # Unknown record type; skip it or handle accordingly.
            continue
    nodes = [el for el in parsed_records if el.get("record_type") == "entity"]
    relationships = [el for el in parsed_records if el.get("record_type") == "relationship"]
    return nodes, relationships


def extract_entities(text: str) -> List[Dict]:
    # Construct prompt
    messages = [
        {"role": "user", "content": create_extraction_prompt(ENTITY_TYPES, text)},
    ]
    # Make the LLM call
    output = chat(messages)
    print (output)
    # Construct JSON from output
    return parse_extraction_output(output)
    #return output


for chapter_i, chapter in enumerate(
    tqdm(chunked_chapters, desc="Processing Books")
):
    for chunk_i, chunk in enumerate(tqdm(chapter, desc=f"Book {chapter_i}", leave=False)):
        nodes, relationships = extract_entities(chunk)
        driver.execute_query(
            import_nodes_query,
            data=nodes,
            book_id=chapter_i,
            graph_id=graph_id,
            text=chunk,
            chunk_id=chunk_i,
        )
        driver.execute_query(
            import_relationships_query, data=relationships,  graph_id=graph_id,
        )

###########################
#Summariser for entities###
###########################

SUMMARIZE_PROMPT = """
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we have the full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

def get_summarize_prompt(entity_name, description_list):
    return SUMMARIZE_PROMPT.format(
        entity_name=entity_name,
        description_list=description_list)

candidates_to_summarize, _, _ = driver.execute_query(
    """MATCH (e:__Entity__{graph_id: "book_002"}) WHERE size(e.description) > 1
    RETURN e.name AS entity_name, e.description AS description_list"""
)

summaries = []
for candidate in tqdm(candidates_to_summarize, desc="Summarizing entities"):
    messages = [
        {
            "role": "user",
            "content": get_summarize_prompt(
                candidate["entity_name"], candidate["description_list"]
            ),
        },
    ]
    summary = chat(messages).content
    summaries.append({"entity": candidate["entity_name"], "summary": summary})

def import_entity_summary(driver, entity_information):
    driver.execute_query("""
    UNWIND $data AS row
    MATCH (e:__Entity__ {name: row.entity})
    SET e.summary = row.summary
    """, data=entity_information)
    
    # If there was only 1 description use that
    driver.execute_query("""
    MATCH (e:__Entity__)
    WHERE size(e.description) = 1
    SET e.summary = e.description[0]
    """)

import_entity_summary(driver, summaries)

candidates_to_summarize, _, _ = driver.execute_query(
    """MATCH (e:__Entity__) WHERE size(e.description) > 1 
     RETURN e.name AS entity_name, e.summary AS summary, e.description AS description_list""")

#############################
#Summariser for relationship#
#############################

rels_to_summarize, _, _ = driver.execute_query(
    """MATCH (s:__Entity__ {graph_id: "book_002"})-[r:RELATIONSHIP]-(t:__Entity__ {graph_id: "book_002"})
    WHERE id(s) < id(t)
    WITH s.name AS source, t.name AS target, 
           collect(r.description) AS description_list,
           count(*) AS count
    WHERE count > 1
    RETURN source, target, description_list"""
)

rel_summaries = []
for candidate in tqdm(rels_to_summarize, desc="Summarizing relationships"):
    entity_name = f"{candidate['source']} relationship to {candidate['target']}"
    messages = [
        {
            "role": "user",
            "content": get_summarize_prompt(
                entity_name, candidate["description_list"]
            ),
        },
    ]
    summary = chat(messages).content
    rel_summaries.append({"source": candidate["source"], "target": candidate["target"], "summary": summary})

def import_rels_summary(driver, rel_summaries):
    driver.execute_query("""
    UNWIND $data AS row
    MATCH (s:__Entity__ {name: row.source, graph_id: "book_002"}), (t:__Entity__ {name: row.target, graph_id: "book_002"})
    MERGE (s)-[r:SUMMARIZED_RELATIONSHIP]-(t)
    SET r.summary = row.summary
    """, data=rel_summaries)
    
    # If there was only 1 description use that
    driver.execute_query("""
    MATCH (s:__Entity__{graph_id: "book_002"})-[e:RELATIONSHIP]-(t:__Entity__{graph_id: "book_002"})
    WHERE NOT (s)-[:SUMMARIZED_RELATIONSHIP]-(t)
    MERGE (s)-[r:SUMMARIZED_RELATIONSHIP]-(t)
    SET r.summary = e.description
    """)
import_rels_summary(driver, rels_summaries)

data, _, _ = driver.execute_query(
"""MATCH (n:__Entity__ {graph_id: "book_002"})-[r:SUMMARIZED_RELATIONSHIP]-(m:__Entity__ {graph_id: "book_002"})
WHERE n.name= "WEDDING-CAKE" AND m.name = "MARRIAGE"
RETURN r.summary AS summary
""")

#################
#Communities#####
#################

def calculate_communities(driver):
    # Drop graph if exist
    try:
        driver.execute_query("""CALL gds.graph.drop('entity1')""")
    except:
      pass
    driver.execute_query("""
    MATCH (source:__Entity__ {graph_id: "book_002"})-[r:RELATIONSHIP]->(target:__Entity__ {graph_id: "book_002"})
    WITH gds.graph.project('entity1', source, target, {}, {undirectedRelationshipTypes: ['*']}) AS g
    RETURN g.graphName AS graph, g.nodeCount AS nodes, g.relationshipCount AS rels
    """)
    
    records, _, _ = driver.execute_query("""
    CALL gds.louvain.write("entity1", {writeProperty:"louvain"})
    """)
    return [el.data() for el in records][0]

community_distribution = calculate_communities(driver)


COMMUNITY_REPORT_PROMPT = """
You are an AI assistant that helps a human analyst to perform general information discovery. Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

# Goal
Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

# Report Structure

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
        ]
    }}

# Grounding Rules

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (1), Entities (5, 7); Relationships (23); Claims (7, 2, 34, 64, 46, +more)]."

where 1, 5, 7, 23, 2, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


# Example Input
-----------
Text:

Entities

id,entity,description
5,VERDANT OASIS PLAZA,Verdant Oasis Plaza is the location of the Unity March
6,HARMONY ASSEMBLY,Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza

Relationships

id,source,target,description
37,VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza is the location of the Unity March
38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly is holding a march at Verdant Oasis Plaza
39,VERDANT OASIS PLAZA,UNITY MARCH,The Unity March is taking place at Verdant Oasis Plaza
40,VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Tribune Spotlight is reporting on the Unity march taking place at Verdant Oasis Plaza
41,VERDANT OASIS PLAZA,BAILEY ASADI,Bailey Asadi is speaking at Verdant Oasis Plaza about the march
43,HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly is organizing the Unity March

Output:
{{
    "title": "Verdant Oasis Plaza and Unity March",
    "summary": "The community revolves around the Verdant Oasis Plaza, which is the location of the Unity March. The plaza has relationships with the Harmony Assembly, Unity March, and Tribune Spotlight, all of which are associated with the march event.",
    "rating": 5.0,
    "rating_explanation": "The impact severity rating is moderate due to the potential for unrest or conflict during the Unity March.",
    "findings": [
        {{
            "summary": "Verdant Oasis Plaza as the central location",
            "explanation": "Verdant Oasis Plaza is the central entity in this community, serving as the location for the Unity March. This plaza is the common link between all other entities, suggesting its significance in the community. The plaza's association with the march could potentially lead to issues such as public disorder or conflict, depending on the nature of the march and the reactions it provokes. [Data: Entities (5), Relationships (37, 38, 39, 40, 41,+more)]"
        }},
        {{
            "summary": "Harmony Assembly's role in the community",
            "explanation": "Harmony Assembly is another key entity in this community, being the organizer of the march at Verdant Oasis Plaza. The nature of Harmony Assembly and its march could be a potential source of threat, depending on their objectives and the reactions they provoke. The relationship between Harmony Assembly and the plaza is crucial in understanding the dynamics of this community. [Data: Entities(6), Relationships (38, 43)]"
        }},
        {{
            "summary": "Unity March as a significant event",
            "explanation": "The Unity March is a significant event taking place at Verdant Oasis Plaza. This event is a key factor in the community's dynamics and could be a potential source of threat, depending on the nature of the march and the reactions it provokes. The relationship between the march and the plaza is crucial in understanding the dynamics of this community. [Data: Relationships (39)]"
        }},
        {{
            "summary": "Role of Tribune Spotlight",
            "explanation": "Tribune Spotlight is reporting on the Unity March taking place in Verdant Oasis Plaza. This suggests that the event has attracted media attention, which could amplify its impact on the community. The role of Tribune Spotlight could be significant in shaping public perception of the event and the entities involved. [Data: Relationships (40)]"
        }}
    ]
}}


# Real Data

Use the following text for your answer. Do not make anything up in your answer.

Text:
{input_text}

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
        ]
    }}

# Grounding Rules

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (1), Entities (5, 7); Relationships (23); Claims (7, 2, 34, 64, 46, +more)]."

where 1, 5, 7, 23, 2, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.

Output:"""

community_info_query = """MATCH (e:__Entity__ {graph_id: "book_002"})
             WHERE e.louvain IS NOT NULL
             WITH e.louvain AS louvain, collect(e) AS nodes
             WHERE size(nodes) > 1
             CALL apoc.path.subgraphAll(nodes[0], {
                whitelistNodes:nodes
             })
             YIELD relationships
             UNWIND nodes as n
             WITH louvain as louvain, collect({
             id: n.name,
             description: n.summary,
             type: [el in labels(n) WHERE el<>'ENTITY__{graph_id: "book_02"}'][0]
             
             }) AS nodes, relationships
             UNWIND relationships as r
             WITH louvain, nodes, collect({
             start: startNode(r).name, 
             type: type(r), 
             end: endNode(r).name, 
             description: r.description
             }) AS rels
             RETURN louvain AS communityId, nodes, rels"""

#this creates an object with louvain id grouped,objects grouped by louvain as COmmunity ID. Each group containes nodes and they sub graph relationships.
#The relationships has start node and end node with "type" of relationship, its a list of such relationhsips. So basically it prints all the nodes with thier louvain id and all the
#Other nodes connected to it.for example this object:
#| 355         | [{type: "__Entity__", description: "The city in Utah where the Foundation's business office is located", id: "SALT LAKE CITY"}, {type: "__Entity__", description: "The state where the Foundation's business office is located", id: "UTAH"}] |
#| [{start: "SALT LAKE CITY", description: "Salt Lake City is located within Utah", end: "UTAH", type: "RELATIONSHIP"}, {start: "UTAH", description: NULL, end: "SALT LAKE CITY", type: "SUMMARIZED_RELATIONSHIP"}]       
# the line above is nodes list and th eline below is relationships teh nodes are associated with
def get_summarize_community_prompt(nodes, relationships):
    input_text = f"""Entities

    {nodes}

    Relationships

    {relationships}
    """
    return COMMUNITY_REPORT_PROMPT.format(
        input_text=input_text,
    )

community_info, _, _ = driver.execute_query(community_info_query)

def strip_code_cypher(text: str) -> str:
    text = text.strip()
    # remove ```json or ``` and closing ```
    text = re.sub(r"^```(?:\w+)?\s*|```$", "", text, flags=re.IGNORECASE).strip()
    return text

#This creates an object looking like this:
#{
#community: {}
#communityID:
#nodes: 
# Like this:
"""
 {'community': {'title': 'Salt Lake City and Utah',
   'summary': 'The community consists of Salt Lake City, Utah. Salt Lake City is located within Utah [Data: Relationships (1)].',
   'rating': 1.0,
   'rating_explanation': 'The impact severity rating is low as it simply describes a geographical relationship.',
   'findings': [{'summary': 'Geographical Relationship',
     'explanation': 'Salt Lake City is located within the state of Utah [Data: Relationships (1)]. This defines a geographical containment relationship where Salt Lake City is a part of the larger entity, Utah. This relationship is important for understanding the geographical context of any activities or entities associated with either location.'}]},
  'communityId': 355,
  'nodes': ['SALT LAKE CITY', 'UTAH']}]
"""
communities = []
for community in tqdm(community_info, desc="Summarizing communities"):
    messages = [
        {
            "role": "user",
            "content": get_summarize_community_prompt(
                community["nodes"], community["rels"]
            ),
        },
    ]
    summary = chat(messages).content
    communities.append(
        {
            "community": json.loads(strip_code_cypher(summary)),
            "communityId": community["communityId"],
            "nodes": [el["id"] for el in community["nodes"]],
        }
    )

import_community_query = """
UNWIND $data AS row
MERGE (c:__Community__ {communityId: row.communityId, graph_id: "book_002"})
SET c.title = row.community.title,
    c.summary = row.community.summary,
    c.rating = row.community.rating,
    c.rating_explanation = row.community.rating_explanation
WITH c, row
UNWIND row.nodes AS node
MERGE (n:__Entity__ {name: node, graph_id: "book_002"})
MERGE (n)-[:IN_COMMUNITY]->(c)
"""

#This basically creats nodes c __Community__ {communityId: row.communityId}
# and set some values. Finnaly it creates or finds node (n:__Entity__ {name: node}
#from which data? I mean from which communityId or name: node?
# Form this data[0] as row:

"""
{'community': {'title': 'The Peerybingle Family and the Enigmatic Stranger',
 'communityId': 128,
 'nodes': ['MRS. PEERYBINGLE',
  'BABY',
  'MAN',
  'TILLY SLOWBOY',
  'INFANT',
  'TILLY',
  'BOXER',
  'MISS SLOWBOY',
  'STRANGER',
  'CARRIER’S WIFE',
  'TOYS',
  'FARMERS',
  'PIGS',
  'MARKET',
  'FAIRY CRICKET',
  'FOWLS',
  'COTTAGES',
  'DAME-SCHOOLS',
  'PIGEONS',
  'CATS',
  'PUBLIC-HOUSES',
  'BARGES',
  'WHARF',
  'THINGS',
  'GUN',
  'WILD BEAST',
  'DEMON',
  'IMAGE',
  'GOD',
  'PRESENCE',
  'FAIRIES',
  'THOUGHTS',
  'KITCHEN',
  'MISTRESS',
  'SLOWBOY',
  'GARRET',
  'BABBY']}
"""
#So basically it search all these nodes or searches them and connects them with relastionship
driver.execute_query(import_community_query, data=communities)


#cheking whats created:
data, _, _ = driver.execute_query(
    """MATCH (c:__Community__{graph_id: "book_002"})
WITH c, count {(c)<-[:IN_COMMUNITY]-()} AS size
ORDER BY size DESC LIMIT 10
RETURN c.title AS title, c.summary AS summary
"""
)


################################################
########Graph retrival##########################
################################################

GRAPH_EXTRACTION_PROMPT = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:
Entity_types: ORGANIZATION,PERSON
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
("entity"{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}ORGANIZATION{tuple_delimiter}The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday)
{record_delimiter}
("entity"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}PERSON{tuple_delimiter}Martin Smith is the chair of the Central Institution)
{record_delimiter}
("entity"{tuple_delimiter}MARKET STRATEGY COMMITTEE{tuple_delimiter}ORGANIZATION{tuple_delimiter}The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply)
{record_delimiter}
("relationship"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}Martin Smith is the Chair of the Central Institution and will answer questions at a press conference{tuple_delimiter}9)
{completion_delimiter}

######################
Example 2:
Entity_types: ORGANIZATION
Text:
TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
######################
Output:
("entity"{tuple_delimiter}TECHGLOBAL{tuple_delimiter}ORGANIZATION{tuple_delimiter}TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones)
{record_delimiter}
("entity"{tuple_delimiter}VISION HOLDINGS{tuple_delimiter}ORGANIZATION{tuple_delimiter}Vision Holdings is a firm that previously owned TechGlobal)
{record_delimiter}
("relationship"{tuple_delimiter}TECHGLOBAL{tuple_delimiter}VISION HOLDINGS{tuple_delimiter}Vision Holdings formerly owned TechGlobal from 2014 until present{tuple_delimiter}5)
{completion_delimiter}

######################
Example 3:
Entity_types: ORGANIZATION,GEO,PERSON
Text:
Five Aurelians jailed for 8 years in Firuzabad and widely regarded as hostages are on their way home to Aurelia.

The swap orchestrated by Quintara was finalized when $8bn of Firuzi funds were transferred to financial institutions in Krohaara, the capital of Quintara.

The exchange initiated in Firuzabad's capital, Tiruzia, led to the four men and one woman, who are also Firuzi nationals, boarding a chartered flight to Krohaara.

They were welcomed by senior Aurelian officials and are now on their way to Aurelia's capital, Cashion.

The Aurelians include 39-year-old businessman Samuel Namara, who has been held in Tiruzia's Alhamia Prison, as well as journalist Durke Bataglani, 59, and environmentalist Meggie Tazbah, 53, who also holds Bratinas nationality.
######################
Output:
("entity"{tuple_delimiter}FIRUZABAD{tuple_delimiter}GEO{tuple_delimiter}Firuzabad held Aurelians as hostages)
{record_delimiter}
("entity"{tuple_delimiter}AURELIA{tuple_delimiter}GEO{tuple_delimiter}Country seeking to release hostages)
{record_delimiter}
("entity"{tuple_delimiter}QUINTARA{tuple_delimiter}GEO{tuple_delimiter}Country that negotiated a swap of money in exchange for hostages)
{record_delimiter}
{record_delimiter}
("entity"{tuple_delimiter}TIRUZIA{tuple_delimiter}GEO{tuple_delimiter}Capital of Firuzabad where the Aurelians were being held)
{record_delimiter}
("entity"{tuple_delimiter}KROHAARA{tuple_delimiter}GEO{tuple_delimiter}Capital city in Quintara)
{record_delimiter}
("entity"{tuple_delimiter}CASHION{tuple_delimiter}GEO{tuple_delimiter}Capital city in Aurelia)
{record_delimiter}
("entity"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}PERSON{tuple_delimiter}Aurelian who spent time in Tiruzia's Alhamia Prison)
{record_delimiter}
("entity"{tuple_delimiter}ALHAMIA PRISON{tuple_delimiter}GEO{tuple_delimiter}Prison in Tiruzia)
{record_delimiter}
("entity"{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}PERSON{tuple_delimiter}Aurelian journalist who was held hostage)
{record_delimiter}
("entity"{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}PERSON{tuple_delimiter}Bratinas national and environmentalist who was held hostage)
{record_delimiter}
("relationship"{tuple_delimiter}FIRUZABAD{tuple_delimiter}AURELIA{tuple_delimiter}Firuzabad negotiated a hostage exchange with Aurelia{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}QUINTARA{tuple_delimiter}AURELIA{tuple_delimiter}Quintara brokered the hostage exchange between Firuzabad and Aurelia{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}QUINTARA{tuple_delimiter}FIRUZABAD{tuple_delimiter}Quintara brokered the hostage exchange between Firuzabad and Aurelia{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}ALHAMIA PRISON{tuple_delimiter}Samuel Namara was a prisoner at Alhamia prison{tuple_delimiter}8)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}Samuel Namara and Meggie Tazbah were exchanged in the same hostage release{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}Samuel Namara and Durke Bataglani were exchanged in the same hostage release{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}Meggie Tazbah and Durke Bataglani were exchanged in the same hostage release{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}FIRUZABAD{tuple_delimiter}Samuel Namara was a hostage in Firuzabad{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}FIRUZABAD{tuple_delimiter}Meggie Tazbah was a hostage in Firuzabad{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}FIRUZABAD{tuple_delimiter}Durke Bataglani was a hostage in Firuzabad{tuple_delimiter}2)
{completion_delimiter}

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""

MAP_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}},
        {{"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}}
    ]
}}

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Points supported by data should list the relevant reports as references as follows:
"This is an example sentence supported by data references [Data: Reports (report ids)]"

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data report in the provided tables.

Do not include information where the supporting evidence for it is not provided.


---Data tables---

{context_data}

---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Points supported by data should list the relevant reports as references as follows:
"This is an example sentence supported by data references [Data: Reports (report ids)]"

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data report in the provided tables.

Do not include information where the supporting evidence for it is not provided.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}},
        {{"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}}
    ]
}}
"""

def get_map_system_prompt(context):
    return MAP_SYSTEM_PROMPT.format(context_data=context)


REDUCE_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

The response should also preserve all the data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}


---Analyst Reports---

{report_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

The response should also preserve all the data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

def get_reduce_system_prompt(report_data, response_type: str = "multiple paragraphs"):
    return REDUCE_SYSTEM_PROMPT.format(report_data=report_data, response_type=response_type)

def global_retriever(query: str, rating_threshold: float = 5) -> str:
    community_data, _, _ = driver.execute_query(
        """
    MATCH (c:__Community__{graph_id: "book_002"})
    WHERE c.rating >= $rating
    RETURN c.summary AS summary
    """,
        rating=rating_threshold,
    )
    print(f"Got {len(community_data)} community summaries")
    intermediate_results = []
    for community in tqdm(community_data, desc="Processing communities"):
        intermediate_messages = [
            {
                "role": "system",
                "content": get_map_system_prompt(community["summary"]),
            },
            {
                "role": "user",
                "content": query,
            },
        ]
        intermediate_response = chat(intermediate_messages).content
        intermediate_results.append(intermediate_response)

    final_messages = [
        {
            "role": "system",
            "content": get_reduce_system_prompt(intermediate_results),
        },
        {"role": "user", "content": query},
    ]
    summary = chat(final_messages)
    return summary

global_retriever=global_retriever('What are the central conflicts in this story?')

######################################
#Local search using vector embeddings#
######################################

entities, _, _ = driver.execute_query(
    """
MATCH (e:__Entity__{graph_id: "book_002"})
RETURN e.summary AS summary, e.name AS name
"""
)

def embed(texts):
    if isinstance(texts, str):
        texts = [texts]  # auto-fix common mistake
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedder.embed_documents(texts)

data = [{"name": el["name"], "embedding": embed(el["summary"])[0]} for el in entities if el.get("summary") is not None]


#this creates a property called embeddings on all the nodes with lable __Entity__
# db.create.setNodeVectorProperty this tells its vector property and not n ordinary property
driver.execute_query(
    """
UNWIND $data AS row
MATCH (e:__Entity__ {name: row.name, graph_id: "book_002"})
CALL db.create.setNodeVectorProperty(e, 'embedding1', row.embedding)
""",
    data=data,
)

#Create vector index in embeddings

driver.execute_query(
    """
CREATE VECTOR INDEX entitie IF NOT EXISTS
FOR (n:__Entity__)
ON (n.embedding1)
""",
    data=data,
)


###
#local search query, entry point is vector similarity search, once the most consine similarity data point is identified
# Thye are expended for their connected nodes, entities, text chunks, sumaries and relationships,


local_search_query="""CALL db.index.vector.queryNodes('entitie', $k, $embedding)
    YIELD node, score
    WITH collect(node) AS nodes, node.name AS name, node.summary AS summary
    WITH collect {
        UNWIND nodes as n
        MATCH (n)<-[:MENTIONS]->(c:__Chunk__)
        WITH c, count(DISTINCT n) as freq
        RETURN c.text as chunkText
        ORDER BY freq DESC 
    } AS textMappings,
    collect {
        UNWIND nodes as n
            MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
            WITH c, c.rank as rank, c.weight AS weight
            RETURN c.summary 
            ORDER BY rank, weight DESC
    } AS report_mapping,
    collect {
        UNWIND nodes as n
            MATCH (n)-[r:SUMMARIZED_RELATIONSHIP]-(m) 
            RETURN r.summary AS descriptionText
            ORDER BY r.rank, r.weight DESC 
    } as insideRels,
    collect {
        UNWIND nodes as n
        RETURN n.summary AS descriptionText
    } as entities
    RETURN {Chunks: textMappings, 
            Reports: report_mapping, 
            Relationships: insideRels,
            Entities: entities
            } AS Texts
    
    """

LOCAL_SEARCH_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Sources (15, 16), Reports (1), Entities (5, 7); Relationships (23); Claims (2, 7, 34, 46, 64, +more)]."

where 15, 16, 1, 5, 7, 23, 2, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}


---Data tables---

{context_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Sources (15, 16), Reports (1), Entities (5, 7); Relationships (23); Claims (2, 7, 34, 46, 64, +more)]."

where 15, 16, 1, 5, 7, 23, 2, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

def get_local_system_prompt(report_data, response_type: str = "multiple paragraphs"):
    return LOCAL_SEARCH_SYSTEM_PROMPT.format(context_data=report_data, response_type=response_type)


k_entities = 5

topChunks = 3
topCommunities = 3
topInsideRels = 3

def local_search(query: str) -> str:
    context, _, _ = driver.execute_query(
        local_search_query,
        embedding=embed(query)[0],
        topChunks=topChunks,
        topCommunities=topCommunities,
        topInsideRels=topInsideRels,
        k=k_entities,
    )
    context_str = str(context[0]["Texts"])
    print (context_str)
    local_messages = [
        {
            "role": "system",
            "content": get_local_system_prompt(context_str),
        },
        {
            "role": "user",
            "content": query,
        },
    ]
    final_answer = chat(local_messages)
    return final_answer

local_search("Who is Mr. Tackleton and what gifts did he gave?")
local_search=local_search("What relationship did Caleb and Berta had?")
local=local_search("Who was daughter of Caleb?")
#Finished!!!!



















local_search_query = """
CALL db.index.vector.queryNodes('entities', $k, $embedding)
YIELD node, score
WITH collect(node) as nodes
WITH collect {
    UNWIND nodes as n
    MATCH (n)<-[:HAS_ENTITY]->(c:__Chunk__)
    WITH c, count(distinct n) as freq
    RETURN c.text AS chunkText
    ORDER BY freq DESC
    LIMIT $topChunks
} AS text_mapping,
collect {
    UNWIND nodes as n
    MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
    WITH c, c.rank as rank, c.weight AS weight
    RETURN c.summary 
    ORDER BY rank, weight DESC
    LIMIT $topCommunities
} AS report_mapping,
collect {
    UNWIND nodes as n
    MATCH (n)-[r:SUMMARIZED_RELATIONSHIP]-(m) 
    WHERE n IN nodes AND m IN nodes
    RETURN r.summary AS descriptionText
    ORDER BY r.rank, r.weight DESC 
    LIMIT $topInsideRels
} as insideRels,
collect {
    UNWIND nodes as n
    RETURN n.summary AS descriptionText
} as entities
RETURN {Chunks: text_mapping, Reports: report_mapping, 
       Relationships: insideRels, 
       Entities: entities} AS text
"""