import os
from pydantic import BaseModel, Field
from typing import Optional, List
import json
from langchain_google_vertexai import ChatVertexAI
from vertexai.generative_models import GenerativeModel
import re
from neo4j import GraphDatabase


class Location(BaseModel):
    """
    Represents a physical location including address, city, state, and country.
    """

    address: Optional[str] = Field(
        ..., description="The street address of the location."
    )
    city: Optional[str] = Field(..., description="The city of the location.")
    state: Optional[str] = Field(
        ..., description="The state or region of the location."
    )
    country: str = Field(
        ...,
        description="The country of the location. Use the two-letter ISO standard.",
    )


class Organization(BaseModel):
    """
    Represents an organization, including its name and location.
    """

    name: str = Field(..., description="The name of the organization.")
    location: Location = Field(
        ..., description="The primary location of the organization."
    )
    role: str = Field(
        ...,
        description="The role of the organization in the contract, such as 'provider', 'client', 'supplier', etc.",
    )

contract_types = [
    "Service Agreement",
    "Licensing Agreement",
    "Non-Disclosure Agreement (NDA)",
    "Partnership Agreement",
    "Lease Agreement"
]

class Contract(BaseModel):
    """
    Represents the key details of the contract.
    """

    contract_type: str = Field(
        ...,
        description="The type of contract being entered into.",
        enum=contract_types,
    )
    parties: List[Organization] = Field(
        ...,
        description="List of parties involved in the contract, with details of each party's role.",
    )
    effective_date: str = Field(
        ...,
        description="The date when the contract becomes effective. Use yyyy-MM-dd format.",
    )
    term: str = Field(
        ...,
        description="The duration of the agreement, including provisions for renewal or termination.",
    )
    contract_scope: str = Field(
        ...,
        description="Description of the scope of the contract, including rights, duties, and any limitations.",
    )
    end_date: Optional[str] = Field(
        ...,
        description="The date when the contract becomes expires. Use yyyy-MM-dd format.",
    )
    total_amount: Optional[float] = Field(
        ..., description="Total value of the contract."
    )
    governing_law: Optional[Location] = Field(
        ..., description="The jurisdiction's laws governing the contract."
    )

system_message = """
You are an information extraction model.  
You MUST output **only valid JSON** that strictly matches the following Pydantic schema.  
Do NOT output markdown, explanations, commentary, or any text outside the JSON.

The JSON MUST match this exact structure:

{
  "contract_type": "<one of: 'Service Agreement', 'Licensing Agreement', 'Non-Disclosure Agreement (NDA)', 'Partnership Agreement', 'Lease Agreement'>",
  "parties": [
    {
      "name": "<organization name>",
      "location": {
        "address": "<street address>",
        "city": "<city>",
        "state": "<state or region>",
        "country": "<2-letter ISO country code>"
      },
      "role": "<provider | client | supplier | other>"
    }
  ],
  "effective_date": "<yyyy-MM-dd>",
  "term": "<description of duration, renewal, termination>",
  "contract_scope": "<summary of rights, duties, limitations>",
  "end_date": "<yyyy-MM-dd or null>",
  "total_amount": <float or null>,
  "governing_law": {
    "address": "<street address>",
    "city": "<city>",
    "state": "<state or region>",
    "country": "<2-letter ISO country code>"
  }
}

RULES:
- ALWAYS return a single JSON object.
- NEVER output Markdown, bullet points, headings, or text outside JSON.
- Dates MUST be normalized to yyyy-MM-dd format.
- All required fields must be present, even if values are unknown (use null where allowed).
- "country" must be a two-letter ISO code (e.g., "US").
- If the input does not contain information for a required field, infer a reasonable placeholder based on context.

"""

# Read the file
with open('./license.txt', 'r') as file:
    contents = file.read()


def strip_code_cypher(text: str) -> str:
    text = text.strip()
    # remove ```json or ``` and closing ```
    text = re.sub(r"^```(?:\w+)?\s*|```$", "", text, flags=re.IGNORECASE).strip()
    return text

def chat(messages, response_format=None):
    llm = ChatVertexAI(model="gemini-2.0-flash")
    return llm.invoke(messages, response_format=response_format)

def extract(document):
    messages = [
        ("system", system_message),
        ("user", document)
    ]

    # No response_format here — tell Gemini manually to output JSON
    response = chat(messages)

    raw = strip_code_cypher(response.content)  # text from Gemini (hopefully JSON)

    # turn JSON string into your Contract model
    result = Contract.model_validate_json(raw)

    return result

data = extract(contents)
print(data)

###########################
#Inspecting Contract object
###########################

for field, value in data.model_dump().items():
    print(field, "=>", value)

from pprint import pprint
pprint(data.model_dump(), width=120)

###############
#Creating a graph
################
driver = GraphDatabase.driver("neo4j://127.0.0.1:7687",
    auth=("neo4j", "qawsedRF123"),
    notifications_min_severity="OFF"
)
driver.execute_query(
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Contract) REQUIRE c.id IS UNIQUE;"
)
driver.execute_query(
    "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE;"
)
driver.execute_query(
    "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Location) REQUIRE l.fullAddress IS UNIQUE;"
)

import_query = """WITH $data AS contract_data
// Create Contract node
MERGE (contract:Contract {id: randomUUID()})
SET contract += {
  contract_type: contract_data.contract_type,
  effective_date: contract_data.effective_date,
  term: contract_data.term,
  contract_scope: contract_data.contract_scope,
  end_date: contract_data.end_date,
  total_amount: contract_data.total_amount,
  governing_law: contract_data.governing_law.state + ' ' +
                 contract_data.governing_law.country
}
WITH contract, contract_data
// Create Party nodes and their locations
UNWIND contract_data.parties AS party
MERGE (p:Organization {name: party.name})
MERGE (loc:Location {
  fullAddress: party.location.address + ' ' +
                party.location.city + ' ' +
                party.location.state + ' ' +
                party.location.country})
SET loc += {
  address: party.location.address,
  city: party.location.city,
  state: party.location.state,
  country: party.location.country
}
// Link party to their location
MERGE (p)-[:LOCATED_AT]->(loc)
// Link parties to the contract
MERGE (p)-[r:HAS_PARTY]->(contract)
SET r.role = party.role
"""
driver.execute_query(import_query, data=data)
driver.execute_query(import_query, {"data": data.model_dump()})