import os, json
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_vertexai import VertexAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import OpenAI
import networkx as nx
#from langchain.chains import GraphQAChain
from langchain_classic.chains import GraphQAChain
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(model='gpt-3.5-turbo-instruct',api_key=OPENAI_API_KEY)


prompt = "convert given json object into unstructured, readable paragraph. output just the final paragraph . json : {}"

text = ""

# Load your JSON data from a file
with open("employee.json", "r") as file:
    json_data = json.load(file)

#print("JSON Data:", json_data)

for employee in json_data:
    response = llm.generate([prompt.format(employee)])
    text += response.generations[0][0].text + "\n"

#print("Converted Text:", text)

documents = [Document(page_content=text)]
print("Documents:", documents)
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents)

df = pd.DataFrame(columns=["node1", "node2", "relation"])

for edge in graph_documents[0].relationships:
    df = pd.concat([df, pd.DataFrame([{'node1': edge.source.id, 'node2': edge.target.id, 'relation': edge.type}])], ignore_index=True)

print("Extracted Relationships DataFrame:", df)


graph = NetworkxEntityGraph()

# Add nodes to the graph
for node in graph_documents[0].nodes:
    graph.add_node(node.id)

# Add edges to the graph
for edge in graph_documents[0].relationships:
    graph._graph.add_edge(
            edge.source.id,
            edge.target.id,
            relation=edge.type,
        )


chain = GraphQAChain.from_llm(
    llm=llm, 
    graph=graph, 
    verbose=True
)



question = """where Bryan Smith works?"""
print(chain.run(question))

"""
> Entering new GraphQAChain chain...
Entities Extracted:
 Bryan Smith
Full Context:
Bryan Smith WORKS_AT Incentivize
Bryan Smith SUPERVISED_BY Rebecca Andrews
Bryan Smith CONTACTED_BY amanda52@example.net
Bryan Smith CONTACTED_BY 815-823-1247x52270
Bryan Smith WORKING_ON Multi-tiered discrete Internet solution
Bryan Smith WORKING_ON Empower Cross-Media Users

> Finished chain.
 Incentivize
"""