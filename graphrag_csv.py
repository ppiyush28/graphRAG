import os
import pandas as pd
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_vertexai import VertexAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import OpenAI
import networkx as nx
#from langchain.chains import GraphQAChain
from langchain_classic.chains import GraphQAChain
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(model='gpt-3.5-turbo-instruct',api_key=OPENAI_API_KEY)

data = pd.read_csv("airlines_flights_data.csv")

def duration(x):
    if x < 10:
        return "near flight"
    elif x < 20:
        return "far flight"
    else:
        return "too far flight"

data['duration'] = data.apply(lambda x:duration(x["duration"]),axis=1)       
print(data.head())



graph = NetworkxEntityGraph()

# Add nodes to the graph
for id, row in data.iterrows():
    graph.add_node(row['source_city'])
    graph.add_node(row['destination_city'])

# Add edges to the graph
for id, row in data.iterrows():
    graph._graph.add_edge(
            row['source_city'],
            row['destination_city'],
            relation=row['duration'],
        )


chain = GraphQAChain.from_llm(
    llm=llm, 
    graph=graph, 
    verbose=True
)



# question = """places to vist near pune"""
# print(chain.run(question))


"""
> Entering new GraphQAChain chain...
Entities Extracted:
 Pune
Full Context:


> Finished chain.
 Some possible places to visit near Pune include: 
- Mumbai - a bustling city with a vibrant culture and many historical attractions
- Lonavala - a hill station known for its beautiful scenery and popular for weekend getaways
- Mahabaleshwar - a hill station known for its stunning views and strawberry farms
- Lavasa - a planned city with picturesque lakes and waterfalls
- Alibaug - a coastal town with clean beaches and historic forts
"""


question = """which is short flight from banglore to delhi in ealry moring"""
print(chain.run(question))

"""
> Entering new GraphQAChain chain...
Entities Extracted:
 Banglore, Delhi
Full Context:
Delhi near flight Mumbai
Delhi far flight Bangalore
Delhi far flight Kolkata
Delhi far flight Hyderabad
Delhi far flight Chennai

> Finished chain.
 Mumbai
"""