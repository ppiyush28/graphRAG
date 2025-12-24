import os
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

# llm = GoogleGenerativeAI(
#     model="gemini-3-pro", #" "gemini-3-pro-preview",
#     max_output_tokens=4000,
#     api_key=os.getenv("GOOGLE_GENAI_API_KEY"),
# )

text = """
Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris. 
"""


documents = [Document(page_content=text)]
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents)



llm_transformer_filtered = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Country", "Organization"],
    allowed_relationships=["NATIONALITY", "LOCATED_IN", "WORKED_AT", "SPOUSE"],
)
graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(
    documents
)



graph = NetworkxEntityGraph()

# Add nodes to the graph
for node in graph_documents_filtered[0].nodes:
    graph.add_node(node.id)

# Add edges to the graph
for edge in graph_documents_filtered[0].relationships:
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



question = """Who is Marie Curie?"""
print(chain.run(question))

"""
> Entering new GraphQAChain chain...
Entities Extracted:
 Marie Curie
Full Context:
Marie Curie NATIONALITY Polish
Marie Curie NATIONALITY French
Marie Curie WORKED_AT University of Paris

> Finished chain.

Marie Curie is a scientist who worked at the University of Paris. Her nationality is both Polish and French.
""" 
