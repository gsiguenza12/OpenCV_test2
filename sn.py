#https://programminghistorian.org/en/lessons/exploring-and-analyzing-network-data-with-python
import csv
import matplotlib.pyplot as plt
from operator import itemgetter
import networkx as nx
from networkx.algorithms import community

# Read in the nodelist file
with open('quakers_nodelist.csv', 'r') as nodecsv:
    nodereader = csv.reader(nodecsv)
    nodes = [n for n in nodereader][1:]

# Get a list of just the node names (the first item in each row)
node_names = [n[0] for n in nodes]

# Read in the edgelist file
with open('quakers_edgelist.csv', 'r') as edgecsv:
    edgereader = csv.reader(edgecsv)
    edges = [tuple(e) for e in edgereader][1:]

# Print the number of nodes and edges in our two lists
print(len(node_names))
print(len(edges))

G = nx.Graph() # Initialize a Graph object
G.add_nodes_from(node_names) # Add nodes to the Graph
G.add_edges_from(edges) # Add edges to the Graph
# print(nx.info(G)) # Print information about the Graph
print(G)
# Create an empty dictionary for each attribute
hist_sig_dict = {}
gender_dict = {}
birth_dict = {}
death_dict = {}
id_dict = {}

for node in nodes: # Loop through the list of nodes, one row at a time
    hist_sig_dict[node[0]] = node[1] # Access the correct item, add it to the corresponding dictionary
    gender_dict[node[0]] = node[2]
    birth_dict[node[0]] = node[3]
    death_dict[node[0]] = node[4]
    id_dict[node[0]] = node[5]

# Add each dictionary as a node attribute to the Graph object
nx.set_node_attributes(G, hist_sig_dict, 'historical_significance')
nx.set_node_attributes(G, gender_dict, 'gender')
nx.set_node_attributes(G, birth_dict, 'birth_year')
nx.set_node_attributes(G, death_dict, 'death_year')
nx.set_node_attributes(G, id_dict, 'sdfb_id')

# Loop through each node, to access and print all the "birth_year" attributes
for n in G.nodes():
    print(n, G.nodes[n]['birth_year'])

for n in G.nodes():
    print(n, nx.clustering(G, n))

nx.draw(G)  # draws the networkx graph containing nodes which are declared till before
plt.show()  # displays the networkx graph on matplotlib canvas
