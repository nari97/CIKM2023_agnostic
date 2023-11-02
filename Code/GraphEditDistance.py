from typing import List

import networkx

from ParseRules import Rule
from networkx import DiGraph


def node_match(n1, n2):
    return n1["id"] == n2["id"]


def edge_match(edge1, edge2):
    return edge1["r"] == edge2["r"]


def create_networkx_graph(rule):
    variable_to_id = {"a": 1, "b": 2, "g": 3}

    G = DiGraph()

    for node in variable_to_id.keys():
        G.add_node(node, id=variable_to_id[node])

    for atom in rule.body_atoms:
        variable1 = atom.variable1.replace("?", "")
        variable2 = atom.variable2.replace("?", "")
        variable1 = "g" if (variable1 == "g" or variable1 == "h") else variable1
        variable2 = "g" if (variable2 == "g" or variable2 == "h") else variable2
        G.add_edge(variable1, variable2, r=atom.relationship)

    return G


def compute_graph_edit_distance(rules: List[Rule]):
    edit_distance_storage = {}

    networkx_graphs = {}
    for rule in rules:
        networkx_graphs[rule] = create_networkx_graph(rule)

    # print("Length of networkx:", len(networkx_graphs))

    ctr1 = 0
    for i in range(len(rules)):
        G1 = networkx_graphs[rules[i]]
        for j in range(0, len(rules)):
            G2 = networkx_graphs[rules[j]]
            worst_case = len(G1.edges) + len(G2.edges)
            edit_distance_storage[(rules[i], rules[j])] = networkx.graph_edit_distance(G1, G2, node_match=node_match, edge_match=edge_match)*1.0/worst_case

    return edit_distance_storage

