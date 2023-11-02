import sys
import traceback

import networkx
from networkx.algorithms.isomorphism import MultiDiGraphMatcher

import ParseRules
from ParseRules import Rule, Atom

universal_node_ids = None


def is_isomorphic(rule1, rule2):
    return rule1.id_print().replace("g", "h") == rule2.id_print().replace("g", "h")


def union_no_isomorphs_rules(rules1, rules2):
    result = []

    for rule_1 in rules1:
        result.append(rule_1)

    for rule_2 in rules2:
        is_isomorphic_flag = False

        # Check if graph_s is isomorphic to any graph in the result list

        for rule_r in result:
            if rule_2 == rule_r:
                is_isomorphic_flag = True
                break

        # If graph_s is not isomorphic to any graph in the result list, add it
        if not is_isomorphic_flag:
            result.append(rule_2)

    return result


def intersection_isomorphs_rules(rules1, rules2):
    result = []

    # Iterate over graphs in graph_big
    for rules_1 in rules1:
        for rules_2 in rules2:
            if rules_1 == rules_2:
                # Add graph_b to the result list and break the inner loop
                result.append(rules_1)
                break

    return result


def union_no_isomorphs(graph1, graph2):
    result = []
    # print(f"Size of nwx_mapping inside:{len(nwx_mapping)}")
    # print(f"Graph1 size: {len(graph1)}; Graph2 size: {len(graph2)}")

    for relation_key in graph1:
        graphs = graph1[relation_key]

        if relation_key not in graph2:
            result.extend(graphs)
        else:

            sub_result = []
            # Check if graph_s is isomorphic to any graph in the result list
            for graph in graphs:
                sub_result.append(graph)
            for graph_s in graph2[relation_key]:
                is_isomorphic_flag = False
                for graph_r in sub_result:
                    matcher = MultiDiGraphMatcher(graph_r, graph_s, edge_match=edge_match, node_match=node_match)
                    if matcher.is_isomorphic():
                        is_isomorphic_flag = True
                        break

                # If graph_s is not isomorphic to any graph in the result list, add it
                if not is_isomorphic_flag:
                    sub_result.append(graph_s)

            result.extend(sub_result)

    for relation_key in graph2:
        if relation_key not in graph1:
            result.extend(graph2[relation_key])

    return result


def intersection_isomorphs(graph1, graph2):
    result = []

    # Iterate over graphs in graph_big
    for relation_key in graph1:
        graphs = graph1[relation_key]

        if relation_key in graph2:
            for graph_1 in graphs:
                # Check if graph_b is isomorphic to any graph in graph_small
                # print("G1:", graph_1.nodes(), graph_1.edges())
                for graph_2 in graph2[relation_key]:
                    # print("G2:", graph_2.nodes(), graph_2.edges())
                    matcher = MultiDiGraphMatcher(graph_1, graph_2, edge_match=edge_match, node_match=node_match)
                    if matcher.is_isomorphic():
                        # Add graph_b to the result list and break the inner loop
                        result.append(graph_1)
                        break

    return result


def set_universal_node_id_mapping(universal_node_ids_local):
    global universal_node_ids

    universal_node_ids = universal_node_ids_local


def get_universal_node_id_mapping(rules_list):
    """
        Takes input of list of list of rules (Multiple Rules) and maps variables found to IDs

        Args:
            rules_list (List[List[Rule]]): Multiple outputs from ParseRules concatenated into a list

        Returns:
            inner_universal_node_ids (dict): Mapping of variables found in rules to their corresponding ID numbers
    """
    inner_universal_node_ids = {}
    for current_rule_list in rules_list:
        for rule in current_rule_list:
            variables = rule.get_variables()

            for variable in variables:
                if variable not in inner_universal_node_ids:
                    inner_universal_node_ids[variable] = len(inner_universal_node_ids)
    return inner_universal_node_ids


def get_networkx_representation(rule):
    """
        Convert given Rule object into its NetworkX Graph implementation

        Args:
            rule (Rule): The given input rule object

        Returns:
            G (networkx.DiGraph): The converted graph

    """
    global universal_node_ids
    variables = rule.get_variables()
    G = networkx.MultiDiGraph()

    for variable in variables:
        G.add_node(variable, id=universal_node_ids[variable])

    for atom in rule.body_atoms:
        G.add_edge(atom.variable1.replace("?", ""), atom.variable2.replace("?", ""), r=atom.relationship_name)

    G.add_edge(rule.head_atom.variable1.replace("?", ""), rule.head_atom.variable2.replace("?", ""),
               r=rule.head_atom.relationship_name)
    return G


def convert_rules_to_networkx_graphs(rules):
    """
        Convert list of Rules to list of networkx.DiGraphs

        Args:
            rules: List of Rule

        Returns:
            graphs: List of Networkx.DiGraph
            networkx_to_rule_mapping: Mapping of networkx.DiGraph to Rule
    """

    networkx_to_rule_mapping = {}
    graphs = []

    for rule in rules:
        nwx_graph = get_networkx_representation(rule)

        networkx_to_rule_mapping[nwx_graph] = rule
        graphs.append(nwx_graph)

    return graphs, networkx_to_rule_mapping


def node_match(node1, node2):
    """
        Function for matching two nodes for isomorphism

        Args:
            node1: 1st node
            node2: 2nd node

        Returns:
            boolean: If the two nodes were a match or not
    """

    global universal_node_ids
    # if "id" not in node1 or "id" not in node2:
    #     return True
    if node1["id"] == universal_node_ids["a"]:
        if node2["id"] == universal_node_ids["a"]:
            return True
        else:
            return False

    elif node1["id"] == universal_node_ids["b"]:
        if node2["id"] == universal_node_ids["b"]:
            return True
        else:
            return False

    else:
        if node2["id"] != universal_node_ids["a"] and node2["id"] != universal_node_ids["b"]:
            return True
        else:
            return False


def edge_match(edge1, edge2):
    counts = set()

    # print(edge1, edge2)
    for key in edge1:
        value = edge1[key]

        r = value['r']
        counts.add(r)
    # print(counts)
    for key in edge2:
        value = edge2[key]
        r = value['r']
        # print(r, r not in counts)
        if r not in counts:
            return False

    return True


def get_graph_split_by_head(graphs_big, nwx_mapping):
    result = {}

    for graph in graphs_big:
        rule = nwx_mapping[graph]

        if rule.head_atom.relationship_name not in result:
            result[rule.head_atom.relationship_name] = []

        result[rule.head_atom.relationship_name].append(graph)

    return result


def filter_rules(rules_big, folder_to_datasets, dataset2):
    relations = []

    with open(f"{folder_to_datasets}/{dataset2}/relation2id.txt") as f:
        f.readline()

        for line in f:
            line = line.strip()
            relations.append(line.split("\t")[0])
    result = []

    for rule in rules_big:
        atoms = []
        atoms.extend(rule.body_atoms)
        atoms.append(rule.head_atom)
        flag = True
        for atom in atoms:
            if atom.relationship_name not in relations:
                flag = False
                break

        if flag:
            result.append(rule)

    return result


def compare_rules_in_paired_datasets(dataset_big, dataset_small, path_to_rules, path_to_datasets):
    models = ["boxe", "complex", "hake", "hole", "quate", "rotate", "rotpro", "toruse", "transe", "tucker"]
    global universal_node_ids
    for model in models:
        r_big = ParseRules.ParseRule(
            filename=f"{path_to_rules}/{dataset_big}/{model}/{dataset_big}_{model}_mat_0_rules.tsv",
            folder_to_datasets=path_to_datasets, model_name=model, dataset_name=dataset_big)
        r_small = ParseRules.ParseRule(
            filename=f"{path_to_rules}/{dataset_small}/{model}/{dataset_small}_{model}_mat_0_rules.tsv",
            folder_to_datasets=path_to_datasets, model_name=model,
            dataset_name=dataset_small)

        r_big.parse_rules_from_file(1.0)
        r_small.parse_rules_from_file(1.0)

        rules_big = r_big.rules
        rules_big = filter_rules(rules_big, folder_to_datasets, dataset2)
        rules_small = r_small.rules
        universal_node_ids = get_universal_node_id_mapping([rules_big, rules_small])
        graphs_big, nwx_mapping_big = convert_rules_to_networkx_graphs(rules_big)
        graphs_small, nwx_mapping_small = convert_rules_to_networkx_graphs(rules_small)

        nwx_mapping = {**nwx_mapping_big, **nwx_mapping_small}
        graph_by_head_big = get_graph_split_by_head(graphs_big, nwx_mapping)
        graph_by_head_small = get_graph_split_by_head(graphs_small, nwx_mapping)
        graph_intersection = intersection_isomorphs(graph1=graph_by_head_big, graph2=graph_by_head_small)

        graph_union = union_no_isomorphs(graph1=graph_by_head_big, graph2=graph_by_head_small)

        # graph_intersection = intersection_isomorphs_rules(rules_big, rules_small)
        # graph_union = union_no_isomorphs_rules(rules_big, rules_small)
        print(
            f"Model:{model}; Number of intersections:{len(graph_intersection)}; Number of unions:{len(graph_union)}; Ratio:{len(graph_intersection) * 1.0 / len(graph_union)}")


def sort_by_relationship(atom):
    if atom.variable1 == "?a" or atom.variable2 == "?a":
        return 1
    else:
        return 10


def test_intersection_and_union_rule():
    r1 = Rule(Atom("1", "?a", "?b", "1"), [Atom("1", "?a", "?h", "1"), Atom("1", "?h", "?b", "1")], 0.1, 0.1,
              "a")  # ISO1
    r2 = Rule(Atom("1", "?a", "?b", "1"), [Atom("1", "?h", "?a", "1"), Atom("1", "?h", "?b", "1")], 0.1, 0.1, "a")
    r3 = Rule(Atom("1", "?a", "?b", "1"), [Atom("1", "?a", "?g", "1"), Atom("1", "?g", "?b", "1")], 0.1, 0.1,
              "a")  # ISO1
    r4 = Rule(Atom("2", "?a", "?b", "2"), [Atom("2", "?h", "?a", "2"), Atom("2", "?h", "?b", "2")], 0.1, 0.1,
              "a")  # ISO2
    r5 = Rule(Atom("2", "?a", "?b", "2"), [Atom("2", "?g", "?a", "2"), Atom("2", "?g", "?b", "2")], 0.1, 0.1,
              "a")  # ISO2
    r6 = Rule(Atom("2", "?a", "?b", "2"), [Atom("2", "?b", "?a", "2")], 0.1, 0.1, "a")
    r7 = Rule(Atom("3", "?a", "?b", "3"), [Atom("3", "?b", "?a", "3")], 0.1, 0.1, "a")
    r8 = Rule(Atom("3", "?a", "?b", "2"), [Atom("2", "?a", "?b", "2")], 0.1, 0.1, "a")
    r9 = Rule(Atom("2", "?a", "?b", "2"), [Atom("1", "?a", "?h", "1"), Atom("1", "?h", "?b", "1")], 0.1, 0.1, "a")
    r10 = Rule(Atom("3", "?a", "?b", "1"), [Atom("2", "?a", "?b", "1"), Atom("1", "?a", "?b", "2")], 0.1, 0.1, "a")
    r11 = Rule(Atom("3", "?a", "?b", "1"), [Atom("32", "?h", "?b", "32"), Atom("146", "?h", "?a", "146")], 0.1, 0.1,
               "a")  # ISO3
    r12 = Rule(Atom("3", "?a", "?b", "1"), [Atom("146", "?h", "?a", "146"), Atom("32", "?h", "?b", "32")], 0.1, 0.1,
               "a")  # ISO3

    for rule in [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12]:
        print(rule.id_print())

    rules1 = [r1, r4, r11, r2, r6, r10]
    rules2 = [r3, r5, r12, r7, r8, r9]

    rules_intersection = intersection_isomorphs_rules(rules1, rules2)

    print("Intersection")
    for rule in rules_intersection:
        print(rule.id_print())

    rules_union = union_no_isomorphs_rules(rules1, rules2)

    print("Union")
    for rule in rules_union:
        print(rule.id_print())


def test_intersection_and_union_nx():
    global universal_node_ids
    r1 = Rule(Atom("1", "?a", "?b", "1"), [Atom("1", "?a", "?h", "1"), Atom("1", "?h", "?b", "1")], 0.1, 0.1,
              "a")  # ISO1
    r2 = Rule(Atom("1", "?a", "?b", "1"), [Atom("1", "?h", "?a", "1"), Atom("1", "?h", "?b", "1")], 0.1, 0.1, "a")
    r3 = Rule(Atom("1", "?a", "?b", "1"), [Atom("1", "?a", "?g", "1"), Atom("1", "?g", "?b", "1")], 0.1, 0.1,
              "a")  # ISO1
    r4 = Rule(Atom("2", "?a", "?b", "2"), [Atom("2", "?h", "?a", "2"), Atom("2", "?h", "?b", "2")], 0.1, 0.1,
              "a")  # ISO2
    r5 = Rule(Atom("2", "?a", "?b", "2"), [Atom("2", "?g", "?a", "2"), Atom("2", "?g", "?b", "2")], 0.1, 0.1,
              "a")  # ISO2
    r6 = Rule(Atom("2", "?a", "?b", "2"), [Atom("2", "?b", "?a", "2")], 0.1, 0.1, "a")
    r7 = Rule(Atom("3", "?a", "?b", "3"), [Atom("3", "?b", "?a", "3")], 0.1, 0.1, "a")
    r8 = Rule(Atom("3", "?a", "?b", "3"), [Atom("2", "?a", "?b", "2")], 0.1, 0.1, "a")
    r9 = Rule(Atom("2", "?a", "?b", "2"), [Atom("1", "?a", "?h", "1"), Atom("1", "?h", "?b", "1")], 0.1, 0.1, "a")
    r10 = Rule(Atom("3", "?a", "?b", "3"), [Atom("1", "?a", "?b", "1"), Atom("2", "?a", "?b", "2")], 0.1, 0.1,
               "a")  # ISO3
    r11 = Rule(Atom("3", "?a", "?b", "3"), [Atom("2", "?a", "?b", "2"), Atom("1", "?a", "?b", "1")], 0.1, 0.1,
               "a")  # ISO3
    r12 = Rule(Atom("3", "?a", "?b", "3"), [Atom("4", "?a", "?b", "4"), Atom("1", "?a", "?b", "2")], 0.1, 0.1, "a")
    for rule in [r1, r2, r3, r4]:
        print(rule.id_print())
    rules1 = [r1, r2, r4, r6, r9, r10]
    rules2 = [r3, r5, r7, r8, r12, r11]

    universal_node_ids = get_universal_node_id_mapping([rules1, rules2])
    graphs_big, nwx_mapping_big = convert_rules_to_networkx_graphs(rules1)
    graphs_small, nwx_mapping_small = convert_rules_to_networkx_graphs(rules2)

    nwx_mapping = {**nwx_mapping_big, **nwx_mapping_small}
    graph_by_head_big = get_graph_split_by_head(graphs_big, nwx_mapping)
    graph_by_head_small = get_graph_split_by_head(graphs_small, nwx_mapping)
    graph_intersection = intersection_isomorphs(graph1=graph_by_head_big, graph2=graph_by_head_small)

    graph_union = union_no_isomorphs(graph1=graph_by_head_big, graph2=graph_by_head_small)

    print("Intersection")
    for graph in graph_intersection:
        if graph in nwx_mapping_big:
            print(nwx_mapping_big[graph].id_print())
        else:
            print(nwx_mapping_small[graph].id_print())

    print("Union")
    for graph in graph_union:
        if graph in nwx_mapping_big:
            print(nwx_mapping_big[graph].id_print())
        else:
            print(nwx_mapping_small[graph].id_print())


if __name__ == "__main__":
    folder_to_rules = sys.argv[1]
    folder_to_datasets = sys.argv[2]
    dataset1 = sys.argv[3]
    dataset2 = sys.argv[4]
    print(f"Dataset 1: {dataset1}, Dataset 2: {dataset2}")
    compare_rules_in_paired_datasets(dataset1, dataset2, path_to_rules=folder_to_rules,
                                     path_to_datasets=folder_to_datasets)
