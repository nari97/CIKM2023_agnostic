import numpy as np
from networkx.algorithms.isomorphism import MultiDiGraphMatcher

import Isomorphism
import ParseRules
from Isomorphism import get_universal_node_id_mapping, convert_rules_to_networkx_graphs
from LatexUtils import create_multi_table, create_regular_table
from matplotlib import pyplot as plt
from ParseRules import Atom, Rule


def get_entity_types(relationship):
    index = 0
    for i in range(len(relationship)):
        val = relationship[i]
        if 97 <= ord(val) <= 122:
            index = i
            break

    if ">" in relationship:
        lhs = relationship[0:index]
        rhs = relationship[index + 2:]
    else:
        lhs = relationship[0:index]
        rhs = relationship[index + 1:]

    return [lhs, rhs]


def check_rules(rules, entity_types):
    match = 0

    for rule in rules:
        type_dict = {}
        atoms = list(rule.body_atoms)
        atoms.append(rule.head_atom)
        for atom in atoms:
            relationship = atom.relationship_name

            lhs, rhs = entity_types[relationship]

            if atom.variable1 not in type_dict:
                type_dict[atom.variable1] = set()

            type_dict[atom.variable1].add(lhs)

            if atom.variable2 not in type_dict:
                type_dict[atom.variable2] = set()

            type_dict[atom.variable2].add(rhs)

        flag = True
        for key in type_dict:
            if len(type_dict[key]) > 1:
                flag = False
                break

        if flag:
            match += 1

    return match


def check_rule_overlap(rules, file_to_write):
    rule_dict = {}

    for model_name in rules:
        rules_by_model = rules[model_name]

        for rule in rules_by_model:
            rp = rule.relationship_print()

            if rp not in rule_dict:
                rule_dict[rp] = set()

            rule_dict[rp].add((model_name, rule.selectivity))

    predicates = {}
    rule_stats = {}
    for rule in rule_dict:
        head_predicate = rule.split("==>")[1].split("(")[0].strip()

        if head_predicate not in predicates:
            predicates[head_predicate] = 0

        longest_so_far = predicates[head_predicate]

        if len(rule_dict[rule]) >= longest_so_far:
            predicates[head_predicate] = len(rule_dict[rule])

            rule_stats[head_predicate] = {"Rule": rule, "Models": rule_dict[rule]}

    predicates_sorted = sort_dict_by_value_desc(predicates)

    for val in predicates_sorted:
        rule = rule_stats[val[0]]["Rule"]
        models_selec = rule_stats[val[0]]["Models"]
        file_to_write.write(f"Predicate: {val[0]}; Rule: {rule}; Count: {val[1]}; Model stats: {models_selec}\n")


def sort_dict_by_value_desc(d):
    sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return [[k, v] for k, v in sorted_items]


def get_hetionet_relation_types():
    hetionet_folder = r"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Datasets\Hetionet"

    entity_types = {}
    with open(f"{hetionet_folder}/relation2id.txt") as f:
        f.readline()
        for line in f:
            relation = line.split("\t")[0]
            entity_types[relation] = get_entity_types(relation)

    return entity_types


def get_biokg_relation_types():
    biokg_folder = r"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Datasets\BioKG"

    all_entities = {}
    all_triples = {}
    for type_of_file in ["metadata.disease", "metadata.drug", "metadata.pathway", "metadata.protein",
                         "properties.genetic_disorder"]:
        with open(f"{biokg_folder}/biokg.{type_of_file}.tsv") as f:
            for line in f:
                line = line.strip()
                entity = line.split("\t")[0]

                all_entities[entity] = type_of_file.split(".")[1]

    with open(f"{biokg_folder}/biokg.properties.pathway.tsv") as f:
        for line in f:
            line = line.strip()
            entity1 = line.split("\t")[0]
            entity2 = line.split("\t")[2]

            all_entities[entity1] = "pathway"
            all_entities[entity2] = "pathway"

    with open(f"{biokg_folder}/biokg.links.tsv") as f:
        for line in f:
            line = line.strip()
            head, relation, tail = line.split("\t")

            if relation not in all_triples:
                all_triples[relation] = []
            all_triples[relation].append([head, relation, tail])

    entity_types = {}
    for relation in all_triples:
        lhs = set()
        rhs = set()

        for triple in all_triples[relation]:
            try:

                if triple[0] not in all_entities:
                    if "R-" in triple[0]:
                        lhs.add("pathway")
                    elif "MIM" in triple[0]:
                        lhs.add("genetic_disorder")
                else:
                    lhs.add(all_entities[triple[0]])

                if triple[2] not in all_entities:
                    if "R-" in triple[2]:
                        rhs.add("pathway")
                    elif "MIM" in triple[2]:
                        rhs.add("genetic_disorder")
                else:
                    rhs.add(all_entities[triple[2]])
            except:
                print(f"{relation},{triple}")

        entity_types[relation] = [lhs.pop(), rhs.pop()]

    return entity_types


def parse_best_rule_str(rule, hc, pca, selec):
    rule = rule.replace("( ", "").replace("  )", "")

    splits = rule.split(" ")

    atom1 = parse_atom_str(splits[0])
    atom2 = None
    if len(splits) == 4:
        atom2 = parse_atom_str(splits[1])

    head_atom = parse_atom_str(splits[-1])

    if atom2 is None:
        body_atoms = [atom1]
    else:
        body_atoms = [atom1, atom2]

    rule_object = Rule(head_atom, body_atoms, hc, pca, "", 1.0)
    rule_object.selectivity = selec

    return rule_object


def parse_atom_str(atom):
    relationship = atom[0:atom.index("(")]

    variable1, variable2 = atom[atom.index("("):].replace("(", "").replace(")", "").replace("g", "z").replace("h",
                                                                                                              "z").split(
        ",")
    return Atom(None, variable1, variable2, relationship)


def get_graph_key(rule_stats, graph):
    for graph_key in rule_stats:
        # print(graph_key)
        matcher = MultiDiGraphMatcher(graph_key, graph, edge_match=Isomorphism.edge_match,
                                      node_match=Isomorphism.node_match)
        if matcher.is_isomorphic():
            return graph_key

    return None


def rule_overlap_between_models(dataset_name):
    folder_to_best_rules = rf"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\BestRules/{dataset_name}"
    print(f"Dataset name: {dataset_name}")
    models = ["boxe", "complex", "hake", "hole", "quate", "rotate", "rotpro", "toruse", "transe", "tucker"]
    universal_node_ids = {'z': 0, 'b': 1, 'a': 2}
    rules_by_model = {}
    nwx_mapping = {}
    cnt_rules = 0
    for model_name in models:
        rules = []
        try:
            with open(f"{folder_to_best_rules}/{model_name}/{dataset_name}_{model_name}_mat_0_best1.tsv") as f:
                for line in f:
                    line = line.strip()
                    rule, hc, pca, selec = line.split("\t")
                    selec = float(selec)
                    hc = float(hc)
                    pca = float(pca)
                    cnt_rules += 1

                    rule_object = parse_best_rule_str(rule, hc, pca, selec)
                    # print(f"Rule: {rule_object.relationship_print()}")
                    rules.append(rule_object)

            Isomorphism.set_universal_node_id_mapping(universal_node_ids)
            graphs_big, nwx_mapping_rules = convert_rules_to_networkx_graphs(rules)

            rules_by_model[model_name] = graphs_big
            nwx_mapping = {**nwx_mapping, **nwx_mapping_rules}
        except FileNotFoundError as e:
            pass

    rule_stats = {}

    for model_name in models:
        try:
            graphs = rules_by_model[model_name]
            # print(f"Model: {model_name}")
            for graph in graphs:
                rule = nwx_mapping[graph]
                relationship = rule.head_atom.relationship_name

                if relationship not in rule_stats:
                    rule_stats[relationship] = {}
                # print(graph, rule.relationship_print(), rule_stats[relationship])
                graph_key = get_graph_key(rule_stats[relationship], graph)

                if graph_key is None:
                    rule_stats[relationship][graph] = []
                    graph_key = graph

                rule_stats[relationship][graph_key].append([model_name, rule.selectivity])
        except Exception as e:
            pass

    print_flags = {}
    for relationship in rule_stats:
        print_flags[relationship] = False

    for relationship in rule_stats:
        ctr = 1
        for graph_key in rule_stats[relationship]:
            if len(rule_stats[relationship][graph_key]) >= 7:
                print("===============================================")
                print(f"Relationship: {relationship}")
                rule = nwx_mapping[graph_key]
                print(f" Rule: {rule.relationship_print()}")
                print(f"  Count: {len(rule_stats[relationship][graph_key])}")
                print(f"  Values: {rule_stats[relationship][graph_key]}")


def rule_overlap_between_models_experiment():
    rule_overlap_between_models("Hetionet")



def get_interpretibility_score_table():
    folder_to_results = r"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\Tables"
    datasets = ["FB15K", "FB15K-237", "WN18", "WN18RR", "YAGO3-10", "Hetionet", "BioKG"]
    data = {}

    for dataset_name in datasets:
        with open(f"{folder_to_results}/{dataset_name}/{dataset_name}_mat_0_selec.csv") as f:
            f.readline()
            for line in f:
                line = line.strip()
                splits = line.split(",")
                model_name = splits[0]
                best1 = round(float(splits[1]), 3)
                best2 = round(float(splits[2]), 3)
                best3 = round(float(splits[3]), 3)
                bestH = round(float(splits[4]), 3)

                max_value = max(best1, best2, best3)

                best = ""
                if max_value == best1:
                    best = "t1"
                elif max_value == best2:
                    best = "t2"
                else:
                    best = "t3"

                if model_name not in data:
                    data[model_name] = {}

                if dataset_name not in data[model_name]:
                    delta = round(bestH - max_value, 3)
                    if delta >= 0:
                        delta = "+" + str(delta)
                    data[model_name][dataset_name] = {"HBM": bestH, "delta": f"{delta}",
                                                      "Top1": best1, "Top2": best2, "Top3": best3}

    data_latex = {}
    for model_name in data:
        data_latex[model_name] = {}
        for dataset_name in data[model_name]:
            data_latex[model_name][dataset_name] = {"HBM": data[model_name][dataset_name]["HBM"],
                                                    "delta": data[model_name][dataset_name]["delta"]}

    table = create_multi_table(data_latex, "")
    latex = table.to_latex(column_format="|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|", index_names=False, index=False)
    latex = latex.replace("\\toprule", "").replace("\\midrule", "").replace("\\bottomrule", "").replace(
        "\n",
        " \\hline \n")

    latex = latex.replace("{l}", "{c||}").replace("0.", ".").replace("1.0", "1")
    # print(latex)

    data_aggregated_by_model = {}

    averages = {}
    stds = {}

    for model_name in data:
        data_aggregated_by_model[model_name] = {"Top1": [], "Top2": [], "Top3": [], "HBM": []}
        averages[model_name] = {}
        stds[model_name] = {}

    # print(data_aggregated_by_model)
    for model_name in data:
        for dataset_name in data[model_name]:
            data_aggregated_by_model[model_name]["Top1"].append(data[model_name][dataset_name]["Top1"])
            data_aggregated_by_model[model_name]["Top2"].append(data[model_name][dataset_name]["Top2"])
            data_aggregated_by_model[model_name]["Top3"].append(data[model_name][dataset_name]["Top3"])
            data_aggregated_by_model[model_name]["HBM"].append(data[model_name][dataset_name]["HBM"])

    for model_name, item in data_aggregated_by_model.items():
        averages[model_name]["Top1"] = np.average(item["Top1"])
        stds[model_name]["Top1"] = np.std(item["Top1"])
        averages[model_name]["Top2"] = np.average(item["Top2"])
        stds[model_name]["Top2"] = np.std(item["Top2"])
        averages[model_name]["Top3"] = np.average(item["Top3"])
        stds[model_name]["Top3"] = np.std(item["Top3"])
        averages[model_name]["HBM"] = np.average(item["HBM"])
        stds[model_name]["HBM"] = np.std(item["HBM"])

    # fig, axs = plt.subplots(nrows=1, ncols=10, figsize=(16, 4), sharey=True)

    top1_avg = []
    top2_avg = []
    top3_avg = []
    hbm_avg = []
    top1_std = []
    top2_std = []
    top3_std = []
    hbm_std = []
    for model_name in averages:
        top1_avg.append(averages[model_name]["Top1"])
        top1_std.append(stds[model_name]["Top1"])
        top2_avg.append(averages[model_name]["Top2"])
        top2_std.append(stds[model_name]["Top2"])
        top3_avg.append(averages[model_name]["Top3"])
        top3_std.append(stds[model_name]["Top3"])
        hbm_avg.append(averages[model_name]["HBM"])
        hbm_std.append(stds[model_name]["HBM"])

    ctr = 0
    barWidth = 5
    width = 28
    br1 = [0 for i in range(0, 10)]
    br1[0] = 1
    for i in range(1, 10):
        br1[i] = br1[i - 1] + width

    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.tight_layout()
    y_min = 0.0
    y_max = 0.65
    y_interval = 0.05
    plt.ylim(y_min, y_max)
    plt.yticks(np.arange(y_min, y_max + y_interval, y_interval))
    plt.bar(br1, top1_avg, color="#FFD0CA", width=barWidth, edgecolor="black", label="Top-1")
    plt.bar(br2, top2_avg, color="#CDFFCA", width=barWidth, edgecolor="black", label="Top-2")
    plt.bar(br3, top3_avg, color="#CAD9FF", width=barWidth, edgecolor="black", label="Top-3")
    plt.bar(br4, hbm_avg, color="#FCFFCA", width=barWidth, edgecolor="black", label="HBM")

    # Adding Xticks
    # plt.xlabel('Models', fontsize=9, fontname="monospace")
    # plt.ylabel('Interpretibility scores averaged by models', fontsize=9, fontname="monospace")
    plt.xticks([val + 6 for val in br1],
               ["BoxE", "ComplEx", "HAKE", "HolE", "QuatE", "RotatE", "RotPro", "TorusE", "TransE", "TuckER"],
               fontsize=10)
    fig.subplots_adjust(left=0.07)
    # plt.ylim(0, 0.7)
    plt.tick_params(axis='y', labelsize=10)
    plt.legend(prop={'size': 8})
    # plt.show()
    plt.savefig(
        f"D:/PhD/Work/EmbeddingInterpretibility/Interpretibility/Results/Plots/macro_interpret_scores_aggregated_dataset.pdf",
        format="pdf",
        bbox_inches='tight', dpi=1000)
    plt.close()


def check_for_rule_type_agreement(dataset_name):
    models = ["boxe", "complex", "hake", "hole", "quate", "rotate", "rotpro", "toruse", "transe", "tucker"]
    folder_to_datasets = r"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Datasets"
    folder_to_mined_rules = r"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\MinedRules"

    if dataset_name == "Hetionet":
        entity_types = get_hetionet_relation_types()
    else:
        entity_types = get_biokg_relation_types()

    rules_by_model = {}

    result = {}
    for model in models:
        rp = ParseRules.ParseRule(
            filename=f"{folder_to_mined_rules}/{dataset_name}/{model}/{dataset_name}_{model}_mat_0_rules.tsv",
            model_name=model,
            dataset_name=dataset_name, folder_to_datasets=folder_to_datasets)
        rp.parse_rules_from_file(beta=1.0)
        rules = rp.rules
        rules_by_model[model] = rules
        match = check_rules(rules, entity_types)

        result[model] = {dataset_name: {"M": match,
                                        "T": len(rules),
                                        "R": round(match * 1.0 / len(rules), 3)}}

    return result


def check_for_rule_type_agreement_bio_datasets_experiment():
    result_het = check_for_rule_type_agreement("Hetionet")
    result_biokg = check_for_rule_type_agreement("BioKG")

    data = {}

    models = result_het.keys()

    for model_name in models:
        data[model_name] = {"Hetionet": result_het[model_name]["Hetionet"], "BioKG": result_biokg[model_name]["BioKG"]}

    table = create_multi_table(data, "Model name")
    latex = table.to_latex(column_format="|c|c|c|c|c|c|c|", index_names=False, index=False)
    latex = latex.replace("\\toprule", "").replace("\\midrule", "").replace("\\bottomrule", "").replace("\n",
                                                                                                        " \\hline \n")

    latex = latex.replace("{l}", "{c||}").replace("0.", ".").replace("1.0", "1")
    # print(table.to_markdown())
    print(latex)


def create_box_plots():
    datasets = ["FB15K", "FB15K-237", "NELL-995", "WN18", "WN18RR", "YAGO3-10", "Hetionet", "BioKG"]
    models = ["boxe", "complex", "hake", "hole", "quate", "rotate", "rotpro", "toruse", "transe", "tucker"]

    fig, axs = plt.subplots(nrows=1, ncols=7, figsize=(12, 3), sharey='all')
    fig.tight_layout()
    model_names = ["BoxE", "ComplEx", "HAKE", "HolE", "QuatE", "RotatE", "RotPro", "TorusE", "TransE", "TuckER"]
    ctr = 0

    for dataset_name in datasets:
        folder_to_rules = f"D:/PhD/Work/EmbeddingInterpretibility/Interpretibility/Results/BestRules"
        folder_datasets = "D:/PhD/Work/EmbeddingInterpretibility/Interpretibility/Datasets"
        selecs = []

        if dataset_name == "NELL-995":
            continue
        for model_name in models:
            selec_array = []
            rules = []
            with open(
                    f"{folder_to_rules}/{dataset_name}/{model_name}/{dataset_name}_{model_name}_mat_0_bestH.tsv") as f:
                for line in f:
                    line = line.strip()
                    rule, hc, pca, selec = line.split("\t")
                    selec_array.append(float(selec))

            selecs.append(selec_array)

        axs[ctr].boxplot(selecs, showfliers=False, whis=3.5, widths=0.75)
        axs[ctr].set_xticklabels(model_names, rotation=90, fontsize=10)
        # axs[ctr].xaxis.set_visible(False)
        axs[ctr].tick_params(axis='y', labelsize=10)
        axs[ctr].set_title(f"{dataset_name}", fontsize=10)
        axs[ctr].set_ylim(0.2, 1.05)
        if ctr > 0:
            axs[ctr].yaxis.set_visible(False)
        ctr += 1

    # Adjust the white space on the left side of the first box plot and the right side of the last box plot
    # axs[0].set_xlim(0.5, 10.5)
    # axs[-1].set_xlim(0.5, 10.5)
    fig.subplots_adjust(left=0.025, bottom=0.018, top=0.93, right=0.997, wspace=0.03)
    plt.show()
    plt.savefig(
        f"D:/PhD/Work/EmbeddingInterpretibility/Interpretibility/Results/BoxPlots/all_datasets_box_mined_rules.pdf",
        format="pdf",
        bbox_inches='tight')
    plt.close()


def create_size_predictions_table():
    input = open("input.txt")
    mr_data = {'boxe': {'FB15K': {'AMR': 0.993, '|M|': 6.437, 'Ties': 0.0},
                        'FB15K237': {'AMR': 0.969, '|M|': 9.146, 'Ties': 0.0},
                        'WN18': {'AMR': 0.964, '|M|': 7.329, 'Ties': 0.0},
                        'WN18RR': {'AMR': 0.626, '|M|': 47.972, 'Ties': 0.0},
                        'YAGO3-10': {'AMR': 0.961, '|M|': 23.403, 'Ties': 0.0}},
               'complex': {'FB15K237': {'AMR': 0.927, '|M|': 21.405, 'Ties': 0.0},
                           'WN18': {'AMR': 0.952, '|M|': 9.778, 'Ties': 0.0},
                           'WN18RR': {'AMR': 0.586, '|M|': 53.128, 'Ties': 0.0},
                           'YAGO3-10': {'AMR': 0.942, '|M|': 35.166, 'Ties': 0.0},
                           'FB15K': {'AMR': 0.98, '|M|': 17.337, 'Ties': 0.0}},
               'hake': {'FB15K': {'AMR': 0.985, '|M|': 12.814, 'Ties': 0.0},
                        'FB15K237': {'AMR': 0.943, '|M|': 16.661, 'Ties': 0.0},
                        'WN18': {'AMR': 0.96, '|M|': 8.253, 'Ties': 0.0},
                        'WN18RR': {'AMR': 0.519, '|M|': 61.727, 'Ties': 0.0},
                        'YAGO3-10': {'AMR': 0.942, '|M|': 35.01, 'Ties': 0.0}},
               'hole': {'FB15K': {'AMR': 0.971, '|M|': 25.192, 'Ties': 0.0},
                        'FB15K237': {'AMR': 0.91, '|M|': 26.253, 'Ties': 0.0},
                        'WN18': {'AMR': 0.964, '|M|': 7.467, 'Ties': 0.0},
                        'WN18RR': {'AMR': 0.607, '|M|': 50.439, 'Ties': 0.0},
                        'YAGO3-10': {'AMR': 0.9, '|M|': 60.545, 'Ties': 0.0}},
               'quate': {'FB15K': {'AMR': 0.978, '|M|': 19.546, 'Ties': 0.0},
                         'FB15K237': {'AMR': 0.938, '|M|': 18.054, 'Ties': 0.0},
                         'WN18': {'AMR': 0.958, '|M|': 8.533, 'Ties': 0.0},
                         'WN18RR': {'AMR': 0.583, '|M|': 53.504, 'Ties': 0.0},
                         'YAGO3-10': {'AMR': 0.924, '|M|': 46.01, 'Ties': 0.0}},
               'rotate': {'FB15K': {'AMR': 0.985, '|M|': 13.065, 'Ties': 0.0},
                          'FB15K237': {'AMR': 0.946, '|M|': 15.702, 'Ties': 0.0},
                          'WN18': {'AMR': 0.959, '|M|': 8.3, 'Ties': 0.0},
                          'WN18RR': {'AMR': 0.705, '|M|': 37.85, 'Ties': 0.0},
                          'YAGO3-10': {'AMR': 0.624, '|M|': 227.967, 'Ties': 0.0}},
               'rotpro': {'FB15K': {'AMR': 0.982, '|M|': 15.486, 'Ties': 0.0},
                          'FB15K237': {'AMR': 0.895, '|M|': 30.875, 'Ties': 0.0},
                          'WN18': {'AMR': 0.973, '|M|': 5.586, 'Ties': 0.0},
                          'WN18RR': {'AMR': 0.768, '|M|': 29.732, 'Ties': 0.0},
                          'YAGO3-10': {'AMR': 0.716, '|M|': 172.092, 'Ties': 0.0}},
               'toruse': {'FB15K': {'AMR': 0.986, '|M|': 12.31, 'Ties': 0.0},
                          'FB15K237': {'AMR': 0.951, '|M|': 14.317, 'Ties': 0.0},
                          'WN18': {'AMR': 0.981, '|M|': 3.811, 'Ties': 0.0},
                          'WN18RR': {'AMR': 0.782, '|M|': 27.994, 'Ties': 0.0},
                          'YAGO3-10': {'AMR': 0.928, '|M|': 43.527, 'Ties': 0.0}},
               'transe': {'FB15K': {'AMR': 0.993, '|M|': 6.407, 'Ties': 0.0},
                          'FB15K237': {'AMR': 0.972, '|M|': 8.203, 'Ties': 0.0},
                          'WN18': {'AMR': 0.984, '|M|': 3.243, 'Ties': 0.0},
                          'WN18RR': {'AMR': 0.853, '|M|': 18.813, 'Ties': 0.0},
                          'YAGO3-10': {'AMR': 0.977, '|M|': 14.077, 'Ties': 0.0}},
               'tucker': {'FB15K': {'AMR': 0.964, '|M|': 31.16, 'Ties': 0.352},
                          'FB15K237': {'AMR': 0.877, '|M|': 36.102, 'Ties': 0.274},
                          'WN18': {'AMR': 0.562, '|M|': 89.705, 'Ties': 0.749},
                          'WN18RR': {'AMR': 0.263, '|M|': 94.565, 'Ties': 0.353},
                          'YAGO3-10': {'AMR': 0.929, '|M|': 43.036, 'Ties': 0.0}}}

    mr2_data = {
        'boxe': {'BioKG': {'AMR': 0.978, '|M|': 9.948, 'Ties': 0}, 'Hetionet': {'AMR': 0.96, '|M|': 7.226, 'Ties': 0},
                 'NELL-995': {'AMR': 0.906, '|M|': 11.284, 'Ties': 0}},
        'complex': {'BioKG': {'AMR': 0.986, '|M|': 7.156, 'Ties': 0},
                    'Hetionet': {'AMR': 0.949, '|M|': 9.319, 'Ties': 0},
                    'NELL-995': {'AMR': 0.744, '|M|': 38.792, 'Ties': 0}},
        'hake': {'BioKG': {'AMR': 0.991, '|M|': 4.45, 'Ties': 0}, 'Hetionet': {'AMR': 0.963, '|M|': 6.336, 'Ties': 0},
                 'NELL-995': {'AMR': 0.654, '|M|': 45.916, 'Ties': 0}},
        'hole': {'BioKG': {'AMR': 0.986, '|M|': 7.28, 'Ties': 0}, 'Hetionet': {'AMR': 0.944, '|M|': 9.857, 'Ties': 0},
                 'NELL-995': {'AMR': 0.615, '|M|': 82.26, 'Ties': 0}},
        'quate': {'BioKG': {'AMR': 0.982, '|M|': 9.13, 'Ties': 0}, 'Hetionet': {'AMR': 0.944, '|M|': 10.733, 'Ties': 0},
                  'NELL-995': {'AMR': 0.663, '|M|': 64.846, 'Ties': 0}},
        'rotate': {'BioKG': {'AMR': 0.946, '|M|': 29.167, 'Ties': 0}, 'Hetionet': {'AMR': 0.96, '|M|': 6.78, 'Ties': 0},
                   'NELL-995': {'AMR': 0.344, '|M|': 163.952, 'Ties': 0}},
        'rotpro': {'BioKG': {'AMR': 0.897, '|M|': 55.333, 'Ties': 0},
                   'Hetionet': {'AMR': 0.962, '|M|': 6.454, 'Ties': 0},
                   'NELL-995': {'AMR': 0.546, '|M|': 116.101, 'Ties': 0}},
        'toruse': {'BioKG': {'AMR': 0.979, '|M|': 11.14, 'Ties': 0}, 'Hetionet': {'AMR': 0.953, '|M|': 8.47, 'Ties': 0},
                   'NELL-995': {'AMR': 0.755, '|M|': 55.159, 'Ties': 0}},
        'transe': {'BioKG': {'AMR': 0.992, '|M|': 3.956, 'Ties': 0},
                   'Hetionet': {'AMR': 0.955, '|M|': 7.733, 'Ties': 0},
                   'NELL-995': {'AMR': 0.819, '|M|': 27.616, 'Ties': 0}},
        'tucker': {'BioKG': {'AMR': 0.856, '|M|': 78.109, 'Ties': 0},
                   'Hetionet': {'AMR': 0.954, '|M|': 8.219, 'Ties': 0},
                   'NELL-995': {'AMR': 0.194, '|M|': 185.302, 'Ties': 0}}}

    data = {}
    for line in input:
        line = line.strip()

        if "n_mat_0" in line:
            dataset, model = line.split("Dataset: ")[1].split(" Model: ")[0], line.split("Model: ")[1].split(" ")[0]
            if dataset == "FB15K-237":
                dataset = dataset.replace("-", "")
            n_mat_0 = round(float(line.split("n_mat_0: ")[1].split(" ")[0]) / 1000000, 2)

            if model not in data:
                data[model] = {}

            if dataset not in data[model]:
                data[model][dataset] = {}

            if dataset == "Hetionet" or dataset == "BioKG" or dataset == "NELL-995":
                data[model][dataset] = {"AMR": str(round(mr2_data[model][dataset]["AMR"], 3)).replace("0.", "."),
                                        "|M|": str(n_mat_0)}
            else:
                data[model][dataset] = {"AMR": str(round(mr_data[model][dataset]["AMR"], 3)).replace("0.", "."),
                                        "|M|": str(n_mat_0)}
    print(data)
    table = create_multi_table(data, "Model name")

    latex = table.to_latex(column_format="|c|c|c|c|c|c|", index=False, index_names=False)

    latex = latex.replace("\\toprule", "").replace("\\midrule", "").replace("\\bottomrule", "").replace("\n",
                                                                                                        " \\hline \n")
    print(latex)


def create_rule_overlap_tables():
    """
    Function to create latex tables for the rule overlap between WN18 vs WN18RR and FB15K vs FB15K-237
    :return:
    """

    fb_file = open(r"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\IsoExp\FB.out")
    wn_file = open("D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\IsoExp\WN.out")

    data = {}
    for line in fb_file:
        line = line.strip()

        if "Model" in line:
            splits = line.split(";")
            model = splits[0].split(":")[1]
            n_rule1 = int(splits[1].split(":")[1])
            n_rule2 = int(splits[2].split(":")[1])
            n_inter = int(splits[4].split(":")[1])

            if model not in data:
                data[model] = {}

            if "FB15K vs FB15K-237" not in data[model]:
                data[model]["FB15K vs FB15K-237"] = {"N_d1": n_rule1,
                                                     "N_d2": n_rule2, "I": n_inter}

    for line in wn_file:
        line = line.strip()

        if "Model" in line:
            splits = line.split(";")
            model = splits[0].split(":")[1]
            n_rule1 = int(splits[1].split(":")[1])
            n_rule2 = int(splits[2].split(":")[1])
            n_inter = int(splits[4].split(":")[1])

            if model not in data:
                data[model] = {}

            if "WN18 vs WN18RR" not in data[model]:
                data[model]["WN18 vs WN18RR"] = {"N_d1": n_rule1,
                                                 "N_d2": n_rule2, "I": n_inter}

    table = create_multi_table(data, outer_key_name="Model name")
    latex = table.to_latex(column_format="|c|c|c|c|c|c|c|", index=False, index_names=False)

    latex = latex.replace("\\toprule", "").replace("\\midrule", "").replace("\\bottomrule", "").replace("\n",
                                                                                                        " \\hline \n").replace(
        "0.", ".")

    print(latex)


def get_relation_types(dataset_name):
    folder = fr"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Datasets\{dataset_name}"

    result = {}
    with open(f"{folder}/relation2id.txt") as f:
        f.readline()
        for line in f:
            relation = line.split("\t")[0].strip()
            result[line.split("\t")[1].strip()] = relation

    return result


def get_predicate_pcts(folder_to_rules, folder_to_datasets, dataset_name, model_name, predicates, filter1, filter2):
    if "Best" not in folder_to_rules:
        rp = ParseRules.ParseRule(
            filename=f"{folder_to_rules}/{dataset_name}/{model_name}/{dataset_name}_{model_name}_mat_0_rules.tsv",
            folder_to_datasets=folder_to_datasets, model_name=None, dataset_name=dataset_name)

        rp.parse_rules_from_file(1.0)
        rules = rp.rules

        filtered_rules = []

        for rule in rules:
            if filter1 <= rule.selectivity < filter2:
                filtered_rules.append(rule)

        rules = filtered_rules
    else:
        rules = []
        with open(f"{folder_to_rules}/{dataset_name}/{model_name}/{dataset_name}_{model_name}_mat_0_best1.tsv") as f:
            for line in f:
                line = line.strip()
                rule, hc, pca, selec = line.split("\t")
                rule_object = parse_best_rule_str(rule, float(hc), float(pca), float(selec))
                rules.append(rule_object)

        for rule in rules:
            for atom in rule.body_atoms:
                for r, r_name in predicates.items():
                    if atom.relationship_name == r_name:
                        atom.relationship = int(r)
                        break

            atom = rule.head_atom
            for r, r_name in predicates.items():
                if atom.relationship_name == r_name:
                    atom.relationship = int(r)
                    break

    predicate_body_pct = {}
    total_body = {}
    for predicate in predicates:
        predicate = int(predicate)
        predicate_body_pct[predicate] = 0
        total_body[predicate] = 0
        for rule in rules:
            if predicate == rule.body_atoms[0].relationship:
                predicate_body_pct[predicate] += 1
            elif len(rule.body_atoms) == 2 and predicate == rule.body_atoms[1].relationship:
                predicate_body_pct[predicate] += 1
            elif rule.head_atom.relationship == predicate:
                predicate_body_pct[predicate] += 1

            unique_predicate = set()

            for atom in rule.body_atoms:
                unique_predicate.add(atom.relationship)

            unique_predicate.add(rule.head_atom.relationship)

            total_body[predicate] += len(unique_predicate)

    for predicate in predicates:
        predicate = int(predicate)

        if total_body[predicate] == 0:
            predicate_body_pct[predicate] = 0.0
        else:
            predicate_body_pct[predicate] = predicate_body_pct[predicate] * 1.0 / total_body[predicate]

    return predicate_body_pct


def predicate_mrs_across_models_experiment(dataset_name):
    models = ["boxe", "complex", "hake", "hole", "quate", "rotate", "rotpro", "toruse", "transe", "tucker"]
    folder_to_datasets = "D:/PhD/Work/EmbeddingInterpretibility/Interpretibility/Datasets"
    folder_to_materializations = "D:/PhD/Work/EmbeddingInterpretibility/Interpretibility/Results/Materializations"
    folder_to_rules = "D:/PhD/Work/EmbeddingInterpretibility/Interpretibility/Results/MinedRules"
    folder_to_best_rules = "D:/PhD/Work/EmbeddingInterpretibility/Interpretibility/Results/BestRules"

    data = {}

    predicates = get_relation_types(dataset_name)

    for model_name in models:
        predicate_in_body_pct_above_t = get_predicate_pcts(folder_to_rules, folder_to_datasets,
                                                           dataset_name, model_name, predicates, 0.75, 1.1)
        predicate_in_body_pct_below_t = get_predicate_pcts(folder_to_rules, folder_to_datasets,
                                                           dataset_name, model_name, predicates, 0.25, 0.75)

        # total = 0
        # for _, cnt in predicate_in_body_pct.items():
        #     total += cnt
        #
        # print(f"Model: {model_name}, Body total: {total}")
        # total = 0
        # for _, cnt in predicate_in_head_pct.items():
        #     total += cnt
        # print("Head total: ", total)

        with open(
                f"{folder_to_materializations}/{dataset_name}/{model_name}/{dataset_name}_{model_name}_relation_mrs.csv") as f:
            for line in f:
                line = line.strip()
                splits = line.split(",")
                predicate_name = predicates[splits[0]]
                predicate = int(splits[0])

                if predicate_name not in data:
                    data[predicate_name] = {}

                # data[predicate_name][model_name] = {"AMR": round(float(splits[1]), 2), "Pb %": round(predicate_in_body_pct[predicate]*100, 1), "Ph %": round(predicate_in_head_pct[predicate]*100, 1)}
                data[predicate_name][model_name] = {"AMR": float(splits[1]),
                                                    "Pb Above %": round(predicate_in_body_pct_above_t[predicate] * 100,
                                                                        1),
                                                    "Pb Below %": round(predicate_in_body_pct_below_t[predicate] * 100,
                                                                        1)}
    # for predicate in data:
    #     total = 0
    #     for model in models:
    #         total += data[predicate][model]["AMR"]
    #
    #     data[predicate]["average"] = round(total/len(models), 3)
    table = create_multi_table(data, "Predicate")
    latex = table.to_latex(index=False, index_names=False)

    latex = latex.replace("\\toprule", "").replace("\\midrule", "").replace("\\bottomrule", "").replace("\n",
                                                                                                        " \\hline \n")

    print(latex)
    # table = table.sort_values(by=['average'])
    table.to_csv(
        fr"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\PredicateMRExp\{dataset_name}_predicate_amr_compared.csv",
        index=False)


def predicate_mrs_all_datasets():
    predicate_mrs_across_models_experiment("WN18")
    predicate_mrs_across_models_experiment("NELL-995")
    predicate_mrs_across_models_experiment("YAGO3-10")


if __name__ == "__main__":
    # check_for_rule_type_agreement_bio_datasets_experiment()
    # count_rules_experiment()
    # same_dataset_rule_comparison_experiment()
    # create_rule_overlap_tables()
    # create_size_predictions_table()
    # create_box_plots()
    # get_interpretibility_score_table()
    rule_overlap_between_models_experiment()
    # predicate_mrs_all_datasets()
    # get_interpretibility_score_table()
