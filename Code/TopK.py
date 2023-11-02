import io
import os
import sys
import zipfile

from GraphEditDistance import compute_graph_edit_distance
from ParseRules import ParseRule


def compute_selectivity(pca, hc, beta=1.0):
    selectivity = ((1 + beta * beta) * pca * hc) / (
            beta * beta * pca + hc)
    return selectivity


def compute_multi_rule_pca_internal(positives, totals):
    if len(totals) == 0:
        return 0

    return len(positives) * 1.0 / len(totals)


def compute_multi_rule_head_coverage_internal(positives, negative_count):
    return len(positives) * 1.0 / negative_count


def compute_combined_metrics_internal(positives, totals, negatives_count, beta=1):
    pca = compute_multi_rule_pca_internal(positives, totals)
    hc = compute_multi_rule_head_coverage_internal(positives, negatives_count)
    selectivity = compute_selectivity(pca, hc, beta)

    return hc, pca, selectivity


def print_combined_rule(rules):
    result = ""

    for rule in rules:
        result += "( "
        for atom in rule.body_atoms:
            result += atom.relationship_print() + " "

        result += " ) V "

    result = result[:-3] + " ==> " + str(rules[0].head_atom.relationship_print())

    return result


def compute_overlap(rules, candidate, edit_distance_dict):
    min_overlap = 10000
    for rule in rules:
        key = (rule, candidate)
        overlap = edit_distance_dict[key]

        min_overlap = min(min_overlap, overlap)

    return min_overlap


def compute_metric(rules, candidate, edit_distance_dict):
    overlap = compute_overlap(rules, candidate, edit_distance_dict)
    pca = candidate.pca_confidence
    metric = overlap * pca
    return metric


def sort_rules_by_selectivity(rules, beta=1.0):
    for rule in rules:
        rule.selectivity = compute_selectivity(rule.pca_confidence, rule.head_coverage, beta=beta)

    rules = sorted(rules, key=lambda x: x.selectivity, reverse=True)
    return rules


def greedy_approximate(rules, negatives_count, edit_distance_dict, rule_to_file, beta):
    best_subset = [rules[0]]
    all_rules = rules.copy()

    # for rule in rules:
    #     print(f"{rule.id_print()}\t{rule.selectivity}")
    rules = sort_rules_by_selectivity(rules, beta=beta)
    if rules[0].pca_confidence >= 1.0 and rules[0].head_coverage >= 1.0:
        return print_combined_rule(best_subset), rules[0].head_coverage, rules[0].pca_confidence, rules[0].selectivity

    rules.remove(rules[0])
    while len(rules) > 0:
        best_rule = None
        best_metric = 0.0

        for rule in rules:
            metric = compute_metric(best_subset, rule, edit_distance_dict)

            if metric > best_metric:
                best_metric = metric
                best_rule = rule

        if best_rule is None:
            break

        best_subset.append(best_rule)
        rules.remove(best_rule)

        if len(best_subset) >= 5:
            break

    rule_positives = set()
    rule_totals = set()

    ctr = 0

    for rule in best_subset:
        this_rule, positives, totals = get_params_from_rule_file(
            rule_to_file[rule.id_print()])
        if rule.id_print() != this_rule:
            print("Rule mismatch for:", str(
                all_rules.index(rule)), rule.id_print(), this_rule)
            exit(0)
        ctr += 1
        rule_positives = rule_positives.union(positives)
        rule_totals = rule_totals.union(totals)

    hc, pca, selectivity = compute_combined_metrics_internal(rule_positives, rule_totals, negatives_count, beta)

    return print_combined_rule(best_subset), hc, pca, selectivity


def rule_to_file_mapping(folder_to_instantiations, predicate):
    folder = f"{folder_to_instantiations}{predicate}/"
    rule_files = os.listdir(folder)

    rule_to_file = {}

    for rule_file in rule_files:
        with zipfile.ZipFile(f"{folder}{rule_file}", 'r') as zip_rule_file:
            with zip_rule_file.open('data.txt', 'r') as data_file:
                data = data_file.read()
                data_str = data.decode("utf-8")
                data_io = io.StringIO(data_str)
                splits = data_io.readline().strip().split(",")
                rule = ",".join(splits[:-2])
                rule = rule.replace("=> ", "==>")
                rule = rule.replace("  ==>", " ==>")
                rule_to_file[rule] = folder + rule_file

    return rule_to_file


def best1(rules, beta):
    best_rule = rules[0]
    hc = best_rule.head_coverage
    pca = best_rule.pca_confidence
    selectivity = compute_selectivity(pca, hc, beta)

    return print_combined_rule([best_rule]), hc, pca, selectivity


def best2(folder_to_instantiations, rules, negatives_count, beta):
    rule_1 = rules[0]
    rule_2 = rules[1]
    head_predicate = str(rules[0].head_atom.relationship)
    r1, positives_1, totals_1 = get_params_from_rule_file(f"{folder_to_instantiations}{head_predicate}/r0.zip")
    r2, positives_2, totals_2 = get_params_from_rule_file(f"{folder_to_instantiations}{head_predicate}/r1.zip")

    if r1 != rule_1.id_print():
        print("Rule mismatch:", rule_1.id_print(), r1)
        exit(0)

    if r2 != rule_2.id_print():
        print("Rule mismatch:", rule_2.id_print(), r2)
        exit(0)

    rule_positives = positives_1.union(positives_2)
    rule_totals = totals_1.union(totals_2)
    hc, pca, selec = compute_combined_metrics_internal(rule_positives, rule_totals, negatives_count, beta=beta)
    return print_combined_rule([rule_1, rule_2]), hc, pca, selec


def best3(folder_to_instantiations, rules, negatives_count, beta):
    rule_1 = rules[0]
    rule_2 = rules[1]
    rule_3 = rules[2]
    head_predicate = str(rules[0].head_atom.relationship)
    r1, positives_1, totals_1 = get_params_from_rule_file(f"{folder_to_instantiations}{head_predicate}/r0.zip")
    r2, positives_2, totals_2 = get_params_from_rule_file(f"{folder_to_instantiations}{head_predicate}/r1.zip")
    r3, positives_3, totals_3 = get_params_from_rule_file(f"{folder_to_instantiations}{head_predicate}/r2.zip")

    if r1 != rule_1.id_print():
        print("Rule mismatch:", rule_1.id_print(), r1)
        exit(0)

    if r2 != rule_2.id_print():
        print("Rule mismatch:", rule_2.id_print(), r2)
        exit(0)

    if r3 != rule_3.id_print():
        print("Rule mismatch:", rule_3.id_print(), r3)
        exit(0)

    rule_positives = positives_1.union(positives_2).union(positives_3)
    rule_totals = totals_1.union(totals_2).union(totals_3)

    hc, pca, selec = compute_combined_metrics_internal(rule_positives, rule_totals, negatives_count, beta)
    return print_combined_rule([rule_1, rule_2, rule_3]), hc, pca, selec


def write_rule(rule, hc, pca, selec, f):
    f.write(rule + "\t" + str(round(hc, 4)) + "\t" + str(round(pca, 4)) + "\t" + str(round(selec, 4)) + "\n")


def count_negatives_by_predicate(folder_to_materializations):
    f = open(folder_to_materializations)

    negative_dict = {}
    total = 0
    for line in f:
        if line == "\n":
            continue
        splits = line.strip().split("\t")
        if int(splits[1]) not in negative_dict:
            negative_dict[int(splits[1])] = 0
        total += 1
        negative_dict[int(splits[1])] += 1

    return negative_dict, total


def get_params_from_rule_file(filename):
    with zipfile.ZipFile(filename, 'r') as zip_file:
        with zip_file.open('data.txt', 'r') as data_file:
            data = data_file.read()
            data_str = data.decode("utf-8")
            data_io = io.StringIO(data_str)

            positives = set()
            totals = set()
            line = data_io.readline()
            splits = line.strip().split(",")
            rule = ",".join(splits[:-2])
            rule = rule.replace("=> ", "==>")
            rule = rule.replace("  ==>", " ==>")
            n_positives = int(data_io.readline().strip())

            for ctr in range(0, n_positives):
                line = data_io.readline().strip()
                h, t = line.split(",")
                inst = (h, t)
                positives.add(inst)

            data_io.readline()

            for line in data_io:
                line = line.strip()
                h, t = line.split(",")
                inst = (h, t)
                totals.add(inst)
            data_io.close()
    return rule, positives, totals




def run_greedy(dataset_name, model_name, folder_to_mined_rules, folder_to_best_rules, folder_to_instantiations,
               folder_to_materializations, folder_to_datasets, mat_type, beta1, beta2):
    # TODO: Add beta2 to best functions for changing beta post processing
    if not os.path.exists(f"{folder_to_best_rules}/{dataset_name}/{model_name}"):
        os.mkdir(f"{folder_to_best_rules}/{dataset_name}/{model_name}")
    folder_to_instantiations = f"{folder_to_instantiations}/{dataset_name}/{model_name}/{dataset_name}_{model_name}_mat_{mat_type}/"
    folder_to_best_rules = f"{folder_to_best_rules}/{dataset_name}/{model_name}/{dataset_name}_{model_name}_mat_{mat_type}"
    folder_to_mined_rules = f"{folder_to_mined_rules}/{dataset_name}/{model_name}/{dataset_name}_{model_name}_mat_{mat_type}_rules.tsv"
    folder_to_materializations = f"{folder_to_materializations}/{dataset_name}/{model_name}/{dataset_name}_{model_name}_mat_{mat_type}.tsv"
    predicates = os.listdir(folder_to_instantiations)

    f_best1 = open(folder_to_best_rules + "_best1.tsv", "w+")
    f_best2 = open(folder_to_best_rules + "_best2.tsv", "w+")
    f_best3 = open(folder_to_best_rules + "_best3.tsv", "w+")
    f_bestH = open(folder_to_best_rules + "_bestH.tsv", "w+")

    negatives_count, negatives_totals = count_negatives_by_predicate(folder_to_materializations)

    count = len(predicates)

    rp = ParseRule(folder_to_mined_rules, folder_to_datasets, model_name, dataset_name, "\t")
    rp.parse_rules_from_file(beta=beta1)
    for predicate in rp.rules_by_predicate:
        print("Predicate:", predicate, "Predicates left:", count - 1)
        rule_to_file = rule_to_file_mapping(folder_to_instantiations, predicate)
        count -= 1
        totals_per_predicate = {}
        positives_per_predicate = {}
        rule_files = os.listdir(f"{folder_to_instantiations}{predicate}/")
        rules = rp.rules_by_predicate[predicate][:25]

        m1_rule, m1_hc, m1_pca, m1_selec = best1(rules.copy(), beta=beta2)
        write_rule(m1_rule, m1_hc, m1_pca, m1_selec, f_best1)

        edit_distance_dict = compute_graph_edit_distance(rules)
        print("\t\tGED computed")

        m4_rule, m4_hc, m4_pca, m4_selec = greedy_approximate(rules.copy(), negatives_count[int(predicate)],
                                                              edit_distance_dict, rule_to_file=rule_to_file, beta=beta2)
        write_rule(m4_rule, m4_hc, m4_pca, m4_selec, f_bestH)
        print("\t\tHeuristic completed")

        if len(rules) >= 2:
            m2_rule, m2_hc, m2_pca, m2_selec = best2(folder_to_instantiations, rules.copy(),
                                                     negatives_count[int(predicate)], beta=beta2)
            write_rule(m2_rule, m2_hc, m2_pca, m2_selec, f_best2)
            print("\t\tBest 2 completed")

        if len(rules) >= 3:
            m3_rule, m3_hc, m3_pca, m3_selec = best3(folder_to_instantiations, rules.copy(),
                                                     negatives_count[int(predicate)], beta=beta2)

            write_rule(m3_rule, m3_hc, m3_pca, m3_selec, f_best3)
            print("\t\tBest 3 completed")

    f_best1.close()
    f_best2.close()
    f_best3.close()
    f_bestH.close()


def run_experiment(dataset_name, model_name, folder_to_mined_rules, folder_to_best_rules, folder_to_instantiations,
                   folder_to_materializations, folder_to_datasets, mat_type, beta1, beta2):
    print("Starting combination")
    print("Dataset:", dataset_name, " Model:", model_name)
    run_greedy(dataset_name=dataset_name, model_name=model_name, folder_to_mined_rules=folder_to_mined_rules,
               folder_to_best_rules=folder_to_best_rules, folder_to_instantiations=folder_to_instantiations,
               folder_to_materializations=folder_to_materializations, folder_to_datasets=folder_to_datasets,
               mat_type=mat_type, beta1=beta1, beta2=beta2)
    print("Finished combination")


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    folder_to_mined_rules = sys.argv[3]
    folder_to_best_rules = sys.argv[4]
    folder_to_instantiations = sys.argv[5]
    folder_to_materializations = sys.argv[6]
    folder_to_datasets = sys.argv[7]
    mat_type = int(sys.argv[8])
    beta1 = float(sys.argv[9])
    beta2 = float(sys.argv[10])

    run_experiment(dataset_name=dataset_name, model_name=model_name, folder_to_mined_rules=folder_to_mined_rules,
                   folder_to_best_rules=folder_to_best_rules, folder_to_instantiations=folder_to_instantiations,
                   folder_to_materializations=folder_to_materializations, folder_to_datasets=folder_to_datasets,
                   mat_type=mat_type, beta1=beta1, beta2=beta2)
    print("Top K done")
