from graphdatascience import GraphDataScience
from tqdm import tqdm

from Interpretibility.Code.ParseRules import ParseRule, Atom, Rule
from datetime import datetime


def compute_multi_rule_support(rules, gds):
    rule_queries = ["" for rule in rules]

    for i in range(0, len(rules)):
        rule = rules[i]

        for atom in rule.body_atoms:
            rule_queries[i] += " MATCH " + atom.neo4j_print()

        rule_queries[
            i] += " MATCH " + rule.head_atom.neo4j_print() + " WITH DISTINCT a,b RETURN a.entityId as headId, b.entityId as tailId"

    query = "CALL {"

    for rule_query in rule_queries:
        query += rule_query + " UNION "

    query = query[:-7] + "} RETURN count(*) as cnt"
    result = gds.run_cypher(query)

    return result.loc[0, "cnt"]


def compute_multi_rule_totals(rules, gds):
    if rules[0].functional_variable == "a":
        func = "b"
    else:
        func = "a"

    rule_queries = ["" for rule in rules]

    for i in range(0, len(rules)):
        rule = rules[i]

        for atom in rule.body_atoms:
            rule_queries[i] += " MATCH " + atom.neo4j_print()

        rule_queries[
            i] += " MATCH " + rule.head_atom.neo4j_print().replace(func,
                                                                   "") + " WITH DISTINCT a,b RETURN a.entityId as headId, b.entityId as tailId"

    query = "CALL {"

    for rule_query in rule_queries:
        query += rule_query + " UNION "

    query = query[:-7] + "} RETURN count(*) as cnt"
    result = gds.run_cypher(query)

    return result.loc[0, "cnt"]


def compute_multi_rule_pca(rules, support, gds):
    totals = compute_multi_rule_totals(rules, gds)
    return support * 1.0 / totals


def get_triples_for_relation(relation, gds):
    result = gds.run_cypher("MATCH ()-[r:`" + str(relation) + "`]->() RETURN count(*) as cnt")
    return result.loc[0, "cnt"]


def compute_combined_head_coverage(support, relation, gds):
    totals = get_triples_for_relation(relation, gds)
    return support * 1.0 / totals


def compute_selectivity(pca, hc, beta=1):
    selectivity = ((1 + beta * beta) * pca * hc) / (
            beta * beta * pca + hc)
    return selectivity


def compute_combined_metrics(rules, gds, beta=1):
    support = compute_multi_rule_support(rules, gds)
    pca = compute_multi_rule_pca(rules, support, gds)
    hc = compute_combined_head_coverage(support, rules[0].head_atom.relationship, gds)
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


def get_postive_triples(rule, gds):
    query = ""
    for atom in rule.body_atoms:
        query += " MATCH " + atom.neo4j_print()

    query += " MATCH " + rule.head_atom.neo4j_print() + " WITH DISTINCT a,b RETURN a.entityId as headId, b.entityId as tailId"

    result = gds.run_cypher(query)

    triples = set()

    for index, row in result.iterrows():
        triple = (row.headId, row.tailId)
        triples.add(triple)

    return triples


def get_total_triples(rule, gds):
    if rule.functional_variable == "a":
        func = "b"
    else:
        func = "a"

    query = ""
    for atom in rule.body_atoms:
        query += " MATCH " + atom.neo4j_print()

    query += " MATCH " + rule.head_atom.neo4j_print().replace(func,
                                                              "") + " WITH DISTINCT a,b RETURN a.entityId as headId, b.entityId as tailId"

    result = gds.run_cypher(query)

    triples = set()

    for index, row in result.iterrows():
        triple = (row.headId, row.tailId)
        triples.add(triple)

    return triples


def compute_pca_python(positives, totals):
    if len(totals) == 0:
        return 0

    return len(positives) * 1.0 / len(totals)


def compute_hc_python(positives, denom):
    return len(positives) * 1.0 / denom


def compute_metrics_python(positives, totals, triples_in_relation):
    hc = compute_hc_python(positives, triples_in_relation)
    pca = compute_pca_python(positives, totals)
    selec = compute_selectivity(pca, hc)

    return hc, pca, selec


def astar_python(rules, all_positives, all_totals, triples_in_relation):
    relationship = rules[0].head_atom.relationship
    best_subset = [rules[0]]

    positives = all_positives[rules[0].id_print()]
    totals = all_totals[rules[0].id_print()]
    hc, pca, max_selectivity = compute_metrics_python(positives, totals, triples_in_relation)
    rules.remove(rules[0])

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("\t\t" + dt_string + "\t" + "Finding best rules")

    while len(rules) > 0:
        best_rule = None
        new_positives = None
        new_totals = None
        # print(len(rules), "Previous selectivity:", max_selectivity)

        for i in range(len(rules)):
            rule = rules[i]
            rule_positives = all_positives[rule.id_print()]
            rule_totals = all_totals[rule.id_print()]
            rule_positives = rule_positives.union(positives)
            rule_totals = rule_totals.union(totals)
            hc, pca, selec = compute_metrics_python(rule_positives, rule_totals, triples_in_relation)
            if selec > max_selectivity:
                max_selectivity = selec
                best_rule = rule
                new_positives = rule_positives
                new_totals = rule_totals

        if best_rule is None:
            break

        best_subset.append(best_rule)
        rules.remove(best_rule)

        positives = new_positives.copy()
        totals = new_totals.copy()

        hc = compute_hc_python(positives, triples_in_relation)

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("\t\t\t" + dt_string + "\t" + "Rules added: " + str(len(best_subset)))

        if hc >= 1 or len(best_subset) >= 5:
            break

    hc, pca, selec = compute_metrics_python(positives, totals, triples_in_relation)
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("\t\t" + dt_string + "\t" + "Best rules found")

    return best_subset, hc, pca, selec


def get_rule_atom_dict(rule):
    rule_atom_dict = {}

    for atom in rule.body_atoms:
        if atom.relationship not in rule_atom_dict:
            rule_atom_dict[atom.relationship] = 0
        rule_atom_dict[atom.relationship] += 1

    return rule_atom_dict


def compute_overlap(rules, candidate):
    max_overlap = 0
    for rule in rules:
        overlap = 0
        rule_atom_dict = get_rule_atom_dict(rule)
        candidate_atom_dict = get_rule_atom_dict(candidate)

        for key, value in candidate_atom_dict.items():
            if key in rule_atom_dict:
                rule_value = rule_atom_dict[key]
                candidate_value = candidate_atom_dict[key]

                if rule_value == candidate_value:
                    overlap += candidate_value
                elif rule_value > candidate_value:
                    overlap += candidate_value
                else:
                    overlap += rule_value

        # print(rule.id_print(), candidate.id_print(), overlap)
        max_overlap = max(max_overlap, overlap)

    return 1 - max_overlap * 1.0 / len(candidate.body_atoms)


def compute_metric(rules, candidate):
    overlap = max(0.001, compute_overlap(rules, candidate))
    hc = candidate.head_coverage
    pca = candidate.pca_confidence
    selectivity = compute_selectivity(pca, hc)
    return compute_selectivity(overlap, selectivity)


def astar2(rules, gds):
    best_subset = [rules[0]]
    if rules[0].pca_confidence >= 1.0 and rules[0].head_coverage >= 1.0:
        hc, pca, selectivity = compute_combined_metrics(best_subset, gds)
        return best_subset, hc, pca, selectivity

    print(rules[0].relationship_print())
    rules.remove(rules[0])
    while len(rules) > 0:
        best_rule = None
        best_metric = 0.0

        for rule in rules:
            metric = compute_metric(best_subset, rule)
            print (len(best_subset), rule.relationship_print(), metric)
            if metric > best_metric:
                best_metric = metric
                best_rule = rule

        if best_rule is None:
            break

        best_subset.append(best_rule)
        rules.remove(best_rule)

        if len(best_subset) >= 5:
            break

    hc, pca, selectivity = compute_combined_metrics(best_subset, gds)

    return best_subset, hc, pca, selectivity


def precompute_positives_and_totals(rules, gds):
    totals = {}
    positives = {}
    ctr = len(rules)
    try:
        for rule in rules:
            print("Rules left:", ctr)
            ctr -= 1
            positives[rule.id_print()] = get_postive_triples(rule, gds)
            totals[rule.id_print()] = get_total_triples(rule, gds)
    except:
        return positives, totals

    return positives, totals


def best1(rules):
    best_rule = rules[0]
    hc = best_rule.head_coverage
    pca = best_rule.pca_confidence
    selectivity = compute_selectivity(pca, hc)

    return best_rule.relationship_print(), hc, pca, selectivity


def best2(rules, gds):
    rule_1 = rules[0]
    rule_2 = rules[1]
    hc, pca, selec = compute_combined_metrics([rule_1, rule_2], gds)
    return print_combined_rule([rule_1, rule_2]), hc, pca, selec


def greedy_python(rules, all_positives, all_totals, triples_in_relation):
    rules, hc, pca, selec = astar_python(rules, all_positives, all_totals, triples_in_relation)
    return print_combined_rule(rules), hc, pca, selec


def greedy_python2(rules, gds):
    rules, hc, pca, selec = astar2(rules, gds)
    return print_combined_rule(rules), hc, pca, selec


def write_rule(rule, hc, pca, selec, f):
    f.write(rule + "\t" + str(round(hc, 4)) + "\t" + str(round(pca, 4)) + "\t" + str(round(selec, 4)) + "\n")


def run_astar2(rule_filename, dataset_name, model_name, database, gds):
    rp = ParseRule(rule_filename, model_name, dataset_name)
    rp.parse_rules_from_file()
    gds.set_database(database)
    mat_type = "materialized" if "materialized" in database else "mispredicted"
    f_greedy = open(
        "D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\BestRules\\" + dataset_name + "\\" + model_name + "_" + mat_type + "_greedy5.tsv",
        "w+")
    f_best1 = open(
        "D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\BestRules\\" + dataset_name + "\\" + model_name + "_" + mat_type + "_best1.tsv",
        "w+")
    f_best2 = open(
        "D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\BestRules\\" + dataset_name + "\\" + model_name + "_" + mat_type + "_best2.tsv",
        "w+")

    for predicate, rules in rp.rules_by_predicate.items():
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(dt_string + "\t" + "Predicate: " + str(predicate))
        m1_rule, m1_hc, m1_pca, m1_selec = best1(rules)
        write_rule(m1_rule, m1_hc, m1_pca, m1_selec, f_best1)
        if len(rules) >= 2:
            m2_rule, m2_hc, m2_pca, m2_selec = best2(rules, gds)
            write_rule(m2_rule, m2_hc, m2_pca, m2_selec, f_best2)

            m3_rule, m3_hc, m3_pca, m3_selec = greedy_python2(rules.copy(), gds)
            write_rule(m3_rule, m3_hc, m3_pca, m3_selec, f_greedy)

    f_greedy.close()


def run_individual_experiment(rule_filename, dataset_name, model_name, database, gds):
    rp = ParseRule(rule_filename, model_name, dataset_name)
    rp.parse_rules_from_file()
    gds.set_database(database)

    mat_type = "materialized" if "materialized" in database else "mispredicted"
    # f_best1 = open(
    #     "D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\BestRules\\" + dataset_name + "\\" + model_name + "_" + mat_type + "_best1.tsv",
    #     "w+")
    # f_best2 = open(
    #     "D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\BestRules\\" + dataset_name + "\\" + model_name + "_" + mat_type + "_best2.tsv",
    #     "w+")
    f_greedy = open(
        "D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\BestRules\\" + dataset_name + "\\" + model_name + "_" + mat_type + "_greedy.tsv",
        "w+")

    for predicate, rules in rp.rules_by_predicate.items():
        # rules = rules[:10]
        # now = datetime.now()
        # dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        # print(dt_string + "\t" + "Predicate: " + str(predicate))
        #
        # now = datetime.now()
        # dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        # print("\t" + dt_string + "\t" + "Computing positives and totals")

        all_positives, all_totals = precompute_positives_and_totals(rules, gds)
        triples_in_relation = get_triples_for_relation(str(predicate), gds)

        # now = datetime.now()
        # dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        # print("\t" + dt_string + "\t" + "Computed positives and totals")
        #
        # now = datetime.now()
        # dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        # print(dt_string + "\t" + " Best 1")
        # m1_rule, m1_hc, m1_pca, m1_selec = best1(rules)
        # write_rule(m1_rule, m1_hc, m1_pca, m1_selec, f_best1)
        if len(rules) >= 2:
            # now = datetime.now()
            # dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            # print(dt_string + "\t" + " Best 2")
            #
            # m2_rule, m2_hc, m2_pca, m2_selec = best2(rules, all_positives, all_totals, triples_in_relation)
            # write_rule(m2_rule, m2_hc, m2_pca, m2_selec, f_best2)
            #
            # now = datetime.now()
            # dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            # print(dt_string + "\t" + " Greedy")

            m3_rule, m3_hc, m3_pca, m3_selec = greedy_python(rules.copy(), all_positives, all_totals,
                                                             triples_in_relation)
            write_rule(m3_rule, m3_hc, m3_pca, m3_selec, f_greedy)

    # f_best1.close()
    # f_best2.close()
    f_greedy.close()


def run_experiment():
    gds = GraphDataScience("bolt://localhost:7687", auth=("neo4j", "pass"))

    # models = ["TransE", "ComplEx"]
    # dataset_name = "FB15K"
    # for model_name in models:
    #     database = model_name.lower() + "materialized"
    #     now = datetime.now()
    #     dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    #     print(dt_string + "\t" + database)
    #     rule_filename = "D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\MinedRules\\" + dataset_name + "\\" + model_name + "_materialized.tsv"
    #     run_individual_experiment(rule_filename, dataset_name, model_name, database, gds)
    #     database = model_name.lower() + "mispredicted"
    #     now = datetime.now()
    #     dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    #     print(dt_string + "\t" + database)
    #     rule_filename = "D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\MinedRules\\" + dataset_name + "\\" + model_name + "_mispredicted.tsv"
    #     run_individual_experiment(rule_filename, dataset_name, model_name, database, gds)

    rule_filename = "D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\\test_rules.tsv"
    run_individual_experiment(rule_filename, "WN11", "", "neo4j", gds)


if __name__ == "__main__":
    # run_experiment()
    gds = GraphDataScience("bolt://localhost:7687", auth=("neo4j", "pass"))

    model_name = ""
    dataset_name = "WN11"
    database = model_name.lower() + "materialized"
    rule_filename = "D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\MinedRules\\" + dataset_name + "\\" + model_name + "_materialized.tsv"
    rule_filename = "D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\\test_rules.tsv"
    run_astar2(rule_filename, dataset_name, model_name, "neo4j", gds)
