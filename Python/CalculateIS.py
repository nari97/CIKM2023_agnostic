import os
import sys

import pandas as pd
from matplotlib import pyplot as plt



def get_relation_to_id(folder_to_datasets, dataset_name, delimiter="\t"):
    folder = f"{folder_to_datasets}/{dataset_name}/"

    f = open(folder + "relation2id.txt")
    relation_to_id = {}

    f.readline()

    for line in f:
        splits = line.strip().split("\t")

        if len(splits) != 2:
            splits = line.strip().split(" ")

        relation_to_id[splits[0]] = int(splits[1])

    return relation_to_id


def get_test_triple_count(folder_to_datasets, dataset_name, delimiter=" "):
    folder = f"{folder_to_datasets}/{dataset_name}/"

    f = open(folder + "test2id.txt")
    test_triple_count_predicate = {}
    test_triple_count_total = 0

    f.readline()

    for line in f:
        splits = line.strip().split("\t")
        if len(splits) != 3:
            splits = line.strip().split(" ")

        if int(splits[2]) not in test_triple_count_predicate:
            test_triple_count_predicate[int(splits[2])] = 0

        test_triple_count_predicate[int(splits[2])] += 1
        test_triple_count_total += 1

    return test_triple_count_predicate, test_triple_count_total


def get_negative_triple_count(dataset_name, model_name, mat_type, delimiter="\t"):
    folder = "../Results/Materializations/" + dataset_name + "/" + model_name + "_" + mat_type + ".tsv"

    f = open(folder)
    test_triple_count_predicate = {}
    test_triple_count_total = 0

    f.readline()

    for line in f:
        if line == "\n":
            continue
        splits = line.strip().split(delimiter)

        if int(splits[1]) not in test_triple_count_predicate:
            test_triple_count_predicate[int(splits[1])] = 0

        test_triple_count_predicate[int(splits[1])] += 1
        test_triple_count_total += 1

    return test_triple_count_predicate, test_triple_count_total


def get_is(f_rule, relation_to_id, test_triple_count_predicate, test_triple_count_total):
    score = 0.0

    for line in f_rule:
        splits = line.strip().split("\t")
        selec = float(splits[-1])
        head = splits[0].split(" ==> ")[1].replace("(?a,?b)", "")
        head_id = relation_to_id[head]
        score += test_triple_count_predicate[head_id] * selec

    return round(score * 1.0 / test_triple_count_total, 4)


def get_is_hc_pca(f_rule, relation_to_id, test_triple_count_predicate, test_triple_count_total):
    score_selec = 0.0
    score_hc = 0.0
    score_pca = 0.0

    for line in f_rule:
        splits = line.strip().split("\t")
        selec = float(splits[-1])
        pca = float(splits[-2])
        hc = float(splits[-3])
        head = splits[0].split(" ==> ")[1].replace("(?a,?b)", "")
        head_id = relation_to_id[head]
        score_selec += test_triple_count_predicate[head_id] * selec
        score_hc += test_triple_count_predicate[head_id] * hc
        score_pca += test_triple_count_predicate[head_id] * pca

    return [round(score_selec * 1.0 / test_triple_count_total, 4), round(score_hc * 1.0 / test_triple_count_total, 4),
            round(score_pca * 1.0 / test_triple_count_total, 4)]


def run_individual(dataset_name, model_name, mat_type, relation_to_id, test_triple_count_predicate,
                   test_triple_count_total, folder_to_best_rules):
    f_best1 = open(
        f"{folder_to_best_rules}/{dataset_name}/{model_name}/{dataset_name}_{model_name}_mat_{mat_type}_best1.tsv", "r")
    f_best2 = open(
        f"{folder_to_best_rules}/{dataset_name}/{model_name}/{dataset_name}_{model_name}_mat_{mat_type}_best2.tsv", "r")
    f_best3 = open(
        f"{folder_to_best_rules}/{dataset_name}/{model_name}/{dataset_name}_{model_name}_mat_{mat_type}_best3.tsv", "r")
    f_best_h = open(
        f"{folder_to_best_rules}/{dataset_name}/{model_name}/{dataset_name}_{model_name}_mat_{mat_type}_bestH.tsv", "r")

    best1_is = get_is_hc_pca(f_best1, relation_to_id, test_triple_count_predicate, test_triple_count_total)
    best2_is = get_is_hc_pca(f_best2, relation_to_id, test_triple_count_predicate, test_triple_count_total)
    best3_is = get_is_hc_pca(f_best3, relation_to_id, test_triple_count_predicate, test_triple_count_total)
    heuristic_is = get_is_hc_pca(f_best_h, relation_to_id, test_triple_count_predicate, test_triple_count_total)

    f_best1.close()
    f_best2.close()
    f_best3.close()
    f_best_h.close()

    return best1_is, best2_is, best3_is, heuristic_is


def calculate_aggregates(dataset_name, mat_type, folder_to_results, folder_to_best_rules, folder_to_datasets):
    model_names = ["boxe", "complex", "hake", "hole", "quate", "rotate", "rotpro", "toruse", "transe", "tucker"]
    print(folder_to_results)
    relation_to_id = get_relation_to_id(folder_to_datasets, dataset_name)
    test_triple_count_predicate, test_triple_count_total = get_test_triple_count(folder_to_datasets, dataset_name)
    # print(test_triple_count_predicate)
    models = []
    selec = {"best1": [], "best2": [], "best3": [], "bestH": []}
    hc = {"best1": [], "best2": [], "best3": [], "bestH": []}
    pca = {"best1": [], "best2": [], "best3": [], "bestH": []}

    for model_name in model_names:
        # test_triple_count_predicate, test_triple_count_total = get_negative_triple_count(dataset_name, model_name,
        #                                                                                 mat_type)
        if not os.path.exists(f"{folder_to_best_rules}/{dataset_name}/{model_name}"):
            print(f"The file '{folder_to_best_rules}/{dataset_name}/{model_name}' does not exist, continuing...")
            continue

        best1, best2, best3, best_h = run_individual(dataset_name=dataset_name, model_name=model_name,
                                                     relation_to_id=relation_to_id, mat_type=mat_type,
                                                     test_triple_count_predicate=test_triple_count_predicate,
                                                     test_triple_count_total=test_triple_count_total,
                                                     folder_to_best_rules=folder_to_best_rules)
        models.append(model_name)
        selec["best1"].append(best1[0])
        selec["best2"].append(best2[0])
        selec["best3"].append(best3[0])
        selec["bestH"].append(best_h[0])

        hc["best1"].append(best1[1])
        hc["best2"].append(best2[1])
        hc["best3"].append(best3[1])
        hc["bestH"].append(best_h[1])

        pca["best1"].append(best1[2])
        pca["best2"].append(best2[2])
        pca["best3"].append(best3[2])
        pca["bestH"].append(best_h[2])


    selec_df = pd.DataFrame(
        {"Model name": models, "Best1": selec["best1"], "Best2": selec["best2"], "Best3": selec["best3"],
         "BestH": selec["bestH"]})

    hc_df = pd.DataFrame(
        {"Model name": models, "Best1": hc["best1"], "Best2": hc["best2"], "Best3": hc["best3"],
         "BestH": hc["bestH"]})

    pca_df = pd.DataFrame(
        {"Model name": models, "Best1": pca["best1"], "Best2": pca["best2"], "Best3": pca["best3"],
         "BestH": pca["bestH"]})

    plot_table(f"{folder_to_results}/{dataset_name}/{dataset_name}_mat_{mat_type}_selec.png", selec_df, dataset_name)
    plot_table(f"{folder_to_results}/{dataset_name}/{dataset_name}_mat_{mat_type}_hc.png", hc_df, dataset_name)
    plot_table(f"{folder_to_results}/{dataset_name}/{dataset_name}_mat_{mat_type}_pca.png", pca_df, dataset_name)

    # f_result_selec.close()
    # f_result_hc.close()
    # f_result_pca.close()


def plot_table(path_to_write, table, dataset_name):
    print(path_to_write)
    fig, ax = plt.subplots(figsize=(10, 3))

    # remove axis
    ax.axis('off')

    # add a title to the figure
    fig.suptitle(f'Dataset name: {dataset_name}')

    # add the dataframe to the axis object as a table
    ax.table(cellText=table.values, colLabels=table.columns, loc='center')

    # save the figure as an image with the specified filename and format
    plt.savefig(f'{path_to_write}', bbox_inches='tight', pad_inches=0.5, dpi=300)
    table.to_csv(f'{path_to_write.replace(".png",".csv")}', index=False)

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    mat_type = sys.argv[2]
    folder_to_results = sys.argv[3]
    folder_to_best_rules = sys.argv[4]
    folder_to_datasets = sys.argv[5]

    calculate_aggregates(dataset_name=dataset_name, mat_type=mat_type, folder_to_results=folder_to_results,
                         folder_to_best_rules=folder_to_best_rules, folder_to_datasets=folder_to_datasets)
    print("Calculation done")
