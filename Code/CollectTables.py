import pickle
import os
import sys

import pandas as pd


def collect_materializations(folder_to_materialization, mat_type, datasets, models):
    for dataset in datasets:
        for model in models:
            try:
                folder = f"{folder_to_materialization}/{dataset}/{model}/{dataset}_{model}_mat_{mat_type}.pickle"
                num_predictions = 0
                if os.path.exists(folder):
                    obj = pickle.load(open(folder))
                    num_predictions = 0

                    for key, value in obj.items():
                        if mat_type == 0:
                            if value[0] == 1 and value[1] == 1:
                                num_predictions += 1

                        elif mat_type == 1:
                            if value[0] == 1 and value[1] == 1 and value[2] == 1:
                                num_predictions += 1

                        else:
                            if value[3] == 1:
                                num_predictions += 1

                print("Dataset:", dataset, "Model:", model, f"n_mat_{mat_type}:", num_predictions)
            except Exception as e:
                print("Dataset:", dataset, "Model:", model, "Error:", e)


def summary(folder_to_rules, mat_type, datasets, models):
    selec_array = []

    for dataset in datasets:
        for model in models:
            folder = f"{folder_to_rules}/{dataset}/{model}/{dataset}_{model}_mat_{mat_type}_bestH.tsv"

            if os.path.exists(folder):
                try:
                    f = open(folder)
                    for line in f:
                        if line == "\n":
                            continue

                        splits = line.strip().split("\t")

                        selec = float(splits[-1])
                        selec_array.append(selec)

                    selec_pd = pd.DataFrame(selec_array)
                    desc = selec_pd.describe()
                    print("Dataset:", dataset, "Model:", model, "Min:", desc.loc["min", 0], "Max:", desc.loc["max", 0],
                          "Q1:", desc.loc["25%", 0], "Q3:", desc.loc["75%", 0], "Median:", selec_pd[0].median())
                except Exception as e:
                    print("Dataset:", dataset, "Model:", model, "Error:", e)


def collect_pbe():
    models = ["TuckER"]
    datasets = ["WN18", "WN18RR", "FB15K", "FB15K-237"]

    folder_to_dataset = "D:\\PhD\\Work\\EmbeddingInterpretibility\\Interpretibility\\Datasets\\"
    folder_to_pbe = "D:\\\PhD\\\Work\\\EmbeddingInterpretibility\\\Interpretibility\\\Results\\Materializations\\"
    for model in models:
        for dataset in datasets:
            test_file_path = f"{folder_to_dataset}{dataset}\\test2id.txt"
            pbe_file_path = f"{folder_to_pbe}{dataset}\\{model}_positives_before_expected.tsv"
            n_test_triples = int(open(test_file_path).readline())

            n_pbe = 0
            with open(pbe_file_path) as f:
                for line in f:
                    splits = line.strip().split("\t")
                    n_pbe += int(splits[1])

            print(f"Model: {model}, Dataset: {dataset}, Misp: {((2 * n_test_triples) - n_pbe) / (2 * n_test_triples)}")


if __name__ == "__main__":
    folder_to_materializations = sys.argv[1]
    folder_to_best_rules = sys.argv[2]
    mat_type = int(sys.argv[3])

    collect_materializations(folder_to_materialization=folder_to_materializations, mat_type=mat_type,
                             datasets=["FB15K", "FB15K-237", "WN18RR"],
                             models=["boxe", "complex", "hake", "hole", "quate", "rotate", "rotpro", "toruse", "transe",
                                     "tucker"])
    collect_materializations(folder_to_materialization=folder_to_materializations, mat_type=mat_type,
                             datasets=["WN18"],
                             models=["boxe", "complex", "hake", "hole", "quate", "rotate", "rotpro", "toruse",
                                     "transe"])
    collect_materializations(folder_to_materialization=folder_to_materializations, mat_type=mat_type,
                             datasets=["YAGO3-10"],
                             models=["boxe", "complex", "hake", "hole", "quate", "rotpro", "toruse", "transe",
                                     "tucker"])

    summary(folder_to_rules=folder_to_best_rules, mat_type=mat_type, datasets=["FB15K", "FB15K-237", "WN18RR"],
            models=["boxe", "complex", "hake", "hole", "quate", "rotate", "rotpro", "toruse", "transe", "tucker"])

    summary(folder_to_rules=folder_to_best_rules, mat_type=mat_type, datasets=["WN18"],
            models=["boxe", "complex", "hake", "hole", "quate", "rotate", "rotpro", "toruse", "transe"])

    summary(folder_to_rules=folder_to_best_rules, mat_type=mat_type, datasets=["YAGO3-10"],
            models=["boxe", "complex", "hake", "hole", "quate", "rotpro", "toruse", "transe",
                    "tucker"])
