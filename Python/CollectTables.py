import pickle
import os
import sys
import pandas as pd


def collect_materializations(folder_to_materialization, mat_type, datasets, models):
    for dataset in datasets:
        print(dataset)
        for model in models:
            print(model)
            try:
                folder = f"{folder_to_materialization}/{dataset}/{model}/{dataset}_{model}_triple_stats.pickle"
                num_predictions = -1
                if os.path.exists(folder):
                    obj = pickle.load(open(folder, "rb"))
                    num_predictions = 0
                    num_cyclic = 0
                    for key, value in obj.items():
                        if mat_type == 0:
                            if value[0] == 1 and value[1] == 1:
                                if key[0] == key[2]:
                                    num_cyclic += 1
                                num_predictions += 1

                        elif mat_type == 1:
                            if value[0] == 1 and value[1] == 1 and value[2] == 1:
                                num_predictions += 1

                        else:
                            if value[3] == 1:
                                num_predictions += 1

                print(f"Dataset:{dataset};Model:{model};|M|:{num_predictions};|C|:{num_cyclic}")
            except Exception as e:
                print(f"Dataset:{dataset};Model:{model};Error:{e}")


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


if __name__ == "__main__":
    folder_to_materializations = sys.argv[1]
    mat_type = 0
    collect_materializations(folder_to_materialization=folder_to_materializations, mat_type=mat_type,
                             datasets=["WN18RR"],
                             models=["boxe"])

