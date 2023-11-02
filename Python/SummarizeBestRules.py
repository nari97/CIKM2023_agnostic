import matplotlib.pyplot as plt

def summary(rule_filename):
    selec_array = []
    selec_to_rule = {}

    f = open(rule_filename)

    for line in f:
        if line == "\n":
            continue

        splits = line.strip().split("\t")

        rule = splits[0]
        selec = float(splits[-1])
        selec_array.append(selec)

        if selec not in selec_to_rule:
            selec_to_rule[selec] = []

        selec_to_rule[selec].append(rule)

    selec_pd = pd.DataFrame(selec_array)

    desc = selec_pd.describe()

    print("\tMin:", desc.loc["min", 0])
    print("\tMax:", desc.loc["max", 0])
    print("\tQ1:", desc.loc["25%", 0])
    print("\tQ3:", desc.loc["75%", 0])
    print("\tMedian:", selec_pd[0].median())

    return selec_to_rule

def print_min_max(selec_to_rule):

    min_key = min(selec_to_rule.keys())
    max_key = max(selec_to_rule.keys())

    print("\tMin rule:", selec_to_rule[min_key][0], "Selec:", min_key)
    print("\tMax rule:", selec_to_rule[max_key][0], "Selec:", max_key)

if __name__ == "__main__":
    folder = "D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\BestRules\\"
    datasets = ["FB15K", "FB15K-237", "WN18", "WN18RR"]
    models = ["TuckER"]

    for dataset in datasets:
        for model in models:
            print("Dataset:", dataset, "Model:", model)
            selec_to_rule = summary(folder + dataset + "/" + model + "_materialized_best1.tsv")
            print(print_min_max(selec_to_rule))
