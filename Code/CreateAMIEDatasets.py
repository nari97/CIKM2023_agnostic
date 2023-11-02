
def createAMIEDatasets():
    folder = r"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Datasets"

    datasets = ["WN18", "WN18RR"]
    files = ["train2id", "valid2id", "test2id"]

    for dataset in datasets:
        print(f"Loading dataset: {dataset}")
        triples = []
        relation2id = {}

        with open(f"{folder}\\{dataset}\\relation2id.txt") as f:
            f.readline()
            for line in f:
                splits = line.strip().split("\t")
                relation2id[splits[1]] = splits[0]

        for ifile in files:
            with open(f"{folder}\\{dataset}\\{ifile}.txt") as f:
                f.readline()
                for line in f:
                    splits = line.strip().split(" ")

                    triples.append([splits[0], splits[2], splits[1]])

        with open(f"{folder}\\{dataset}\\{dataset}_AMIE_triples_relations_named.tsv", "w+") as f:
            for triple in triples:
                f.write(f"{triple[0]}\t{relation2id[triple[1]]}\t{triple[2]}\n")


if __name__ == "__main__":
    createAMIEDatasets()