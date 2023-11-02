import random


def get_splits(path_to_data):
    path_to_test = rf"{path_to_data}/test2id.txt"
    path_to_validation = rf"{path_to_data}/valid2id.txt"

    triples = {}

    f_test = open(path_to_test)
    f_valid = open(path_to_validation)

    f_test.readline()
    f_valid.readline()

    for f in [f_test, f_valid]:
        for line in f:
            splits = line.strip().split("\t")
            if len(splits) < 2:
                splits = line.strip().split(" ")

            s = splits[0]
            r = splits[2]
            t = splits[1]

            if r not in triples:
                triples[r] = []

            triples[r].append((s, t))

    cnt = 0
    for r in triples:
        print(f"Relation {r}: {len(triples[r])}")
        cnt += len(triples[r])

    print(f"Total triple count: {cnt}, {cnt // 2}")

    f_new_valid = open(f"{path_to_data}/new_valid2id.txt", "w+")
    f_new_test = open(f"{path_to_data}/new_test2id.txt", "w+")

    triples_test = []
    triples_valid = []
    ctr_valid = 2

    for r in triples:
        random.shuffle(triples[r])

        midpoint = len(triples[r]) // 2

        first_half = triples[r][0:midpoint]
        second_half = triples[r][midpoint:]

        if len(triples[r]) % 2 == 1:
            print(r)
            if ctr_valid > 0:
                print("valid")
                triples_for_valid = second_half
                triples_for_test = first_half
                ctr_valid -= 1
            else:
                print("test")
                triples_for_valid = first_half
                triples_for_test = second_half
        else:
            print("equal")
            triples_for_valid = first_half
            triples_for_test = second_half

        for triple in triples_for_valid:
            s, o = triple
            triples_valid.append((s, o, r))

        for triple in triples_for_test:
            s, o = triple
            triples_test.append((s, o, r))

    f_new_test.write(f"{len(triples_test)}\n")
    f_new_valid.write(f"{len(triples_valid)}\n")
    for triple in triples_test:
        f_new_test.write(f"{triple[0]}\t{triple[1]}\t{triple[2]}\n")

    for triple in triples_valid:
        f_new_valid.write(f"{triple[0]}\t{triple[1]}\t{triple[2]}\n")

    f_new_test.close()
    f_new_valid.close()


if __name__ == "__main__":
    get_splits(r"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Datasets\NELL-995")
