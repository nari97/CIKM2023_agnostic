import pickle
import sys


def create_materialization(path_to_materialization, path_to_result, mat_kind=0):
    triple_dict = pickle.load(open(path_to_materialization, "rb"))

    result_f = open(path_to_result, "w+")
    # Index 0 is is_negative
    # Index 1 is ranked_better
    # Index 2 is correctly predicted
    # Index 3 is top-k

    # mat_kind=0 => including mispredictions
    # mat_kind=1 => excluding mispredictions
    # mat_kind=2 => top-k
    for key, value in triple_dict.items():
        #print(key, value)
        if mat_kind == 0:
            if value[0] == 1 and value[1] == 1:
                
                result_f.write("%s\t%s\t%s\n" % (key[0], key[1], key[2]))

        elif mat_kind == 1:
            if value[0] == 1 and value[1] == 1 and value[2] == 1:
                result_f.write("%s\t%s\t%s\n" % (key[0], key[1], key[2]))

        else:
            if value[3] == 1:
                result_f.write("%s\t%s\t%s\n" % (key[0], key[1], key[2]))

    result_f.close()
    with open(path_to_result, 'r') as f:
        contents = f.read()

    # Remove the newline character from the last line
    contents = contents.rstrip('\n')

    # Write the contents back to the file
    with open(path_to_result, 'w+') as f:
        f.write(contents)


if __name__ == "__main__":
    # path_to_materialization = sys.argv[1]
    # path_to_result = sys.argv[2]
    # mat_kind = int(sys.argv[3])
    #
    path_to_materialization = "D:/PhD/Work/EmbeddingInterpretibility/Interpretibility/Results/Materializations/FB15K/tucker/FB15K_tucker_triple_stats.pickle"
    path_to_result = "D:/PhD/Work/EmbeddingInterpretibility/Interpretibility/Results/Materializations/FB15K/tucker/FB15K_tucker_mat_0.tsv"
    mat_kind=0
    create_materialization(path_to_materialization, path_to_result, mat_kind)
