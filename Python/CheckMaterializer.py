import pickle

def check_materializer(path):
    df = pickle.load(open(path, "rb"))
    ctr = 0
    for key in df.keys():
        if ctr == 15:
            break
        print(key, df[key])
        ctr += 1

if __name__ == "__main__":
    path = r"C:\Users\nk1581\Downloads\WN18_boxe_inference.pickle"
    check_materializer(path)