import pandas as pd

def parse_rules(file_name):
    f_rule = open(file_name, encoding="utf-8")

    headers = ["Rule", "Head Coverage", "Std Confidence", "PCA Confidence", "Positive Examples", "Body size",
               "PCA Body size", "Functional variable"]

    rules = []
    f_rule.readline()

    for line in f_rule:
        splits = line.strip().split("\t")



def getRulesByParameters(fileName, parameter = None, value = None):
    f = open(fileName, encoding = "utf-8")

    headers = ["Rule", "Head Coverage", "Std Confidence", "PCA Confidence", "Positive Examples", "Body size", "PCA Body size", "Functional variable"]
    i = 0
    rules = []
    for line in f:
        line = line.replace("\t", " ")
        vals = line.strip().split(" ")
        #print (vals)

        newvals = []
        for val in vals:
            if val != '':
                newvals.append(val)
        vals = newvals
        newvals = []
        st = ""
        for i in range(len(vals)-1, 0, -1):
            #print (vals[i])
            if i>len(vals)-8:
                newvals = [vals[i]] + newvals
            else:
                st = vals[i] + " " + st
        #print (st)
        newvals = [vals[0] + " " + st[:-1]] + newvals
        rules.append(newvals)
    f.close()


    data = pd.DataFrame(rules, columns=headers)

    if parameter == None:
        return data
    else:
        return data[data[parameter] >= value]

def getBestRules(rules, beta = 1):

    r = {}

    for i in range(len(rules)):
        rule = rules.loc[i]['Rule']
        s_beta = f_beta(beta, float(rules.loc[i]['Head Coverage']), float(rules.loc[i]['PCA Confidence']))

        head = rule[rule.index("=>")+3:]

        if head in r:
            r[head].append((rule, s_beta))
        else:
            r[head] = [(rule, s_beta)]

    #print (r)
    for key in r:
        r[key] = sorted(r[key], key = mysrt, reverse = True)

    #print (r)
    fr = {}

    for key in r:
        fr[key] = r[key][0]
    return fr

def mysrt(s):
    return float(s[-1])

def f_beta(beta, hc, conf):
    return ((1+beta*beta)*conf*hc)/(beta*beta*conf+hc)

def eAvg(rules, relations):
    sum = 0
    #print (rules)
    for key in rules:
        sum+= float(rules[key][1])

    return sum/relations

def eCounts(rules, counts):
    sum = 0
    total = 0

    for key in counts:
        total += counts[key]

    #print (counts)
    for key in rules:

        try:
            sum+=(counts[key[:key.index("(")]]*float(rules[key][1]))
        except KeyError:
            print (key)
            sum+=0
        #total+=counts[int(key[:key.index("(")])]

    if total == 0:
        return 0

    #print (total)
    return sum/total


def getNegativeCounts(fileName):
    #Returns negcounts
    f = open(fileName)

    counts = {}

    for line in f:
        splits = line.strip().split("\t")
        counts[splits[0]] = int(splits[1])

    return counts

def getRelationCount(fileName):
    f = open(fileName)

    for line in f:
        return int(line.strip())

    return None

def resolveRules(rules, fileName, fileName2):

    f = open(fileName, "r")
    r_dict = {}
    i = 0
    for line in f:
        if i == 0:
            i+=1
            continue
        
        split = line.strip().split("\t")
        #print (split)
        r_dict[split[1]] = split[0]

    f.close()
    f = open(fileName2, "w+")
    for key in rules:
        splits = rules[key][0].split(" ")
        
        if len(splits) == 3:
            anti = r_dict[splits[0][:splits[0].index("(")]] + splits[0][splits[0].index("("):]
            prec = r_dict[splits[2][:splits[2].index("(")]] + splits[2][splits[2].index("("):]

            f.write(anti + " => " + prec + "\t" + str(rules[key][1]) + "\n")

        if len(splits) == 4:
            anti1 = r_dict[splits[0][:splits[0].index("(")]] + splits[0][splits[0].index("("):]
            anti2 = r_dict[splits[1][:splits[1].index("(")]] + splits[1][splits[1].index("("):]
            prec = r_dict[splits[3][:splits[3].index("(")]] + splits[3][splits[3].index("("):]

            f.write(anti1 + " " + anti2 + " => " + prec + "\t" + str(rules[key][1]) + "\n")

    f.close()

def positivesPerDataset(datasets):

    pbe = {}
    for data in datasets:
        pbe[data] = {}
        r_count = getRelationCount("Datasets/" + data +"/relation2id.txt")

        for j in range(r_count):
            pbe[data][str(j)] = 0

        f = open("Datasets/" + data +"/new_test2id.txt")
        i = 0
        for line in f:
            if i == 0:
                i+=1
                continue
            vals = None
            if data == "FB13":
                vals = line.strip().split("\t")
            else:
                vals = line.strip().split(" ")

            if vals[2] in pbe[data]:
                pbe[data][vals[2]] +=1
            else:
                pbe[data][vals[2]] = 1
        f.close()
    
    return pbe

def getNegativeCountByRelation(filename):
    #Returns negative counts of a tsv file
    f = open(filename, "r")
    counts = {}
    for line in f:
        if line == "\n":
            continue
        splits = line.strip().split("\t")
        if splits[1] in counts:
            counts[splits[1]]+=1
        else:
            counts[splits[1]] = 1

    #print (counts)
    f.close()

    return counts

def getPositivesRankedBelowRatio(pbe, model):

    f = open(model + "_pbr.txt")

    pbr_c = {}

    #print (pbe)
    for key in pbe:
        pbr_c[key] = 0

    total = 0

    for key in pbe:
        total+= pbe[key]

    for line in f:
        l = line.strip().split(" ")
        pbr_c[l[0]] = int(l[1])

    #print (pbr_c)
    num = 0
    for key in pbe:
        num+= (2*pbe[key] - pbr_c[key])

    f.close()
    
    return num/(2*total)

def getNegativesRatio(allCounts, mCounts):

    mTotal = 0
    aTotal = 0

    for key in mCounts:
        mTotal+= int(mCounts[key])

    for key in allCounts:
        aTotal+= int(allCounts[key])

    return mTotal/aTotal
    
def writeLine(f, dataset, vals):

    st = dataset + " & "
    for val in vals:

        if not isinstance(val, str):
           val = round(val,3)
        st += str(val) + " & "

    f.write(st + "\n")

if __name__ == "__main__":
    datasets = ["WN18RR"]
    models = ["ComplEx"]

    #pbe = positivesPerDataset(datasets)

    em = pd.DataFrame(columns = ["index"] + models)
    em["index"] = datasets
    em = em.set_index("index")
    #print (em)
    beta = 1
    for dataset in datasets:

        relations = getRelationCount("../Datasets/" + str(dataset) + "/relation2id.txt")
        print (dataset + " : " + "Total number of relations : ", relations)

        pbe_arr = []
        neg_arr = []
        rules_arr = []

        for model in models:
            rules = getRulesByParameters("../Results/MinedRules/" + str(dataset) + "/" + str(model) + "_test.tsv")
            r = getBestRules(rules, beta)
            print (r)
            #pbe_arr.append(getPositivesRankedBelowRatio(pbe[dataset], "Results/Materialisations/" + dataset + "/" + model))
            mCounts = getNegativeCountByRelation("../Results/Materializations/" + dataset + "/" + model + "_materialized.tsv")
            # allCounts = getNegativeCounts("Results/Materialisations/" + dataset + "/" + model + "_negcounts.txt")
            # neg_arr.append(getNegativesRatio(allCounts, mCounts))
            # rules_arr.append(len(rules))



            print (dataset + "//" + model, " : ", {"eAvg" : round(eAvg(r, relations),3), "eWeighted" : round(eCounts(r, mCounts),3)})
            #
            # em[model].loc[dataset] = round(eCounts(r, mCounts),3)

            #resolveRules(r, "Datasets\\" + str(dataset)+"\\relation2id.txt", "Results/BestRules/"  + str(dataset) +  "/" + model + ".txt")