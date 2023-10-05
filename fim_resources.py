import sys, re, itertools, datetime
import numpy, scipy.sparse
import matplotlib.pyplot as plt
import pdb

DATASETS = {}
DATASETS["abalone"] = {"in_file": "abalone.data", "format": "data_txt"}
DATASETS["covertype"] = {"in_file": "covtype.data", "format": "data_txt"}
DATASETS["house"] = {"in_file": "house.data", "format": "data_txt"}
DATASETS["pizzas"] = {"in_file": "pizzas.csv", "format": "matrix"}
DATASETS["plants"] = {"in_file": "plants.data", "format": "trans_txt", "params": {"contains_ids": True}}

# datetime can be used for tracking running time
# tic = datetime.datetime.now()
# [running some function that takes time]
# tac = datetime.datetime.now()
# print("Function running time: %s" % (tac-tic))
# print("The function took %s seconds to complete" % (tac-tic).total_seconds())

### Below are functions for loading datasets from various formats, returning a list of transactions along with a list of the labels of the items. Each transaction is stored as a (frozen)set of item ids (integers).
def load_trans_num(in_file, sep=","):
    with open(in_file) as fp:
        tracts = [frozenset([int(s.strip()) for s in line.strip().split(sep)]) for line in fp if not re.match("#", line)]
    U = sorted(set().union(*tracts))
    return tracts, U

def load_trans_txt(in_file, sep=",", contains_ids=False):
    with open(in_file) as fp:
        tracts = [frozenset([s.strip() for s in line.strip().split(sep)[1*contains_ids:]]) for line in fp if not re.match("#", line)]
    U = sorted(set().union(*tracts))
    map_items = dict([(v,k) for (k,v) in enumerate(U)])
    tracts = [frozenset([map_items[i] for i in t]) for t in tracts]
    return tracts, U

def load_matrix(in_file, sep=","):
    with open(in_file) as fp:
        firstline = fp.readline()
        parts = firstline.strip().strip("#").split(sep)
        try:
            s = [k for (k,v) in enumerate(parts) if int(v) != 0]
            U = range(len(s))
            tracts = [frozenset(s)]
        except ValueError:
            U = parts
            tracts = []
        tracts.extend([frozenset([k for (k,v) in enumerate(line.strip().split(sep)) if int(v) != 0]) for line in fp if not re.match("#", line)])
    if len(U) <= max(set().union(*tracts)):
        print("Something went wrong while reading!")
        return [], []
    return tracts, U

def load_sparse_num(in_file, sep=","):
    tracts = {}
    with open(in_file) as fp:
        for line in fp:
            if not re.match("#", line):
                i, j = map(int, line.strip().split(sep)[:2])
                if i not in tracts:
                    tracts[i] = []
                tracts[i].append(j)
    U = range(max(set().union(*tracts.values()))+1)
    tracts = [frozenset(tracts.get(ti, [])) for ti in range(max(tracts.keys())+1)]
    return tracts, U

def load_sparse_txt(in_file, sep=","):
    tracts = {}
    with open(in_file) as fp:
        for line in fp:
            if not re.match("#", line):
                i, j = line.strip().split(sep)[:2]
                i = int(i)
                if i not in tracts:
                    tracts[i] = []
                tracts[i].append(j)
    U = sorted(set().union(*tracts.values()))
    map_items = dict([(v,k) for (k,v) in enumerate(U)])
    tracts = [frozenset([map_items[i] for i in tracts.get(ti,[])]) for ti in range(max(tracts.keys())+1)]
    return tracts, U

def load_data_txt(in_file, sep=","):
    bin_file = ".".join(in_file.split(".")[:-1]+["bininfo"])
    bininfo, U = read_bininfo(bin_file)
    tracts = []
    with open(in_file) as fp:
        for line in fp:
            if not re.match("#", line):
                parts = line.strip().split(sep)
                t = []
                for k, part in enumerate(parts):
                    if k in bininfo:
                        if "bool" in bininfo[k]:
                            if bininfo[k]["with_false"]:
                                t.append(bininfo[k]["offset"]+1*(part in bininfo[k]["bool"]))
                            elif part in bininfo[k]["bool"]:
                                t.append(bininfo[k]["offset"])
                        elif "cats" in bininfo[k]:
                            if part in bininfo[k]["cats"]:
                                t.append(bininfo[k]["offset"]+bininfo[k]["cats"].index(part))
                        elif "bounds" in bininfo[k]:
                            off = 0
                            v = float(part)
                            while off < len(bininfo[k]["bounds"]) and v > bininfo[k]["bounds"][off]:
                                off += 1
                            t.append(bininfo[k]["offset"]+off)
                tracts.append(frozenset(t))
    return tracts, U

def read_bininfo(bin_file):
    bininfo = {}
    with open(bin_file) as fp:
        for line in fp:
            tmp = re.match("^(?P<pos>[0-9]+) *(?P<name>[^ ]+) *(?P<type>\w+) *(?P<quote>[\'\"])(?P<details>.*)(?P=quote)", line)
            if tmp is not None:
                if tmp.group("type") == "BOL":
                    pos_trim = [v.strip() for v in tmp.group("details").split(",") if v.strip() != ""]
                    bininfo[int(tmp.group("pos"))] = {"bool": pos_trim, "name": tmp.group("name"), "with_false": len(tmp.group("details").split(",")) != len(pos_trim)}
                elif tmp.group("type") == "CAT":
                    bininfo[int(tmp.group("pos"))] = {"cats": tmp.group("details").split(","), "name": tmp.group("name")}
                else:
                    tt = re.search("equal\-(?P<type>(width)|(height)) *k=(?P<k>[0-9]+)", tmp.group("details"))
                    if tt is not None:
                        bininfo[int(tmp.group("pos"))] = {"type": "equal-%s" % tt.group("type"), "k": int(tt.group("k")), "name": tmp.group("name")}
                    else:
                        try:  
                            bininfo[int(tmp.group("pos"))] = {"type": "fixed", "bounds": sorted(map(float, tmp.group("details").split(","))), "name": tmp.group("name")}
                        except ValueError:
                            pass

    fields = []
    ks = sorted(bininfo.keys())
    for k in ks:
        bininfo[k]["offset"] = len(fields)
        if "bool" in bininfo[k]:
            if bininfo[k]["with_false"]:
                fields.extend(["%s_%s" % (bininfo[k]["name"], v) for v in ["False", "True"]])
            else:
                fields.extend(["%s_%s" % (bininfo[k]["name"], v) for v in ["True"]])
        elif "cats" in bininfo[k]:
            fields.extend(["%s_%s" % (bininfo[k]["name"], v) for v in bininfo[k]["cats"]])
        elif "bounds" in bininfo[k]:
            fields.append("%s_:%s" % (bininfo[k]["name"], bininfo[k]["bounds"][0]))
            fields.extend(["%s_%s:%s" % (bininfo[k]["name"], bininfo[k]["bounds"][i], bininfo[k]["bounds"][i+1]) for i in range(len(bininfo[k]["bounds"])-1)])
            fields.append("%s_%s:" % (bininfo[k]["name"], bininfo[k]["bounds"][-1]))
        elif "k" in bininfo[k]:
            fields.extend(["%s_bin%d" % (bininfo[k]["name"], v) for v in range(bininfo[k]["k"])])
    return bininfo, fields

# Task 1: Implementing support count function
def supportCountForItemSet (dataset,itemset):
    support_count = 0
    # print(itemset)
    for transaction in dataset[0]:
        if(set([itemset]).issubset(tranasaction)):
            support_count +=1
    return support_count
# Task 2: Implement a simple level-wise search
def generate_fist_frequent_itemsets(dataset):

    data = dataset[0]
    # get the unique value for the set
    unique_values = {item for frozenset_item in data for item in frozenset_item}
    # Converting the unique values to a list
    prev_frequent_itemsets = list(unique_values) # for the first level

    print(f"prev_frequent_itemsets or First level of mining set:: {prev_frequent_itemsets}")
    return prev_frequent_itemsets


# def generate_candidates(dataset,first_frequent_itemsets, k):
#     candidates = set()
#     data = dataset[0]
#     # Extract frequent (k-1)-itemsets
#
#     # Join step: Join (k-1)-itemsets with themselves to generate k-item candidates
#     for i in range(len(first_frequent_itemsets)):
#         for j in range(i + 1, len(first_frequent_itemsets)):
#             itemset1 = frequent_k_minus_1_itemsets[i]
#             itemset2 = frequent_k_minus_1_itemsets[j]
#
#             # Check if the first (k-2) items are the same, and join them if they are
#             if itemset1[:-1] == itemset2[:-1]:
#                 new_candidate = tuple(sorted(set(itemset1).union(set(itemset2))))
#
#                 # Check if all (k-1)-subsets are frequent
#                 if all(subset in frequent_k_minus_1_itemsets for subset in combinations(new_candidate, k - 1)):
#                     candidates.add(new_candidate)
#
#     return candidates
def simple_level_wise_search(dataset, min_support_threshold):
    k=1 # initialize the K value
    # Generate the first level of itemset
    first_frequent_itemsets = generate_fist_frequent_itemsets(dataset)
    # generating candidate k-itemsets by extending frequent k − 1-itemsets by appending one item
    # generate_candidates(dataset,first_frequent_itemsets,k)

    print(dataset)
    exit()
if __name__ == "__main__":
    # Task 1:
    # loading the transactions from abalone dataset
    dataset = load_data_txt("abalone.data")
    tranasaction = dataset[0][0] # Taking the transcation from dataset
    support_count = 0
    min_support_threshold = 2
    for itemset in tranasaction:
        # print(itemset)
        # exit()
        if(support_count==0):
            support_count = supportCountForItemSet(dataset,itemset)
        print(f"Support count is: {support_count}")

    # Task 2:
        simple_level_wise_search(dataset,min_support_threshold)
        exit()

    if len(sys.argv) < 2:
        print("Please choose a dataset: %s" % ", ".join(DATASETS.keys()))
        exit()
    which = sys.argv[1]
        
    if which not in DATASETS:
        print("Unknown setup (%s)!" % which)
        exit()
    try:
        method_load =  eval("load_%s" % DATASETS[which]["format"])
    except AttributeError:
        raise Exception('No known method to load this data type (%)!' % DATASETS[which]["format"])
    tracts, U = method_load(DATASETS[which]["in_file"], **DATASETS[which].get("params", {}))

    print("Dataset '%s' loaded. The dataset contains %d transactions over %d items." % (which, len(tracts), len(U)))
    print("The first three transactions are:")
    for ti in range(3):
        print("transaction id: (%d)\titem ids: {%s}\titem labels: {%s}" % (ti+1, ", ".join(["%d" % ii for ii in tracts[ti]]), ", ".join([U[ii] for ii in tracts[ti]])))



