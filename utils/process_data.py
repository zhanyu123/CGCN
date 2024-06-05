from collections import defaultdict as ddict

def read_dictionary(file_path):
    d = {}
    with open(file_path, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d


def read_triplets(file_path, entity_dict, relation_dict):
    l = []
    with open(file_path, 'r+') as f:
        for line in f:
            triplet_line = line.strip().split('\t')
            s = entity_dict[triplet_line[0]]
            p = relation_dict[triplet_line[1]]
            o = entity_dict[triplet_line[2]]
            l.append([s, p, o])
    return l


def process(dataset, num_rel):

    sr2o = ddict(set)
    for subj, rel, obj in dataset['train_triplets']:
        sr2o[(subj, rel)].add(obj)
        sr2o[(obj, rel + num_rel)].add(subj)

    sr2o_train = {k: list(v) for k, v in sr2o.items()}
    for split in ['valid_triplets', 'test_triplets']:
        for subj, rel, obj in dataset[split]:
            sr2o[(subj, rel)].add(obj)
            sr2o[(obj, rel + num_rel)].add(subj)
    sr2o_all = {k: list(v) for k, v in sr2o.items()}
    triplets = ddict(list)


    for (subj, rel), obj in sr2o_train.items():
        triplets['train_triplets'].append({'triple': (subj, rel, -1), 'label': sr2o_train[(subj, rel)]})

    for split in ['valid_triplets', 'test_triplets']:
        for subj, rel, obj in dataset[split]:
            triplets[f"{split}_tail"].append({'triple': (subj, rel, obj), 'label': sr2o_all[(subj, rel)]})
            triplets[f"{split}_head"].append({'triple': (obj, rel + num_rel, subj), 'label': sr2o_all[(obj, rel + num_rel)]})
    triplets = dict(triplets)
    return triplets

