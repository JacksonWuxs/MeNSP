import random
import numpy 
import os


def frozen(seed):
    numpy.random.seed(seed)
    random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass

    try:
        import tensorflow
        tensorflow.random.set_seed(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = str(seed)
        os.environ["PYTHONASHSEED"] = str(seed)
    except ImportError:
        pass


def manual(root):
    any_ = False
    for task in range(1, 9):
        random = "%s/g%d/" % (root, task)
        human = "%s/human/g%d.tsv" % (root, task)
        if os.path.isdir(random) and os.path.exists(human):
            resp = 10000
            any_ = True
            with open(human, encoding="utf8") as human, \
                 open(random + "train.tsv", "w", encoding="utf8") as train, \
                 open(random + "resp_idx.txt", "a+", encoding="utf8") as idx, \
                 open(random + "resp_meta.txt", "a+", encoding="utf8") as meta:
                for row in human:
                    label, text = row.strip().split("\t")
                    idx.write(str(resp) + '\n')
                    meta.write(text + '\n')
                    train.write("g%d\t%d\t%s\n" % (task, resp, label))
                    resp += 1
    return any_

def oneshot(root):
    for task in range(1, 9):
        subset = {}
        with open("%s/g%s/train.tsv" % (root, task), encoding="utf8") as f:
            for row in f:
                label = int(row.strip().split("\t")[-1])
                if label not in subset:
                    subset[label] = row
        with open("%s/g%s/train.tsv" % (root, task), "w", encoding="utf8") as f:
            for row in subset.values():
                f.write(row)


def split_dataset(target, train_rate=3, valid_rate=0, seed=42):
    scores = {}
    with open(target + "/full.tsv", encoding="utf8") as full:
        for row in full:
            scores.setdefault(row.rsplit("\t")[-1].strip(), []).append(row)

    with open(target + "/train.tsv", "w", encoding="utf8") as train,\
         open(target + "/valid.tsv", "w", encoding="utf8") as valid,\
         open(target + "/test.tsv", "w", encoding="utf8") as test:
        for records in scores.values():
            random.seed(seed)
            random.shuffle(records)
            size = len(records)
            if isinstance(train_rate, float):
                train_rate = int(train_rate * size)
            if isinstance(valid_rate, float):
                valid_rate = int(valid_rate * size)
            for idx, row in enumerate(records):
                if idx < train_rate:
                    train.write(row)
                elif idx < train_rate + valid_rate:
                    valid.write(row)
                else:
                    test.write(row)



def load_problems(fpath):
    with open(fpath, "r", encoding="utf8") as f:
        f.readline()
        return f.readlines()


def merge_responses(fpath, prob, drop_dup=True):
    with open(fpath, "r", encoding="utf8") as f:
        f.readline()
        num_dup, num_inc = 0, 0
        last, pairs, responses = "", {}, []
        for row in f:
            if "\t" not in row:
                if "\t" in last:
                    row = last + row
                    last = ""
                else:
                    last += row
                    continue
            text, score = row.strip().replace("\n", "").split("\t")
            text = text.strip()
            if (text.startswith('"') and text.endswith('"')) or \
               (text.startswith("'") and text.endswith("'")):
                text = text[1:-1]
            if text[-1].isalpha():
                text = text + '.'
            text = text.replace("'s", "")
            if text in pairs:
                num_dup += 1
                if pairs[text] != score:
                    num_inc += 1
                    #print("Inconsistent Score Warning - Problem: %s | Answer: %s" % (prob, text))
            pairs[text] = score
            responses.append((text, score))

    print("Merge Subset - Problem: %s | #Unique: %s | #Duplicate: %s | #Inconsist: %s" % (prob, len(pairs), num_dup, num_inc))
    if drop_dup:
        return pairs.items()
    return responses


def shuffle_dataset(data):
    with open(data, encoding="utf8") as f:
        records = f.readlines()
    random.shuffle(records)
    with open(data, 'w', encoding="utf8") as f:
        for row in records:
            f.write(row)


def prepare_datasets(root, seed=42):
    problems = load_problems(root + r"/problems.txt")
    tasks = [_ for _ in os.listdir(root) if os.path.isdir(root + "/" + _) and _.startswith("g")]
    assert len(tasks) == len(problems)
    for pmeta, pname in zip(problems, tasks):
        task_root = root + "/" + pname
        data = task_root + "/%s_txt/%s_fulltxt.txt" % (pname, pname)
        data = task_root + "/%s_txt/%s.tsv" % (pname, pname)
        with open(task_root + "/prob_idx.txt", "w", encoding="utf8") as pindex_file, \
             open(task_root + "/prob_meta.txt", "w", encoding="utf8") as pmeta_file:
            pindex_file.write(pname + '\n')
            pmeta_file.write(pmeta)
        with open(task_root + "/resp_idx.txt", "w", encoding="utf8") as rindex_file, \
             open(task_root + "/resp_meta.txt", "w", encoding="utf8") as rmeta_file, \
             open(task_root + "/full.tsv", "w", encoding="utf8") as full:
            responses = merge_responses(data, pname)
            for ridx, (resp, score) in enumerate(responses):
                rindex_file.write(str(ridx) + '\n')
                rmeta_file.write(resp + '\n')
                full.write('%s\t%s\t%s\n' % (pname, ridx, score))
        shuffle_dataset(task_root + "/full.tsv")
        split_dataset(task_root, seed=seed)
        
        
                    
    
