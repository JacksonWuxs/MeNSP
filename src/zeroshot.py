import math
import sys
import os
import copy
import pickle
import random
import warnings
import collections

import numpy as np
import torch as tc
from sklearn import metrics
import matplotlib.pyplot as plt

from datautils.education import EducationMeta, EducationData
from models.baselines import TfidfSVMClassifier
from models.final import TwoStages
from models.plms import PLM
from utils import frozen, prepare_datasets, oneshot, manual


SEED = int(sys.argv[1])
CUDA = str(sys.argv[2])
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA 
warnings.filterwarnings("ignore")
PLMNAME = "bert-large-cased"

device = "cuda:%s" % CUDA if tc.cuda.is_available() else "cpu"


def evalulate(truth, predict):
    if isinstance(truth, (list, tuple)):
        truth = np.array(truth)
    if isinstance(predict, (list, tuple)):
        predict = np.array(predict)
    elif isinstance(predict, tc.Tensor):
        predict = predict.cpu().numpy()
    assert isinstance(truth, np.ndarray)
    assert isinstance(predict, np.ndarray)
    assert len(truth) == len(predict)
    if len(predict.shape) != 1:
        predict = np.argmax(predict, axis=-1)
    return {"Acc": metrics.accuracy_score(truth, predict),
            "Kappa": metrics.cohen_kappa_score(truth, predict),
            "F1": metrics.f1_score(truth, predict, average="weighted")}


def pipeline(seed, prefix=""):
    tokenizer = PLM.load_tokenizer(PLMNAME)
    print("\n\n\n")
    for task in range(4, 7):
        data = "../datasets/g%s" % task
        
        meta = EducationMeta(data, tokenizer, template="%s The student answer: '{:rubric:}' is [::MASK::] to the rubrics: '{:response:}' %s")
        full_data = EducationData(meta, "full")
        train_data = EducationData(meta, "train")
        test_data = EducationData(meta, "test")
        shot = len(train_data) // 3
        if shot > 1:
            frozen(seed)
            yrandom = random.choices([0, 1], k=len(test_data.get_labels()))
            rslt = evalulate(test_data.get_labels(), yrandom)
            print(prefix, "Seed=%d" % seed, "Task=%d" % task, "Model=Random", "Shot=0", rslt, sep="\t")


        frozen(seed)
        model = TwoStages(PLMNAME, device)
        model.set_rubrics(full_data.get_rubrics())
        if shot > 1:
            yhat = model.predict(test_data.get_responses(), test_data.get_labels())
            rslt = evalulate(test_data.get_labels(), yhat)
            print(prefix, "Seed=%d" % seed, "Task=%d" % task, "Model=MeNSP", "Shot=0", rslt, sep="\t")

        model.fit(train_data.get_responses(), train_data.get_labels())
        yhat = model.predict(test_data.get_responses(), test_data.get_labels())
        rslt = evalulate(test_data.get_labels(), yhat)
        print(prefix, "Seed=%d" % seed, "Task=%d" % task, "Model=MeNSP", "Shot=%d" % shot, rslt, sep="\t")

        for clf in ["RF", "GB", "Vote"]:
            frozen(seed)
            baseline = TfidfSVMClassifier(clf=clf)
            baseline.fit(train_data)
            yhat_test = baseline.predict(test_data)
            rslt = evalulate(test_data.get_labels(), yhat_test)
            print(prefix, "Seed=%d" % seed, "Task=%d" % task, "Model=%s" % clf, "Shot=%d" % shot, rslt, sep="\t")




if __name__ == "__main__":
    ROOT = "../datasets"
    frozen(SEED)
    for seed in random.sample(range(1, 65536), k=5):
        prepare_datasets(ROOT, seed)
        pipeline(seed, "Random-3shot")
        oneshot(ROOT)
        pipeline(seed, "Random-1shot")
        if manual(ROOT):
            pipeline(seed, "Manual-3shot")
            oneshot(ROOT)
            pipeline(seed, "Manual-1shot")
