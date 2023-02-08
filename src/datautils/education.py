import re
import collections
import string


import torch as tc
import numpy as np
import transformers
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

from .core.dataset import BaseMeta, BaseData
from .core.templates import Template
from .core.verbalizers import Continue, Category, Simple, Functional


class TextCleaner:
    def __init__(self):
        self._stopwords = set(stopwords.words("english")) | set(string.punctuation)
        self._stemmer = PorterStemmer()

    def __call__(self, text):
        tokens = word_tokenize(text.lower())
        return " ".join([self._stemmer.stem(_) for _ in tokens if _ not in self._stopwords])


class EducationMeta(BaseMeta):
    def __init__(self, root, tokenizer, template="%s {:rubric:} %s {:response:} %s"):
        self._cleaner = TextCleaner()
        cls = tokenizer.cls_token
        sep = tokenizer.sep_token
        args = (cls,) + (sep,) * (template.count("%s") - 1)
        prob_temp = Template(
                           prompt="The problem is {:description:}. The rubric is {:rubric:}.",
                           tokenizer=tokenizer,
                           maxlen=64,
                           )
        resp_temp = Template(
                           prompt="The student response to the problem is {:response:}",
                           tokenizer=tokenizer,
                           maxlen=48,
                            )
        pair_temp = Template(
                           prompt=template % args,
                           tokenizer=tokenizer,
                           maxlen=510,
                           slot_args={
                               "rubric": {"maxlen": 256},
                               "background": {"maxlen": 256},
                               "response": {"maxlen": 256},
                               }
                            )
        prob_verbs = [Simple(_, "Empty") for _ in ["background", "problem", "grad0", "grad1", "grad2", "grad3"]]
        resp_verbs = (Simple("response"),)
        BaseMeta.__init__(self, root, "\t", prob_verbs, resp_verbs, prob_temp, resp_temp, pair_temp)
        pfeat = self.probs.get_features(0)
        self._label_mask = [1 if pfeat["grad%d" % _] else 0 for _ in range(4)]

    def get_feed_dict(self, pid, rid):
        pid = self.probs.get_sample_index(pid)
        rid = self.resps.get_sample_index(rid)
        feed_dict = {"problem": pid, "response": rid}
        pfeat = self.probs.get_features(pid)
        rfeat = self.resps.get_features(rid)

        comb_ids, comb_masks, comb_segs = [], [], []
        for grad in range(4):
            if pfeat["grad%d" % grad] is None:
                pfeat["grad%d" % grad] = "nothing"
            temp_feat = {"background": pfeat["background"],
                         "problem": pfeat["problem"],
                         "rubric": pfeat["grad%d" % grad]}
            _, ids, masks, segs = self.pair_temp.construct(**(temp_feat | rfeat))
            comb_ids.append(ids)
            comb_masks.append(masks)
            comb_segs.append(segs)
        feed_dict["ids"] = np.array([comb_ids])
        feed_dict["masks"] = np.array([comb_masks])
        feed_dict["segs"] = np.array([comb_segs])
        feed_dict["label_mask"] = np.array(self._label_mask)
        return feed_dict

    def get_choices(self,):
        pfeat = self.probs.get_features(0)
        return [_ for _ in range(4) if pfeat["grad%d" % _] is not None]

    

class EducationData(BaseData):
    def __init__(self, metaset, subset, sampling=None):
        BaseData.__init__(self, subset, metaset, "\t", sampling)

    def get_labels(self):
        return [float(row.split(self.seg)[2]) for row in self.data]

    def get_responses(self):
        responses = []
        for row in self.data:
            _, resp, score = row.split(self.seg)
            rid = self.resps.get_sample_index(resp)
            rspn = self.resps.get_features(rid)["response"]
            responses.append(rspn)
        return responses

    def get_problem(self):
        return self.probs.get_features(0)

    def get_rubrics(self):
        meta_prob = self.get_problem()
        rubrics = []
        for grad in range(4):
            if meta_prob["grad%d" % grad] is not None:
                rubrics.append(meta_prob["grad%d" % grad])
        return rubrics
