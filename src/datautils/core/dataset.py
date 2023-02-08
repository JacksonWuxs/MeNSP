import os

import numpy as np
import torch as tc

from .lookup import LookupTable
from .corpus import CorpusSearchIndex
from .templates import Template
from .verbalizers import Simple
from .cache import HashCache


class _BaseType:
    def __init__(self, root, category, split, verbalizers, template):
        assert os.path.exists(root)
        assert isinstance(split, str)
        assert isinstance(verbalizers, (list, tuple))
        assert all(isinstance(_, Simple) for _ in verbalizers)
        assert isinstance(template, Template)
        self._root = root
        self._category = category
        self._segment = split
        self._template = template
        self._verbalizers = verbalizers
        
        self._names = LookupTable.from_txt(root + r"/%s_idx.txt" % category)
        self._meta = CorpusSearchIndex(root + r"/%s_meta.txt" % category)
        self._cache = HashCache(self._verbalize, len(self._names), "freq")
        assert len(self._names) == len(self._meta)

    def __len__(self):
        return len(self._names)

    def get_sample_index(self, instance_name):
        if isinstance(instance_name, str):
            return self._names[instance_name]
        return instance_name

    def get_features(self, idx, use_cache=True):
        if use_cache:
            return dict(self._cache.collect(idx))
        return dict(self._verbalize(idx))

    def get_profile(self, idx, use_cache=True):
        features = self.get_features(idx, use_cache)
        return self._template.construct(**features)

    def _verbalize(self, idx):
        assert isinstance(idx, int) and idx < len(self._names)
        features = self._meta[idx].split(self._segment)
        assert len(features) == len(self._verbalizers), "%s not match %s" % (features, len(self._verbalizers))
        pairs = []
        for feat, verb in zip(features, self._verbalizers):
            pairs.append(verb.verbalize(feat))
        return pairs
        

class BaseMeta:
    def __init__(self, root, split, prob_verbs, resp_verbs, prob_temp, resp_temp, pair_temp):
        assert isinstance(pair_temp, Template)
        self.root = os.path.abspath(root).replace(r"\\", "/")
        self.pair_temp = pair_temp
        self.probs = _BaseType(self.root, "prob", split, prob_verbs, prob_temp)
        self.resps = _BaseType(self.root, "resp", split, resp_verbs, resp_temp)
        
    def get_feed_dict(self, pid, rid):
        pid = self.probs.get_sample_index(pid)
        rid = self.resps.get_sample_index(rid)
        feed_dict = {"problem": pid, "response": rid}
        pfeat = self.probs.get_features(pid)
        rfeat = self.resps.get_features(rid)
        ids, masks, segs = self.pair_temp.construct(**(pfeat | rfeat))
        feed_dict["pair_ids"] = np.array(ids)
        feed_dict["pair_masks"] = np.array(masks)
        feed_dict["pair_segs"] = np.array(segs)
        return feed_dict

    def get_profiles(self, who="both"):
        if who in ("both", "problem"):
            for idx in range(len(self.users)):
                yield self.probs.get_profile(idx, False)

        if who in ("both", "response"):
            for idx in range(len(self.items)):
                yield self.resps.get_profile(idx, False)
        

class BaseData(tc.utils.data.Dataset):
    def __init__(self, subset, metaset, split, sampling):
        assert isinstance(split, str)
        assert isinstance(metaset, BaseMeta) and subset in ("train", "test", "valid", "full")
        super(tc.utils.data.Dataset).__init__()
        self.meta, self.probs, self.resps = metaset, metaset.probs, metaset.resps
        self.data = CorpusSearchIndex(metaset.root + r"/%s.tsv" % subset, sampling=sampling)
        self.seg = split

    def __iter__(self):
        for row in self.data:
            record = row.split(self.seg)
            yield (self.users.get_sample_index(record[0]),
                   self.items.get_sample_index(record[1]),
                   float(record[2]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pid, rid, rate = self.data[idx].split(self.seg)[:3]
        info = self.meta.get_feed_dict(pid, rid)
        info["score"] = float(rate)
        return info
                   
        
