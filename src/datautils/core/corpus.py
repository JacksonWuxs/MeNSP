import os


class CorpusSearchIndex:
    def __init__(self, file_path, encoding="utf8", cache_freq=5, sampling=None):
        self.fpath = file_path
        self.coding = encoding
        self.cache_freq = cache_freq
        self._lookup, self._numrow = [0], 0
        with open(self.fpath, encoding=self.coding) as f:
            while self._numrow != sampling:
                row = f.readline()
                if len(row) == 0:
                    break
                self._numrow += 1
                if self._numrow % cache_freq == 0:
                    self._lookup.append(f.tell())

    def __iter__(self):
        with open(self.fpath, encoding=self.coding) as f:
            for idx, row in enumerate(f, 1):
                yield row.strip()
                if idx == self._numrow:
                    break

    def __len__(self):
        return self._numrow

    def __getitem__(self, index):
        with open(self.fpath, encoding=self.coding) as f:
            cacheid = index // self.cache_freq
            f.seek(self._lookup[cacheid])
            for idx, row in enumerate(f, cacheid * self.cache_freq):
                if idx == index:
                    return row.strip()
        raise IndexError("Index %d is out of boundary" % index)
