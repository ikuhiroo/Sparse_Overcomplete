import pickle
import bz2
PROTOCOL = pickle.HIGHEST_PROTOCOL
class ZpkObj:
    def __init__(self, obj):
        self.zpk_object = bz2.compress(pickle.dumps(obj, PROTOCOL), 9)

    def load(self):
        return pickle.loads(bz2.decompress(self.zpk_object))
