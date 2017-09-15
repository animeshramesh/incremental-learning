import pickle

def load(filename):
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def dump(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
