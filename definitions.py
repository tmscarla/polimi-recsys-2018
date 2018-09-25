import os
import pickle
import json

# Define project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Save and load any kind of object
def save_obj(obj, name, path=''):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, path=''):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)
