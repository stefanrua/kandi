import pickle
import pandas
import matplotlib.pyplot as plt

datapath = 'data/'

def read(filename):
    fname = f"{datapath}{filename}"
    with open(fname, 'rb') as f:
        #return pickle.load(f)
        return pandas.read_pickle(f)

def write(df, filename):
    fname = f"{datapath}{filename}"
    with open(fname, 'wb') as f:
        pickle.dump(df, f)
        print('Wrote', fname)

def savefig(fname):
    plt.savefig(f"{fname}.pdf")
    plt.savefig(f"{fname}.png", dpi=300)
    print('Wrote', fname)
