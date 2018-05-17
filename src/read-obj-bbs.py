# Reads in labels from jackson-town-square-2017-12-14.csv

import pandas as pd

fname = '../data/jackson-town-square-2017-12-14.csv'

data = pd.read_csv(fname, header = 0)

print data.shape
print data.frame.nunique()
print data.ind.nunique()
print data.head(5)