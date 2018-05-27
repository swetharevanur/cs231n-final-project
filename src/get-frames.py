# Reads in labels from jackson-town-square-2017-12-14.csv

import pandas as pd

def getVehicleFrames(fname):
	data = pd.read_csv(fname, header = 0)
	uniqueFrames = []
	uniqueFrames.append(data.frame.unique())
	return uniqueFrames