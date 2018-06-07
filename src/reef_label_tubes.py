import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from loader import DataLoader
from program_synthesis.heuristic_generator import HeuristicGenerator
from program_synthesis.synthesizer import Synthesizer
from program_synthesis.verifier import Verifier

def weak_label(train_primitive_matrix):
	# Load data
	dl = DataLoader()
	# train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
	# train_ground, val_ground, test_ground, mode, frameNums = dl.load_data(mode = 'auto', numFramesToLoad = 1000)
	_, val_primitive_matrix, _, \
	_, val_ground, _, mode, frameNums = dl.load_data(mode = 'auto', numFramesToLoad = 1000, need_split = False)
	# Pass into reef
	return_matrix = reef_label(train_primitive_matrix, val_primitive_matrix, val_ground, None)
	return(return_matrix)

def reef_label(train_primitive_matrix, val_primitive_matrix, val_ground, train_ground, frame_nums):
	validation_accuracy = []
	training_accuracy = []
	validation_coverage = []
	training_coverage = []
	training_marginals = []
	idx = None

	hg = HeuristicGenerator(train_primitive_matrix, val_primitive_matrix, val_ground, train_ground, b=0.5)
	for i in range(3,26):
		if (i-2)%5 == 0:
			print("Running iteration: ", str(i-2))
						
		#Repeat synthesize-prune-verify at each iterations
		if i == 3:
			hg.run_synthesizer(max_cardinality=1, idx=idx, keep=3, model='dt')
		else:
			hg.run_synthesizer(max_cardinality=1, idx=idx, keep=1, model='dt')
		hg.run_verifier()
												
		#Save evaluation metrics
		va,ta, vc, tc = hg.evaluate()
		validation_accuracy.append(va)
		training_accuracy.append(ta)
		training_marginals.append(hg.vf.train_marginals)
		validation_coverage.append(vc)
		training_coverage.append(tc)

		#Find low confidence datapoints in the labeled set
		hg.find_feedback()
		idx = hg.feedback_idx
				
		#Stop the iterative process when no low confidence labels
		if idx == []:
			break
	return_matrix = np.vstack((frame_nums, training_marginals[-1]))
	return_matrix = dict(return_matrix.swapaxes(0,1))
	return(return_matrix)
