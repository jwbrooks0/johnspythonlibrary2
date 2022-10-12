import reservoirpy as _rpy

_rpy.verbosity(0)  # no need to be too verbose here
_rpy.set_seed(42)  # make everyhting reproducible !

from reservoirpy.nodes import Reservoir as _Reservoir
from reservoirpy.nodes import Ridge as _Ridge
import matplotlib.pyplot as _plt


def RC(s1, s2, N=100, lr=0.5, sr=0.9, ridge=1e-7):
	"""
	

	Examples
	--------
	
	Example 1 ::
		
		# use a sine wave to reproduce the same wave staggered by 1 time step
		import numpy as np
		X = np.sin(np.linspace(0, 6*np.pi, 101)).reshape(-1, 1)
		RC(X[:100], X[1:])
	
	Example 2 ::
		
		# use a sine wave to reproduce a cosine wave at twice the frequency
		import numpy as np
		N = 10000
		s1 = np.sin(np.linspace(0, 30*np.pi, N))
		s2 = np.cos(np.linspace(0, 60*np.pi, N))
		RC(s1, s2)
		
	Example 3 ::
		
		# use a sine wave to reproduce a cosine wave at twice the frequency
		import numpy as np
		N = 10000
		s1 = np.sin(np.linspace(0, 30*np.pi, N))
		s2 = 2 * np.cos(np.linspace(0, 60*np.pi, N)) + (np.random.rand(N)-0.5) * 0.5
		RC(s1, s2, lr=0.1, sr=0.1)
		
	"""
	## reshape data according to reservoirpy's requirements
	s1 = s1.reshape(-1, 1)
	s2 = s2.reshape(-1, 1)
	
	## create reservoir and readout
	reservoir = _Reservoir(N, lr=lr, sr=sr)
	readout = _Ridge(ridge=ridge)
	
	## split data in training and test sections
	M = len(s1)
	M_half = M // 2
	s1_training_data = s1[:M_half]
	s1_testing_data = s1[M_half : 2 * M_half]
	s2_training_data = s2[:M_half]
	s2_testing_data = s2[M_half : 2 * M_half]
	
	## train RC
	train_states = reservoir.run(s1_training_data, reset=True)
	readout = readout.fit(train_states, s2_training_data, warmup=10)

	## test RC
	test_states = reservoir.run(s1_testing_data)
	s2_predicted_data = readout.run(test_states)
	
	## plot results
	fig, ax = _plt.subplots(2, 2)
	ax[0,0].plot(s1_training_data)
	ax[0,1].plot(s1_testing_data)
	ax[1,0].plot(s2_training_data)
	ax[1,1].plot(s2_testing_data, label="Actual")
	ax[1,1].plot(s2_predicted_data, ls='--', label="Predicted")
	ax[0,0].set_title('Training data\n(First half)', fontsize=10)
	ax[0,0].set_ylabel('Signal 1\nReference data', fontsize=10)
	ax[0,1].set_title('Test data\n(Second half)', fontsize=10)
	ax[1,0].set_ylabel('Signal 2\nTarget data', fontsize=10)
	ax[1,1].legend()
	fig.set_tight_layout(True)
	