import numpy as np
import pandas as pd

def CreateNN(nn):
	"""Create Neural Network
	"""
	
	net = {} #neural network
	net['nn'] = nn #structure
	net['M'] = len(nn)-1 #number of layers
	net['layers'] = nn[1:] #structure without inputs
	net = w_Create(net) #initialize random weight vector
	net['w'] = net['w0'].copy() #weight vector used for calculation
	net['N'] = len(net['w0']) #number of weights
	return net
	
def w_Create(net):
	"""	Creates random weight vector of NN and defines sets needed for
		derivative calculation
	"""
	
	M = net['M'] #number of layers
	layers = net['layers'] #NN structure
	inputs = net['nn'][0] #number of inputs

	X = []	#set of input layers
	U = [] 	#set of output layers (output of layer is used for cost function calculation

	IW = {}	#input-weight matrices
	LW = {} #LW[m,l] connection weight matrix layer m

	L_b = {}# L_b[m]: set of layers with a backward connection to layer m
	L_f = {}# L_f[m]: set of layers with a forward connection to layer m

	CX_LW = {} #CX_LW[u]: set of all input layers, u has a connection to
	CU_LW = {} #CU_LW[x]: set of all output layers, x has a connection to
	b = {} #b[m]: bias vector of layer m

	'''Inputs'''
	I = {}	#set of inputs with a connection to layer 1
	I[1]=[1] 	#Inputs only connect to layer 1
	
	#input-weight matrix
	#random values [-0.5,0.5]
	IW[1, 1] = np.random.randn(layers[0], inputs)
	X.append(1) 	#first layer is input layer
	
	'''Internal Connection Weight Matrices'''
	for m in range(1,M+1):
		L_b[m] = [] #L_b[m]: Set of layers that have a backward connection to layer m
		L_f[m] = [] #L_f[m]: Set of layers that have a forward connection to layer m

		if m>1:
			l = m-1
			LW[m, l] = np.random.randn(layers[m-1], layers[l-1]) #connection weight matrix
			L_b[l].append(m) #layer m has backward connection to layer l
			L_f[m].append(l) #layer l has forward connection to layer m

		b[m] = np.random.randn(layers[m-1])#create bias vector for layer m
	
	if M not in U:
		U.append(M) # #add M to output layers if not yet done

	for u in U:
		CX_LW[u] = []
	for x in range(1,M+1):
		CU_LW[x] = []

	#Add to NN
	net['U'] = U
	net['X'] = X
	net['L_b'] = L_b
	net['L_f'] = L_f
	net['I']=I
	net['CX_LW'] = CX_LW
	net['CU_LW'] = CU_LW
	net['w0'] = Wb2w(net,IW,LW,b)
	return net
	
def Wb2w(net,IW,LW,b):
	"""	Converts Input Weight matrices IW, connection weight matrices LW
		and bias vectors b to weight vector w
	Returns:
		w: 		weight vector
	"""

	I = net['I'] #set of inputs with a connection to layer 1
	L_f = net['L_f'] # L_f[m]: set of layers with a forward connection to layer m
	M = net['M'] #number of layers of NN
	
	w = np.array([]) #empty weight vector

	for m in range(1,M+1):
		#input weights
		if m == 1:
			for i in I[m]:
				w = np.append(w,IW[m,i].flatten('F'))
		#internal connection weights
		for l in L_f[m]:
				w = np.append(w,LW[m,l].flatten('F'))
		#bias weights
		w = np.append(w,b[m])
	
	return w
	
def w2Wb(net):
	"""	Converts weight vector w to Input Weight matrices IW, connection weight matrices LW
		and bias vectors b

	Args:
		net: 	neural network (containing weight vector w)		
	Returns:
		IW		input-weight matrices
		LW 		LW[m,l] connection weight matrix layer m -> layer l
		b		b[m]: bias vector of layer m
	"""

	I = net['I'] #set of inputs with a connection to layer 1
	L_f = net['L_f'] # L_f[m]: set of layers with a forward connection to layer m
	M = net['M'] #number of layers of NN
	layers = net['layers'] #structure of the NN
	inputs = net['nn'][0] #number of inputs
	w_temp = net['w'].copy() #weight vector
		
	IW = {}	#input-weight matrices
	LW = {} #LW[m,l,d] connection weight matrix layer m -> layer l with delay d
	b = {} #b[m]: bias vector of layer m
	
	for m in range(1,M+1):
	
		#input weights
		if m==1:
			for i in I[m]:
				w_i = inputs*layers[m-1]
				vec = w_temp[0:w_i]
				w_temp = w_temp[w_i:]
				IW[m,i] = np.reshape(vec,(layers[m-1],int(len(vec)/layers[m-1])),order='F')
		
		#internal connection weights
		for l in L_f[m]:
			w_i = layers[l-1]*layers[m-1]
			vec = w_temp[0:w_i]
			w_temp = w_temp[w_i:]
			LW[m,l] = np.reshape(vec,(layers[m-1],int(len(vec)/layers[m-1])),order='F')
		
		#bias weights
		w_i = layers[m-1]
		b[m] =w_temp[0:w_i]
		w_temp = w_temp[w_i:]

	return IW,LW,b


def NNOut_(P,net,IW,LW,b,a={},q0=0):
	"""	Calculates NN Output for given Inputs P
		For internal use only
	
	Args:
		P:		NN Inputs
		net: 	neural network
		IW:		input-weight matrices
		LW:		LW[m,l] connection weight matrix layer m -> layer l
		b:		b[m]: bias vector of layer m
		a:		Layer Outputs of NN. for use of known historical data
		q0:		Use data starting from datapoint q0 P[q0:]
	Returns:
		Y_NN: 	Neural Network output for input P
		a:		Layer Outputs of NN
		n:		sum output of layers
	"""

	I = net['I'] #set of inputs with a connection to layer 1
	L_f = net['L_f'] # L_f[m]: set of layers with a forward connection to layer m
	M = net['M'] #number of layers of NN
	outputs = net['nn'][-1] #number of outputs
	
	n = {} #sum output of layers
	Q = P.shape[1] #number of input datapoints

	Y_NN = np.zeros((outputs, Q)) #NN Output
	
	for q in range(q0+1,Q+1):
	#for all datapoints
		a[q, 1] = 0
		for m in range(1,M+1):
		#for all layers m
			n[q, m] = 0 #sum output datapoint q, layer m

			if m==1:
				for i in I[m]:
					if q > 0:
						n[q, m] = n[q, m] + np.dot(IW[m, i], P[:, q-1])

			for l in L_f[m]:
					if q > 0:
						n[q, m] = n[q, m] + np.dot(LW[m, l], a[q, l])
			#bias
			n[q, m] = n[q, m] + b[m]
			
			#Calculate layer output
			if m == M:
				a[q, M] = n[q, M] #linear layer for output
			else:
				a[q, m] = np.tanh(n[q, m])
		Y_NN[:, q-1] = a[q, M]
	Y_NN = Y_NN[:, q0:]
	return Y_NN,n,a
	
def NNOut(P,net,P0=None,Y0=None):
	"""	Calculates NN Output for given Inputs P
		User Function

	Args:
		P:		NN Inputs
		net: 	neural network
		P0:		previous input Data
		Y0:		previous output Data
	Returns:
		Y_NN: 	Neural Network output for input P
	"""
	Y = np.zeros((net['layers'][-1], int(np.size(P)/net['nn'][0])))
	data, net = prepare_data(P,Y,net,P0=P0,Y0=Y0)
	IW,LW,b = w2Wb(net) #input-weight matrices,connection weight matrices, bias vectors
	Y_NN = NNOut_(data['P'],net,IW,LW,b,a=data['a'],q0=data['q0'])[0]
	
	#scale normalized Output
	Y_NN_scaled = Y_NN.copy()
	for y in range(np.shape(Y_NN)[0]):
		Y_NN_scaled[y] = Y_NN[y]*net['normY'][y]
	
	if np.shape(Y_NN_scaled)[0]==1:
		Y_NN_scaled = Y_NN_scaled[0]
	return Y_NN_scaled
	
	
def RTRL(net,data):
	"""	Implementation of the Real Time Recurrent Learning Algorithm based on:
		Williams, Ronald J.; Zipser, David: A Learning Algorithm for Continually Running
		Fully Recurrent Neural Networks. In: Neural Computation, Nummer 2, Vol. 1
		(1989), S. 270-280.
		
	Args:
		net:	neural network
		data: 	Training Data
	Returns:
		J: 		Jacobian Matrix. derivatives of e with respect to the weight vector w
		E:		Mean squared Error of the Neural Network compared to Training data
		e:		error vector: difference of NN Output and target data['Y']
	"""
	P = data['P']	#Training data Inputs
	Y = data['Y']	#Training data Outputs
	a = data['a']	#Layer Outputs
	q0 = data['q0']	#Use training data [q0:]
	
	I = net['I'] #set of inputs with a connection to layer 1
	L_f = net['L_f'] # L_f[m]: set of layers with a forward connection to layer m
	L_b = net['L_b'] # L_f[m]: set of layers with a forward connection to layer m
	M = net['M'] #number of layers of NN
	inputs = net['nn'][0] #number of inputs
	outputs = net['nn'][-1] #number of outputs
	layers = net['layers'] #structure of the NN

	U = net['U'] #set of input layers
	X = net['X'] #set of output layers (output of layer is used for cost function calculation

	CU_LW = net['CU_LW'] #CU_LW[x]: set of all output layers, x has a connection to		
	IW,LW,b = w2Wb(net) #input-weight matrices,connection weight matrices, bias vectors
	
	########################
	# 1. Calculate NN Output
	Y_NN,n,a = NNOut_(P,net,IW,LW,b,a=a,q0=q0)
	
	########################
	# 2. Calculate Cost function E
	Y_delta = Y - Y_NN #error matrix
	e = np.reshape(Y_delta,(1,np.size(Y_delta)),order='F')[0] #error vector
	E = np.dot(e,e.transpose()) #Cost function (mean squared error)
	
	#########################
	# 3. Backpropagation RTRL
	
	Q = P.shape[1] #number of input datapoints
	Q0 = Q-q0 #number of datapoints without "old data"
	
	#Definitions
	dAu_db = {}		#derivative of layer output a(u) with respect to bias vector b
	dAu_dIW = {}	#derivative of layer output a(u) with respect to input weights IW
	dAu_dLW = {}	#derivative of layer output a(u) with respect to connections weights LW
	dA_dw = {}		#derivative of layer outputs a with respect to weight vector w
	S = {}			#Sensitivity Matrix
	Cs = {}			#Cs[u]: Set of layers m with an existing sensitivity matrix S[q,u,m]
	CsX = {}		#CsX[u]: Set of input layers x with an existing sensitivity matrix
					#S[q,u,x]
					#Cs and CsX are generated during the Backpropagation
			
	#Initialize
	J = np.zeros((Q0*layers[-1],net['N']))	#Jacobian matrix
	for q in range(1,q0+1):
		for u in U:
			dAu_dLW[q,u] = np.zeros((layers[u-1],net['N']))
	
	###
	#Begin RTRL
	for q in range(q0+1,Q+1):
	
		#Initialize
		U_ = [] #set needed for calculating sensitivities
		for u in U:
			Cs[u] = []
			CsX[u] = []
			dA_dw[q,u] = 0
			
		#Calculate Sensitivity Matrices
		for m in range(M,1-1,-1):
		# decrement m in backpropagation order
		
			for u in U_:
				S[q,u,m] = 0 #Sensitivity Matrix layer u->m
				for l in L_b[m]:
					S[q,u,m] = S[q,u,m] \
						+ np.dot(np.dot(S[q,u,l],LW[l,m]),np.diag(1-(np.tanh(n[q,m]))**2))
				if m not in Cs[u]:
					Cs[u].append(m) #add m to set Cs[u]
					if m in X:
						CsX[u].append(m) #if m ind X, add to CsX[u]
			
			if m in U:
				if m == M:
					#output layer is linear, no transfer function
					S[q,m,m] = np.diag(np.ones(outputs)) #Sensitivity Matrix S[M,M]
				else:
					S[q,m,m] = np.diag(1-(np.tanh(n[q,m]))**2) #Sensitivity Matrix S[m,m]
				
				U_.append(m) #add m to U'
				Cs[m].append(m) #add m to Cs
				if m in X:
					CsX[m].append(m) #if m ind X, add to CsX[m]
		
		#Calculate derivatives
		for u in sorted(U):
			#static derivative calculation
			dAe_dw = np.empty((layers[u-1],0)) #static derivative vector: explicit derivative layer outputs with respect to weight vector
			for m in range(1,M+1):
				#Input weights
				if m==1:
					for i in I[m]:
							if ((q,u,m) not in S.keys()) or (0>=q):
							#if no sensivity matrix exists or d>=q: derivative is zero
								dAu_dIW[m,i] = \
									np.kron(P[:,q-1].transpose(),\
											np.zeros((layers[u-1],layers[m-1])))
							else:
								#derivative output layer u with respect to IW[m,i]
								dAu_dIW[m,i] = \
									np.kron(P[:,q-1].transpose(),S[q,u,m])
							dAe_dw = np.append(dAe_dw,dAu_dIW[m,i],1) #append to static derivative vector
	
				# #Connection weights
				for l in L_f[m]:
						if ((q,u,m) not in S.keys()) or (0>=q):
						#if no sensivity matrix exists or d>=q: derivative is zero
							dAu_dLW[m,l] = \
								np.kron(a[q,l].transpose(),\
										np.zeros((layers[u-1],layers[m-1])))
						else:
							dAu_dLW[m,l] = \
								np.kron(a[q,l].transpose(),S[q,u,m])
								#derivative output layer u with respect to LW[m,i]
						dAe_dw = np.append(dAe_dw,dAu_dLW[m,l],1) #append to static derivative vector
				
				#Bias weights
				if ((q,u,m) not in S.keys()):
					dAu_db[m] = np.zeros((layers[u-1],layers[m-1])) #derivative is zero
				else:
					dAu_db[m] = S[q,u,m] #derivative output layer u with respect to b[m]
				dAe_dw = np.append(dAe_dw,dAu_db[m],1) #append to static derivative vector
				
			#dynamic derivative calculation
			dAd_dw=0 #dynamic derivative, sum of all x
			for x in CsX[u]:
				sum_u_ = 0 #sum of all u_
				for u_ in CU_LW[x]:
					sum_d = 0 #sum of all d
					sum_u_ = sum_u_+sum_d
				if sum_u_ != 0:
					dAd_dw = dAd_dw + np.dot(S[q,u,x],sum_u_) #sum up dynamic derivative
					
			#static + dynamic derivative
			dA_dw[q,u] = dAe_dw + dAd_dw # total derivative output layer u with respect to w
			
		# Jacobian Matrix
		J[range(((q-q0)-1)*outputs,(q-q0)*outputs),:] = -dA_dw[q,M]
		
		# Delete entries older than q-max_delay in dA_dw
		if q > 0:
			new_dA_dw = dict(dA_dw)
			for key in dA_dw.keys():
				if key[0] == q:
					del new_dA_dw[key]
			dA_dw = new_dA_dw
		
		# Reset S
		S = {}
		
	return J,E,e

	
def train_LM(P,Y,net,k_max=100,E_stop=1e-10,dampfac=3.0,dampconst=10.0,\
			verbose = False,min_E_step=1e-09):
	"""	Implementation of the Levenberg-Marquardt-Algorithm (LM) based on:
		Levenberg, K.: A Method for the Solution of Certain Problems in Least Squares.
		Quarterly of Applied Mathematics, 2:164-168, 1944.
		and
		Marquardt, D.: An Algorithm for Least-Squares Estimation of Nonlinear Parameters.
		SIAM Journal, 11:431-441, 1963.
		
	Args:

		P:		NN Inputs
		Y:		NN Targets
		net: 	neural network
		k_max:	maxiumum number of iterations
		E_stop:	Termination Error, Training stops when the Error <= E_stop
		dampconst:	constant to adapt damping factor of LM
		dampfac:	damping factor of LM
		min_E_step: minimum step for error. When reached 5 times, training terminates.        
	Returns:
		net: 	trained Neural Network 
	"""
	#create data dict
	data,net = prepare_data(P,Y,net)
	
	#Calculate Jacobian, Error and error vector for first iteration
	J,E,e = RTRL(net,data)
	k = 0
	ErrorHistory=np.zeros(k_max+1) #Vektor for Error hostory
	ErrorHistory[k]=E
	if verbose:
		print('Iteration: ',k,'		Error: ',E,'	scale factor: ',dampfac)
	
	early=0

	while True:
	#run loop until either k_max or E_stop is reached

		JJ = np.dot(J.transpose(),J) #J.transp * J
		w = net['w'] #weight vector
		while True:
		#repeat until optimizing step is successful
			#gradient
			g = np.dot(J.transpose(),e)
			
			#calculate scaled inverse hessian
			try:
				G = np.linalg.inv(JJ+dampfac*np.eye(net['N'])) #scaled inverse hessian
			except np.linalg.LinAlgError:
				# Not invertible. Go small step in gradient direction
				w_delta = 1.0/1e10 * g
			else:
				# calculate weight modification
				w_delta = np.dot(-G,g)
			
			net['w'] = w + w_delta #new weight vector
			
			Enew = calc_error(net,data) #calculate new Error E			
			if Enew<E and abs(E-Enew)>=min_E_step:
			#Optimization Step successful!
				dampfac= dampfac/dampconst#adapt scale factor
				early=0 #reset the early stopping criterium
				break #go to next iteration
			else:
			#Optimization Step NOT successful!\
				dampfac = dampfac*dampconst#adapt scale factor
				if abs(E-Enew)<=min_E_step:
					early=early+1
					
					if verbose:
						print('E-Enew<=min_E_step Encountered!!')
						if early>=5.0:
							print('5 Times * E-Enew<=min_E_step Encountered!!')
					break                    
		
		#Calculate Jacobian, Error and error vector for next iteration
		J,E,e = RTRL(net,data)
		k = k+1
		ErrorHistory[k] = E
		if verbose:
			print('Iteration: ',k,'		Error: ',E,'	scale factor: ',dampfac)
	
		#Ceck if termination condition is fulfilled
		if k>=k_max:
			print('Maximum number of iterations reached')
			break
		elif E<=E_stop:
			print('Termination Error reached')
			break
		elif early>=5.0:
			print('Error decreased 5 times by minimum step. Force training exit.')
			break
        
	net['ErrorHistory'] = ErrorHistory[:k]
	return net
	
	
def calc_error(net,data):
	"""	Calculate Error for NN based on data
		
	Args:
		net:	neural network
		data: 	Training Data
	Returns:
		E:		Mean squared Error of the Neural Network compared to Training data
	"""
	P = data['P']	#Training data Inputs
	Y = data['Y']	#Training data Outputs
	a = data['a']	#Layer Outputs
	q0 = data['q0']	#Use training data [q0:]
	
	IW,LW,b = w2Wb(net) #input-weight matrices,connection weight matrices, bias vectors
	
	########################
	# 1. Calculate NN Output
	Y_NN,n,a = NNOut_(P,net,IW,LW,b,a=a,q0=q0)
	
	########################
	# 2. Calculate Cost function E
	Y_delta = Y - Y_NN #error matrix
	e = np.reshape(Y_delta,(1,np.size(Y_delta)),order='F')[0] #error vector
	E = np.dot(e,e.transpose()) #Cost function (mean squared error)
	
	return E
	
def prepare_data(P,Y,net,P0=None,Y0=None):
	"""	Prepare Input Data for the use for NN Training and check for errors
		
	Args:
		P:		neural network Inputs
		Y: 		neural network Targets
		net: 	neural network
		P0:		previous input Data
		Y0:		previous output Data
	Returns:
		data:	dict containing data for training or calculating putput
	"""	
	
	#Convert P and Y to 2D array, if 1D array is given
	if P.ndim==1:
		P = np.array([P])
	if Y.ndim==1:
		Y = np.array([Y])
		
	#Ceck if input and output data match structure of NN	
	if np.shape(P)[0] != net['nn'][0]:
		raise ValueError("Dimension of Input Data does not match number of inputs of the NN")
	if np.shape(Y)[0] != net['nn'][-1]:
		raise ValueError("Dimension of Output Data does not match number of outputs of the NN")
	if np.shape(P)[1] != np.shape(Y)[1]:
		raise ValueError("Input and output data must have same number of datapoints Q")
	
	#check if previous data is given
	if (P0 is not None) and (Y0 is not None):

		#Convert P and Y to 2D array, if 1D array is given
		if P0.ndim==1:
			P0 = np.array([P0])
		if Y0.ndim==1:
			Y0 = np.array([Y0])
			
		#Ceck if input and output data match structure of NN
		if np.shape(P0)[0] != net['nn'][0]:
			raise ValueError("Dimension of previous Input Data P0 does not match number of inputs of the NN")
		if np.shape(Y0)[0] != net['nn'][-1]:
			raise ValueError("Dimension of previous Output Data Y0 does not match number of outputs of the NN")
		if np.shape(P0)[1] != np.shape(Y0)[1]:
			raise ValueError("Previous Input and output data P0 and Y0 must have same number of datapoints Q0")

		q0 = np.shape(P0)[1]#number of previous Datapoints given 
		a = {} #initialise layer outputs
		for i in range(1,q0+1):
			for j in range(1,net['M']):
				a[i,j]=np.zeros(net['nn'][j]) #layer ouputs of hidden layers are unknown -> set to zero
			a[i,net['M']]=Y0[:,i-1]/net['normY'] #set layer ouputs of output layer

		#add previous inputs and outputs to input/output matrices
		P_ = np.concatenate([P0,P],axis=1)
		Y_ = np.concatenate([Y0,Y],axis=1)
	else:
		#keep inputs and outputs as they are and set q0 and a to default values
		P_ = P.copy()
		Y_ = Y.copy()
		q0=0
		a={}
	#normalize
	P_norm = P_.copy()
	Y_norm = Y_.copy()
	if 'normP' not in net.keys():
		normP = np.ones(np.shape(P_)[0])
		for p in range(np.shape(P_)[0]):
			normP[p] = np.max([np.max(np.abs(P_[p])),1.0])
			P_norm[p] = P_[p]/normP[p]
		normY = np.ones(np.shape(Y_)[0])
		for y in range(np.shape(Y_)[0]):
			normY[y] = np.max([np.max(np.abs(Y_[y])),1.0])
			Y_norm[y] = Y_[y]/normY[y]
		net['normP'] = normP
		net['normY'] = normY
	else:
		for p in range(np.shape(P_)[0]):
			P_norm[p] = P_[p]/net['normP'][p]
		normY = np.ones(np.shape(Y)[0])
		for y in range(np.shape(Y_)[0]):
			Y_norm[y] = Y_[y]/net['normY'][y]
		
	#Create data dict
	data = {}		
	data['P'] = P_norm
	data['Y'] = Y_norm
	data['a'] = a
	data['q0'] = q0
	
	return data,net

def saveNN(net,filename):
	"""	Save neural network object to file
		
	Args:
		net: 	neural network object
		filename:	path of csv file to save neural network
	
	"""	
	import csv

	#create csv write
	file = open(filename,"w")
	writer = csv.writer(file, lineterminator='\n')


	#write network structure nn
	writer.writerow(['nn'])
	writer.writerow(net['nn'])

	#write factor for input data normalization normP
	writer.writerow(['normP'])
	writer.writerow(net['normP'])
	
	#write factor for output data normalization normY
	writer.writerow(['normY'])
	writer.writerow(net['normY'])
	
	#write weight vector w
	writer.writerow(['w'])
	file.close()
	
	file = open(filename,"ab")
	np.savetxt(file,net['w'],delimiter=',',fmt='%.55f')
	
	#close file
	file.close()
	
	return
	
def loadNN(filename):
	"""	Load neural network object from file
		
	Args:
		filename:	path to csv file to save neural network
	Returns:
		net: 	neural network object
	"""	
	import csv

	#read csv
	data= list(csv.reader(open(filename,"r")))

	#read network structure nn
	nn = list(np.array(data[1],dtype=np.int))

	#read factor for input data normalization normP
	normP = np.array(data[3],dtype=np.float)
	
	#read factor for output data normalization normY
	normY = np.array(data[5],dtype=np.float)
	
	#read weight vector w
	w = pd.read_csv(filename,sep=',',skiprows=range(6))['w'].values
	
	#Create neural network and assign loaded weights and factors
	net = CreateNN(nn)
	net['normP'] = normP
	net['normY'] = normY
	net['w'] = w
	
	return net

def inputdata(path):
	"""Prepare Input Data
	"""
	df = pd.read_csv(path, sep=',')
	P = np.array([df['BDFE_cat'].values, df['BDFE_sub'].values, df['x_cat'].values, df['x_sub'].values, df['d'].values, df['Vbur_cat'].values, df['Vbur_sub'].values])
	Y = np.array(df['gibbs'].values)

	return P, Y

def trainNN(nn, P, Y):
	"""train Neural Network
	"""
	np.random.seed(64)
	net = CreateNN(nn)
	# Train NN with training data P=input and Y=target
	# Set maximum number of iterations k_max to 500
	# The Training will stop after 500 iterations or when the Error <=E_stop
	net = train_LM(P, Y, net, verbose=True, k_max=500, E_stop=0.25)

	return net

def R(name, Y, y):
	"""evaluate Neural Network
	"""

	mse = np.average((Y - y) ** 2)
	rmse = np.sqrt(mse)
	mae = np.sum(np.absolute(Y - y)) / len(Y)
	R2 = 1 - mse / np.var(Y)
	print(name, 'RMSE:', rmse, 'MAE:', mae, 'R^2:', R2, '\n', sep='\n')

	import matplotlib.pyplot as plt
	plt.scatter(x=Y, y=y, color='blue')
	plt.plot([0, 25], [0, 25], color='black', linewidth=0.25)
	plt.show()
	return


P, Y = inputdata('data/train.csv')
Ptest, Ytest = inputdata('data/test.csv')
Ptest2, Ytest2 = inputdata('selectivity.csv')
Ptest3, Ytest3 = inputdata('cumoexp.csv')

net = trainNN([7,6,6,6,1], P, Y)
# net = loadNN('mynetwork.csv')

saveNN(net,'mynetwork.csv')

# Calculate outputs of the trained NN for train and test data
y = NNOut(P, net)
R('training', Y, y)
# y = y.reshape(-1,1)
# print('y:', y)

ytest = NNOut(Ptest, net)
R('test', Ytest, ytest)
# ytest = ytest.reshape(-1,1)
# print('ytest:', ytest)

ytest2 = NNOut(Ptest2,net)
R('CH3O_selectivity', Ytest2, ytest2)
ytest2 = ytest2.reshape(-1,1)
print('CH3O_selectivity:', ytest2)

ytest3 = NNOut(Ptest3,net)
R('CumO_exp', Ytest3, ytest3)
# ytest3 = ytest3.reshape(-1,1)
# print('CumO_exp:', ytest3)
