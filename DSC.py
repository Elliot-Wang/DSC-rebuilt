import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import layers
from sklearn import cluster
from munkres import Munkres
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize


def get_data(data_set):
    if data_set == 'Yale':
        data = sio.loadmat('DSC/Data/YaleBCrop025.mat')
        img = data['Y']
        I = []
        Label = []
        for i in range(img.shape[2]):
            for j in range(img.shape[1]):
                temp = np.reshape(img[:, j, i],[42, 48])
                Label.append(i)
                I.append(temp)
        I = np.array(I)
        Label = np.array(Label[:])
        Img = np.transpose(I,[0, 2, 1])
        Img = np.expand_dims(Img[:], 3)
    elif data_set == 'ORL':
        data = sio.loadmat('DSC/Data/ORL_32x32.mat')
        Img = data['fea']
        Label = data['gnd'] 
    elif data_set == 'C20':
        data = sio.loadmat('DSC/Data/COIL20.mat')
        Img = data['fea']
        Label = data['gnd']
        Img = np.reshape(Img,(Img.shape[0],32,32,1))
    elif data_set == 'C100':
        data = sio.loadmat('DSC/Data/COIL100.mat')
        Img = data['fea']
        Label = data['gnd']
        Img = np.reshape(Img,(Img.shape[0],32,32,1))
    else:
        return None, None
    
    return Img, Label

def get_path(data_set):
    if data_set == 'Yale':
        path = {'model':'DSC/pretrain-model-EYaleB/model-102030-48x42-yaleb.ckpt', 
        'logs':'DSC/pretrain-model-EYaleB/logs'}
        path['res'] = path['model']
    elif data_set == 'ORL':
        path = {'model':'DSC/pretrain-model-ORL/model-335-32x32-orl.ckpt', 
        'logs':'DSC/pretrain-model-ORL/logs'}
        path['res'] = path['model']
    elif data_set == 'C20':
        path = {'model':'DSC/COIL20CodeModel/model.ckpt', 
        'logs':'DSC/COIL20CodeModel/logs'}
        path['res'] = path['model']
    elif data_set == 'C100':
        path = {'model':'DSC/pretrain-model-COIL100/model50.ckpt', 
        'logs':'DSC/pretrain-model-COIL100/logs'}
        path['res'] = path['model']
    else:
        path = None
    return path

class ConvAE(object):
	def __init__(self, net_struct, path, regular, denoise = False):
	    #n_hidden is a arrary contains the number of neurals on every layer
		self.n_input = net_struct['input']
		self.batch_size = net_struct['batch']
		self.n_hidden = net_struct['hidden']
		self.kernel_size = net_struct['kernel']

		self.reg = regular
		self.path = path
		self.iter = 0
		weights = self._initialize_weights()

		reg_1 = regular['reg_1']
		reg_2 = regular['reg_2']

		# model feed variable
		self.x = tf.placeholder(tf.float32, [None, self.n_input[0], self.n_input[1], 1], name='input_x')
		self.learning_rate = tf.placeholder(tf.float32, [], name='leaning_rate')
		# encoder layers
		if denoise == False:
			x_input = self.x
			latent, shape = self._build_enc(x_input, weights)
		else:
			x_input = tf.add(self.x, \
                    tf.random_normal(shape=tf.shape(self.x),
									mean = 0, stddev = 0.2, dtype=tf.float32))
			latent,shape = self._build_enc(x_input, weights)
        # self-representive module
		self.z_conv = tf.reshape(latent, [self.batch_size, -1], name='z_conv')
		self._set_self_ex_layer()
        # decoder layers
		latent_dec = tf.reshape(self.z_ssc, tf.shape(latent), name='z_self_re')
		self.x_r_ft = self._build_dec(latent_dec, weights, shape)
        # save parameters
		self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))]) 
        # loss
		self.self_ex_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.z_conv, self.z_ssc), 2))
		self.reconst_loss =  tf.reduce_sum(tf.pow(tf.subtract(self.x_r_ft, self.x), 2.0))
		self.z_reg_loss = tf.reduce_sum(tf.pow(self.Coef, 2))
		self.loss = self.self_ex_loss * reg_2 + reg_1 * self.z_reg_loss + self.reconst_loss
        # summary
		tf.summary.scalar("self_expressive_loss", self.self_ex_loss)
		tf.summary.scalar("reconstruct_loss", self.reconst_loss)
		tf.summary.scalar("coefficient_lose", self.z_reg_loss)
		self.get_summary = tf.summary.merge_all()
		self.summary_writer = tf.summary.FileWriter(path['logs'], graph=tf.get_default_graph())
        # 优化器
		self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
        # 初始化变量
		self.sess = tf.InteractiveSession()
		self.init = tf.global_variables_initializer()
		self._init_var()

	def _initialize_weights(self):
		Weights = dict()
		n_layers = len(self.n_hidden)
		
		Weights['enc_w0'] = tf.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
			initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
		Weights['enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype = tf.float32), name='enc_b0')
		
		for i in range(1, n_layers):
			name_wi = 'enc_w' + str(i)
			Weights[name_wi] = tf.get_variable(name_wi, shape=[self.kernel_size[i], self.kernel_size[i], self.n_hidden[i-1], \
						self.n_hidden[i]], initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
			name_bi = 'enc_b' + str(i)
			Weights[name_bi] = tf.Variable(tf.zeros([self.n_hidden[i]], dtype = tf.float32), name=name_bi)
		
		Weights['Coef'] = tf.Variable(1.0e-4 * tf.ones([self.batch_size, self.batch_size], tf.float32), name = 'Coef_matrix')

		for i in range(n_layers - 1):
			name_wi = 'dec_w' + str(i)
			Weights[name_wi] = tf.get_variable(name_wi, shape=[self.kernel_size[n_layers-1-i], self.kernel_size[n_layers-1-i], 
						self.n_hidden[n_layers-1-i],self.n_hidden[n_layers-1-i]], initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
			name_bi = 'dec_b' + str(i)
			Weights[name_bi] = tf.Variable(tf.zeros([self.n_hidden[n_layers-1-i]], dtype = tf.float32), name = name_bi)
			
		name_wi = 'dec_w' + str(n_layers - 1)
		Weights[name_wi] = tf.get_variable(name_wi, shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
			initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg)
		name_bi = 'dec_b' + str(n_layers - 1)
		Weights[name_bi] = tf.Variable(tf.zeros([1], dtype = tf.float32), name = name_bi)
		return Weights	
	

	# Building the encoder
	def _build_enc(self,x, weights):
		shapes = []
		shapes.append(x.shape.as_list())
		layer_i = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[1,2,2,1], padding='SAME'), weights['enc_b0'])
		layer_i = tf.nn.relu(layer_i)
		shapes.append(layer_i.shape.as_list())
		
		for i in range(1, len(self.n_hidden)):
			layer_i = tf.nn.bias_add(tf.nn.conv2d(layer_i, weights['enc_w' + str(i)], strides=[1,2,2,1],padding='SAME'),weights['enc_b' + str(i)])
			layer_i = tf.nn.relu(layer_i)
			shapes.append(layer_i.shape.as_list())
		
		layer_n = layer_i
		return  layer_n, shapes

	# Building the decoder
	def _build_dec(self, layer_i, weights, shapes):	
		for i in range(len(self.n_hidden)):
			shape_de = shapes[len(self.n_hidden) - i - 1] 
			layer_i = tf.add(tf.nn.conv2d_transpose(layer_i, weights['dec_w' + str(i)], tf.stack([tf.shape(self.x)[0],shape_de[1],shape_de[2],shape_de[3]]),\
					 strides=[1,2,2,1],padding='SAME'), weights['dec_b' + str(i)])
			layer_i = tf.nn.relu(layer_i)
		return layer_i

    # self representive
	def _set_self_ex_layer(self):
		self.Coef = tf.Variable(1.0e-4 * tf.ones([self.batch_size, self.batch_size],tf.float32), name = 'Coef')			
		self.z_ssc = tf.matmul(self.Coef, self.z_conv)
		return

    # train once and record loops, save summary
	def train_once(self, X, lr):
		C, l1_cost, l2_cost, summary, _ = self.sess.run((self.Coef, self.z_reg_loss, self.self_ex_loss, self.get_summary, self.optimizer), \
													feed_dict = {self.x: X, self.learning_rate: lr})
		self.summary_writer.add_summary(summary, self.iter)
		self.iter = self.iter + 1
		return C, l1_cost, l2_cost 
	
	def _init_var(self):
		self.sess.run(self.init)

	def get_z_conv(self, X):
		return self.sess.run(self.z_conv, feed_dict = {self.x:X})

	def save_model(self):
		save_path = self.saver.save(self.sess, self.path['model'])
		print ("model saved in file: %s" % save_path)

	def restore_model(self):
		self.saver.restore(self.sess, self.path['model'])
		print ("model restored")

# train with given class numbers
# the more class numbers, the lower accuracy
def test_face(Img, Label, CAE, num_class):       
    # why set alpha
    alpha = max(0.4 - (num_class-1)/10 * 0.1, 0.1)
    print('alhpa is ', alpha)

    acc_= []
    for i in range(39 - num_class): 
        face_10_subjs = np.array(Img[64 * i: 64 * (i + num_class), :])
        face_10_subjs = face_10_subjs.astype(float)
        
        label_10_subjs = np.array(Label[64 * i: 64 * (i + num_class)]) 
        # label value normolization
        label_10_subjs = label_10_subjs - label_10_subjs.min() + 1 
        # need squeeze ??
        label_10_subjs = np.squeeze(label_10_subjs)

        CAE.initlization()
        CAE.restore() # restore from pre-trained model    
        
        max_step =  50 + num_class*25 # 100+num_class*20
        display_step = max_step
        lr = 1.0e-3
        # fine-tune network
        for epoch in range(max_step):
            # train once
            cost, Coef = CAE.partial_fit(face_10_subjs, lr)
            # display cost and accuracy every certain times                  
            if epoch % display_step == 0:
                print( "epoch: %.1d" % epoch, "cost: %.8f" % (cost/float(CAE.batch_size)))
                # 可以研究这里的cost 和 accuracy的关系！
                Coef = thrC(Coef,alpha)
                y_x, _ = post_proC(Coef, label_10_subjs.max(), 10, 3.5)
                acc_x = 1 - err_rate(label_10_subjs, y_x)
                print ("experiment: %d" % i, "our accuracy: %.4f" % acc_x)
        acc_.append(acc_x)    
    
    acc_ = np.array(acc_)
    mean_acc = 1 - np.mean(acc_)
    median_acc = 1 - np.median(acc_)
    print("%d subjects:" % num_class)    
    print("Mean: %.4f%%" % (mean_acc * 100))
    print("Median: %.4f%%" % (median_acc * 100))
    print(acc_) 
    
    return mean_acc, median_acc

def best_map(L1, L2):
    #L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-1*G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2   

def thrC(C, rho):
    if rho < 1:
        Cp = np.zeros(C.shape)
        S = np.abs(np.sort(-1*np.abs(C), axis=0))
        Ind = np.argsort(-1*np.abs(C), axis=0)

        for i in range(C.shape[0]):
            cL1 = np.sum(S[:,i]).astype(float)
            csum = 0
            t = 0
            while(True):
                csum = csum + S[t,i]
                if csum > rho*cL1:
                    Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
                    break
                t = t + 1
    else:
        Cp = C

    return Cp

def build_aff(C):
    N = C.shape[0]
    Cabs = np.abs(C)
    ind = np.argsort(-Cabs,0)
    for i in range(N):
        Cabs[:,i]= Cabs[:,i] / (Cabs[ind[0,i],i] + 1e-6)
    Cksym = Cabs + Cabs.T
    return Cksym

def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5*(C + C.T)
    r = d*K + 1
    U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
    U = U[:,::-1]    
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis = 1)
    Z = U.dot(U.T)
    Z = Z * (Z>0)    
    L = np.abs(Z ** alpha) 
    L = L/L.max()   
    L = 0.5 * (L + L.T)    
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L

def err_rate(gt_s, s):
    c_x = best_map(gt_s,s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate 

def build_laplacian(C):
    C = 0.5 * (np.abs(C) + np.abs(C.T))
    W = np.sum(C,axis=0)         
    W = np.diag(1.0/W)
    L = W.dot(C)    
    return L