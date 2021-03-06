# Code Authors: Pan Ji,     University of Adelaide,         pan.ji@adelaide.edu.au
#               Tong Zhang, Australian National University, tong.zhang@anu.edu.au
# Copyright Reserved!
import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
from munkres import Munkres
import time

class ConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_constant1 = 1.0, re_constant2 = 1.0, batch_size = 200, reg = None, \
                denoise = False, model_path = None, restore_path = None, \
                logs_path = 'DSC/pretrain-model-EYaleB/logs'):
        self.n_input = n_input
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0
        
        #input required to be fed
        self.x = tf.placeholder(tf.float32, [None, n_input[0], n_input[1], 1], name='input_image')
        self.learning_rate = tf.placeholder(tf.float32, [], name='learing_rate')
        # initialize weights
        weights = self._initialize_weights()
        # encoder
        if denoise == False:
            x_input = self.x
            latent, shape = self.encoder(x_input, weights)
        else:
            x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
                                               mean = 0,
                                               stddev = 0.2,
                                               dtype=tf.float32))
            latent, shape = self.encoder(x_input, weights)
        # self representive
        z = tf.reshape(latent, [batch_size, -1], name='z')  
        self.z = z
        Coef = weights['Coef']
        self.Coef = Coef
        z_c = tf.matmul(Coef, z, name='convert_z')
        latent_c = tf.reshape(z_c, tf.shape(latent), name='represent_reconstruct')
     
        # decoder
        self.x_r = self.decoder(latent_c, weights, shape)
        # l_2 reconstruction loss 
        self.reconst_cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x_r, self.x), 2.0), name='2_reconst_cost')
        tf.summary.scalar("recons_loss", self.reconst_cost)
                
        self.reg_losses = tf.reduce_sum(tf.pow(self.Coef,2.0), name='regulation_losses')
        tf.summary.scalar("reg_loss", reg_constant1 * self.reg_losses )
        
        self.selfexpress_losses = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(z_c, z), 2.0), name='2_self_represent_losses')
        tf.summary.scalar("selfexpress_loss", re_constant2 * self.selfexpress_losses )
        
        self.loss = self.reconst_cost + reg_constant1 * self.reg_losses + re_constant2 * self.selfexpress_losses  
        self.merged_summary_op = tf.summary.merge_all()

        #GradientDescentOptimizer and AdamOptimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, name='optimizer').minimize(self.loss) 
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)        
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))])              
        
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['enc_w0'] = tf.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg['reg'])
        all_weights['enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype = tf.float32))

        all_weights['enc_w1'] = tf.get_variable("enc_w1", shape=[self.kernel_size[1], self.kernel_size[1], self.n_hidden[0],self.n_hidden[1]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg['reg'])
        all_weights['enc_b1'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype = tf.float32))

        all_weights['enc_w2'] = tf.get_variable("enc_w2", shape=[self.kernel_size[2], self.kernel_size[2], self.n_hidden[1],self.n_hidden[2]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg['reg'])
        all_weights['enc_b2'] = tf.Variable(tf.zeros([self.n_hidden[2]], dtype = tf.float32))        
        
        all_weights['Coef'] = tf.Variable(1.0e-4 * tf.ones([self.batch_size, self.batch_size], tf.float32), name = 'Coef')
        
        all_weights['dec_w0'] = tf.get_variable("dec_w0", shape=[self.kernel_size[2], self.kernel_size[2], self.n_hidden[1],self.n_hidden[2]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg['reg'])
        all_weights['dec_b0'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype = tf.float32))

        all_weights['dec_w1'] = tf.get_variable("dec_w1", shape=[self.kernel_size[1], self.kernel_size[1], self.n_hidden[0],self.n_hidden[1]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg['reg'])
        all_weights['dec_b1'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype = tf.float32))

        all_weights['dec_w2'] = tf.get_variable("dec_w2", shape=[self.kernel_size[0], self.kernel_size[0],1, self.n_hidden[0]],
            initializer=layers.xavier_initializer_conv2d(),regularizer = self.reg['reg'])
        all_weights['dec_b2'] = tf.Variable(tf.zeros([1], dtype = tf.float32))
        
        return all_weights
        
    # Building the encoder
    def encoder(self, x, weights):
        shapes = []
        # Encoder Hidden layer with relu activation #1
        shapes.append(x.get_shape().as_list())
        layer1 = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[1,2,2,1],padding='SAME'), weights['enc_b0'], name='encoder_layer1')
        layer1 = tf.nn.relu(layer1)
        shapes.append(layer1.get_shape().as_list())
        layer2 = tf.nn.bias_add(tf.nn.conv2d(layer1, weights['enc_w1'], strides=[1,2,2,1],padding='SAME'),weights['enc_b1'], name='encoder_layer2')
        layer2 = tf.nn.relu(layer2)
        shapes.append(layer2.get_shape().as_list())
        layer3 = tf.nn.bias_add(tf.nn.conv2d(layer2, weights['enc_w2'], strides=[1,2,2,1],padding='SAME'),weights['enc_b2'], name='encoder_layer3')
        layer3 = tf.nn.relu(layer3)
        return  layer3, shapes
    
    # Building the decoder
    def decoder(self, z, weights, shapes):
        # Encoder Hidden layer with relu activation #1
        shape_de1 = shapes[2]
        layer1 = tf.add(tf.nn.conv2d_transpose(z, weights['dec_w0'], tf.stack([tf.shape(self.x)[0], shape_de1[1], shape_de1[2], shape_de1[3]]),\
         strides=[1,2,2,1], padding='SAME'), weights['dec_b0'], name='decoder_layer1')
        layer1 = tf.nn.relu(layer1)
        shape_de2 = shapes[1]
        layer2 = tf.add(tf.nn.conv2d_transpose(layer1, weights['dec_w1'], tf.stack([tf.shape(self.x)[0], shape_de2[1], shape_de2[2], shape_de2[3]]),\
         strides=[1,2,2,1], padding='SAME'), weights['dec_b1'], name='decoder_layer2')
        layer2 = tf.nn.relu(layer2)
        shape_de3= shapes[0]
        layer3 = tf.add(tf.nn.conv2d_transpose(layer2, weights['dec_w2'], tf.stack([tf.shape(self.x)[0], shape_de3[1], shape_de3[2], shape_de3[3]]),\
         strides=[1,2,2,1], padding='SAME'), weights['dec_b2'], name='decoder_layer3')
        layer3 = tf.nn.relu(layer3)
        return layer3
    
    # train once and get cost and Coef
    def partial_fit(self, X, lr):
        cost, summary, _, Coef = self.sess.run((self.reconst_cost, self.merged_summary_op, self.optimizer, self.Coef), \
                                                feed_dict = {self.x: X, self.learning_rate: lr})
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return cost, Coef
    
    #initialize all variables and layers
    def initlization(self):
        self.sess.run(self.init)
    
    # get reconstruct x, e.x. input
    def reconstruct(self, X):
        return self.sess.run(self.x_r, feed_dict = {self.x:X})
    
    # get z
    def transform(self, X):
        return self.sess.run(self.z, feed_dict = {self.x:X})
    
    # save weights except Coef
    def save_model(self):
        save_path = self.saver.save(self.sess, self.model_path)
        print ("model saved in file: %s" % save_path)

    # restore model from saved model file
    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print ("model restored")

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

# load face images and labels
def get_data(addr):
    data = sio.loadmat(addr)
    # img = [2016, 64, 38]
    # img = [48*42, 64, 38]
    img = data['Y']
    I = []
    Label = []
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            temp = np.reshape(img[:, i, j],[42, 48])
            Label.append(j)
            I.append(temp)
    I = np.array(I)
    Label = np.array(Label[:])
    Img = np.transpose(I, [0, 2, 1])
    Img = np.expand_dims(Img[:], 3)
    return Img, Label
    
if __name__ == '__main__':
    
    Img, Label = get_data('DSC/Data/YaleBCrop025.mat')
    
    # face image clustering
    n_input = [48, 42]
    kernel_size = [5, 3, 3]
    n_hidden = [10, 20, 30]
    net_strc = {}
    net_strc['input'] = n_input
    net_strc['kernel'] = kernel_size
    net_strc['hidden'] = n_hidden
    
    all_subjects = [10, 15, 20, 25, 30, 35, 38]
    
    avg = []
    med = []

    model_path = 'DSC/pretrain-model-EYaleB/model-102030-48x42-yaleb.ckpt' 
    restore_path = 'DSC/pretrain-model-EYaleB/model-102030-48x42-yaleb.ckpt' 
    logs_path = 'DSC/pretrain-model-EYaleB/logs' 
    # train loop
    for i in range(len(all_subjects)):
        num_class = all_subjects[i]
        batch_size = num_class * 64
        reg1 = 1.0
        reg2 = 1.0 * 10 ** (num_class / 10.0 - 3.0)           
        tf.reset_default_graph()
        CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, reg_constant1=reg1, re_constant2=reg2, \
                     kernel_size=kernel_size, batch_size=batch_size, model_path=model_path, \
                         restore_path=restore_path, logs_path=logs_path)
    
        avg_i, med_i = test_face(Img, Label, CAE, num_class)
        avg.append(avg_i)
        med.append(med_i)

    # performance index printing
    for i in range(len(all_subjects)):
        num_class = all_subjects[i]
        print ('%d subjects:' % num_class)
        print ('Mean: %.4f%%' % (avg[i]*100), 'Median: %.4f%%' % (med[i]*100))  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
