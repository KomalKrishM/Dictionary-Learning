import torch 
import tensorflow as tf
from argparse import ArgumentParser
import torch.nn.functional as F
import numpy as np
from hyperopt import tpe, hp, fmin, STATUS_OK,Trials


parser = ArgumentParser(description='Generalized_lp')
parser.add_argument('--sparsity', type=int, default=15, help='% of non-zeros in the sparse vector')
parser.add_argument('--input_SNR', type=int, default=20, help='Input SNR')
parser.add_argument('--lam', type=float, default=0.1, help='lambda')
parser.add_argument('--m', type=int, default=250, help='measurement_vector_length')
parser.add_argument('--n', type=int, default=500, help='sparse_vector_length')
parser.add_argument('--p_val_in_lp', type=float, default=0.7, help='p_value_in_lp_penalty')
parser.add_argument('--iterations', type=int, default=500, help='num_of_FISTA_iterations')
parser.add_argument('--no_of_examples', type=int, default=10000, help='num_of_examples')
args = parser.parse_args()


# def soft_thr(input_, theta_):
#     return F.relu(input_-theta_)-F.relu(-input_-theta_)

def soft_thr(input_, theta_):
    return tf.nn.relu(input_-theta_)-tf.nn.relu(-input_-theta_)

def gst(y,lam,p,J):
  x_0   = tf.abs(y)
  for j in range(J):
    x_k = tf.abs(y) - lam*p*(x_0)**(p-1)
    x_0 = x_k
  return tf.sign(y)*x_k

def GST(y,lam,p,J):
  thr     = (2*lam*(1-p))**(1/(2-p)) + lam*p*(2*lam*(1-p))**((p-1)/(2-p))
  a       = y*tf.cast(tf.abs(y)>thr,dtype=tf.float32)
  k       = tf.where(a)
  non_z_t = tf.gather_nd(y,k)
  l       = tf.gather_nd(lam,k)
  x       = gst(non_z_t,l,p,J)
  b       = tf.scatter_nd(k,x,a.shape)
  return b

def pes(x,x_est):
  d   = []
  for i in range(x.shape[1]):
    M = max(np.sum(x[:,i] != 0),np.sum(x_est[:,i] != 0))
    d.append((M - np.sum((x[:,i]!=0) * (x_est[:,i]!=0)))/M)
  return np.mean(d),np.std(d)

def forward(y, x_test, thr_, gama, d, B, W, numIter, p, J, tau, eps):
    x           = []
    x.append(d)
    # RSNR_iter = []
    # inp_pow   = np.linalg.norm(x_test)
    t           = 1
    x_k         = d.clone()
    weights     = tf.ones(d.shape)*0.03

    for iter in range(numIter):
      xold    = tf.identity(x_k)
      x_k     = GST(np.matmul(B,y) + np.matmul(W,d), weights*thr_, p, J)
      
      t0      = t
      t       = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.0
      d       = x_k + ((t0 - 1.0) / t) * (x_k - xold)
      # weights = tau/(np.abs(d)**p + eps)
      weights = tf.maximum(weights - (tf.abs(d)**p)*(1/gama),0.0)

      x.append(d)

      # err   = np.linalg.norm(x_k - x_test)
      # RSNR  = 20*np.log10(err/inp_pow)
      # RSNR_iter.append(RSNR)

    return x#, RSNR_iter

def data_gen(y, snr, rng):
   
  y             = torch.tensor(y,dtype=torch.float32)
  noise_Std_Dev = y.std(dim=0)*(10**(-snr/20))
  noise         = rng.normal(0,noise_Std_Dev,y.shape).astype(np.float32)
  y             = y + noise

  return y

# def objective(params):
rng       = np.random.RandomState(2021)
m         = args.m
n         = args.n    
N         = args.no_of_examples
snr       = args.input_SNR
p_s       = args.sparsity/100
numIter   = args.iterations
p         = args.p_val_in_lp

x         = rng.normal(0,1,(n,N)).astype(np.float32)*rng.binomial(1,p_s,(n,N))
D         = rng.normal(0,1,(m,n)).astype(np.float32)
D         = D/np.linalg.norm(D,axis=0)
y         = D@x
y         = data_gen(y, snr, rng)
x         = torch.tensor(x, dtype=torch.float32)

y_test    = y[:,0:9000]
x_test    = x[:,0:9000]
y_train   = y[:,9000:]
x_train   = x[:,9000:]

lam   = 1 #params['lambda']   #0.234 #params['lambda']   #args.lam    
gama  = 20024.721 #params['gamma'] #96.984 #params['gamma']  
scale = (np.linalg.norm(D, 2) ** 2 )*1.001
thr   = lam/scale
    # print(lam)
    # print(gama)

J   = 2
# tau = params['TAU'] #1
# eps = params['EPS'] #1
tau = 0.07
eps = 2.74

# print('TAU is:', tau)
# print('EPS is:', eps)


    ########### TRAINING ############


B = (np.transpose (D) / scale).astype (np.float32)
W = np.eye (x_train.shape[0], dtype=np.float32) - np.matmul (B, D)
d = torch.zeros([x_train.shape[0], x_train.shape[1]])

x_est = forward(y_train, x_train, thr, gama, d, B, W, numIter, p, J, tau, eps)
x_est = x_est[-1].numpy()

x_train    = x_train.numpy()
RSNR_train = []
N_train    = y_train.shape[1]
for i in range(N_train):
    err  = np.linalg.norm(x_est[:, i] - x_train[:, i])
    RSNR = 20*np.log10(np.linalg.norm(x_train[:, i])/err)
    RSNR_train.append(RSNR)

PES_mean, PES_std = pes(x_train, x_est)

print('RSNR is:', np.mean(RSNR_train))
print('PES is:', PES_mean)

#     return -1*np.mean(RSNR_train) + PES_mean

# # space = {'lambda': hp.lognormal('lambda', 0,1),'gamma': 1+hp.lognormal('gamma', 0,2)}  ####
# space = {'gamma': 1+hp.lognormal('gamma', 0,1)}  ####

# best  = fmin(fn=objective,space = space, algo=tpe.suggest, max_evals=100)

# print(best)


# B = (np.transpose (D) / scale).astype(np.float32)
# W = torch.eye (x_test.shape[0]) - np.matmul (B, D)
# d = torch.zeros([x_test.shape[0], x_test.shape[1]])

# x_est = forward(y_test, x_test, thr, d, B, W, numIter, p, J, tau, eps)
# x_est = x_est[-1].numpy()
# # np.save('thr_weights_iteration_wise_for_RMCP_ISTA.npy', thr_weights)

# x_test = x_test.numpy()
# RSNR_test = []
# N_test    = y_test.shape[1]
# for i in range(N_test):
#     err   = np.linalg.norm(x_est[:, i] - x_test[:, i])
#     RSNR  = 20*np.log10(np.linalg.norm(x_test[:, i])/err)
#     RSNR_test.append(RSNR)

# PES_mean, PES_std = pes(x_test, x_est)

# print('RSNR is:', np.mean(RSNR_test))
# print('PES is:', PES_mean)

# return -1*np.mean(RSNR_test) + PES_mean

# space = {
#         'TAU': hp.uniform('TAU', 0.0,5.0), 'EPS': hp.uniform('EPS', 0.0,5.0)}             #'p_val': hp.uniform('p_val', 0.01,1), , 'EPS': hp.uniform('EPS', 0.0001,5)

# best  = fmin(fn=objective,space = space, algo=tpe.suggest, max_evals=100)

# print(best)


# print('Input SNR is', snr)
# print('Testing: my gen_lp avg SNR is ', np.mean(RSNR_test))
# print('Testing: my gen_lp std deviation in SNR is ', np.std(RSNR_test))
# print('Testing: my gen_lp peak SNR is ', np.max(RSNR_test))

# print('Testing: my gen_lp avg PES is ', PES_mean)
# print('Testing: my gen_lp std deviation in PES is ', PES_std)
# print('-'*40)

# from matplotlib import pyplot as plt
# plt.figure()
# plt.plot(range(200),R_SNR_iter,label = 'lp')
# plt.xlabel('Number of iterations')
# plt.ylabel('R-SNR in dB')
# plt.title('R_SNR vs Iterations')
# plt.legend()
# plt.savefig('R_SNR vs Iterations for lp.png')
# plt.show()


# # Write the content into a log text file
# f = open("Gen_Re_lp/logs1/Gen_LP1 measurement {} sparsity {}.txt".format(m, args.sparsity), "w")
# # f.write('Lamda is {}\n'.format( lam))
# # f.write('gamma is {}\n'.format( gam))
# # f.write('threshold is {}\n'.format( thr_weights))
# # f.write('\nInput SNR is {}\n'.format( input_SNR))
# # f.write('Sparsity is {}\n'.format( sparsity))
# f.write('-'*40)
# f.write('\navg SNR is {}, std SNR is {}\n'.format( round(np.mean(RSNR_test), 3), round(np.std(RSNR_test), 3)))
# f.write('avg PES is {}, std PES is {}\n'.format( round(PES_mean, 3), round(PES_std, 3)))
# f.close()

# print('index of the maximum reconstruction is:', RSNR_test.index(max(RSNR_test)))
# from matplotlib import pyplot as plt
# plt.figure()
# markerline1, stemlines, baseline = plt.stem(range(500),x_test[:,389],markerfmt ='D',label='ORIGINAL')
# markerline, stemlines, baseline = plt.stem(range(500),x_est[:,389],markerfmt ='*',label='GEN_LP')
# markerline1.set_markerfacecolor('none')
# plt.xlabel('SPARSE COEFFICIENT INDEX')
# plt.ylabel('AMPLITUDE')
# # plt.title('Estimated vs Original i/p SNR {}dB sparsity {}'.format(input_SNR, sparsity))
# plt.legend()
# plt.savefig('Gen_Re_lp/figures/Gen_LP with {} sparse and measurement {}.png'.format(args.sparsity, m))
# plt.show()



