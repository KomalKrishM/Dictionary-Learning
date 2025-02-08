import torch.nn.functional as F
import numpy as np
import torch
from argparse import ArgumentParser
import copy
import scipy.io
from hyperopt import tpe, hp, fmin, STATUS_OK,Trials

def firm_thr(input_, theta_, gamma):
    T = theta_*gamma
    return (1/(gamma-1))*F.relu(-input_-T)-(gamma/(gamma-1))*F.relu(-input_-theta_)+(gamma/(gamma-1))*F.relu(input_-theta_)-(1/(gamma-1))*F.relu(input_-T)

def soft_thr(input_, theta_):
    return F.relu(input_-theta_)-F.relu(-input_-theta_)

parser = ArgumentParser(description='L1')
parser.add_argument('--sparsity', type=int, default=10, help='% of non-zeros in the sparse vector')
parser.add_argument('--input_SNR', type=int, default=30, help='Input SNR')
parser.add_argument('--lam', type=float, default=0.1, help='lambda')
parser.add_argument('--m', type=int, default=64, help='measurement_vector_length')
parser.add_argument('--n', type=int, default=100, help='sparse_vector_length')
parser.add_argument('--k', type=int, default=100, help='sparse_A_examples')
parser.add_argument('--iterations', type=int, default=300, help='num_of_FISTA_iterations')
parser.add_argument('--no_of_examples', type=int, default=5000, help='num_of_examples')
args = parser.parse_args()

def pes(x,x_est):
  d   = []
  for i in range(x.shape[1]):
    M = max(np.sum(x[:,i] != 0),np.sum(x_est[:,i] != 0))
    d.append((M - np.sum((x[:,i]!=0) * (x_est[:,i]!=0)))/M)
  return np.mean(d),np.std(d)

def pes_dict(x,x_est):
  d   = []
  M = max(np.sum(x != 0),np.sum(x_est != 0))
  d.append((M - np.sum((x!=0) * (x_est!=0)))/M)
  return np.mean(d),np.std(d)

def forward(y, thr_, d, B, W, numIter):
    x   = []
    x.append(d)
    # RSNR_iter = []
    # inp_pow = np.linalg.norm(x_test)
    t   = 1
    x_k = d.clone()

    for iter in range(numIter):
      xold = x_k.clone()
      # x_k = firm_thr(np.matmul(B,y) + np.matmul(W,d), thr_, gam)
      x_k  = soft_thr(np.matmul(B,y) + np.matmul(W,d), thr_)
      t0   = t
      t    = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.0
      d    = x_k + ((t0 - 1.0) / t) * (x_k - xold)
      x.append(d)
      # x.append(x_k)
      # err  = np.linalg.norm(x_k - x_test)
      # RSNR = 20*np.log10(err/inp_pow)
      # RSNR_iter.append(RSNR)

    return x#, RSNR_iter

def sparse_coding(y_test, x_test, D, numIter):

    # D     = phi@A
    lam   = 0.05 #params['lambda'] #args.lam
    scale = (np.linalg.norm(D, 2) ** 2 )*1.001
    thr   = lam/scale
    B     = (np.transpose (D) / scale).astype (np.float32)
    W     = np.eye (x_test.shape[0], dtype=np.float32) - np.matmul (B, D)
    d     = torch.zeros([x_test.shape[0], x_test.shape[1]])

    x_est = forward(y_test, thr, d, B, W, numIter)
    x_est = x_est[-1].numpy()
    
    RSNR_test = []
    N_test    = y_test.shape[1]
    for i in range(N_test):
        err  = np.linalg.norm(x_est[:, i] - x_test[:, i])
        RSNR = 20*np.log10(np.linalg.norm(x_test[:, i])/err)
        RSNR_test.append(RSNR)
    PES_mean, PES_std = pes(x_test, x_est)

    print('SPARSE CODING STAGE1')
    print('Input SNR is', snr)
    print('Testing: my l1 avg SNR is ', np.mean(RSNR_test))
    print('Testing: my l1 std deviation in SNR is ', np.std(RSNR_test))
    # print('Testing: my l1 peak SNR is ', np.max(RSNR_test))

    print('Testing: my l1 avg PES is ', PES_mean)
    print('Testing: my l1 std deviation in PES is ', PES_std)
    print('-'*40)

    return x_est

def sparse_dict_coding(y, A_test, D, numIter):
    lam   = 2 #params['lambda'] #args.lam
    scale = (np.linalg.norm(D, 2) ** 2 )*1.001
    thr   = lam/scale
    B     = (np.transpose (D) / scale).astype (np.float32)
    W     = np.eye (D.shape[1], dtype=np.float32) - np.matmul (B, D)
    # d     = torch.zeros([D.shape[1], y.shape[1]])
    d     = torch.zeros([D.shape[1],1])

    A_est = forward(y, thr, d, B, W, numIter)
    A_est = A_est[-1].numpy()

    # RSNR_test = []
    # # N_test    = y.shape[1]
    # # for i in range(N_test):
    # #     err  = np.linalg.norm(A_est[:, i] - A_test[:, i])
    # #     RSNR = 20*np.log10(np.linalg.norm(A_test[:, i])/err)
    # #     RSNR_test.append(RSNR)
    # err  = np.linalg.norm(A_est - A_test)
    # RSNR = 20*np.log10(np.linalg.norm(A_test)/err)
    # RSNR_test.append(RSNR)

    # PES_mean, PES_std = pes_dict(A_test, A_est)

    # print('Dictionary Update')
    # print('Input SNR is', snr)
    # print('Testing: my l1 avg SNR is ', np.mean(RSNR_test))
    # print('Testing: my l1 std deviation in SNR is ', np.std(RSNR_test))
    # print('Testing: my l1 peak SNR is ', np.max(RSNR_test))

    # print('Testing: my l1 avg PES is ', PES_mean)
    # print('Testing: my l1 std deviation in PES is ', PES_std)
    # print('-'*40)

    return A_est


######## DICTIONARY LEARNING STAGE #########
def dict_learning(y_test, x_test, Phi, A, numIter, rng):
    main_loops = 30
    # A_est = np.eye (Phi.shape[1], dtype=np.float32)
    A_init = A + rng.normal(0,0.1,(A.shape[0],A.shape[1]))
    A_init = A_init/np.linalg.norm(Phi@A_init,axis=0)
    D = Phi@A_init
    for k in range(main_loops):
        rsnr_test = []
        x_k = sparse_coding(y_test, x_test, D, numIter)
        E = y_test - Phi@A_init@x_k
        for j in range(A_init.shape[1]):
            Ej = E + Phi@A_init[:,j][:,np.newaxis]@x_k[j,:][np.newaxis,:]
            A_init[:,j][:,np.newaxis] = sparse_dict_coding(Ej@x_k[j,:][np.newaxis,:].T, A[:,j][:,np.newaxis], Phi, numIter)
            A_init[:,j][:,np.newaxis] = A_init[:,j][:,np.newaxis]/np.linalg.norm(Phi@A_init[:,j][:,np.newaxis],axis=0)
            E = Ej - Phi@A_init[:,j][:,np.newaxis]@x_k[j,:][np.newaxis,:]
            err  = np.linalg.norm(A_init[:,j] - A[:,j])
            RSNR = 20*np.log10(np.linalg.norm(A[:,j])/err)
            rsnr_test.append(RSNR)

        PES_mean, PES_std = pes(A, A_init)
        print('Dictionary Update Completed')
        print('Input SNR is', snr)
        print('Testing: my l1 avg SNR is ', np.mean(rsnr_test))
        print('Testing: my l1 std deviation in SNR is ', np.std(rsnr_test))
        # print('Testing: my l1 peak SNR is ', np.max(rsnr_test))

        print('Testing: my l1 avg PES is ', PES_mean)
        print('Testing: my l1 std deviation in PES is ', PES_std)
        print('-'*40)
        D = Phi@A_init
        
    return A_init, x_k

def data_gen(D, snr, x, rng):
  
  y             = D@x 
  y             = torch.tensor(y,dtype=torch.float32)
  # noise_Std_Dev = y.std(dim=0)*(10**(-snr/20))
  # noise         = rng.normal(0,noise_Std_Dev,y.shape).astype(np.float32)
  # y             = y + noise

  return y

# def objective(params):

rng       = np.random.RandomState(2021)
m         = args.m
n         = args.n    
k         = args.k
N         = args.no_of_examples
snr       = args.input_SNR
p_s       = args.sparsity/100
numIter   = args.iterations

x_test    = rng.normal(0,1,(n,N)).astype(np.float32)*rng.binomial(1,p_s,(n,N))
A         = rng.normal(0,1,(k,n)).astype(np.float32)*rng.binomial(1,p_s,(k,n))
phi       = scipy.io.loadmat('PHI.mat')['x'].astype(np.float32) #+ rng.normal(0,1,(m,k)).astype(np.float32)
# print(phi)
# phi       = rng.normal(0,1,(m,k)).astype(np.float32)
# print(phi)
A         = A/np.linalg.norm(phi@A,axis=0)
# D         = rng.normal(0,1,(m,n)).astype(np.float32)
# D         = D/np.linalg.norm(D,axis=0)


# print((phi).T@(phi))
# D = scipy.linalg.orth(phi@A)
# print(D)
# print(D.T@D)


y_test    = data_gen(phi@A, snr, x_test, rng)

A_est, x_est = dict_learning(y_test, x_test, phi, A, numIter, rng)


# ###### DENOISING STAGE ########
# x_est        = sparse_dict_coding(y_test, Phi@A, numIter)

# denoised_img = Phi@A@x_est

#     print('RSNR is:', np.mean(RSNR_test))
#     print('PES is:', PES_mean)

#     return -1*np.mean(RSNR_test) + PES_mean

# space = {'lambda': hp.uniform('lambda', 0.01,10)}

# best  = fmin(fn=objective,space = space, algo=tpe.suggest, max_evals=100)

# print(best)



# Write the content into a log text file
# f = open("l1/logs/L1 measurement {} sparsity {}.txt".format(m, args.sparsity), "w")
# f.write('-'*40)
# f.write('\navg SNR is {}, std SNR is {}\n'.format( round(np.mean(RSNR_test), 3), round(np.std(RSNR_test), 3)))
# f.write('avg PES is {}, std PES is {}\n'.format( round(PES_mean, 3), round(PES_std, 3)))
# f.close()


# print('index of the maximum reconstruction is:', RSNR_test.index(max(RSNR_test)))
# from matplotlib import pyplot as plt
# plt.figure()
# markerline1, stemlines, baseline = plt.stem(range(500),x_test[:,111],markerfmt ='D',label='ORIGINAL')
# markerline, stemlines, baseline = plt.stem(range(500),x_est[:,111],markerfmt ='*',label='L1')
# markerline1.set_markerfacecolor('none')
# plt.xlabel('SPARSE COEFFICIENT INDEX')
# plt.ylabel('AMPLITUDE')
# # plt.title('Estimated vs Original i/p SNR {}dB sparsity {}'.format(input_SNR, sparsity))
# plt.legend()
# plt.savefig('l1/figures/L1 with {} sparse and measurement {}.png'.format(args.sparsity, m))
# plt.show()

# def cleanDictionary(data,dictionary,coefMatrix):
#     T1=3
#     T2=0.99
#     error=np.sum(np.square(data-np.dot(dictionary,coefMatrix)),0)
#     G=np.dot(np.transpose(dictionary,[1,0]),dictionary)
#     G=G-np.diag(np.diag(G))
#     for j in range(dictionary.shape[1]):
#         if np.max(G[j,:])>T2 or np.sum(np.abs(coefMatrix[j,:])>1e-7)<=T1:
#             index=np.argmax(error)
#             error[index]=0
#             dictionary[:,j]=data[:,index]/np.sqrt(np.sum(data[:,index]*data[:,index]))
#             G = np.dot(np.transpose(dictionary, [1, 0]), dictionary)
#             G = G - np.diag(np.diag(G))
#     return dictionary