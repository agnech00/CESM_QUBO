#D-wave classes
import dimod
import neal
#scientific python classes
import numpy as np
from numpy import linalg as LA

#input 
dim_matrix=8
num_of_bits=2
num_of_reads=100
beta=100
zero_energy=False

#p and P matrices
p=np.empty([num_of_bits])
p[0]=-1
for b in range(2,num_of_bits+1):
    p[b-1]=2**(-b+1) 
P=np.empty([dim_matrix,dim_matrix*num_of_bits])
P[:][:]=0
for i in range(dim_matrix):
    P[i][i*num_of_bits:(i+1)*num_of_bits]=p[:]

print(P)    
#Writing the array in the binary basis
def bin_matrix(A):
    QA=np.matmul(P.T,np.matmul(A,P))
    return QA

#anealer type
sampler = neal.SimulatedAnnealingSampler()#simulator

#Random array A. The matrix we want diagonalize!
I=np.eye(dim_matrix,dim_matrix)
np.random.seed(12345)
A=10*(np.random.rand(dim_matrix,dim_matrix)-0.5)
for i in range(dim_matrix):
    for j in range(dim_matrix):
        A[j][i]=A[i][j]
w,v=LA.eigh(A)
print(w[0],v[0])

#First step of diagonalization
lam=10**6
tmp_lam=np.trace(A)/dim_matrix
print("Starting lambda= ", tmp_lam)
i=0
while tmp_lam < lam:
    i=i+1
    lam=tmp_lam
    print("Computing for =",lam)
    QA=bin_matrix(A-lam*I)
    #Transform our matrix in one managable by the D-wave
    A_bqm = dimod.BinaryQuadraticModel.from_qubo(QA,offset=0.0)
    #sampling
    sampleset = sampler.sample(A_bqm,num_reads=num_of_reads,beta_range=[0.1, 4.2])
    #sample database
    rsample=sampleset.record
    #print(rsample)
    if(zero_energy):
        #here we select the state with energy 0
        x=np.array(rsample[0][0])
    else:
        E0=rsample[0][1]
        x=np.empty([dim_matrix*num_of_bits])
        for i in range(num_of_reads):
            x=x+np.exp(-beta*(rsample[i][1]-E0))*rsample[i][0]
        x=x/num_of_reads           
    tmp_v=np.matmul(P,x)
    tmp_v_norm=np.sqrt(np.matmul(tmp_v.T,tmp_v))
    tmp_v=tmp_v/tmp_v_norm
    tmp_lam=np.matmul(tmp_v.T,np.matmul(A,tmp_v))  
    if(tmp_lam < lam):
        v=tmp_v
    print(i,lam,v,tmp_lam)

#for i in range (3): 
#    print(rsample[1][i])
#print(sampleset.info)

