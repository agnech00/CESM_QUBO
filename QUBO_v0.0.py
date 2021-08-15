#D-wave classes
import dimod
import neal
#scientific python classes
import numpy as np
from numpy import linalg as LA
#plot
import matplotlib.pyplot as plt

#input 
dim_matrix=10
num_of_bits=2
num_of_reads=50
beta=100
zero_energy=False
epsilon_tol=0.0001

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
def bin_matrix(A,prec):
    QA=prec**2*np.matmul(P.T,np.matmul(A,P))
    return QA
def bin_vector(v,prec):
    Qv=prec*np.matmul(v.T,P)
    return Qv

#anealer type
sampler = neal.SimulatedAnnealingSampler()#simulator

#Random array A. The matrix we want diagonalize!
I=np.eye(dim_matrix,dim_matrix)
np.random.seed(1234)
A=10*(np.random.rand(dim_matrix,dim_matrix)-0.5)
for i in range(dim_matrix):
    for j in range(dim_matrix):
        A[j][i]=A[i][j]
w,vw=LA.eigh(A)
vw0=vw[::,0]
print(w,vw0)
print (np.matmul(vw0,np.matmul(A,vw0)))

#array for output
out=np.empty([1000,3])
itr=np.empty([1000])

#Inital guess
lam=10**6
tmp_lam=np.trace(A)/dim_matrix
print("Starting lambda= ", tmp_lam)
j=0
while tmp_lam < lam:
    j=j+1
    lam=tmp_lam
    QA=bin_matrix(A-lam*I,1)
    #Transform our matrix in one managable by the D-wave
    A_bqm = dimod.BinaryQuadraticModel.from_qubo(QA,offset=0.0)
    #sampling
    sampleset = sampler.sample(A_bqm,num_reads=num_of_reads,beta_range=[0.1, 4.2],seed=3456)
    #sample database
    rsample=sampleset.record
    #print(rsample)
    energy0=10**6
    for i in range(num_of_reads):
        energy=rsample[i][1]
        print
        if(energy<energy0):
            energy0=energy
            indx=i
    print(energy0,indx)
    if(zero_energy):
        #here we select the state with energy 0
        x=np.array(rsample[indx][0])
    else:
        E0=rsample[indx][1]
        x=np.empty([dim_matrix*num_of_bits])
        for i in range(num_of_reads):
            x=x+np.exp(-beta*(rsample[i][1]-E0))*rsample[i][0]
        x=x/num_of_reads
    if(np.matmul(x.T,x)<10**(-16)):
        dvw=np.sqrt(np.sum((v-vw0)**2))
        out[j][0]=-np.log10(dvw)
        out[j][1]=-np.log10(abs(w[0]-lam))
        out[j][2]=-np.log10(1)
        itr[j]=j
        print("exit loop")
        break
    print(x)
    tmp_v=np.matmul(P,x)
    tmp_v_norm=np.sqrt(np.matmul(tmp_v.T,tmp_v))
    print(tmp_v)
    tmp_v=tmp_v/tmp_v_norm
    tmp_lam=np.matmul(tmp_v.T,np.matmul(A,tmp_v))  

    dvw=np.sqrt(np.sum((tmp_v-vw0)**2))
    out[j][0]=-np.log10(dvw)
    out[j][1]=-np.log10(abs(w[0]-tmp_lam))
    out[j][2]=-np.log10(1)
    itr[j]=j
    if(tmp_lam < lam):
        v=tmp_v
    print("Iteration= ",j," old value= ",lam," new value= ",tmp_lam,v)
print("End of the first part")
print("Eigenvalue = ",lam)
print("Eigenvector= ",v)

#starting iterative descendent
precision=0.1      
while precision > epsilon_tol:
#    print()
    j=j+1
    H=A-lam*I
    vH=2.0*np.matmul(v.T,H)
    QH=bin_matrix(H,precision)
    QvH=bin_vector(vH,precision)
    QA=QH
    for i in range(dim_matrix*num_of_bits):
        QA[i][i]=QA[i][i]+QvH[i]
    #Transform our matrix in one managable by the D-wave
    A_bqm = dimod.BinaryQuadraticModel.from_qubo(QA,offset=0.0)
    #sampling
    sampleset = sampler.sample(A_bqm,num_reads=num_of_reads,beta_range=[0.1, 4.2])
    #sample database
    rsample=sampleset.record
    #print(rsample)
    energy0=10**6
    for i in range(num_of_reads):
        energy=rsample[i][1]
#        print
        if(energy<energy0):
#            print(energy)
            energy0=energy
            indx=i
#    print("energy,index= ",energy0,indx)
    #if(zero_energy):
        #here we select the state with energy 0
    x=np.array(rsample[indx][0])
    #else:
    #    E0=rsample[indx][1]
    #    x=np.empty([dim_matrix*num_of_bits])
    #    for i in range(num_of_reads):
    #        x=x+np.exp(-beta*(rsample[i][1]-E0))*rsample[i][0]
    #    x=x/num_of_reads           
    delta=precision*np.matmul(P,x)
    delta=delta-np.matmul(delta.T,v)*v
    Hd=np.matmul(H,delta)
    #print("<v,d>=",np.matmul(v.T,delta))
    if(np.matmul(delta.T,delta)<10**(-16)):
        dvw=np.sqrt(np.sum((v-vw0)**2))
        out[j][0]=-np.log10(dvw)
        out[j][1]=-np.log10(abs(w[0]-lam))
        out[j][2]=-np.log10(precision)
        itr[j]=j
        #        print("brake",x,delta)
        precision=precision*0.1
        print("Iteration= ",j,"no precision improvement -->",precision)
        continue
    tmin=-np.matmul(v.T,Hd)/np.matmul(delta,Hd)
    tmin=max(tmin,1)
#    print("delta",tmin,delta)
    delta=delta*tmin
    tmp_v=v+delta    
    tmp_norm=np.sqrt(np.matmul(tmp_v.T,tmp_v))
    tmp_v=tmp_v/tmp_norm
#    print(v,tmp_v)
    tmp_lam=np.matmul(tmp_v.T,np.matmul(A,tmp_v))
    if(tmp_lam<lam):
        v=tmp_v
        lam=tmp_lam
        dvw=np.sqrt(np.sum((v-vw0)**2))
        out[j][0]=-np.log10(dvw)
        out[j][1]=-np.log10(abs(w[0]-lam))
        out[j][2]=-np.log10(precision)
        itr[j]=j
        print("Iteration= ",j," new value= ",tmp_lam,energy0)
    else:
        dvw=np.sqrt(np.sum((v-vw0)**2))
        out[j][0]=-np.log10(dvw)
        out[j][1]=-np.log10(abs(w[0]-lam))
        out[j][2]=-np.log10(precision)
        itr[j]=j
        precision=precision*0.1
        print("Iteration= ",j,"no precision improvement -->",lam,dvw,precision)

print("Final values")
print("Eigenvalue = ",lam,w[0])
print("Eigenvector= ",v)
print("Eigenvector= ",vw0)

#plotting the results
xdata=np.empty([j])
ydata0=np.empty([j])
ydata1=np.empty([j])
ydata2=np.empty([j])
for i in range(j):
    xdata[i]=i
    ydata0[i]=out[i][0]
    ydata1[i]=out[i][1]
    ydata2[i]=out[i][2]

plt.plot(xdata,ydata0,label='eigvec prec')
plt.plot(xdata,ydata1,label='eigval prec')
plt.plot(xdata,ydata2,label='precision')
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Precision")
plt.show()
