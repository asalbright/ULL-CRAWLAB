import pychrono as chrono
import numpy as np

my_vect1 = chrono.ChVectorD()
my_vect1.x=5
my_vect1.y=2
my_vect1.z=3
my_vect2 = chrono.ChVectorD(3,4,5)
my_vect4 = my_vect1*10 + my_vect2
my_len = my_vect4.Length()
print ('vector length =', my_len)
my_quat = chrono.ChQuaternionD(1,2,3,4)
my_qconjugate = ~my_quat
print ('quat. conjugate  =', my_qconjugate)
print ('quat. dot product=', my_qconjugate ^ my_quat)
print ('quat. product=',     my_qconjugate % my_quat)
my_vec = chrono.ChVectorD(1,2,3)
my_vec_rot = my_quat.Rotate(my_vec)
mlist = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]
ma = chrono.ChMatrixDynamicD() 
ma.SetMatr(mlist)   # Create a Matrix from a list. Size is adjusted automatically.
npmat = np.asarray(ma.GetMatr()) # Create a 2D npy array from the list extracted from ChMatrixDynamic
w, v = LA.eig(npmat)  # get eigenvalues and eigenvectors using numpy
mb = chrono.ChMatrixDynamicD(4,4)
prod = v * npmat  
mb.SetMatr(v.tolist())    