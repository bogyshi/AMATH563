import numpy as np
import struct
import cvxpy

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

#np.fromfile('/home/bdvr/Documents/GitHub/AMATH563/hw1/data/t10k-images-idx3-ubyte',)
hrm = read_idx('/home/bdvr/Documents/GitHub/AMATH563/hw1/data/t10k-images-idx3-ubyte')
hrm2 = read_idx('/home/bdvr/Documents/GitHub/AMATH563/hw1/data/t10k-labels-idx1-ubyte')

print(hrm[0])
print(len(hrm[0][1]))
print(len(hrm[0]))
print(hrm[1,26])
print(hrm2)
print(hrm2[0])
#images are 28x28, so each entry in the images are 28 'columns' per row, with each column holding another 28 values.
#labels in hrm 2 seem to make sense
