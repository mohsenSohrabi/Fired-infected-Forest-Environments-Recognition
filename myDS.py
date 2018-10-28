import h5py
import os
import numpy as np 
def read_data():
    
    hdf5_path=os.getcwd()+"/Dataset/DS.hdf5"
    with h5py.File(hdf5_path,'r')as hdf5_file:
        X_train=hdf5_file["train_img"][()]
        y_train=hdf5_file["train_labels"][()]
    
        X_val=hdf5_file["val_img"][()]
        y_val=hdf5_file["val_labels"][()]
    
        X_test=hdf5_file["test_img"][()]
        y_test=hdf5_file["test_labels"][()]
    
    
    X_train=np.array(X_train)
    y_train=np.array(y_train)

    X_val=np.array(X_val)
    y_val=np.array(y_val)

    X_test=np.array(X_test)
    y_test=np.array(y_test)

    return X_train,y_train,X_val,y_val,X_test,y_test
