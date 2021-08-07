
import pandas as pd
import numpy as np
from math import ceil
import time
import glob,os
import matplotlib.pyplot as plt
import pickle
import pdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
import random




class PrepareData:
    def __init__(self,dtype):
        
        if dtype[0] == 'L':
            self.inputportion = int(dtype[-1])/10
        
        if dtype[0] == 'T':
            self.Tinput = int(dtype[-3])
            self.Toutput = int(dtype[-1])
            
        self.dtype = dtype
        
    ### Type 1 Data: Create sequential data based on distance:
    # Input: sequence data in the first n% of the road width + aux
    #Output: coordiantes in the rest (1-n)% of the road width    
    def LengthSeq(self,Xscaled,Yscaled,distscaled,
                 usero1traj,usero2traj,usero3traj,
                 useraux):
        X, Xoutput, Y, Youtput, Aux,\
        o1, o1output, o2, o2output,\
        o3, o3output, dist, distoutput =\
        list(), list(), list(), list(),\
        list(), list(), list(), list(),\
        list(), list(), list(), list(), list()
        for i in range(len(Xscaled)):
            # find the end of this pattern
            xdistancecovered = Xscaled[i][0] - Xscaled[i][-1] #total distance covered by user
            if xdistancecovered>0:
                xinput = xdistancecovered * self.inputportion     #distance that is used as input to the model                   
                breakpoint = Xscaled[i][0] - xinput       #x is descending
                n_steps = len([x for x in Xscaled[i] if x>breakpoint])

                # gather input and output parts of the pattern
                seq_x, seq_Xout = Xscaled[i][0:n_steps], Xscaled[i][n_steps:]
                seq_y, seq_Yout = Yscaled[i][0:n_steps], Yscaled[i][n_steps:]
                seq_o1, seq_o1out = usero1traj[i][0:n_steps], usero1traj[i][n_steps:]
                seq_o2, seq_o2out= usero2traj[i][0:n_steps], usero2traj[i][n_steps:]
                seq_o3, seq_o3out= usero3traj[i][0:n_steps], usero3traj[i][n_steps:]
                seq_dist, seq_distout = distscaled[i][0:n_steps], distscaled[i][n_steps:] 

                X.append(seq_x)
                Xoutput.append(seq_Xout)

                Y.append(seq_y)
                Youtput.append(seq_Yout)

                o1.append(seq_o1)
                o1output.append(seq_o1out)

                o2.append(seq_o2)
                o2output.append(seq_o2out)

                o3.append(seq_o3)
                o3output.append(seq_o3out)

                dist.append(seq_dist)
                distoutput.append(seq_distout)

                Aux.append(useraux[i])
        
        return np.array(X),np.array(Xoutput),np.array(Y),\
            np.array(Youtput),np.array(o1),\
            np.array(o1output),np.array(o2),\
            np.array(o2output),np.array(o3),\
            np.array(o3output),np.array(dist),\
            np.array(distoutput), np.array(Aux)  

    ### Type 2 Data: Create sequential data based on time:
     #Input: sequence data in the first t1 seconds of cross + aux <br>
    #Output: coordiantes in the next t2 seconds of cross
    def TimeSeq(self,Xscaled,Yscaled,distscaled,
                 usero1traj,usero2traj,usero3traj,
                 useraux):
        X, Xoutput, Y, Youtput, Aux,\
        o1, o1output, o2, o2output,\
        o3, o3output, dist, distoutput =\
        list(), list(), list(), list(),\
        list(), list(), list(), list(),\
        list(), list(), list(), list(),list()\

        TinpInd =  int(self.Tinput/0.1)
        ToutInd =  int(self.Toutput/0.1)
        for i in range(len(Xscaled)):
            start = 0
            inp = start + TinpInd
            out = inp + ToutInd

            while out < (len(Xscaled[i])+1):
                # gather input and output parts of the pattern
                seq_x, seq_Xout = Xscaled[i][start:inp], Xscaled[i][inp:out]
                seq_y, seq_Yout = Yscaled[i][start:inp], Yscaled[i][inp:out]
                seq_o1, seq_o1out = usero1traj[i][start:inp], usero1traj[i][inp:out]
                seq_o2, seq_o2out= usero2traj[i][start:inp], usero2traj[i][inp:out]
                seq_o3, seq_o3out= usero3traj[i][start:inp], usero3traj[i][inp:out]
                seq_dist, seq_distout = distscaled[i][start:inp], distscaled[i][inp:out] 

                X.append(seq_x)
                Xoutput.append(seq_Xout)

                Y.append(seq_y)
                Youtput.append(seq_Yout)

                o1.append(seq_o1)
                o1output.append(seq_o1out)

                o2.append(seq_o2)
                o2output.append(seq_o2out)

                o3.append(seq_o3)
                o3output.append(seq_o3out)

                dist.append(seq_dist)
                distoutput.append(seq_distout)

                Aux.append(useraux[i])

                start += 1
                inp = start + TinpInd
                out = inp + ToutInd
        return np.array(X),np.array(Xoutput),np.array(Y),\
                np.array(Youtput),np.array(o1),\
                np.array(o1output),np.array(o2),\
                np.array(o2output),np.array(o3),\
                np.array(o3output),np.array(dist),\
                np.array(distoutput), np.array(Aux)

    def run(self,Xscaled,Yscaled,distscaled,
                 usero1traj,usero2traj,usero3traj,
                 useraux):
        if self.dtype[0] == 'L':
            f = self.LengthSeq
        else:
            if self.dtype[0] == 'T':
                f = self.TimeSeq
            else:
                raise TypeError(f"Sequence Type {self.dtype} is not accepted.")
        X, Xoutput,\
        Y, Youtput,\
        o1, o1output,\
        o2, o2output,\
        o3, o3output,\
        dist,distoutput, Aux = f(Xscaled,Yscaled,distscaled,
             usero1traj,usero2traj,usero3traj,
             useraux)
        
        if self.dtype[0] == 'L':
            # padding sequences to have same length by adding negative values:
            # For Lengthwise only, timewise are already equal
            X = pad_sequences(X, dtype='float32',value=-0.01,maxlen=150)   
            Y = pad_sequences(Y,dtype='float32',value=-0.01,maxlen=150)
            o1 = pad_sequences(o1,dtype='float32',value=-0.01,maxlen=150)
            o2 = pad_sequences(o2,dtype='float32',value=-0.01,maxlen=150)
            o3 = pad_sequences(o3,dtype='float32',value=-0.01,maxlen=150)
            dist = pad_sequences(dist,dtype='float32',value=-0.01,maxlen=150)
            Xoutput = pad_sequences(Xoutput,padding='post', dtype='float32',value=-0.01,maxlen=150)   
            Youtput = pad_sequences(Youtput, padding='post',dtype='float32',value=-0.01,maxlen=150)
            o1output = pad_sequences(o1output,padding='post',dtype='float32',value=-0.01,maxlen=150)
            o2output = pad_sequences(o2output,padding='post',dtype='float32',value=-0.01,maxlen=150)
            o3output = pad_sequences(o3output,padding='post',dtype='float32',value=-0.01,maxlen=150)
            distoutput = pad_sequences(distoutput,padding='post',dtype='float32',value=-0.01,maxlen=150)

        inputseq = []
        outputseq = []
        for i in range(len(X)):
            mrg_input=np.transpose(np.vstack((X[i],Y[i],o1[i],o2[i],o3[i],dist[i])))

            mrg_output=np.transpose(np.vstack((Xoutput[i],Youtput[i],o1output[i],\
                                               o2output[i],o3output[i],distoutput[i])))

            inputseq.append(mrg_input)
            outputseq.append(mrg_output)

        inputseq=np.array(inputseq)
        outputseq=np.array(outputseq)

        maxlen = max(inputseq.shape[1],Aux.shape[1])

        inputseq = pad_sequences(inputseq, maxlen = maxlen,
                                      dtype='float32',value=-0.01)

        inputaux= pad_sequences(Aux, maxlen = maxlen,
                                      dtype='float32',value=-0.01)

        inp = np.dstack((inputseq,inputaux))

        return inp,outputseq





