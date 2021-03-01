
# load training data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

er=np.load('dataset_example/dataset_er_symbolic.npy','r')
sf=np.load('dataset_example/dataset_sf_symbolic.npy','r')
sw=np.load('dataset_example/dataset_sw_symbolic.npy','r')
##################### 3 classes #####################################
er_label=np.zeros((np.shape(er)[0],3))
er_label[:,0]=1
sf_label=np.zeros((np.shape(sf)[0],3))
sf_label[:,1]=1
sw_label=np.zeros((np.shape(er)[0],3))
sw_label[:,2]=1

er_train, er_test, yer_train, yer_test = train_test_split(er, er_label, test_size=0.20, shuffle=True)
sf_train, sf_test, ysf_train, ysf_test = train_test_split(sf, sf_label, test_size=0.20, shuffle=True)
sw_train, sw_test, ysw_train, ysw_test = train_test_split(sw, sw_label, test_size=0.20, shuffle=True)

x_train=np.concatenate((er_train,sf_train,sw_train),axis=0)
x_test=np.concatenate((er_test,sf_test,sw_test),axis=0)
y_train=np.concatenate((yer_train, ysf_train, ysw_train),axis=0)
y_test=np.concatenate((yer_test,ysf_test,ysw_test),axis=0)

x_train=np.reshape(x_train,(np.shape(x_train)[0],np.shape(x_train)[1],np.shape(x_train)[2],1))
x_test=np.reshape(x_test,(np.shape(x_test)[0],np.shape(x_test)[1],np.shape(x_test)[2],1))

print(np.shape(x_train),np.shape(x_test),np.shape(y_train),np.shape(y_test))
print('train_Class1:',np.sum(y_train[:,0]),'train_Class2:',np.sum(y_train[:,1]),'train_Class3:',np.sum(y_train[:,2]))
print('test_Class1:',np.sum(y_test[:,0]),'test_Class2:',np.sum(y_test[:,1]),'test_Class3:',np.sum(y_test[:,2]))

# # ##################### 2 classes #####################################
# er_label=np.zeros((np.shape(er)[0],3))
# er_label[:,0]=1
# sf_label=np.zeros((np.shape(sf)[0],3))
# sf_label[:,1]=1

# er_train, er_test, yer_train, yer_test = train_test_split(er, er_label, test_size=0.20, shuffle=True)
# sf_train, sf_test, ysf_train, ysf_test = train_test_split(sf, sf_label, test_size=0.20, shuffle=True)

# x_train=np.concatenate((er_train,sf_train),axis=0)
# x_test=np.concatenate((er_test,sf_test),axis=0)
# y_train=np.concatenate((yer_train,ysf_train),axis=0)
# y_test=np.concatenate((yer_test,ysf_test),axis=0)

# x_train=np.reshape(x_train,(np.shape(x_train)[0],np.shape(x_train)[1],np.shape(x_train)[2],1))
# x_test=np.reshape(x_test,(np.shape(x_test)[0],np.shape(x_test)[1],np.shape(x_test)[2],1))

# print(np.shape(x_train),np.shape(x_test),np.shape(y_train),np.shape(y_test))
# print('train_Class1:',np.sum(y_train[:,0]),'train_Class2:',np.sum(y_train[:,1]))
# print('test_Class1:',np.sum(y_test[:,0]),'test_Class2:',np.sum(y_test[:,1]))



#Conv2D
import tensorflow as tf
# from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization, LeakyReLU, PReLU, Conv1D, Conv2D, GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, MaxPooling1D, Flatten, Activation
from tensorflow.keras.layers import GRU, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
import os
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split


MODEL_SAVE_FOLDER_PATH = './model_save_folder_path/'
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
    os.mkdir(MODEL_SAVE_FOLDER_PATH)

length_list=[1,2,4,6,8,10,15,20,50,100,200] 
node_list=[1,2,4,6,8,10]

total_error_result=np.zeros((len(length_list),len(node_list)))
total_loss_result=np.zeros((len(length_list),len(node_list)))

for j in range(len(node_list)):
    for k in range(len(length_list)):

        model_path = MODEL_SAVE_FOLDER_PATH + 'model_er_sf_length{a}_node{b}_sym.hdf5'.format(a=length_list[k],b=node_list[j])

        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                        verbose=0, save_best_only=True)

        cb_early_stopping = EarlyStopping(monitor='val_loss', patience=60)

        model = Sequential()

        model.add(Conv2D(80,
                        (3,2),
                        input_shape=(length_list[k], node_list[j],1),
                        padding='same',
                        activation='linear',
                        strides=1,
                        data_format="channels_last")) #2048, 4
        model.add(Activation('selu'))
        model.add(Conv2D(80,
                        (3,2),
                        padding='same',
                        activation='linear',
                        strides=1)) #2048, 4 
        model.add(Activation('selu'))
        model.add(Conv2D(40,
                        (3,2),
                        padding='same',
                        activation='linear',
                        strides=1)) #2048, 4
        model.add(Activation('selu'))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(80, activation='selu'))
        # model.add(Dropout(0.3))
        model.add(Dense(80, activation='selu'))
        # model.add(Dropout(0.3))
        model.add(Dense(3, activation='softmax'))

        adam = tf.keras.optimizers.Adam(lr=0.0005)
        model.compile(loss='categorical_crossentropy', optimizer=adam,
                        metrics=['accuracy'])

        print('###########################Training start(length{a}, node{b})#############################'.format(a=length_list[k],b=node_list[j]))
        model.summary()

        history=model.fit(x_train[:,0:length_list[k],0:node_list[j],:], y_train[:,:], validation_data=(x_test[:,0:length_list[k],0:node_list[j],:],y_test),epochs=500, batch_size=10, verbose=1 ,shuffle = True, callbacks=[cb_early_stopping,cb_checkpoint])

        model = load_model(model_path) #load best model

        # training_acc=model.evaluate(x_train[0:sample_list[l],0:length_list[k],0:node_list[j],:], y_train[0:sample_list[l],:])[1]
        # training_loss=model.evaluate(x_train[0:sample_list[l],0:length_list[k],0:node_list[j],:], y_train[0:sample_list[l],:])[0]
        test_acc=model.evaluate(x_test[:,0:length_list[k],0:node_list[j],:],y_test[0:sample_list[l],:])[1]
        test_loss=model.evaluate(x_test[:,0:length_list[k],0:node_list[j],:],y_test[0:sample_list[l],:])[0]
        # total_acc=model.evaluate(input_data[:,0:length_list[k],0:node_list[j],:], output_data)[1]
        # total_loss=model.evaluate(input_data[:,0:length_list[k],0:node_list[j],:], output_data)[0]

        total_error_result[k,j]=1-test_acc
        total_loss_result[k,j]=test_loss



        np.savetxt('error_sweep.txt', total_error_result)           
        np.savetxt('loss_sweep.txt', total_loss_result) 
np.savetxt('error_sweep.txt', total_error_result)           
np.savetxt('loss_sweep.txt', total_loss_result)


import numpy as np
import matplotlib.pyplot as plt

sweep_result=np.loadtxt('error_sweep.txt')

yticks=length_list.copy()
xticks=node_list.copy()


fig, ax = plt.subplots(figsize=(10,10))
im = ax.imshow(sweep_result[:,:],aspect='auto')

ax.set_xticks(np.arange(len(xticks)))
ax.set_yticks(np.arange(len(yticks)))
# ax.set_xticklabels(xticks,size=30,rotation = 90)
ax.set_xticklabels(xticks,size=30)
ax.set_yticklabels(yticks,size=30)

ax.set_xlabel('n',size=40)
ax.set_ylabel('t', size=40)
clb=fig.colorbar(im, ax=ax, orientation='vertical')
clb.mappable.set_clim(0,0.5)
clb.ax.tick_params(labelsize=30)
############################################################################
# plt.title(r'$N=100$ $\langle k \rangle =20$ $\lambda = 0.03$', fontsize=20)
# plt.title(r'$\rangle$', fontsize=20)
###########################################################################
clb.ax.set_title('Error rate', fontsize=30,pad=20)
for (j,i),label in np.ndenumerate(sweep_result):
    plt.text(i,j,round(label,2),ha='center',va='center',fontsize = 20, color='w')
