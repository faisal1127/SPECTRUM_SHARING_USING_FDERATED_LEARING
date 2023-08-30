import flwr as fl
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers.legacy import SGD
from keras import backend as K
import matplotlib.pyplot as plt

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# Load and compile Keras model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(6,)),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(256, activation='sigmoid'),
    keras.layers.Dense(1, activation='sigmoid')
])
sgd = SGD(lr=0.001, decay=0.001, momentum=0.9, nesterov=True)

model.compile( loss = "binary_crossentropy", optimizer = sgd, metrics=['accuracy',f1_m,precision_m, recall_m])

# Load dataset
ds=pd.read_csv("C:/Users/skmuskaan/Downloads/datasetda.csv")
#ds.drop(ds[(ds['Pd'] >1) | (ds['Noise'] ==0)].index, inplace=True)

x=ds.iloc[1:70000,2:-1].values
y=ds.iloc[1:70000,-1].values

#binarize the labels
lb = LabelEncoder()
y = lb.fit_transform(y)

#split data into training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Normalize the data
ss_x=StandardScaler()
x_train=ss_x.fit_transform(x_train)
x_test=ss_x.transform(x_test)

x_plot=['ROUND-1','ROUND-2','ROUND-3','ROUND-4','ROUND-5']
y_plot_loss=[]
y_plot_val_loss=[]
y_plot_accuracy=[]
y_plot_val_accuracy=[]
y_plot_precision=[]
y_plot_val_precision=[]
y_plot_recall=[]
y_plot_val_recall=[]
y_plot_fscore=[]
y_plot_val_fscore=[]
y_plot_global_accuracy=[]
# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self,config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=0)
        hist = r.history
        print("Fit history : " ,hist)
        y_plot_loss.append(hist['loss'])
        y_plot_val_loss.append(hist['val_loss'])
        y_plot_accuracy.append(hist['accuracy'])
        y_plot_val_accuracy.append(hist['val_accuracy'])
        y_plot_precision.append(hist['precision_m'])
        y_plot_val_precision.append(hist['val_precision_m'])
        y_plot_recall.append(hist['recall_m'])
        y_plot_val_recall.append(hist['val_recall_m'])
        y_plot_fscore.append(hist['f1_m'])
        y_plot_val_fscore.append(hist['val_f1_m'])
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)
        y_plot_global_accuracy.append(accuracy)
        print("Eval accuracy : ", accuracy)
        
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:"+str(5012),
        client=FlowerClient(),
        grpc_max_message_length = 1024*1024*1024
)

fig, ax = plt.subplots(2, 3)
ax[0, 0].set_title("ACCURACY")
ax[0, 0].plot(x_plot, y_plot_accuracy, label='accuracy')
ax[0, 0].plot(x_plot, y_plot_val_accuracy, label='val_accuracy')
ax[0, 0].legend()
ax[0, 1].plot(x_plot, y_plot_loss, label='loss')
ax[0, 1].plot(x_plot, y_plot_val_loss, label='val_loss')
ax[0, 1].set_title("LOSS")
ax[0, 1].legend()
ax[0, 2].set_title("PRECISION")
ax[0, 2].plot(x_plot, y_plot_precision, label='precision')
ax[0, 2].plot(x_plot, y_plot_val_precision, label='val_precision')
ax[0, 2].legend()
ax[1, 0].set_title("RECALL")
ax[1, 0].plot(x_plot, y_plot_recall, label='recall')
ax[1, 0].plot(x_plot, y_plot_val_recall, label='val_recall')
ax[1, 0].legend()
ax[1, 1].set_title("F_SCORE")
ax[1, 1].plot(x_plot, y_plot_fscore, label='f_score')
ax[1, 1].plot(x_plot, y_plot_val_fscore, label='val_f_score')
ax[1, 1].legend()
ax[1, 2].set_title("GLOBAL ACCURACY")
ax[1, 2].plot(x_plot, y_plot_global_accuracy)
plt.show()



# determining the name of the file
file_name = 'C:\\Users\\skmuskaan\\Desktop\\File_from_client_1.xlsx'
dataframe = ds.drop(ds[(ds['Pd'] >1) | (ds['Noise'] ==0)].index)
dataframe = dataframe[(dataframe['Decision'] == "h0")]
dataframe1=dataframe.sample(10)

# saving the excel
dataframe1.to_excel(file_name)
print('--------------------------------------------------------------------------')
print('FILE TO BE SENT HAS BEEN CREATED SUCESSFULLY')
print('--------------------------------------------------------------------------')

#client.py sending

'''import socket
c = socket.socket()
try:
    c.connect(('localhost',1127))
    print("CONNECTED WITH SERVER")
except:
    print("UNABLE TO CONNECT")
    exit(0)
fname=input("Enter File Name : ")
path="C:\\Users\\skmuskaan\\Desktop\\"
ffname=path+fname
try:
    f=open(ffname,'rb')
except FileNotFoundError:
    print("No Such File Present")
    c.close()
    exit(0)
c.send(fname.encode('utf-8'))
l=f.read(1024)
while l:
    c.send(l)
    l=f.read(1024)
print('--------------------------------------------------------------------------')
print('FILE SENT SUCESSFULLY')
print('--------------------------------------------------------------------------')
fname=str(c.recv(1024).decode('utf-8'))
fname="C:\\Users\\skmuskaan\\Desktop\\c1_received _from_c2"+fname
f=open(fname,'wb')
print("FILE OPENED - RECIEVING DATA...")
while True:
    data=c.recv(1024)
    if not data:
        print('--------------------------------------------------------------------------')
        print('FILE RECEIVED SUCESSFULLY')
        print('--------------------------------------------------------------------------')
        break
    f.write(data)
f.close()
c.close()'''
import socket
c = socket.socket()
try:
    c.connect(('localhost',1129))
    print("CONNECTED WITH SERVER")
except:
    print("UNABLE TO CONNECT")
    exit(0)
fname=input("Enter File Name : ")
path="C:\\Users\\skmuskaan\\Desktop\\"
ffname=path+fname
try:
    f=open(ffname,'rb')
except FileNotFoundError:
    print("No Such File Present")
    c.close()
    exit(0)
c.send(fname.encode('utf-8'))
l=f.read(1024)
while l:
    c.send(l)
    l=f.read(1024)
print("File Sent Sucessfully")
f.close()
c.close()
import socket
s = socket.socket()
s.bind(('localhost',1128))
s.listen(1)
c,address=s.accept()
print("CONNECTED TO CLIENT -",address)
fname=str(c.recv(1024).decode())
fname="C:\\Users\\skmuskaan\\Desktop\\c2_received _from_c1\\"+fname
f=open(fname,'wb')
print("FILE OPENED - RECIEVING DATA...")
while True:
    data=c.recv(1024)
    if not data:
        print("FILE SUCESSFULLY RECEIVED")
        break
    f.write(data)
f.close()
c.close()
