## packages required
import torch
import tensorflow
import sklearn
import numpy as np
import pandas as pd
import joblib
import math
import keras
from tqdm import tqdm
from keras.layers import Input, InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    Conv1D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, AveragePooling1D, MaxPooling1D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.datasets import mnist
from sklearn.datasets import load_iris
from numpy import reshape
from keras import backend as K
import esm
import collections
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from statistics import mean, stdev
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import umap
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def esm_embeddings(peptide_sequence_list):
  # NOTICE: ESM for embeddings is quite RAM usage, if your sequence is too long,
  #         or you have too many sequences for transformation in a single converting,
  #         you conputer might automatically kill the job.

  # load the model
  # NOTICE: if the model was not downloaded in your local environment, it will automatically download it.
  model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
  batch_converter = alphabet.get_batch_converter()
  model.eval()  # disables dropout for deterministic results

  # load the peptide sequence list into the bach_converter
  batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list)
  batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
  ## batch tokens are the embedding results of the whole data set

  # Extract per-residue representations (on CPU)
  with torch.no_grad():
      # Here we export the last layer of the EMS model output as the representation of the peptides
      # model'esm2_t6_8M_UR50D' only has 6 layers, and therefore repr_layers parameters is equal to 6
      results = model(batch_tokens, repr_layers=[6], return_contacts=True)
  token_representations = results["representations"][6]

  # Generate per-sequence representations via averaging
  # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
  sequence_representations = []
  for i, tokens_len in enumerate(batch_lens):
      sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
  # save dataset
  # sequence_representations is a list and each element is a tensor
  embeddings_results = collections.defaultdict(list)
  for i in range(len(sequence_representations)):
      # tensor can be transformed as numpy sequence_representations[0].numpy() or sequence_representations[0].to_list
      each_seq_rep = sequence_representations[i].tolist()
      for each_element in each_seq_rep:
          embeddings_results[i].append(each_element)
  embeddings_results = pd.DataFrame(embeddings_results).T
  return embeddings_results


# training dataset loading
dataset = pd.read_excel('../data/orginal/all_train.xlsx',na_filter = False) # take care the NA sequence problem
sequence_list = dataset['SequenceID']



embeddings_results = pd.DataFrame()
for seq in tqdm(sequence_list):
    format_seq = [seq,seq] # the setting is just following the input format setting in ESM model, [name,sequence]
    tuple_sequence = tuple(format_seq)
    peptide_sequence_list = []
    peptide_sequence_list.append(tuple_sequence) # build a summarize list variable including all the sequence information
    # employ ESM model for converting and save the converted data in csv format
    one_seq_embeddings = esm_embeddings(peptide_sequence_list)
    embeddings_results= pd.concat([embeddings_results,one_seq_embeddings])

embeddings_results.to_csv('new_neuro_train_esm2_t6_8M_UR50D_unified_320_dimension.csv')

# loading the y dataset for model development
y_train = dataset['Label']
y_train = np.array(y_train,dtype='float16') # transformed as np.array for DNN model



# test dataset loading
dataset = pd.read_excel('../data/orginal/all_test.xlsx',na_filter = False) # take care the NA sequence problem
sequence_list = dataset['SequenceID']
embeddings_results = pd.DataFrame()
# embedding all the peptide one by one
for seq in tqdm(sequence_list):
    format_seq = [seq,seq] # the setting is just following the input format setting in ESM model, [name,sequence]
    tuple_sequence = tuple(format_seq)
    peptide_sequence_list = []
    peptide_sequence_list.append(tuple_sequence) # build a summarize list variable including all the sequence information
    # employ ESM model for converting and save the converted data in csv format
    one_seq_embeddings = esm_embeddings(peptide_sequence_list)
    embeddings_results= pd.concat([embeddings_results,one_seq_embeddings])

embeddings_results.to_csv('new_neuro_test_esm2_t6_8M_UR50D_unified_320_dimension.csv')


# loading the y dataset for model development
y_test = dataset['Label']
y_test = np.array(y_test,dtype='float16') # transformed as np.array for DNN model



# assign the dataset
X_train_data_name = 'new_neuro_train_esm2_t6_8M_UR50D_unified_320_dimension.csv'
X_train_data = pd.read_csv(X_train_data_name,header=0, index_col = 0,delimiter=',')

X_test_data_name = 'new_neuro_test_esm2_t6_8M_UR50D_unified_320_dimension.csv'
X_test_data = pd.read_csv(X_test_data_name,header=0, index_col = 0,delimiter=',')

X_train = np.array(X_train_data,dtype='float16')
X_test = np.array(X_test_data,dtype='float16')

# normalize the X data range

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train) # normalize X to 0-1 range
X_test = scaler.transform(X_test)



# check the dimension of the dataset before model development
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)




X_train = np.array(X_train_data,dtype='float16')
X_test = np.array(X_test_data,dtype='float16')
# concatenate the dataset
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)



# result collection list
ACC_collecton = []
BACC_collecton = []
Sn_collecton = []
Sp_collecton = []
MCC_collecton = []
AUC_collecton = []

def ESM_DNN(X_train, y_train, X_test, y_test):


    inputShape = (320, 1)
    input = Input(inputShape)

    # 展平输入
    x = Flatten()(input)

    # 定义两个全连接层，每个全连接层后面连接一个BatchNormalization层和一个Dropout层，用于减少过拟合。
    x = Dense(64, activation='relu', name='fc1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.15)(x)
    x = Dense(32, activation='relu', name='fc2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.15)(x)

    # 最后一个全连接层有2个输出，使用softmax激活函数进行多分类。
    x = Dense(2, activation='softmax', name='fc3')(x)

    # 创建模型
    model = Model(inputs=input, outputs=x, name='Predict')

    # 定义SGD优化器
    momentum = 0.5
    sgd = SGD(lr=0.01, momentum=momentum, decay=0.0, nesterov=False)

    # 编译模型
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # 定义学习率下降函数
    def step_decay(epoch):
        initial_lrate = 0.05
        drop = 0.6
        epochs_drop = 3.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate

    # 创建LearningRateScheduler回调函数
    lrate = LearningRateScheduler(step_decay)

    # 创建EarlyStopping回调函数
    early_stop = EarlyStopping(monitor='val_accuracy', patience=40, restore_best_weights=True)

    # 将回调函数加入到callbacks_list中
    callbacks_list = [lrate, early_stop]

    # 训练模型
    model_history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                              epochs=200, callbacks=callbacks_list, batch_size=8, verbose=1)

    return model, model_history


for i in range(20):

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=i)
    # normalize the X data range
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train) # normalize X to 0-1 range
    X_test = scaler.transform(X_test)


    model, model_history = ESM_DNN(X_train, y_train, X_test , y_test)
    # confusion matrix
    predicted_class= []
    predicted_protability = model.predict(X_test,batch_size=1)
    for i in range(predicted_protability.shape[0]):
      index = np.where(predicted_protability[i] == np.amax(predicted_protability[i]))[0][0]
      predicted_class.append(index)
    predicted_class = np.array(predicted_class,dtype='float16')
    y_true = y_test
    # np.ravel() return a flatten 1D array
    TP, FP, FN, TN = confusion_matrix(y_true, predicted_class).ravel() # shape [ [True-Positive, False-positive], [False-negative, True-negative] ]
    ACC = (TP+TN)/(TP+TN+FP+FN)
    ACC_collecton.append(ACC)
    Sn_collecton.append(TP/(TP+FN))
    Sp_collecton.append(TN/(TN+FP))
    MCC = (TP*TN-FP*FN)/math.pow(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)),0.5)
    MCC_collecton.append(MCC)
    BACC_collecton.append(0.5*TP/(TP+FN)+0.5*TN/(TN+FP))
    AUC = roc_auc_score(y_test, predicted_protability[:,1])
    AUC_collecton.append(AUC)
    name = "neuro_tensorflow_model" + str(i)
    model.save(name,save_format = 'tf')






print(ACC_collecton)
print(BACC_collecton)
print(Sn_collecton)
print(Sp_collecton)
print(MCC_collecton)
print(AUC_collecton)






#Implementing 10-fold cross validation

k = 10
kf = KFold(n_splits=k, shuffle = True, random_state=1)
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)

# result collection list
ACC_collecton = []
BACC_collecton = []
Sn_collecton = []
Sp_collecton = []
MCC_collecton = []
AUC_collecton = []

for train_index , test_index in kf.split(y_train):
    X_train_CV , X_valid_CV = X_train.iloc[train_index,:],X_train.iloc[test_index,:]
    y_train_CV , y_valid_CV = y_train.iloc[train_index] , y_train.iloc[test_index]
    model, model_history = ESM_DNN(X_train_CV, y_train_CV, X_valid_CV, y_valid_CV)
    # confusion matrix
    predicted_class= []
    predicted_protability = model.predict(X_valid_CV,batch_size=1)
    for i in range(predicted_protability.shape[0]):
      index = np.where(predicted_protability[i] == np.amax(predicted_protability[i]))[0][0]
      predicted_class.append(index)
    predicted_class = np.array(predicted_class,dtype='float16')
    y_true = y_valid_CV

    # np.ravel() return a flatten 1D array
    TP, FP, FN, TN = confusion_matrix(y_true, predicted_class).ravel() # shape [ [True-Positive, False-positive], [False-negative, True-negative] ]
    ACC = (TP+TN)/(TP+TN+FP+FN)
    ACC_collecton.append(ACC)
    Sn_collecton.append(TP/(TP+FN))
    Sp_collecton.append(TN/(TN+FP))
    MCC = (TP*TN-FP*FN)/math.pow(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)),0.5)
    MCC_collecton.append(MCC)
    BACC_collecton.append(0.5*TP/(TP+FN)+0.5*TN/(TN+FP))
    AUC = roc_auc_score(y_valid_CV, predicted_protability[:,1])
    AUC_collecton.append(AUC)



print(mean(ACC_collecton),'±',stdev(ACC_collecton))
print(mean(BACC_collecton),'±',stdev(BACC_collecton))
print(mean(Sn_collecton),'±',stdev(Sn_collecton))
print(mean(Sp_collecton),'±',stdev(Sp_collecton))
print(mean(MCC_collecton),'±',stdev(MCC_collecton))
print(mean(AUC_collecton),'±',stdev(AUC_collecton))
print(ACC_collecton)



# result collection list
ACC_collecton = []
BACC_collecton = []
Sn_collecton = []
Sp_collecton = []
MCC_collecton = []
AUC_collecton = []
model, model_history = ESM_DNN(X_train, y_train, X_test , y_test)
# confusion matrix
predicted_class= []
predicted_protability = model.predict(X_test,batch_size=1)
for i in range(predicted_protability.shape[0]):
  index = np.where(predicted_protability[i] == np.amax(predicted_protability[i]))[0][0]
  predicted_class.append(index)
predicted_class = np.array(predicted_class,dtype='float16')
y_true = y_test

# np.ravel() return a flatten 1D array
TP, FP, FN, TN = confusion_matrix(y_true, predicted_class).ravel() # shape [ [True-Positive, False-positive], [False-negative, True-negative] ]
ACC = (TP+TN)/(TP+TN+FP+FN)
ACC_collecton.append(ACC)
Sn_collecton.append(TP/(TP+FN))
Sp_collecton.append(TN/(TN+FP))
MCC = (TP*TN-FP*FN)/math.pow(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)),0.5)
MCC_collecton.append(MCC)
BACC_collecton.append(0.5*TP/(TP+FN)+0.5*TN/(TN+FP))
AUC = roc_auc_score(y_test, predicted_protability[:,1])
AUC_collecton.append(AUC)

print(ACC_collecton[0])
print(BACC_collecton[0])
print(Sn_collecton[0])
print(Sp_collecton[0])
print(MCC_collecton[0])
print(AUC_collecton[0])


model.save('neuro_tensorflow_model',save_format = 'tf')




# training dataset loading
dataset = pd.read_excel('../data/orginal/all_train.xlsx',na_filter = False) # take care the NA sequence problem
sequence_list = dataset['SequenceID']

# loading the y dataset for model development
y_train = dataset['Label']
y_train = np.array(y_train) # transformed as np.array for DNN model
# test dataset loading
dataset = pd.read_excel('../data/orginal/all_test.xlsx',na_filter = False) # take care the NA sequence problem
sequence_list = dataset['SequenceID']
# loading the y dataset for model development
y_test = dataset['Label']
y_test = np.array(y_test) # transformed as np.array for DNN model
# assign the dataset
X_train_data_name = 'new_neuro_train_esm2_t6_8M_UR50D_unified_320_dimension.csv'
X_train_data = pd.read_csv(X_train_data_name,header=0, index_col = 0,delimiter=',')

X_test_data_name = 'new_neuro_test_esm2_t6_8M_UR50D_unified_320_dimension.csv'
X_test_data = pd.read_csv(X_test_data_name,header=0, index_col = 0,delimiter=',')

X_train = np.array(X_train_data)
X_test = np.array(X_test_data)

# normalize the X data range

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train) # normalize X to 0-1 range
X_test = scaler.transform(X_test)
# check the dimension of the dataset before model development
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)



# concatenate the dataset
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)
embedding = umap.UMAP(n_neighbors=20).fit_transform(X, y=y) # two dimension
df = pd.DataFrame()
df["comp-1"] = embedding[:,0]
df["comp-2"] = embedding[:,1]
y_new_label=[]
for i in y:
    if i == 0:
        y_new_label.append('Active')
    if i == 1:
        y_new_label.append('Inactive')
df["y"] = y_new_label
graph = sns.scatterplot(data=df, x="comp-1", y="comp-2", hue=y_new_label,
                palette='YlOrRd_r', legend='full')
graph_for_output = graph.get_figure()
graph_for_output.savefig('13.neuro_UMAP.png', dpi=300)
df.to_excel('13.neuro_UMAP.xlsx')