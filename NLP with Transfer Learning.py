#!/usr/bin/env python
# coding: utf-8

# 
# In this notebook, You will do amazon review classification with BERT.[Download data from [this](https://www.kaggle.com/snap/amazon-fine-food-reviews/data) link]
# <pre> 
# It contains 5 parts as below.
#     1. Preprocessing 
#     2. Creating a BERT model from the Tensorflow HUB.
#     3. Tokenization
#     4. getting the pretrained embedding Vector for a given review from the BERT.
#     5. Using the embedding data apply NN and classify the reviews.
#     6. Creating a Data pipeline for BERT Model. 
# 

# # Preparation

# In[1]:


from google.colab import drive 
drive.mount('/content/drive')


# In[ ]:


#all imports
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import utils
from tensorflow.keras.models import Model


# In[ ]:


tf.test.gpu_device_name()


# <font size=4>Grader function 1 </font>

# In[ ]:


def grader_tf_version():
    assert((tf.__version__)>'2')
    return True
grader_tf_version()


# # <pre><font size=6>Part-1: Preprocessing</font></pre>

# In[ ]:


#Read the dataset - Amazon fine food reviews
reviews = pd.read_csv(r"/content/drive/MyDrive/28 NLP with Transfer Learning/Reviews.csv")
#check the info of the dataset
reviews.info()


# In[ ]:


reviews.head(2)


# In[ ]:


data = reviews[['Text', 'Score' ]]
data = data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
data.head()


# In[ ]:


import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm


# In[ ]:


if not os.path.isfile('/content/drive/MyDrive/28 NLP with Transfer Learning/processed_reviews.csv'):
  data = reviews[['Text', 'Score' ]]

  data.drop(data[data['Score'] == 3].index, inplace = True)
  data = data.reset_index()
  data = data[['Text', 'Score' ]]

  for i in tqdm(data.index):
    if data.Score.loc[i] > 3:
      data.Score.loc[i] = 1
    else:
      data.Score.loc[i] = 0
  data.to_csv("processed_reviews.csv")
else:
  reviews = pd.read_csv("/content/drive/MyDrive/28 NLP with Transfer Learning/processed_reviews.csv")
  reviews = reviews.drop(['Unnamed: 0'],axis=1)


# In[ ]:


reviews.head()


# <font size=4>Grader function 2 </font>

# In[ ]:


def grader_reviews():
    temp_shape = (reviews.shape == (525814, 2)) and (reviews.Score.value_counts()[1]==443777)
    assert(temp_shape == True)
    return True
grader_reviews()


# In[ ]:


def get_wordlen(x):
    return len(x.split())
reviews['len'] = reviews.Text.apply(get_wordlen)
reviews = reviews[reviews.len<50]
reviews_sample = reviews.sample(n=100000, random_state=30)


# In[ ]:


len(reviews), len(reviews_sample)


# In[ ]:


#remove HTML from the Text column and save in the Text column only

import re
def remove_html(string):
  pattern = re.compile('<[^>]*>')
  clean_string = re.sub(pattern, '', string)
  return clean_string


# In[ ]:


if not os.path.isfile('/content/drive/MyDrive/28 NLP with Transfer Learning/preprocessed.csv'):
  preprocessed_text = []
  for text in tqdm(reviews.Text):
    clean_string = remove_html(text)
    preprocessed_text.append(clean_string)
  reviews['Preprocessed_text'] = preprocessed_text
  reviews = reviews[['Preprocessed_text', 'Score', 'len']]

  #saving to disk. if we need, we can load preprocessed data directly. 
  reviews.to_csv('/content/drive/MyDrive/28 NLP with Transfer Learning/preprocessed.csv', index=False)

else:
  reviews = pd.read_csv("/content/drive/MyDrive/28 NLP with Transfer Learning/preprocessed.csv")
reviews.head(3)


# In[ ]:


#split the data into train and test data(20%) with Stratify sampling, random state 33, 


# In[ ]:


y = reviews.Score
X = reviews[['Preprocessed_text', 'len']]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, stratify = y)


# In[ ]:


print(X_train.shape, len(y_train)), 
print(X_test.shape, len(y_test))


# In[ ]:


y_train_cat = utils.to_categorical(y_train, num_classes=2)
y_test_cat = utils.to_categorical(y_test, num_classes=2)


# In[ ]:


#plot bar graphs of y_train and y_test
import matplotlib.pyplot as plt


# In[ ]:


plt.subplot(121)
ax = y_train.value_counts().plot(kind='bar',
                                    figsize=(10,5),
                                    title="Stratify Sampling - y_train",
                                    color=('c', 'black'))
ax.set_xlabel("y_train")
ax.set_ylabel("Frequency")
#plt.show()

plt.subplot(122)
ax = y_test.value_counts().plot(kind='bar',
                                    figsize=(10,5),
                                    title="Stratify Sampling - y_test",
                                    color=('c', 'black'))
ax.set_xlabel("y_test")
ax.set_ylabel("Frequency")
plt.show()


# # <pre><font size=6>Part-2: Creating BERT Model</font> 

# In[ ]:


## Loading the Pretrained Model from tensorflow HUB
tf.keras.backend.clear_session()

# maximum length of a seq in the data we have, for now i am making it as 55.
max_seq_length = 55

#BERT takes 3 inputs

#this is input words. Sequence of words represented as integers
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")

#mask vector if you are padding anything
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")

#segment vectors. If you are giving only one sentence for the classification, total seg vector is 0. 
#If you are giving two sentenced with [sep] token separated, first seq segment vectors are zeros and 
#second seq segment vector are 1's
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")

#bert layer 
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

#Bert model
#We are using only pooled output not sequence out. 
#If you want to know about those, please read https://www.kaggle.com/questions-and-answers/86510
bert_model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=pooled_output)


# In[ ]:


bert_model.summary()


# In[ ]:


bert_model.output


# In[ ]:





# # <pre><font size=6>Part-3: Tokenization</font></pre>

# In[ ]:


#getting Vocab file
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()


# ### **Getting AttributeError while using tokenizer with tensorflow >2 , lets downgrade the tensorflow and reverse it after tokenization**
# 
# ***AttributeError:*** *module 'tensorflow' has no attribute 'gfile'*

# In[ ]:


get_ipython().system('pip install tensorflow==1.15.0')
import tensorflow as tf
print(tf.__version__)

import tensorflow as tf
tf.enable_eager_execution()


# In[ ]:


#import tokenization - We have given tokenization.py file


# In[ ]:


get_ipython().system('pip install sentencepiece ')
get_ipython().system('python3 "/content/drive/MyDrive/28 NLP with Transfer Learning/tokenization.py"')


# In[ ]:


get_ipython().system('pip install bert-tensorflow==1.0.1')
import bert
from bert import tokenization


# In[ ]:


# Create tokenizer " Instantiate FullTokenizer" 
# name must be "tokenizer"
# the FullTokenizer takes two parameters 1. vocab_file and 2. do_lower_case 
# we have created these in the above cell ex: FullTokenizer(vocab_file, do_lower_case)
# please check the "tokenization.py" file the complete implementation


# In[ ]:


if not os.path.isfile('/content/drive/MyDrive/28 NLP with Transfer Learning/tokenizer.pickle'):
  tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)


# In[ ]:


tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)


# In[ ]:


#Reference: https://intellipaat.com/community/491/keras-text-preprocessing-saving-tokenizer-object-to-file-for-scoring

if not os.path.isfile('/content/drive/MyDrive/28 NLP with Transfer Learning/tokenizer.pickle'):
  #saving
  with open('/content/drive/MyDrive/28 NLP with Transfer Learning/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

  #loading
  with open('/content/drive/MyDrive/28 NLP with Transfer Learning/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# In[ ]:


#tf.io.gfile.GFile


# <font size=4>Grader function 3 </font>

# In[ ]:


#it has to give no error 
def grader_tokenize(tokenizer):
    out = False
    try:
        out=('[CLS]' in tokenizer.vocab) and ('[SEP]' in tokenizer.vocab)
    except:
        out = False
    assert(out==True)
    return out
grader_tokenize(tokenizer)


# In[ ]:


# Create train and test tokens (X_train_tokens, X_test_tokens) from (X_train, X_test) using Tokenizer and 

# add '[CLS]' at start of the Tokens and '[SEP]' at the end of the tokens. 

# maximum number of tokens is 55(We already given this to BERT layer above) so shape is (None, 55)

# if it is less than 55, add '[PAD]' token else truncate the tokens length.(similar to padding)

# Based on padding, create the mask for Train and Test ( 1 for real token, 0 for '[PAD]'), 
# it will also same shape as input tokens (None, 55) save those in X_train_mask, X_test_mask

# Create a segment input for train and test. We are using only one sentence so all zeros. This shape will also (None, 55)

# type of all the above arrays should be numpy arrays

# after execution of this cell, you have to get 
# X_train_tokens, X_train_mask, X_train_segment
# X_test_tokens, X_test_mask, X_test_segment


# #### Example
# <img src='https://i.imgur.com/5AhhmgU.png'>

# In[ ]:


def create_tokens(data:'list of strings',
                  max_seq_length:'Max length of the documents' =55) -> "":
  data_mask = []
  data_segment = []
  data_tokens_encoded = []
  token_list_updated = []

  for sent in tqdm(data):
    #Tokenization
    token = tokenizer.tokenize(sent)
    #X_train_tokens.append(token)
    #Adding Special Tokens
    diff = max_seq_length - len(token)
    if len(token) >= (max_seq_length - 2):
      token_update = token[0: (max_seq_length - 2)]
      token_update.insert(0, '[CLS]')
      token_update.insert(len(token_update), '[SEP]')
      token_list_updated.append(token_update)

      #Positional Encoding
      token_encoded = np.array(tokenizer.convert_tokens_to_ids(token_update))
      data_tokens_encoded.append(token_encoded)

      #Masking
      token_mask = np.array([1]*len(token_update))
      data_mask.append(token_mask)

      #Segment
      token_segment = np.array([0]*max_seq_length)
      data_segment.append(token_segment)

    else:

      token.insert(0, '[CLS]')
      token.insert(len(token), '[SEP]')

      #Masking
      token_mask = np.array([1]*len(token)+ [0]*(max_seq_length - len(token)))
      data_mask.append(token_mask)
      #Segment
      token_segment = np.array([0]*max_seq_length)
      data_segment.append(token_segment)

      for i in range(diff-2):
        token.append('[PAD]')
      token_list_updated.append(token)

      #Positional Encoding
      token_encoded = np.array(tokenizer.convert_tokens_to_ids(token))
      data_tokens_encoded.append(token_encoded)

  data_tokens = np.array(data_tokens_encoded)
  data_mask = np.array(data_mask)
  data_segment = np.array(data_segment)

  return data_tokens, data_mask, data_segment


# In[ ]:


if not os.path.isfile('/content/drive/MyDrive/28 NLP with Transfer Learning/train_data.pkl'):  
  data_training = X_train.Preprocessed_text.values.tolist()
  X_train_tokens, X_train_mask, X_train_segment = create_tokens(data_training)
  #save all your results to disk so that, no need to run all again. 
  pickle.dump((X_train, X_train_tokens, X_train_mask, X_train_segment, y_train),open('/content/drive/MyDrive/28 NLP with Transfer Learning/train_data.pkl','wb'))
else:
  #you can load from disk
  X_train, X_train_tokens, X_train_mask, X_train_segment, y_train = pickle.load(open("/content/drive/MyDrive/28 NLP with Transfer Learning/train_data.pkl", 'rb')) 
  


# In[ ]:


if not os.path.isfile('/content/drive/MyDrive/28 NLP with Transfer Learning/val_data.pkl'):
  data_test = X_test.Preprocessed_text.values.tolist()
  X_test_tokens, X_test_mask, X_test_segment = create_tokens(data_test)
  #save all your results to disk so that, no need to run all again.
  pickle.dump((X_test, X_test_tokens, X_test_mask, X_test_segment, y_test),open('/content/drive/MyDrive/28 NLP with Transfer Learning/val_data.pkl','wb'))
else:
  X_test, X_test_tokens, X_test_mask, X_test_segment, y_test = pickle.load(open("/content/drive/MyDrive/28 NLP with Transfer Learning/val_data.pkl", 'rb')) 


# <font size=4>Grader function 4 </font>

# In[ ]:


max_seq_length = 55


# In[ ]:


def grader_alltokens_train():
    out = False
    
    if type(X_train_tokens) == np.ndarray:
        
        temp_shapes = (X_train_tokens.shape[1]==max_seq_length) and (X_train_mask.shape[1]==max_seq_length) and         (X_train_segment.shape[1]==max_seq_length)
        
        segment_temp = not np.any(X_train_segment)
        
        mask_temp = np.sum(X_train_mask==0) == np.sum(X_train_tokens==0)
        
        no_cls = np.sum(X_train_tokens==tokenizer.vocab['[CLS]'])==X_train_tokens.shape[0]
        
        no_sep = np.sum(X_train_tokens==tokenizer.vocab['[SEP]'])==X_train_tokens.shape[0]
        
        out = temp_shapes and segment_temp and mask_temp and no_cls and no_sep
      
    else:
        print('Type of all above token arrays should be numpy array not list')
        out = False
    assert(out==True)
    return out

grader_alltokens_train()


# <font size=4>Grader function 5 </font>

# In[ ]:


def grader_alltokens_test():
    out = False
    if type(X_test_tokens) == np.ndarray:
        
        temp_shapes = (X_test_tokens.shape[1]==max_seq_length) and (X_test_mask.shape[1]==max_seq_length) and         (X_test_segment.shape[1]==max_seq_length)
        
        segment_temp = not np.any(X_test_segment)
        
        mask_temp = np.sum(X_test_mask==0) == np.sum(X_test_tokens==0)
        
        no_cls = np.sum(X_test_tokens==tokenizer.vocab['[CLS]'])==X_test_tokens.shape[0]
        
        no_sep = np.sum(X_test_tokens==tokenizer.vocab['[SEP]'])==X_test_tokens.shape[0]
        
        out = temp_shapes and segment_temp and mask_temp and no_cls and no_sep
      
    else:
        print('Type of all above token arrays should be numpy array not list')
        out = False
    assert(out==True)
    return out
grader_alltokens_test()


# **Updating back Tensorflow to latest**

# In[ ]:


get_ipython().system('pip install --upgrade tensorflow')


# # <pre><font size=6>Part-4: Getting Embeddings from BERT Model</font>

# <pre><font size=3>Part-4: Getting Embeddings from BERT Model</font>
# We already created the BERT model in the part-2 and input data in the part-3. 
# We will utlize those two and will get the embeddings for each sentence in the 
# Train and test data.</pre>

# In[ ]:


bert_model.input


# In[ ]:


bert_model.output


# In[ ]:


if not os.path.isfile('/content/drive/MyDrive/28 NLP with Transfer Learning/final_output.pkl'):
  # get the train output, BERT model will give one output so save in
  # X_train_pooled_output
  X_train_pooled_output=bert_model.predict([X_train_tokens,X_train_mask,X_train_segment])

  # get the test output, BERT model will give one output so save in
  # X_test_pooled_output
  X_test_pooled_output=bert_model.predict([X_test_tokens,X_test_mask,X_test_segment])

  #save all your results to disk so that, no need to run all again. 
  pickle.dump((X_train_pooled_output, X_test_pooled_output),open('/content/drive/MyDrive/28 NLP with Transfer Learning/final_output.pkl','wb'))

else:
  X_train_pooled_output, X_test_pooled_output= pickle.load(open('/content/drive/MyDrive/28 NLP with Transfer Learning/final_output.pkl', 'rb'))


# <font size=4>Grader function 6 </font>

# In[ ]:


#now we have X_train_pooled_output, y_train
#X_test_pooled_ouput, y_test

#please use this grader to evaluate
def greader_output():
    assert(X_train_pooled_output.shape[1]==768)
    assert(len(y_train)==len(X_train_pooled_output))
    assert(X_test_pooled_output.shape[1]==768)
    assert(len(y_test)==len(X_test_pooled_output))
    assert(len(y_train.shape)==1)
    assert(len(X_train_pooled_output.shape)==2)
    assert(len(y_test.shape)==1)
    assert(len(X_test_pooled_output.shape)==2)
    return True
greader_output()


# In[ ]:


pd.DataFrame(X_test_pooled_output)


# In[ ]:





# # <pre><font size=6>Part-5: Training a NN with 768 features</font>

# In[ ]:


##imports
import tensorflow
import tensorboard
get_ipython().run_line_magic('load_ext', 'tensorboard')
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Dense, Activation, Conv1D
from tensorflow.keras.layers import  Dropout, Flatten, Reshape, BatchNormalization, MaxPooling1D


# ### **5.1 Callbacks**

# In[ ]:


#Reference:https://stackoverflow.com/questions/59666138/sklearn-roc-auc-score-with-multi-class-ovr-should-have-none-average-available
#Code Reference: https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
#Reference:https://learning.oreilly.com/library/view/deep-learning-quick/9781788837996/a22485be-e397-4b46-86b2-29b7878953f5.xhtml#:~:text=Let's%20use%20one%20more%20callback,Keras%20is%20actually%20really%20simple.

from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping,ReduceLROnPlateau

class RocCallback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_epoch_end(self, epoch, logs={}):
     
        y_pred_train = np.argmax(model.predict(self.x),axis=1)
        roc_train = roc_auc_score(self.y, y_pred_train, average='weighted', multi_class='ovr')

        y_pred_val = np.argmax(model.predict(self.x_val),axis=1)
        roc_val = roc_auc_score(self.y_val, y_pred_val, average='weighted', multi_class='ovr')
        print('ROC-AUC Train: %s - ROC-AUC Test: %s' % (str(round(roc_train,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

ROC_AUC = RocCallback(training_data=(X_train_pooled_output, y_train),
                  validation_data=(X_test_pooled_output, y_test))

#Tensorboard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="/content/drive/MyDrive/26 RNN LSTM /log/model", histogram_freq=1)

#Learning Rate
#https://stackoverflow.com/questions/39779710/setting-up-a-learningratescheduler-in-keras
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
def scheduler(epoch, lr):
  if epoch % 10 ==0:
    return lr*0.95
  else:
    return lr
lr_scheduler = LearningRateScheduler(scheduler, verbose=1)

#Early Stop
earlystop = EarlyStopping(monitor='accuracy', patience=50, verbose=1)

#Reduce on Plateau
decay_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=10, 
                                                verbose=0, mode='auto', min_delta=0.001, 
                                                cooldown=0, min_lr=1e-6)


#callback_list = [lr_scheduler, decay_lr, checkpoint, tensorboard_callback]


# In[ ]:


#Reference:https://www.codegrepper.com/code-examples/python/auc+callback+keras

#import tensorflow as tf
from sklearn.metrics import roc_auc_score

def auroc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)


# ### **5.2 Importing various optimizers for experiment:**

# In[ ]:


import tensorflow as tf
optimizer_SGD = tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.8, nesterov=True, 
    name='SGD')

optimizer_aadam = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')

optimizer_adamax = tf.keras.optimizers.Adamax(
    learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
    name='Adamax')

optimizer_RMSprop = tf.keras.optimizers.RMSprop(
    learning_rate=0.01, rho=0.9, momentum=0.8, epsilon=1e-07, centered=False,
    name='RMSprop')

optimizer_adagrad = tf.keras.optimizers.Adagrad(
    learning_rate=0.1, initial_accumulator_value=0.1, epsilon=1e-07,
    name='Adagrad')

optimizer_adadelta = tf.keras.optimizers.Adadelta(
    learning_rate=0.1, rho=0.95, epsilon=1e-07, name='Adadelta')


# ### **5.3 Function for Plotting Loss and AUC:**

# In[ ]:


#Reference:https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
# plot diagnostic learning curves
import matplotlib.pyplot as plt
def summarize_diagnostics(history):
  #plot Loss
  plt.figure(figsize=(10,10))
  plt.subplot(211)
  plt.title('Cross Entropy Loss')
  plt.plot(history.history['loss'], color='blue', label='train')
  plt.plot(history.history['val_loss'], color='orange', label='test')
  plt.xlabel("Epochs")

  # plot accuracy
  plt.subplot(212)
  plt.title('Classification AUC')
  plt.plot(history.history['auroc'], color='blue', label='train')
  plt.plot(history.history['val_auroc'], color='orange', label='test')
  plt.xlabel("Epochs")

  plt.show()


# ### **5.4 Models Architecture**

# In[ ]:


X_in = Input(shape=(X_train_pooled_output[0].shape))
X_in_reshaped = Reshape((768,1))(X_in)

X_conv1d_O = Conv1D(filters=132, kernel_size=16, activation='relu')(X_in_reshaped)


#Adding Batch Normalization and Dropout Layers
X_normalized_1 = BatchNormalization()(X_conv1d_O)
X_dropout_1 = Dropout((0.2))(X_normalized_1)

X_maxpooled_1 = MaxPooling1D(pool_size=2, strides=2, padding='same')(X_dropout_1)

X_conv1d_I = Conv1D(filters=64, kernel_size=12, activation='tanh')(X_maxpooled_1)


#Adding Batch Normalization and Dropout Layers
X_normalized_2 = BatchNormalization()(X_conv1d_I)
X_dropout_2 = Dropout((0.5))(X_normalized_2)

X_maxpooled_2 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X_dropout_2)

X_dropout_3 = Dropout((0.2))(X_maxpooled_2)
X_conv1d_1 = Conv1D(filters=32, kernel_size=8, activation='relu')(X_dropout_3)
X_conv1d_2 = Conv1D(filters=16, kernel_size=4, activation='relu')(X_conv1d_1)
X_flatten = Flatten()(X_conv1d_2)
X_dropout = Dropout((0.2))(X_flatten)

X_dense_1 = Dense(128, activation='relu',kernel_initializer=tf.keras.initializers.glorot_normal(seed=30))(X_dropout)
X_normalized_3 = BatchNormalization()(X_dense_1)
X_dropout_3 = Dropout((0.5))(X_normalized_3)

X_dense_2 = Dense(32, activation='relu',kernel_initializer=tf.keras.initializers.glorot_normal(seed=30))(X_dropout_3)
X_normalized_4 = BatchNormalization()(X_dense_2)

X_dense_3 = Dense(16, activation='relu',kernel_initializer=tf.keras.initializers.glorot_normal(seed=30))(X_normalized_4)


X_out = Dense(2, activation='softmax', kernel_initializer=tf.keras.initializers.glorot_normal(seed=30) )(X_dense_3)

model = Model(inputs=X_in, outputs=X_out)

print(model.summary())


# In[ ]:


#Saving Best Model and Representation of results
filepath = "/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_best_1.hdf5"
checkpoint = ModelCheckpoint(filepath= filepath, save_weights_only=True,
                              monitor='val_auroc', verbose=1,
                              save_best_only=True, mode='max') 

log_dir = "logs/model_1"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq=1)
callback_list = [checkpoint, tensorboard_callback, decay_lr]



optimizer_adagrad = tf.keras.optimizers.Adagrad(
    learning_rate=0.000001 , initial_accumulator_value=0.1, epsilon=1e-07,
    name='Adagrad')

callback_list = [tensorboard_callback, checkpoint]

optimizer_aadam = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')

#Model Compilation
model.compile(optimizer=optimizer_aadam, loss='categorical_crossentropy', metrics=['accuracy', auroc])


# In[ ]:


epochs = 100

#Training the model
history = model.fit(X_train_pooled_output, y_train_cat, validation_data=(X_test_pooled_output, y_test_cat), 
           batch_size=1000, epochs=epochs, verbose='auto', callbacks = callback_list)


model.save_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_100Epoch.h5')


# In[ ]:


import tensorflow.keras.backend as k
k.set_value(model.optimizer.lr, 0.0001)

model.load_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_100Epoch.h5')

epochs = 100

#Training the model
history = model.fit(X_train_pooled_output, y_train_cat, validation_data=(X_test_pooled_output, y_test_cat), 
           batch_size=1000, epochs=epochs, verbose='auto', callbacks = callback_list)


model.save_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_200poch.h5')


# In[ ]:


import tensorflow.keras.backend as k
k.set_value(model.optimizer.lr, 0.00001)

model.load_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_200poch.h5')

epochs = 100

#Training the model
history = model.fit(X_train_pooled_output, y_train_cat, validation_data=(X_test_pooled_output, y_test_cat), 
           batch_size=1000, epochs=epochs, verbose='auto', callbacks = callback_list)


model.save_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_300poch.h5')


# In[ ]:


import tensorflow.keras.backend as k
k.set_value(model.optimizer.lr, 0.000001)

model.load_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_300poch.h5')

epochs = 100

#Training the model
history = model.fit(X_train_pooled_output, y_train_cat, validation_data=(X_test_pooled_output, y_test_cat), 
           batch_size=1000, epochs=epochs, verbose='auto', callbacks = callback_list)


model.save_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_400poch.h5')


# In[ ]:


import tensorflow.keras.backend as k
k.set_value(model.optimizer.lr, 0.0000001)

model.load_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_400poch.h5')

epochs = 100

#Training the model
history = model.fit(X_train_pooled_output, y_train_cat, validation_data=(X_test_pooled_output, y_test_cat), 
           batch_size=1000, epochs=epochs, verbose='auto', callbacks = callback_list)


model.save_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_500poch.h5')


# In[ ]:


import tensorflow.keras.backend as k
k.set_value(model.optimizer.lr, 0.005)

model.load_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_500poch.h5')

epochs = 10

#Training the model
history = model.fit(X_train_pooled_output, y_train_cat, validation_data=(X_test_pooled_output, y_test_cat), 
           batch_size=1000, epochs=epochs, verbose='auto', callbacks = callback_list)


model.save_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_510poch.h5')


# In[ ]:


import tensorflow.keras.backend as k
k.set_value(model.optimizer.lr, 0.005)

model.load_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_best_1.hdf5')

epochs = 10

#Training the model
history = model.fit(X_train_pooled_output, y_train_cat, validation_data=(X_test_pooled_output, y_test_cat), 
           batch_size=1000, epochs=epochs, verbose='auto', callbacks = callback_list)


model.save_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_520poch.h5')


# In[ ]:


import tensorflow.keras.backend as k
k.set_value(model.optimizer.lr, 0.005)

model.load_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_best_1.hdf5')

epochs = 10

#Training the model
history = model.fit(X_train_pooled_output, y_train_cat, validation_data=(X_test_pooled_output, y_test_cat), 
           batch_size=1000, epochs=epochs, verbose='auto', callbacks = callback_list)


model.save_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_530poch.h5')


# In[ ]:


import tensorflow.keras.backend as k
k.set_value(model.optimizer.lr, 0.001)

model.load_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_best_1.hdf5')

epochs = 10

#Training the model
history = model.fit(X_train_pooled_output, y_train_cat, validation_data=(X_test_pooled_output, y_test_cat), 
           batch_size=1000, epochs=epochs, verbose='auto', callbacks = callback_list)


model.save_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_540poch.h5')


# In[ ]:


import tensorflow.keras.backend as k
k.set_value(model.optimizer.lr, 0.001)

model.load_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_best_1.hdf5')

epochs = 10

#Training the model
history = model.fit(X_train_pooled_output, y_train_cat, validation_data=(X_test_pooled_output, y_test_cat), 
           batch_size=1000, epochs=epochs, verbose='auto', callbacks = callback_list)


model.save_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_550poch.h5')


#  

# In[ ]:


#Saving Entire the model for future use:
model.save('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_final', save_traces=True)


#Loading Model Archtecture
final_model = models.load_model('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_final', custom_objects={'auroc':auroc})

#Loading Weights from best model
final_model.load_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_best_1.hdf5')


# In[ ]:


#Loading Best Model
model.load_weights('/content/drive/MyDrive/28 NLP with Transfer Learning/Model Output/model_best_1.hdf5')


# In[1]:


#y_pred_test= np.argmax(model.predict(X_test_pooled_output), axis=1)
#print(list(y_pred_test))


# In[2]:


#print(list(y_test))


# In[ ]:


from prettytable import PrettyTable


x = PrettyTable()
x.field_names = ["Model", "#Epoch","Model Description","Train AUC", "Test AUC"]
x.add_row(["Model_best_1 ",550,"Conv + Dense ", 0.9374 , 0.9504])


print(x)


# # <Pre><font size=6>Part-6: Creating a Data pipeline for BERT Model</font> 

# <Pre><font size=3>Part-6: Creating a Data pipeline for BERT Model</font> 
# 
# 1. Download data 
# 2. Read the csv file
# 3. Remove all the html tags
# 4. Now do tokenization [Part 3 as mentioned above]
#     * Create tokens,mask array and segment array
# 5. Get Embeddings from BERT Model [Part 4 as mentioned above] , let it be X_test
#    * Print the shape of output(X_test.shape).You should get (352,768)
# 6. Predit the output of X_test with the Neural network model which we trained earlier.
# 7. Print the occurences of class labels in the predicted output
# 
# </pre>

# In[ ]:


#1. Download data from here
#2. Read the csv file
data = pd.read_csv("/content/drive/MyDrive/28 NLP with Transfer Learning/test.csv")

#3. Remove all the html tags
data = data.Text.apply(remove_html).to_list()

#4. Now do tokenization [Part 3 as mentioned above]
    #* Create tokens,mask array and segment array
#5. Get Embeddings from BERT Model [Part 4 as mentioned above] , let it be X_test
   #* Print the shape of output(X_test.shape).You should get (352,768)
if not os.path.isfile('/content/drive/MyDrive/28 NLP with Transfer Learning/test_data.pkl'):
  test_tokens, test_mask, test_segment = create_tokens(data, max_seq_length=55) 
  Test_pooled_output = bert_model.predict([test_tokens,test_mask,test_segment])
  pickle.dump((test_tokens, test_mask, test_segment, Test_pooled_output),open('/content/drive/MyDrive/28 NLP with Transfer Learning/test_data.pkl','wb'))
else:
  test_tokens, test_mask, test_segment, Test_pooled_output = pickle.load(open("/content/drive/MyDrive/28 NLP with Transfer Learning/test_data.pkl", 'rb')) 
print("The output shape is : {0} .".format(Test_pooled_output.shape))
print("--"*25)

#6. Predit the output of X_test with the Neural network model which we trained earlier.
test_predict_prob = model.predict(Test_pooled_output)
test_class_pred = test_predict_prob.argmax(axis=-1)

#7. Print the occurences of class labels in the predicted output
print("Predicted Class Labels Are :\n \n {0} ".format(test_class_pred))
print("--"*25)

print("\n Class Labels Occurances: \n 0 - {0} times \n 1 - {1} times ".format(np.bincount(test_class_pred)[0], np.bincount(test_class_pred)[1] ))


# END◼
