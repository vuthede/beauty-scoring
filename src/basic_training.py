
# coding: utf-8

# In[2]:


import cv2
from os.path import isfile, isdir, join
from os import listdir
import numpy as np


# # 1.Train 60, test 40 method

# ## 1.1 Load data

# In[10]:


DATA_ROOT_PATH = "../data/beauty-score-data"
IMAGES_PATH = join(DATA_ROOT_PATH, "Images")
TRAIN_PATH = join(DATA_ROOT_PATH, "train_test_files", "split_of_60%training and 40%testing", "train.txt")
TEST_PATH = join(DATA_ROOT_PATH, "train_test_files", "split_of_60%training and 40%testing", "test.txt")


train_image_paths = []
train_labels = []
valid_image_paths = []
valid_labels = []

with open(TRAIN_PATH, "r") as f:
    for line in f:
        file_name, label = line.split()
        train_image_paths.append(join(IMAGES_PATH, file_name))
        train_labels.append(float(label))

with open(TEST_PATH, "r") as f:
    for line in f:
        file_name, label = line.split()
        valid_image_paths.append(join(IMAGES_PATH, file_name))
        valid_labels.append(float(label))
        
train_image_paths = np.array(train_image_paths)
train_labels = np.array(train_labels)
valid_image_paths = np.array(valid_image_paths)
valid_labels = np.array(valid_labels)


# ## 1.2 Model

# In[6]:


from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Reshape, Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.metrics import confusion_matrix
import random
import timeit
from sklearn.utils import class_weight
from keras import optimizers

batch_size = 50
width_image = 256
channels = 3 #RGB
num_epochs = 200



def generate_batch_data(data_in, data_out, batch_size):
    while True:
        for i in range(0, len(data_in), batch_size):
            x, y = process_data(data_in[i:i+batch_size], data_out[i:i+batch_size])
            x = x / 255.0
            yield (x, y)
            

##Function to transform these above array into data used for training and testing
def process_data(image_paths, labels):
    input = []
    output = []
    for path,label in zip(image_paths, labels):
        try:
            img = cv2.imread(path)
            img = cv2.resize(img, (256, 256))
        except:
            print("error in fikle:", path)
        input.append(img)
        output.append(label)
    
    input = np.array(input)
    output = np.array(output)
    return input, output


model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1),activation='relu', input_shape=(width_image, width_image, channels)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='sgd')
    
valid_x, valid_y = process_data(valid_image_paths, valid_labels)
model.fit_generator(generate_batch_data(train_image_paths, train_labels, batch_size),
                    steps_per_epoch = int(len(train_image_paths) / batch_size), epochs=num_epochs, 
                    validation_data = (valid_x, valid_y), shuffle=True)


# In[ ]:




