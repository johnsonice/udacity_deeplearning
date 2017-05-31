
# coding: utf-8

# ### Generate Wiki Text 

# ##### Download data from https://metamind.io/research/the-wikitext-long-term-dependency-language-modeling-dataset/

# In[1]:

#mini-demo
from urllib.request import urlretrieve
import os 
from os.path import isfile, isdir
import zipfile 
from tqdm import tqdm
import numpy as np #vectorization
import random #generate probability distribution 
import tensorflow as tf #ml
import datetime #clock training time


# ### First download data 

# In[2]:

#### process bar
class DLProgress(tqdm):
    last_block = 0
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

## download file 
data_path = './wikitext'
if isdir(data_path):
    print('Data already exist')
else:
    if not isdir(data_path):
        os.mkdir(data_path)
    zip_file = os.path.join(data_path,'wikitext-103-v1.zip')
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='wikidata') as pbar:
        #urlretrieve('https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip',
        #            zip_file,
        #            pbar.hook)
        urlretrieve('https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip',
                    zip_file,
                    pbar.hook)
    with zipfile.ZipFile(os.path.join(data_path,'wikitext-103-v1.zip')) as myzip:
        myzip.extractall(data_path)
    ## remove zip file 
    os.remove(data_path+'/wikitext-103-v1.zip')

data_file_path = "./wikitext/wikitext-2"
train_file = os.path.join(data_file_path,'wiki.train.tokens')
validate_file = os.path.join(data_file_path,'wiki.valid.tokens')


# ### Read data

# In[3]:

#lets open the text
#native python file read function
text = open(train_file,encoding='utf8').read()
print('text length in number of characters:', len(text))
print('head of text:')
print(text[:1000]) #all tokenized words, stored in a list called text


# #### Create a id to character and character to id map dictionary

# In[4]:

## get the set of characters and sort them 
chars = sorted(list(set(text)))               ## all unique characters
char_size = len(chars)
print('number of characters:', char_size)
print(chars[:20])


# In[5]:

## chrate char to id and id to char map 
char2id = {c:i for i,c in enumerate(chars)}
id2char = {i:c for i,c in enumerate(chars)}


# In[6]:

#Given a probability of each character, return a likely character, one-hot encoded
#our prediction will give us an array of probabilities of each character
#we'll pick the most likely and one-hot encode it
def sample(prediction):
    '''
    prediction: is a list of characters probilities
    '''
    r = random.uniform(0,1)  ## it is just a random number from 0-1
    s = 0 
    char_id = len(prediction)-1  ## this is because it starts with 0
    #for each char prediction probability 
    for i in range(len(prediction)):
        s+= prediction[i]
        if s >= r:
            char_id = i 
            break 
    
    char_one_hot = np.zeros(shape[char_size])  ## one hot encode characters 
    char_one_hot[char_id] = 1.0
    return char_one_hot


# #### Create X and y sets and one hot encode them  

# In[7]:

## 
#vectorize our data to feed it into model
len_per_section = 50
skip = 2
sections = []
next_chars = []
#fill sections list with chunks of text, every 2 characters create a new 50 
#character long section
#because we are generating it at a character level
for i in range(0, len(text) - len_per_section, skip):
    sections.append(text[i: i + len_per_section])
    next_chars.append(text[i + len_per_section])
#Vectorize input and output
#matrix of section length by num of characters
X = np.zeros((len(sections), len_per_section, char_size))
#label column for all the character id's, still zero
y = np.zeros((len(sections), char_size))


# In[8]:

len(sections)
import sys


# this encoding method is not very good. takes a lot of memory. we may want to do it in tensorflow. 
# That is how we did it in 0_basic lstm notebook 

# In[18]:

#for each char in each section, convert each char to an ID
#for each section convert the labels to ids 

for i, section in enumerate(sections):
    for j, char in enumerate(section):
        X[i, j, char2id[char]] = 1
    y[i, char2id[next_chars[i]]] = 1
print(y)


# In[ ]:



