# -*- coding: utf-8 -*-
"""w266_project_baselineMDL.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1D8ytcv_DzVMGCur3OknagfefEfPGXGhT

# Required installs
"""

!pip install -q transformers
!pip install -q sentencepiece

"""# Imports"""

import tensorflow as tf
import tensorflow_datasets as tfds
import transformers

from transformers import PegasusTokenizer, TFPegasusModel, TFPegasusForConditionalGeneration
from transformers import T5Tokenizer, TFT5Model, TFT5ForConditionalGeneration

"""# Utility functions

Wrap text around for aesthetics.
"""

from IPython.display import HTML, display

def set_css():
  display(HTML('''
  <style>
    pre {
        white-space: pre-wrap;
    }
  </style>
  '''))
get_ipython().events.register('pre_run_cell', set_css)

"""# CNN/DM dataset loading

Download and load raw data. Data is in binary format in a tf.Dadaset structure --> we process it later
"""

data, info = tfds.load('cnn_dailymail', with_info=True)

"""Extract train, val, and test data"""

train_data, val_data, test_data = data['train'], data['validation'], data['test']

"""The datasets have each an "article" and the "highlight", let's extract here only the articles that we want to summarize"""

X_train = train_data.map(lambda x: x['article'])
X_val = val_data.map(lambda x: x['article'])
X_test = test_data.map(lambda x: x['article'])

"""Print a few example articles"""

num_example = 2
for c, elem in enumerate(X_train):
  print(elem.numpy().decode())
  print('\n')
  if c>=num_example-1:
    print('--------------------')
    print('Each element of X_train is a:', elem.dtype)
    break

"""# Pegasus model

Download and create the model, get associated tokenizer
"""

pegasus_model = TFPegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
pegasus_tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')

"""Check model summary (pretty impressive!)"""

pegasus_model.summary()

"""Construct list of articles to summarize and tokenize"""

num_articles = 2
summarize = []
for c, elem in enumerate(X_train):
  summarize.append(elem.numpy().decode())
  if c>=num_example-1:
    break

pegasus_inputs = pegasus_tokenizer(summarize, return_tensors='tf', padding=True)
print('There are', pegasus_inputs['input_ids'].shape[0], 'articles of length', pegasus_inputs['input_ids'].shape[1], 'to summarize')

"""Run inference, use the model on CNN/DM data to summarize inputs articles"""

pegasus_summary_ids = pegasus_model.generate(pegasus_inputs['input_ids'], min_length=100, num_beams=5, no_repeat_ngram_size=1)

print([pegasus_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
       for g in pegasus_summary_ids])

"""# T5 Model

Download and create the model, get associated tokenizer
"""

t5_model = TFT5ForConditionalGeneration.from_pretrained('t5-large')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-large')

"""Check model summary"""

t5_model.summary()

"""Tokenize the list of articles + add 'summarize:' because T5 needs to know the task"""

num_articles = 2
summarize = []
for c, elem in enumerate(X_train):
  summarize.append('summarize: ' + elem.numpy().decode())
  if c>=num_example-1:
    break

t5_inputs = t5_tokenizer(summarize, return_tensors='tf', padding=True)
print('There are', t5_inputs['input_ids'].shape[0], 'articles of length', t5_inputs['input_ids'].shape[1], 'to summarize')

"""Run inference, use the model on CNN/DM data to summarize inputs articles"""

t5_summary_ids = t5_model.generate(t5_inputs['input_ids'], num_beams=3, no_repeat_ngram_size=1)

print([t5_tokenizer.decode(g, skip_special_tokens=True, 
                           clean_up_tokenization_spaces=False) for g in t5_summary_ids])