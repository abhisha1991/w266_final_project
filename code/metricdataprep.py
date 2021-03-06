# -*- coding: utf-8 -*-
"""MetricDataPrep.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1l7Y4trxWsNjyuqPXepRo7BMletoIHXlT

# NLP With Deep Learning (W266)

Submission by *Carolina Arriaga, Ayman, Abhi Sharma*

Winter 2021 | UC Berkeley

## Notebook Overview

This notebook contains the data prep needed by the team to be able to draw conclusions on the relationship between `coherence, fluency, consistency, relevance` and the metrics that were authored in the `SummaryScorer` notebook.

References list:

https://arxiv.org/pdf/2007.12626.pdf

https://github.com/Yale-LILY/SummEval

# Data Prep
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content
!pwd

"""## Utilities"""

import numpy as np
import pandas as pd

def get_link_for_model(num):
  return 'https://storage.googleapis.com/sfr-summarization-repo-research/M{}.tar.gz'.format(num)

def get_link_for_human_annotation():
  return 'https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl'
  
# there are 24 models - 0 to 23 as per the link here
# https://github.com/Yale-LILY/SummEval#model-outputs
total_models = 24

def get_links_for_all_models():
  links = ''
  for i in range(total_models):
    links += get_link_for_model(i) + ' '
  return links

!wget {get_link_for_human_annotation()}

!wget {get_links_for_all_models()}

!for f in *.tar.gz; do tar -xf "$f"; done

from os import listdir
from os.path import isfile, join

def get_jsonl_files_for_model(model_num):
  assert type(model_num) == int
  path = "M{}/aligned".format(model_num)
  files = [f for f in listdir(path) if isfile(join(path, f))]
  return [path + '/' + f for f in files if f.endswith('jsonl')]

get_jsonl_files_for_model(23)

import json

def get_model_result_list(model_num, all_model_variants=True):
  file_list = get_jsonl_files_for_model(model_num)
  assert len(file_list) > 0
  if not all_model_variants:
    file_list = [file_list[0]]
  
  corrupted_val = "cnndm/dailymail/stories/9f270039c861e75ee2f01e4e2898a9ea04a96b26.story"
  data = []  
  for f in file_list:
    with open(f, 'r') as jsonl_file:
        json_list = list(jsonl_file)

    for json_str in json_list:
        model_result = json.loads(json_str)
        model_result['model_id'] = 'M' + str(model_num)
        model_result['model_variant'] = f
        # there is a single corrupted record in model #2, so we remove that
        if model_num == 2 and model_result['filepath'] == corrupted_val:
          continue
        data.append(model_result)
  
  return data

def get_all_model_data(all_model_variants=True):
  data = []
  for i in range(total_models):
    data_for_model = get_model_result_list(i, all_model_variants)
    # verify keys in result for every model
    first = data_for_model[0]
    assert 'reference' in first.keys()
    assert 'decoded' in first.keys()
    assert 'id' in first.keys()
    assert 'filepath' in first.keys()
    assert 'model_id' in first.keys()
    assert 'model_variant' in first.keys()
    data.extend(data_for_model)
  return data

"""## Model Summaries"""

import pandas as pd

data = get_all_model_data()
model_summ = pd.DataFrame(data)
model_summ.head()

model_summ.shape

# check for null and empty vals in df
np.where(pd.isnull(model_summ))

# check for null and empty vals in df
np.where(model_summ.applymap(lambda x: x == ''))

# there are some rows with empty summary outputs from the model
# we will keep these as is as this is what the model's natural output is
model_summ.iloc[23072]

"""## Annotator Data"""

def get_annotation_data(with_mturk=False):
  with open('/content/model_annotations.aligned.jsonl', 'r') as json_file:
    json_list = list(json_file)

  data = []
  for json_str in json_list:
      annotation = json.loads(json_str)
      result = {}
      result['id'] = annotation['id']
      result['model_id'] = annotation['model_id']
      result['decoded'] = annotation['decoded']
      result['filepath'] = annotation['filepath']

      # there are 3 expert and 5 mturk outputs
      expert = 'expert_annotations'
      turk = 'turker_annotations'
      assert len(annotation[expert]) == 3
      assert len(annotation[turk]) == 5

      dims =  ["coherence", "consistency", "fluency", "relevance"]
      
      ### add expert individual and avg scores ###
      # go through each dim
      for d in dims:
        summ = 0
        # go through each expert
        for e in range(len(annotation[expert])):
          assert d in annotation[expert][e].keys()

          result['expert_{}_{}'.format(e, d)] = annotation[expert][e][d]
          summ += annotation[expert][e][d]
        
        result['all_expert_avg_{}'.format(d)] = 1.0 * summ/len(annotation[expert])

      if with_mturk:
        ### add turk individual and avg scores ###
        # go through each dim
        for d in dims:
          summ = 0
          # go through each turk
          for t in range(len(annotation[turk])):
            assert d in annotation[turk][t].keys()

            result['turk_{}_{}'.format(t, d)] = annotation[turk][t][d]
            summ += annotation[turk][t][d]
          
          result['all_turk_avg_{}'.format(d)] = 1.0 * summ/len(annotation[turk])

      data.append(result)
  return data

import pandas as pd

data = get_annotation_data(with_mturk=True)
annotations = pd.DataFrame(data)
annotations.head()

annotations.shape

# check for null and empty vals in df
np.where(pd.isnull(annotations))

# check for null and empty vals in df
np.where(annotations.applymap(lambda x: x == ''))

# note that annotations don't have all 24 models present in them
annotations[['model_id', 'id']].groupby(['model_id']).count()

"""## Join Annotator Data & Model Summaries

Key for join: combination of `id` and `model_id`
"""

joined = pd.merge(annotations, model_summ, on = ['id', 'model_id'])

# the reason we have more than 1600 rows here is because of the model variants
# we will filter these later where the variant's decoded should equal the annotation's decoded
joined.shape

np.where(pd.isnull(joined))

np.where(joined.applymap(lambda x: x == ''))

joined.head()

# models that were in the model summaries but were not in the annotations 
# this shows that the annotations didn't include some models
assert len(list(model_summ.model_id.unique())) == total_models
missing_models = [x for x in list(model_summ.model_id.unique()) if x not in list(joined.model_id.unique())] 
print(missing_models)
# there were 16 models remaining in the annotations, for which we have 16x100 = 1600 data points 
print(total_models - len(missing_models))

# the annotation's summary output for the model should match the model variant's summary output
# only keep those rows where it matches as the annotation was done only for that variant
joined = joined.loc[joined['decoded_x'] == joined['decoded_y']]

# we see the rows decrease because the scoring (annotation) was done on a subset of the variants of a single model
joined.shape

# shows that some variants of a single model produced the same summary as another variant of the same model
# for example, 38 examples of M5's variant produced the same summary output as the baseline M5 model 
joined[['model_id', 'model_variant']].value_counts()

# clean up dataframe and store for future use
joined = joined.rename(columns={"id": "story_id", "decoded_x": "decoded"})
joined = joined.drop(columns=['filepath_x', 'filepath_y', 'decoded_y'])

joined.head()

"""# Store Data as CSV



"""

import datetime
from google.colab import files

now = datetime.datetime.now()
filename = now.strftime("%Y-%m-%d-%H-%M-%S")

compression_opts = dict(method='zip', archive_name='data.csv')

joined.to_csv('{}.zip'.format(filename), index=False, compression = compression_opts)
files.download('{}.zip'.format(filename))

"""# Misc Utils"""

!pip install -q datasets

all_articles = datasets.load_dataset("cnn_dailymail", "3.0.0")

import datasets

def get_cnndm_by_id(dataset, id, return_article_only=True):
  id = id.replace('dm-test-', '')
  id = id.replace('dm-train-', '')
  id = id.replace('dm-dev-', '')
  id = id.replace('dm-val-', '')

  id = id.replace('cnn-test-', '')
  id = id.replace('cnn-train-', '')
  id = id.replace('cnn-dev-', '')
  id = id.replace('cnn-val-', '')
  try:
    highlight = dataset.filter(lambda x: x['id'] == id)['highlights'][0]
    article = dataset.filter(lambda x: x['id'] == id)['article'][0]
  except:
    return None
  if return_article_only:
    return article
    
  return article, highlight

id = 'fbbafa743a8c2ecd2cedf65c6c61956b2db8ec5c'
print(get_cnndm_by_id(all_articles['test'], id))

