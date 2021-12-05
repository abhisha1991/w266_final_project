########## REQUIREMENTS ##########
# Run this script inside the lm-scorer folder

# LM-SCORER
# # # # Clone the repo
# !git clone https://github.com/simonepri/lm-scorer
# !sleep 60
# # # # CD into the created folder
# # %cd lm-scorer
# # # # Create a virtualenv and install the required dependencies using poetry
# !poetry install


# Libraries
import torch
from lm_scorer.models.auto import AutoLMScorer as LMScorer

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import re

################################
# First calculate unigram product
################################
## Use lm-scorer to calculate the product of the unigram probabilities
# Available models
# list(LMScorer.supported_model_names())
# => ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", distilgpt2"]

# Load model to cpu or cuda
device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 1
scorer = LMScorer.from_pretrained("gpt2-large", device=device, batch_size=batch_size)
 

################################
# Calculate the Sentence probability based on GPT-2 LM
################################

def softmax(x):
	exps = np.exp(x)
	return np.divide(exps, np.sum(exps))
	
def cloze_finalword(text):
	'''
	This is a version of cloze generator that can handle words that are not in the model's dictionary.
	'''
	whole_text_encoding = tokenizer.encode(text)
	# Parse out the stem of the whole sentence (i.e., the part leading up to but not including the critical word)
	text_list = text.split()
	stem = ' '.join(text_list[:-1])
	stem_encoding = tokenizer.encode(stem)
	# cw_encoding is just the difference between whole_text_encoding and stem_encoding
	# note: this might not correspond exactly to the word itself
	# e.g., in 'Joe flicked the grasshopper', the difference between stem and whole text (i.e., the cw) is not 'grasshopper', but
	# instead it is ' grass','ho', and 'pper'. This is important when calculating the probability of that sequence.
	cw_encoding = whole_text_encoding[len(stem_encoding):]
	# print (cw_encoding)
	# print (whole_text_encoding)

	# Run the entire sentence through the model. Then go "back in time" to look at what the model predicted for each token, starting at the stem.
	# e.g., for 'Joe flicked the grasshopper', go back to when the model had just received 'Joe flicked the' and
	# find the probability for the next token being 'grass'. Then for 'Joe flicked the grass' find the probability that
	# the next token will be 'ho'. Then for 'Joe flicked the grassho' find the probability that the next token will be 'pper'.

	# Put the whole text encoding into a tensor, and get the model's comprehensive output
	tokens_tensor = torch.tensor([whole_text_encoding])
	
	with torch.no_grad():
		outputs = model(tokens_tensor)
		predictions = outputs[0]   

	logprobs = []
	# start at the stem and get downstream probabilities incrementally from the model(see above)
	# I should make the below code less awkward when I find the time
	start = -1-len(cw_encoding)
	for j in range(start,-1,1):
			# print (j)
			raw_output = []
			for i in predictions[-1][j]:
					raw_output.append(i.item())
	
			logprobs.append(np.log(softmax(raw_output)))
			
	# if the critical word is three tokens long, the raw_probabilities should look something like this:
	# [ [0.412, 0.001, ... ] ,[0.213, 0.004, ...], [0.002,0.001, 0.93 ...]]
	# Then for the i'th token we want to find its associated probability
	# this is just: raw_probabilities[i][token_index]
	conditional_probs = []
	for cw,prob in zip(cw_encoding,logprobs):
			# print (prob[cw])
			conditional_probs.append(prob[cw])
	# now that you have all the relevant probabilities, return their product.
	return np.exp(np.sum(conditional_probs))


# Load pre-trained model (weights) - this takes the most time
model = GPT2LMHeadModel.from_pretrained('gpt2-large', output_hidden_states = True, output_attentions = True)
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')



####
# Calculate the Sentence Log Odds Ratio (SLOR)
####

def slor(probability_sentence, probability_unigram, text):
  return (np.log(probability_sentence) - np.log(probability_unigram))/len(text.split())

def normilize_slor(list_of_slor):
  mean_value = statistics.mean(list_of_slor)

sentences = ['I love hotdogs','I like kage', 'like jam I']
                  # 3                 2           1

def sentence_slor_scores(sentences):
  slor_scores = []
  for sentence in sentences:
    
    # Compute sentence conditional prob
    sentence_score = cloze_finalword(sentence)
    # print(sentence_score)
    # print(np.log(sentence_score), "log sentence" )

    # Compute sentence score as the product of tokens' probabilities
    unigram_probs_sentence = scorer.sentence_score(sentence, reduce="prod")
    # print(unigram_probs_sentence)
    # print(np.log(unigram_probs_sentence), "log unigram")

    # Sentence Log Odds Ratio
    slor_score = slor(sentence_score, unigram_probs_sentence, sentence)
    # print(f'Text: {sentence} - SLOR: {slor_score}')

    slor_scores.append(-slor_score)
  return slor_scores
