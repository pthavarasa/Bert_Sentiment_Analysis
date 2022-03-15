#!/usr/bin/env python
# coding: utf-8


import re
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertModel
from transformers import BertTokenizer

import sys
import pickle


MAX_LEN = 64

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)

# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits

class Bert():
	def __init__(self):
		# Load the BERT tokenizer
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
		if torch.cuda.is_available():       
			self.device = torch.device("cuda")
			#print(f'There are {torch.cuda.device_count()} GPU(s) available.')
			#print('Device name:', torch.cuda.get_device_name(0))

		else:
			#print('No GPU available, using the CPU instead.')
			self.device = torch.device("cpu")
		self.saved_model = CustomUnpickler(open('bert_sentiment_model.pickle', 'rb')).load()
		#self.saved_model = torch.load('bert_sentiment_model', map_location=torch.device('cpu'))
	# Create a function to tokenize a set of texts
	def preprocessing_for_bert(self, data):
		"""Perform required preprocessing steps for pretrained BERT.
		@param    data (np.array): Array of texts to be processed.
		@return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
		@return   attention_masks (torch.Tensor): Tensor of indices specifying which
					  tokens should be attended to by the model.
		"""
		# Create empty lists to store outputs
		input_ids = []
		attention_masks = []

		# For every sentence...
		for sent in data:
			# `encode_plus` will:
			#    (1) Tokenize the sentence
			#    (2) Add the `[CLS]` and `[SEP]` token to the start and end
			#    (3) Truncate/Pad sentence to max length
			#    (4) Map tokens to their IDs
			#    (5) Create attention mask
			#    (6) Return a dictionary of outputs
			encoded_sent = self.tokenizer.encode_plus(
				text=self.text_preprocessing(sent),  # Preprocess sentence
				add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
				max_length=MAX_LEN,                  # Max length to truncate/pad
				pad_to_max_length=True,         # Pad sentence to max length
				#return_tensors='pt',           # Return PyTorch tensor
				return_attention_mask=True      # Return attention mask
				)
			
			# Add the outputs to the lists
			input_ids.append(encoded_sent.get('input_ids'))
			attention_masks.append(encoded_sent.get('attention_mask'))

		# Convert lists to tensors
		input_ids = torch.tensor(input_ids)
		attention_masks = torch.tensor(attention_masks)

		return input_ids, attention_masks



	def bert_predict(self, model, test_dataloader):
		"""Perform a forward pass on the trained BERT model to predict probabilities
		on the test set.
		"""
		# Put the model into the evaluation mode. The dropout layers are disabled during
		# the test time.
		model.eval()

		all_logits = []

		# For each batch in our test set...
		for batch in test_dataloader:
			# Load batch to GPU
			b_input_ids, b_attn_mask = tuple(t.to(self.device) for t in batch)[:2]

			# Compute logits
			with torch.no_grad():
				logits = model(b_input_ids, b_attn_mask)
			all_logits.append(logits)
		
		# Concatenate logits from each batch
		all_logits = torch.cat(all_logits, dim=0)

		# Apply softmax to calculate probabilities
		probs = F.softmax(all_logits, dim=1).cpu().numpy()

		return probs



	def text_preprocessing(self, text):
		"""
		- Remove entity mentions (eg. '@united')
		- Correct errors (eg. '&amp;' to '&')
		@param    text (str): a string to be processed.
		@return   text (Str): the processed string.
		"""
		# Remove '@name'
		text = re.sub(r'(@.*?)[\s]', ' ', text)

		# Replace '&amp;' with '&'
		text = re.sub(r'&amp;', '&', text)

		# Remove trailing whitespace
		text = re.sub(r'\s+', ' ', text).strip()

		return text



	# Run `preprocessing_for_bert` on the test set
	#print('Tokenizing data...')
	def predict(self, text):
		test_inputs, test_masks = self.preprocessing_for_bert(np.array([text]))

		# Create the DataLoader for our test set
		test_dataset = TensorDataset(test_inputs, test_masks)
		test_sampler = SequentialSampler(test_dataset)
		test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)


		probs = self.bert_predict(self.saved_model, test_dataloader)


		#return np.argmax(probs, axis=1).tolist()[0]
		return str(np.argmax(probs, axis=1).tolist()[0]) +","+ str(probs.max())

#m = MyModel()
#print(m.predict("good"))
