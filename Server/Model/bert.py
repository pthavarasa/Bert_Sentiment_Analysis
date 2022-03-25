#!/usr/bin/env python
# coding: utf-8

import time
import random
import numpy as np
import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

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

        return self.classifier(last_hidden_state_cls)

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
		self.saved_model = CustomUnpickler(open('bert.pickle', 'rb')).load()
		#self.saved_model = torch.load('bert_sentiment_model', map_location=torch.device('cpu'))
		self.loss_fn = nn.CrossEntropyLoss()
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
	    return F.softmax(all_logits, dim=1).cpu().numpy()
		
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

		return 

	def set_seed(self, seed_value=42):
		"""Set seed for reproducibility.
		"""
		random.seed(seed_value)
		np.random.seed(seed_value)
		torch.manual_seed(seed_value)
		torch.cuda.manual_seed_all(seed_value)
		
	def train(self, model, train_dataloader, optimizer, scheduler, val_dataloader=None, epochs=4, evaluation=False):
		"""Train the BertClassifier model.
		"""
		# Start training loop
		print("Start training...\n")
		for epoch_i in range(epochs):
			# =======================================
			#               Training
			# =======================================
			# Print the header of the result table
			print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
			print("-"*70)

			# Measure the elapsed time of each epoch
			t0_epoch, t0_batch = time.time(), time.time()

			# Reset tracking variables at the beginning of each epoch
			total_loss, batch_loss, batch_counts = 0, 0, 0

			# Put the model into the training mode
			model.train()

			# For each batch of training data...
			for step, batch in enumerate(train_dataloader):
				batch_counts +=1
				# Load batch to GPU
				b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

				# Zero out any previously calculated gradients
				model.zero_grad()

				# Perform a forward pass. This will return logits.
				logits = model(b_input_ids, b_attn_mask)

				# Compute loss and accumulate the loss values
				loss = self.loss_fn(logits, b_labels)
				batch_loss += loss.item()
				total_loss += loss.item()

				# Perform a backward pass to calculate gradients
				loss.backward()

				# Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

				# Update parameters and the learning rate
				optimizer.step()
				scheduler.step()

				# Print the loss values and time elapsed for every 20 batches
				if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
					# Calculate time elapsed for 20 batches
					time_elapsed = time.time() - t0_batch

					# Print training results
					print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

					# Reset batch tracking variables
					batch_loss, batch_counts = 0, 0
					t0_batch = time.time()

			# Calculate the average loss over the entire training data
			avg_train_loss = total_loss / len(train_dataloader)

			print("-"*70)
			# =======================================
			#               Evaluation
			# =======================================
			if evaluation == True:
				# After the completion of each training epoch, measure the model's performance
				# on our validation set.
				val_loss, val_accuracy = self.evaluate(model, val_dataloader)

				# Print performance over the entire training data
				time_elapsed = time.time() - t0_epoch
				
				print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
				print("-"*70)
			print("\n")
		
		print("Training complete!")
		
	def evaluate(self, model, val_dataloader):
		"""After the completion of each training epoch, measure the model's performance
		on our validation set.
		"""
		# Put the model into the evaluation mode. The dropout layers are disabled during
		# the test time.
		model.eval()

		# Tracking variables
		val_accuracy = []
		val_loss = []

		# For each batch in our validation set...
		for batch in val_dataloader:
			# Load batch to GPU
			b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

			# Compute logits
			with torch.no_grad():
				logits = model(b_input_ids, b_attn_mask)

			# Compute loss
			loss = self.loss_fn(logits, b_labels)
			val_loss.append(loss.item())

			# Get the predictions
			preds = torch.argmax(logits, dim=1).flatten()

			# Calculate the accuracy rate
			accuracy = (preds == b_labels).cpu().numpy().mean() * 100
			val_accuracy.append(accuracy)

		# Compute the average accuracy and loss over the validation set.
		val_loss = np.mean(val_loss)
		val_accuracy = np.mean(val_accuracy)

		return val_loss, val_accuracy
		
	def initialize_model(self, train_dataloader, epochs=4):
		"""Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
		"""
		# Instantiate Bert Classifier
		bert_classifier = BertClassifier(freeze_bert=False)

		# Tell PyTorch to run the model on GPU
		bert_classifier.to(self.device)

		# Create the optimizer
		optimizer = AdamW(bert_classifier.parameters(),
						  lr=5e-5,    # Default learning rate
						  eps=1e-8    # Default epsilon value
						  )

		# Total number of training steps
		total_steps = len(train_dataloader) * epochs

		# Set up the learning rate scheduler
		scheduler = get_linear_schedule_with_warmup(optimizer,
													num_warmup_steps=0, # Default value
													num_training_steps=total_steps)
		return bert_classifier, optimizer, scheduler
	
	def train_model(self, data_path, model_save_path):
		df = pd.read_csv(data_path, sep = ';', skiprows=1, header=None)
		df.dropna(inplace=True)
		df.reset_index(drop=True, inplace=True)
		
		df[2].replace({"positive": 1, "negative": 0}, inplace=True)
		
		X = df[1][:].values
		y = df[2][:].values

		X_train, X_val, y_train, y_val =\
			train_test_split(X, y, test_size=0.1, random_state=2020)
		
		train_inputs, train_masks = self.preprocessing_for_bert(X_train)
		val_inputs, val_masks = self.preprocessing_for_bert(X_val)
		
		# Convert other data types to torch.Tensor
		train_labels = torch.tensor(y_train)
		val_labels = torch.tensor(y_val)

		# For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
		batch_size = 32

		# Create the DataLoader for our training set
		train_data = TensorDataset(train_inputs, train_masks, train_labels)
		train_sampler = RandomSampler(train_data)
		train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

		# Create the DataLoader for our validation set
		val_data = TensorDataset(val_inputs, val_masks, val_labels)
		val_sampler = SequentialSampler(val_data)
		val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
		
		# Concatenate the train set and the validation set
		full_train_data = torch.utils.data.ConcatDataset([train_data, val_data])
		full_train_sampler = RandomSampler(full_train_data)
		full_train_dataloader = DataLoader(full_train_data, sampler=full_train_sampler, batch_size=32)

		# Train the Bert Classifier on the entire training data
		self.set_seed(42)
		bert_classifier, optimizer, scheduler = self.initialize_model(train_dataloader, epochs=2)
		self.train(bert_classifier, full_train_dataloader, optimizer, scheduler, epochs=2)
		torch.save(bert_classifier, model_save_path)

	def predict(self, text):
	    test_inputs, test_masks = self.preprocessing_for_bert(np.array([text]))

	    # Create the DataLoader for our test set
	    test_dataset = TensorDataset(test_inputs, test_masks)
	    test_sampler = SequentialSampler(test_dataset)
	    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)


	    probs = self.bert_predict(self.saved_model, test_dataloader)
	    sentiment = "Positif" if np.argmax(probs, axis=1).tolist()[0] else "Negatif"


	    #return np.argmax(probs, axis=1).tolist()[0]
		#return str(np.argmax(probs, axis=1).tolist()[0]) +","+ str(probs.max())
	    return {"text": text, "sentiment": sentiment, "precision": float(probs.max())}