from util import *

# Add your import statements here
from nltk.tokenize.treebank import TreebankWordTokenizer



class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		#Fill in code here

		#For each sentence
		for sentence in text:
			tokens = []
			token = ""
			sentence = sentence.strip()

			count = 0

			#Going through the sentence character wise
			for c in sentence:
				
				count+=1
					 
				if c!=" ":
					token+=c

				else:
					tokens.append(token)
					token = ""

				if count==len(sentence):
					tokens.append(token)
					token = ""

			tokenizedText.append(tokens)

		return tokenizedText


	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		#Fill in code here
		t = TreebankWordTokenizer()

		#Using the tokenizer
		for sentence in text:
			sentence = sentence.strip()
			tokens = t.tokenize(sentence)
			tokenizedText.append(tokens)

		return tokenizedText