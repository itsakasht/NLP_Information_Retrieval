from util import *

# Add your import statements here
from nltk.tokenize import PunktTokenizer

class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = []

		#Fill in code here
		
		#A simplistic top-down approach would be to segment sentences based on the tokens ‘.’, ‘!’ and ‘?’. 
		sentence = ""
		count = 0

		#Going character-wise to isolate sentences.
		for c in text:
			count+=1
			sentence = sentence + c

			if c=='!' or c=='.' or c=='?' or count == len(text):
				segmentedText.append(sentence)
				sentence = ""

		return segmentedText

	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		segmentedText = []

		#Fill in code here
		
		#Using the Punkt Tokenizer.
		sentence_punkt = PunktTokenizer()

		segmentedText = sentence_punkt.tokenize(text.strip())
		
		return segmentedText