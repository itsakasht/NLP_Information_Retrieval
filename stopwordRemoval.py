from util import *

# Add your import statements here
from nltk.corpus import stopwords

class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = []

		#Fill in code here
		stopwords_english = stopwords.words('english')
		
		for sentence in text:
			#Stopwords checked and removed.
			filtered_sentence = [w for w in sentence if not w.lower() in stopwords_english]
			stopwordRemovedText.append(filtered_sentence)

		return stopwordRemovedText




	