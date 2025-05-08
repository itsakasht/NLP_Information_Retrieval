from util import *

# Add your import statements here
from nltk.stem import *
from nltk.stem.porter import *

class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = []

		#Fill in code here

		#Porter stemmer used for stemming.
		stemmer = PorterStemmer()

		for sentence in text:
			
			temp = [stemmer.stem(token) for token in sentence]
			reducedText.append(temp)

		return reducedText


