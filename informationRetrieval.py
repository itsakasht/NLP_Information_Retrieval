# from util import *

# # Add your import statements here


# class InformationRetrieval():

# 	def __init__(self):
# 		self.index = None

# 	def buildIndex(self, docs, docIDs):
# 		"""
# 		Builds the document index in terms of the document
# 		IDs and stores it in the 'index' class variable

# 		Parameters
# 		----------
# 		arg1 : list
# 			A list of lists of lists where each sub-list is
# 			a document and each sub-sub-list is a sentence of the document
# 		arg2 : list
# 			A list of integers denoting IDs of the documents
# 		Returns
# 		-------
# 		None
# 		"""

		

# 		#Fill in code here
# 		# index = {}
# 		# counter = 0
# 		# for doc in docs:
# 		# 	index[docIDs[counter]] = doc
# 		# 	counter+=1

# 		# self.index = index

# 		# index here is going to be a dictionary


# 	def rank(self, queries):
# 		"""
# 		Rank the documents according to relevance for each query

# 		Parameters
# 		----------
# 		arg1 : list
# 			A list of lists of lists where each sub-list is a query and
# 			each sub-sub-list is a sentence of the query
		

# 		Returns
# 		-------
# 		list
# 			A list of lists of integers where the ith sub-list is a list of IDs
# 			of documents in their predicted order of relevance to the ith query
# 		"""

# 		doc_IDs_ordered = []

# 		#Fill in code here
# 		for query in queries:

# 			continue
	
# 		return doc_IDs_ordered

# from util import *
import math

class InformationRetrieval():

    def __init__(self):
        self.index = {}
        self.docIDs = []
        self.doc_vectors = {}
        self.idf = {}
        self.doc_norms = {}

    def buildIndex(self, docs, docIDs):
        index = {}
        self.docIDs = docIDs
        total_docs = len(docIDs)

        # Flatten each document into a list of tokens
        for idx, doc in enumerate(docs):
            doc_tokens = []
            for sentence in doc:
                doc_tokens.extend(sentence)

            term_freq = {}
            for term in doc_tokens:
                if term not in term_freq:
                    term_freq[term] = 0
                term_freq[term] += 1

            for term in term_freq:
                if term not in index:
                    index[term] = {}
                index[term][docIDs[idx]] = term_freq[term]

        # Compute IDF for each term
        for term in index:
            df = len(index[term])
            self.idf[term] = math.log10(total_docs / df)

        # Precompute document vectors and norms
        for idx, doc in enumerate(docs):
            doc_tokens = []
            for sentence in doc:
                doc_tokens.extend(sentence)

            term_freq = {}
            for term in doc_tokens:
                if term not in term_freq:
                    term_freq[term] = 0
                term_freq[term] += 1

            vec = {}
            norm = 0

            for term, freq in term_freq.items():
                tfidf = freq * self.idf.get(term, 0)
                vec[term] = tfidf
                norm += tfidf ** 2

            norm = math.sqrt(norm)
            self.doc_vectors[docIDs[idx]] = vec
            self.doc_norms[docIDs[idx]] = norm

        self.index = index

    def rank(self, queries):
        doc_IDs_ordered = []

        for query in queries:
            query_tokens = []
            for sentence in query:
                query_tokens.extend(sentence)

            query_tf = {}
            for term in query_tokens:
                if term not in query_tf:
                    query_tf[term] = 0
                query_tf[term] += 1

            query_vec = {}
            for term, freq in query_tf.items():
                if term in self.idf:
                    query_vec[term] = freq * self.idf[term]

            query_norm = math.sqrt(sum(val**2 for val in query_vec.values()))

            scores = {}
            for docID in self.docIDs:
                numerator = 0
                doc_vec = self.doc_vectors[docID]

                for term in query_vec:
                    if term in doc_vec:
                        numerator += query_vec[term] * doc_vec[term]

                denominator = self.doc_norms[docID] * query_norm

                if denominator != 0:
                    scores[docID] = numerator / denominator
                else:
                    scores[docID] = 0

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ranked_docIDs = [docID for docID, score in ranked]
            doc_IDs_ordered.append(ranked_docIDs)

        return doc_IDs_ordered


