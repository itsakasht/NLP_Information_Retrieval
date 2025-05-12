from util import *

# Add your import statements here
import math
import numpy as np

class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		num = set(query_doc_IDs_ordered[:k]).intersection(set(true_doc_IDs))
		precision = len(num)/k

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1

		#Fill in code here
		# Store the precision at k for each query
		query_precision = []
		qrels = build_qrels_dict(qrels)

		# Iterate over each query
		for i in range(len(query_ids)):
			query_id = query_ids[i]
			# Get the list of relevant document IDs for the current query from qrels
			query_docs = qrels.get(query_id, {})
			relevant_docs = {}

			for doc_id in query_docs:
				if query_docs[doc_id] > 0:  # Assume relevance > 0 is relevant
					relevant_docs[doc_id] = query_docs[doc_id]
		
	
			# Get the document IDs from the ordered list of predicted docs
			query_doc_IDs_ordered = doc_IDs_ordered[i]

			precision_at_k = self.queryPrecision(query_doc_IDs_ordered, query_id, list(relevant_docs.keys()), k)
			query_precision.append(precision_at_k)

		# Calculate the Mean Precision over all queries
		meanPrecision = sum(query_precision) / len(query_precision) if query_precision else 0

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		num = set(query_doc_IDs_ordered[:k]).intersection(set(true_doc_IDs))
		recall = len(num)/len(true_doc_IDs)

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		#Fill in code here
		# Store the recall at k for each query
		query_recall = []
		qrels = build_qrels_dict(qrels)

		# Iterate over each query
		for i in range(len(query_ids)):
			query_id = query_ids[i]
			# Get the list of relevant document IDs for the current query from qrels
			query_docs = qrels.get(query_id, {})
			relevant_docs = {}

			for doc_id in query_docs:
				if query_docs[doc_id] > 0:  # Assume relevance > 0 is relevant
					relevant_docs[doc_id] = query_docs[doc_id]
		
	
			# Get the document IDs from the ordered list of predicted docs
			query_doc_IDs_ordered = doc_IDs_ordered[i]

			recall_at_k = self.queryRecall(query_doc_IDs_ordered, query_id, list(relevant_docs.keys()), k)
			query_recall.append(recall_at_k)

		# Calculate the Mean Recall over all queries
		meanRecall = sum(query_recall) / len(query_recall) if query_recall else 0

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1
		
		#Fill in code here
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

		if precision + recall == 0:
			fscore = 0
		else:
			fscore = 1.25*precision*recall/(0.25*precision + recall)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here

		query_metric = []
		qrels = build_qrels_dict(qrels)

		# Iterate over each query
		for i in range(len(query_ids)):
			query_id = query_ids[i]
			# Get the list of relevant document IDs for the current query from qrels
			query_docs = qrels.get(query_id, {})
			relevant_docs = {}

			for doc_id in query_docs:
				if query_docs[doc_id] > 0:  # Assume relevance > 0 is relevant
					relevant_docs[doc_id] = query_docs[doc_id]
		
	
			# Get the document IDs from the ordered list of predicted docs
			query_doc_IDs_ordered = doc_IDs_ordered[i]

			metric_at_k = self.queryFscore(query_doc_IDs_ordered, query_id, list(relevant_docs.keys()), k)
			query_metric.append(metric_at_k)

		# Calculate the mean fscore over all queries
		meanFscore = sum(query_metric) / len(query_metric) if query_metric else 0

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_docs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1

		#Fill in code here
		relevance = np.zeros(k)

		#get the relevance of extracted documents
		for i, doc_id in enumerate(query_doc_IDs_ordered[:k]):
			relevance[i] = true_docs.get(doc_id, 0)

		relevance_descending = sorted(relevance, reverse=True)

		DCG = 0
		IDCG = 0
		
		#Calculate DCG and IDCG
		for i in range(k):
			DCG += relevance[i]/math.log2(i+2)
			IDCG += relevance_descending[i]/math.log2(i+2)

		if IDCG!=0:
			nDCG = DCG/IDCG
		else:
			nDCG = 0
		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		#Fill in code here
		query_metric = []
		qrels = build_qrels_dict(qrels)

		# Iterate over each query
		for i in range(len(query_ids)):
			query_id = query_ids[i]
			# Get the list of relevant document IDs for the current query from qrels
			query_docs = qrels.get(query_id, {})
			relevant_docs = {}

			for doc_id in query_docs:
				if query_docs[doc_id] > 0:  # Assume relevance > 0 is relevant
					relevant_docs[doc_id] = query_docs[doc_id]
		
	
			# Get the document IDs from the ordered list of predicted docs
			query_doc_IDs_ordered = doc_IDs_ordered[i]

			metric_at_k = self.queryNDCG(query_doc_IDs_ordered, query_id, relevant_docs, k)
			query_metric.append(metric_at_k)

		# Calculate the Mean nDCG over all queries
		meanNDCG = sum(query_metric) / len(query_metric) if query_metric else 0

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1

		#Fill in code here
		precision_list = []

		for i in range(1, k+1):
			precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i)
			precision_list.append(precision)

		avgPrecision = sum(precision_list) / len(precision_list) if precision_list else 0

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1

		#Fill in code here
		query_metric = []
		qrels = build_qrels_dict(qrels)

		# Iterate over each query
		for i in range(len(query_ids)):
			query_id = query_ids[i]
			# Get the list of relevant document IDs for the current query from qrels
			query_docs = qrels.get(query_id, {})
			relevant_docs = {}

			for doc_id in query_docs:
				if query_docs[doc_id] > 0:  # Assume relevance > 0 is relevant
					relevant_docs[doc_id] = query_docs[doc_id]
		
	
			# Get the document IDs from the ordered list of predicted docs
			query_doc_IDs_ordered = doc_IDs_ordered[i]

			metric_at_k = self.queryAveragePrecision(query_doc_IDs_ordered, query_id, list(relevant_docs.keys()), k)
			query_metric.append(metric_at_k)

		# Calculate the Mean Average Precision over all queries
		meanAveragePrecision = sum(query_metric) / len(query_metric) if query_metric else 0

		return meanAveragePrecision

