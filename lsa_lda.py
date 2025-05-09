from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class InformationRetrieval():

    def __init__(self):
        self.documents = []
        self.docIDs = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.count_matrix = None
        self.svd_model = None
        self.lda_model = None

    def buildIndex(self, docs, docIDs):
        """
        Store preprocessed documents and their IDs.
        Each document is a list of sentences, each sentence is a list of tokens.
        """
        self.documents = [' '.join([' '.join(sent) for sent in doc]) for doc in docs]
        self.docIDs = docIDs

    def rank_lsa(self, queries):
        query_texts = [' '.join([' '.join(sent) for sent in query]) for query in queries]

        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        query_matrix = self.vectorizer.transform(query_texts)

        self.svd_model = TruncatedSVD(n_components=100, random_state=42)
        doc_lsa = self.svd_model.fit_transform(self.tfidf_matrix)
        query_lsa = self.svd_model.transform(query_matrix)

        sims = cosine_similarity(query_lsa, doc_lsa)
        ranked_docs = np.argsort(-sims, axis=1)

        return [[self.docIDs[i] for i in row] for row in ranked_docs]

    def rank_lda(self, queries):
        query_texts = [' '.join([' '.join(sent) for sent in query]) for query in queries]

        count_vectorizer = CountVectorizer()
        self.count_matrix = count_vectorizer.fit_transform(self.documents)
        query_count = count_vectorizer.transform(query_texts)

        self.lda_model = LatentDirichletAllocation(n_components=20, random_state=42)
        doc_lda = self.lda_model.fit_transform(self.count_matrix)
        query_lda = self.lda_model.transform(query_count)

        sims = cosine_similarity(query_lda, doc_lda)
        ranked_docs = np.argsort(-sims, axis=1)

        return [[self.docIDs[i] for i in row] for row in ranked_docs]