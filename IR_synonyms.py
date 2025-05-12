from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import numpy as np
import nltk

class InformationRetrievalWithWordNet:

    def __init__(self):
        self.documents = []
        self.docIDs = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.svd_model = None

    #Function to get the synonyms from the wordnet synsets.
    def get_synonym(self, word):
        synsets = wordnet.synsets(word)
        if synsets:
            # Use the most frequent lemma of the first synset
            return synsets[0].lemmas()[0].name().replace('_', ' ')
        return word

    def normalize_text(self, text):
        tokens = word_tokenize(text.lower())
        return ' '.join([self.get_synonym(word) for word in tokens])

    #Documents processed and stored in an index.
    def buildIndex(self, docs, docIDs):
        self.documents = []
        for doc in docs:
            flat_doc = ' '.join([' '.join(sent) for sent in doc])
            normalized = self.normalize_text(flat_doc)
            self.documents.append(normalized)
        self.docIDs = docIDs

    #LSA
    def rank_lsa(self, queries):

        #Queries processed
        query_texts = []
        for query in queries:
            flat_query = ' '.join([' '.join(sent) for sent in query])
            normalized = self.normalize_text(flat_query)
            query_texts.append(normalized)

        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        query_matrix = self.vectorizer.transform(query_texts)

        D, T = self.tfidf_matrix.shape
        k = max(2, int(min(D, T) / 5))  # Ensure at least 2 components

        #Docs and queries tranformed to the latent structure space.
        self.svd_model = TruncatedSVD(n_components=k, random_state=42)
        doc_lsa = self.svd_model.fit_transform(self.tfidf_matrix)
        query_lsa = self.svd_model.transform(query_matrix)

        #Similarity calculated
        sims = cosine_similarity(query_lsa, doc_lsa)
        ranked_docs = np.argsort(-sims, axis=1)

        # import matplotlib.pyplot as plt

        # # Step 1: Create and fit the SVD model on your TF-IDF matrix
        # svd = TruncatedSVD(n_components=1400)  # Try a large enough number to see the drop-off
        # svd.fit(self.tfidf_matrix)

        # # Step 2: Get the singular values
        # singular_values = svd.singular_values_

        # # Step 3: Plot the singular values
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(1, len(singular_values) + 1), singular_values, marker='o')
        # plt.title('Scree Plot (Singular Values vs. Components)')
        # plt.xlabel('Component Index')
        # plt.ylabel('Singular Value Magnitude')
        # plt.grid(True)
        # plt.show()

        return [[self.docIDs[i] for i in row] for row in ranked_docs]
