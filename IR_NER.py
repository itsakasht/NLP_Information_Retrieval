from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import numpy as np

class InformationRetrieval():

    def __init__(self):
        self.documents = []
        self.docIDs = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.nlp = spacy.load("en_core_web_sm")  # spaCy NER model

    def buildIndex(self, docs, docIDs):
        """
        Store preprocessed documents and their IDs.
        Each document is a list of sentences, each sentence is a list of tokens.
        We also append named entities to each document to improve retrieval.
        """
        self.documents = []
        self.docIDs = docIDs

        for doc in docs:
            flat_text = ' '.join([' '.join(sent) for sent in doc])
            ner_text = self._append_named_entities(flat_text)
            self.documents.append(ner_text)

        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    def rank(self, queries):
        """
        Rank documents for each query using cosine similarity.
        Each query is a list of sentences, each sentence a list of tokens.
        """
        query_texts = []
        for query in queries:
            flat_query = ' '.join([' '.join(sent) for sent in query])
            ner_query = self._append_named_entities(flat_query)
            query_texts.append(ner_query)

        query_matrix = self.vectorizer.transform(query_texts)
        sims = cosine_similarity(query_matrix, self.tfidf_matrix)
        ranked_docs = np.argsort(-sims, axis=1)
        return [[self.docIDs[i] for i in row] for row in ranked_docs]

    def _append_named_entities(self, text):
        """
        Extract named entities using spaCy and append to the text to emphasize them.
        """
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents]
        return text + ' ' + ' '.join(entities) if entities else text
