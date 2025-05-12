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

        self.svd_model = TruncatedSVD(n_components=280, random_state=42)
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
    
    def rank_bm25(self, queries):
        query_texts = [' '.join([' '.join(sent) for sent in query]) for query in queries]

        # Create or reuse CountVectorizer
        if self.vectorizer is None or self.term_freq_matrix is None:
            self.vectorizer = CountVectorizer()
            self.term_freq_matrix = self.vectorizer.fit_transform(self.documents).toarray()
            self.vocab = self.vectorizer.get_feature_names_out()
            self.doc_lens = np.sum(self.term_freq_matrix, axis=1)
            self.avgdl = np.mean(self.doc_lens)
            self.doc_freq = np.sum(self.term_freq_matrix > 0, axis=0)

        query_term_matrix = self.vectorizer.transform(query_texts).toarray()
        N = len(self.documents)
        k1 = 1.5
        b = 0.75

        rankings = []
        for query_vec in query_term_matrix:
            scores = []
            for doc_vec, dl in zip(self.term_freq_matrix, self.doc_lens):
                score = 0
                for i, q in enumerate(query_vec):
                    if q == 0:
                        continue
                    df = self.doc_freq[i]
                    idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
                    tf = doc_vec[i]
                    denom = tf + k1 * (1 - b + b * dl / self.avgdl)
                    score += idf * tf * (k1 + 1) / (denom + 1e-10)
                scores.append(score)
            ranked = np.argsort(-np.array(scores))
            rankings.append([self.docIDs[i] for i in ranked])

        return rankings
    
    def rank_TFQueryExpansion(self, queries, top_k=5, expansion_terms=5):
        """
        Rank documents using TF-IDF + Query Expansion (Pseudo-Relevance Feedback).
        
        Parameters:
        - top_k: number of top documents assumed relevant (pseudo-relevance)
        - expansion_terms: number of top terms to add to each query
        """
        query_texts = [' '.join([' '.join(sent) for sent in query]) for query in queries]

        # Step 1: Fit TF-IDF on documents
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        doc_term_matrix = self.tfidf_matrix.toarray()
        feature_names = self.vectorizer.get_feature_names_out()

        results = []
        for qtext in query_texts:
            # Step 2: Get original query vector
            q_vec = self.vectorizer.transform([qtext]).toarray()

            # Step 3: Get initial retrieval scores
            sims = cosine_similarity(q_vec, doc_term_matrix)[0]
            top_doc_indices = np.argsort(-sims)[:top_k]

            # Step 4: Compute average TF-IDF vector over top documents
            avg_top_vec = np.mean(doc_term_matrix[top_doc_indices], axis=0)

            # Step 5: Pick top terms not already in query
            query_tokens = set(qtext.split())
            ranked_terms = np.argsort(-avg_top_vec)
            new_terms = []
            for idx in ranked_terms:
                term = feature_names[idx]
                if term not in query_tokens:
                    new_terms.append(term)
                if len(new_terms) >= expansion_terms:
                    break

            # Step 6: Expand the query
            expanded_query = qtext + ' ' + ' '.join(new_terms)
            expanded_q_vec = self.vectorizer.transform([expanded_query])

            # Step 7: Final ranking using expanded query
            final_sims = cosine_similarity(expanded_q_vec, doc_term_matrix)
            ranked_docs = np.argsort(-final_sims, axis=1)
            results.append([self.docIDs[i] for i in ranked_docs[0]])

        return results
    
