# # Add your import statements here
from collections import defaultdict

def build_qrels_dict(qrels_list):
    """
    Convert list-of-dicts qrels into a nested dict:
    {
        query_id (int): {
            doc_id (int): relevance_score (int)
        }
    }
    """
    qrels_dict = defaultdict(dict)
    for item in qrels_list:
        query_id = int(item["query_num"])
        doc_id = int(item["id"])
        score = int(item["position"])
        qrels_dict[query_id][doc_id] = score
    return qrels_dict

# import json
# from nltk.tokenize.treebank import TreebankWordTokenizer
# from nltk.corpus import stopwords

# # # Open the JSON file
# # with open("../../cranfield/cran_docs.json", "r") as f:
# #     data = json.load(f)

# # Extract documents
# documents = [item["body"] for item in data]

# #Frequency calculation function
# def frequency(word, tokens):

#     count = 0
#     for token in tokens:
#         if token == word:
#             count+=1

#     return 100*count/len(tokens)

# #Tokenizer
# t = TreebankWordTokenizer()
# frequencies = {}
# stopwords_new = []

# for doc in documents:
#     #Docs tokenized
#     tokens = t.tokenize(doc)
#     for token in tokens:

#         #frequencies calculated
#         if token in frequencies:
#             frequencies[token] += frequency(token, tokens)
#         else:
#             frequencies[token] = frequency(token, tokens)

# #Stopwords updated
# for key, vals in frequencies.items():
#     if vals>6:
#         stopwords_new.append(key)

# stopwords_english = stopwords.words('english')

# #Stopwords compared.
# common = list(set(stopwords_new) and set(stopwords_english))
# print(common)

# # Add any utility functions here