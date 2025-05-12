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
