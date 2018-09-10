import re, sys, numpy, math
from collections import Counter
from scipy.sparse import lil_matrix
import scipy.sparse.linalg

doc_counters = []
corpus_counts = Counter()

doc_text = []
print ("reading")

# for TF-IDF
document_frequency = Counter()

with open(sys.argv[1], encoding="utf-8") as reader:
    for line in reader:
        fields = line.rstrip().split("\t")
        if len(fields) == 3:
            tag = fields[1]
            tokens = fields[2].lower().split()
            
            doc_counter = Counter(tokens)
            corpus_counts.update(doc_counter)
            document_frequency.update( doc_counter.keys() )

            doc_counters.append(doc_counter)
            
            doc_text.append(fields[2])

num_docs = len(doc_counters)

## construct a vocabulary list in reverse order by corpus count
vocabulary = [ w for w, c in corpus_counts.most_common() if c > 5 ]
# maps strings to integers
reverse_vocab = { word:i for (i, word) in enumerate(vocabulary) }
vocab_size = len(vocabulary)

idf_weights = { word:-math.log( document_frequency[word] / num_docs ) for word in vocabulary }

print("constructing matrix")
doc_word_counts = lil_matrix((num_docs, vocab_size))

for doc_id, doc_counter in enumerate(doc_counters):
    words = list([word for word in doc_counter if word in reverse_vocab])
    counts = [doc_counter[word] for word in words]

    weighted_counts = [ idf_weights[word] * doc_counter[word] for word in words ]
    word_ids = [reverse_vocab[word] for word in words]
    
    # have info based on documents (rows). easier to set rows
    doc_word_counts[doc_id,word_ids] = weighted_counts

doc_word_counts = doc_word_counts.tocsr()

print("running SVD")
doc_vectors, singular_values, word_vectors = scipy.sparse.linalg.svds(doc_word_counts, 100)

word_vectors = word_vectors.T

def rank_words(x):
    return sorted(zip(x, vocabulary))

def rank_docs(x):
    return sorted(zip(x, doc_text))

def l2_norm(matrix):
    row_norms = numpy.sqrt(numpy.sum(matrix ** 2, axis = 1))
    return matrix / row_norms[:, numpy.newaxis]
# cosine sim becomes dot product
# so word_vectors.dot(word_vectors[154,:])
# gives you matrix of cosine similarities

# words that occur in most similar contexts to python
# sorted_words[-10:]
# [(0.6332042503921438, 'databricks'), (0.6366757752082062, 'open'), 
# (0.646656838057807, 'server'), (0.6489888390810046, 'package'), 
# (0.651368378524946, 'ide'), (0.6608083695292203, 'studio'), 
# (0.6664462671950556, 'ides'), (0.6733420856087191, 'rstudio'), 
# (0.6861970891874979, 'mac'), (0.9999999999999998, 'python')]

# looking up "r"
# >>> reverse_vocab["r"]
# 171
# >>> sorted_words = rank_words(word_vectors.dot( word_vectors[171,:]))
# >>> sorted_words[-10:]
# [(0.5578353259297886, 'ggplot2'), 
# (0.5608580506753871, 'postgres'), 
# (0.5609395631290913, 'constants'), 
# (0.5687079034804157, 'mac'), 
# (0.6185301636704101, 'oracle'), 
# (0.6279484374038404, 'package'), 
# (0.6354393705297439, 'ide'), 
# (0.6529268897910219, 'law'), (0.7137968712936943, 'ides'), (1.0, 'r')]

# >>> doc_vectors[1,-1]
# 0.005451273761441059
# >>> singular_values[-1]
# 1540.2000204745748
# >>> singular_values[-1] * doc_vectors[1,-1]
# 8.396051958984032
# >>> word_vectors[0:10,-1]
# array([0.62016613, 0.31118427, 0.26832617, 0.20399631, 0.25420701,
#        0.21965151, 0.20298278, 0.16092729, 0.13567712, 0.12362675])
# >>> word_vectors[0:10,-1] * singular_values[-1] * doc_vectors[1,-1]
# array([5.20694702, 2.61271927, 2.25288045, 1.71276363, 2.13433526,
#        1.84420549, 1.70425396, 1.35115388, 1.13915218, 1.03797662])
# >>> vocabulary[:10]
# ['the', 'to', 'a', 'i', 'of', 'is', 'and', 'in', 'you', 'for']
# >>> doc_counters[1]
# Counter({'the': 7, 'to': 6, 'you': 3, 'code': 3, 'images': 2, 'show': 2, 'have':2, 'files': 2, 'how': 2, 'if': 1, 'need': 1, 'work': 1, 'on': 1, 'using': 1, 'python': 1, 'preferred': 1, 'library': 1, 'is': 1, 'pil': 1, 'here': 1, 'i': 1, 'a': 1, 'function': 1, 'do': 1, 'modifications': 1, 'delimited': 1, 'this': 1, 'makes': 1, 'no': 1, 'effort': 1, 'manage': 1, 'multiple': 1, 'or': 1, 'name': 1, 'converted': 1, 'but': 1, 'it': 1, 'does': 1, 'modify': 1, 'in': 1, 'ways': 1, 'asked': 1, 'test': 1, 'save': 1, 'image': 1, 'see': 1, 'docs': 1})
#
# So we expect 'the' to occur 5 times
#    'to' to occur 2 times
#    'a' to occur 2 times