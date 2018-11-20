import numpy, sys

num_clusters = int(sys.argv[2])

vocabulary = [] # int to string
reverse_vocabulary = {} # string to int

# mapping strings to ints
## This function builds a vocabulary as we see each symbol
def get_id(s):
    if s in reverse_vocabulary:
        return reverse_vocabulary[s]
    else:
        symbol_id = len(vocabulary)
        reverse_vocabulary[s] = symbol_id
        vocabulary.append(s)
        return symbol_id

## Edges will be a list of (left ID, right ID, count) tuples
edges = []

## load the network edges
with open(sys.argv[1], encoding="utf-8") as network_file:
    for line in network_file:
        if line.startswith("#"):
            continue
        
        fields = line.rstrip().split("\t")
        if len(fields) == 3:
            left_id = get_id(fields[0])
            right_id = get_id(fields[1])
            # non-binarized
            count = int(fields[2])
            # binarize
            # count = 1
            
            edges.append( (left_id, right_id, count) )

num_symbols = len(vocabulary)

## Use a symmetric Dirichlet to initialize cluster weights for each
##  symbol to close to uniform
## THETA MATRIX
symbol_cluster_weights = numpy.random.dirichlet( numpy.ones(num_clusters), num_symbols )
## Calculate the column square roots

## BUFFER MATRIX
symbol_cluster_buffer = numpy.zeros( ( num_symbols, num_clusters) )

def top_symbols(cluster):
    return sorted(zip(symbol_cluster_weights[:,cluster], vocabulary), reverse=True)[:20]

def display():
    cluster_sums = numpy.sum( symbol_cluster_buffer, axis=0 )

    for cluster in numpy.argsort( - cluster_sums ):
        print(cluster_sums[cluster])
        print(" ".join([ "{} ({:.2f})".format(name, score) for score, name in top_symbols(cluster)]))


# convergence constants
ite = 0
maxiter = 30
diff = sys.maxsize
oldrun = None
eps = 0.01

while (ite < maxiter) and (diff > eps):
    # zero out cluster buffer every iteration
    # otherwise will just be like copying graph and appending again
    symbol_cluster_buffer = numpy.zeros( ( num_symbols, num_clusters) )
    iteration_sum = 0
    
    for left_id, right_id, count in edges:
        ## elementwise product
        product = symbol_cluster_weights[left_id,:] * symbol_cluster_weights[right_id,:]
        row_sum = numpy.sum(product)
        iteration_sum += row_sum
        
        if row_sum > 0.0:
            update = count * product / row_sum
            symbol_cluster_buffer[left_id,:] += update
            symbol_cluster_buffer[right_id,:] += update
        else:
            print("zero edge: {} {} {}".format(left_id, right_id, count))
    
    if not oldrun:
        diff = iteration_sum
        oldrun = iteration_sum
    else:
        diff = iteration_sum - oldrun
        oldrun = iteration_sum

    ite += 1
    # print(iteration_sum)
    
    cluster_square_roots = numpy.sqrt(numpy.sum(symbol_cluster_buffer, axis=0))
    symbol_cluster_weights = symbol_cluster_buffer / cluster_square_roots[numpy.newaxis, :]

print("Final Iterations: " + str(ite))
display()
    
