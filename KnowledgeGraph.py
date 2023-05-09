import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import mwclient
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AffinityPropagation
from sklearn.exceptions import ConvergenceWarning
import warnings
import multiprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Step 1: Preprocess data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return lemmatized_tokens

def extract_keyterms(tokens):
    tagged_tokens = nltk.pos_tag(tokens)
    keyterms = []
    for token, tag in tagged_tokens:
        if tag.startswith('N') and token not in stop_words:
            keyterms.append(token)
    return keyterms

def filter_keyterms(text):
    text = open(text).read()
    tokens = preprocess(text)
    keyterms = extract_keyterms(tokens)
    # Filter keyterms that occur more than once
    keyterm_freq = defaultdict(int)
    for keyterm in keyterms:
        keyterm_freq[keyterm] += 1
    filtered_keyterms = [keyterm for keyterm in keyterms if keyterm_freq[keyterm] > 5]
    filtered_keyterms = list(set(filtered_keyterms))
    return filtered_keyterms


# Step 4: Construct knowledge graph

# not all the mapping are stored
# not all the keyterms are nodes with probs
# Whether this is good or not i do not know might be better to represent as previous form, 
import time

def process_keyterm(kt, mapping_dict):
    site = mwclient.Site('en.wikipedia.org')
    vectorizers = {}
    candidate_mappings = {}
    total_count = 0  # Total occurrence count of keyterm in linked phrases
    # Get entity candidates for each keyterm
    page = site.pages[kt]
    if page.exists:
        # Add the keyterm as a node in the mapping dictionary
        if kt not in mapping_dict:
            mapping_dict[kt] = {}
        count = 0
        for link in page.links():
            if count < 50: # Limited to 50 as KG becomes too big , study could be done into the optimum also they are chosen randomly and obviously this is not ideal 
                count = count + 1
                candidate = link.page_title
                if candidate != kt:
                    # Count co-occurrence of candidate entity with keyterm in linked phrases
                    co_occurrence_count = link.text().count(kt)
                    
                    total_count += co_occurrence_count
                    if candidate in candidate_mappings:
                            candidate_mappings[candidate]['count'] += co_occurrence_count
                    else:
                            candidate_mappings[candidate] = {'count': co_occurrence_count, 'prior_prob': 0, 'cos_sim': 0}
            else:
                break
            time.sleep(1.5)  # Add a delay of 2 seconds between requests
        print("direct")
        # Calculate prior probability and cosine similarity for each candidate entity
        for candidate, values in candidate_mappings.items():
            count = values['count']
            if count == 0:
                prior_prob = 0
            else:
                prior_prob = count / total_count
            if kt not in mapping_dict:
                mapping_dict[kt] = {}
            if candidate not in mapping_dict[kt]:
                mapping_dict[kt][candidate] = {'prior_prob': prior_prob, 'cos_sim': 0}
            if candidate in mapping_dict:
                if kt not in mapping_dict[candidate]:
                    mapping_dict[candidate][kt] = {'prior_prob': prior_prob, 'cos_sim': 0}
                else:
                    mapping_dict[candidate][kt]['prior_prob'] = prior_prob
            else:
                mapping_dict[candidate] = {kt: {'prior_prob': prior_prob}}

                

    else:
        # Add the keyterm as a node in the mapping dictionary
        if kt not in mapping_dict:
            mapping_dict[kt] = {}
        count = 0
        for link in site.search(kt):
            if count < 50:
                count = count + 1
                candidate = link.get('title')
                if candidate != kt:
                    link = site.pages[candidate]
                    # Count co-occurrence of candidate entity with keyterm in linked phrases
                    co_occurrence_count = link.text().count(kt)
                    
                    total_count += co_occurrence_count
                    if candidate in candidate_mappings:
                            candidate_mappings[candidate]['count'] += co_occurrence_count
                    else:
                            candidate_mappings[candidate] = {'count': co_occurrence_count, 'prior_prob': 0, 'cos_sim': 0}
            else:
                break
            time.sleep(1.5)  # Add a delay of 2 seconds between requests
        print("undirect")
        # Calculate prior probability and cosine similarity for each candidate entity
        for candidate, values in candidate_mappings.items():
            count = values['count']
            if count == 0:
                prior_prob = 0
            else:
                prior_prob = count / total_count
            if kt not in mapping_dict:
                mapping_dict[kt] = {}
            if candidate not in mapping_dict[kt]:
                mapping_dict[kt][candidate] = {'prior_prob': prior_prob, 'cos_sim': 0}
            if candidate in mapping_dict:
                if kt not in mapping_dict[candidate]:
                    mapping_dict[candidate][kt] = {'prior_prob': prior_prob, 'cos_sim': 0}
                else:
                    mapping_dict[candidate][kt]['prior_prob'] = prior_prob
            else:
                mapping_dict[candidate] = {kt: {'prior_prob': prior_prob}}

        
    return mapping_dict



def construct_knowledge_graph_parallel(keyterms, num_processes=None, G=None, doc_text=None):
    if not G:
        G = nx.Graph()
    site = mwclient.Site('en.wikipedia.org')

    # Build a mapping dictionary for keyterms
    mapping_dict = multiprocessing.Manager().dict()
    vectorizer = CountVectorizer()
    processes = []
    for kt in keyterms:
        p = multiprocessing.Process(target=process_keyterm, args=(kt, mapping_dict))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

      # Add nodes to the graph
    for entity in mapping_dict:
        print("loop")
        G.add_node(entity)
        # Get entity description from Wikipedia, this sections isn't really needed but helps for human interpretation as probabilites have already been calculated
        page = site.pages[entity]
        if page.exists:
            if page.redirect:
                target_title = page.redirects_to().name
                page = site.pages[target_title]
                description = extract_summary(page.text())[0:300] + '...' if len(extract_summary(page.text())) > 300 else extract_summary(page.text()) # can experiment with this value
                G.nodes[entity]['description'] = "This is not the direct entity description as it was redirected to " + target_title + ": " +  description
            else:
                description = extract_summary(page.text())[0:300] + '...' if len(extract_summary(page.text())) > 300 else extract_summary(page.text()) 
                G.nodes[entity]['description'] = description

        else:
            count = 0
            for result in site.search(entity):
                if count < 1:
                    count = count + 1
                    search_page = result.get('title')
                else:
                    break
            page = site.pages[search_page]
            description = extract_summary(page.text())[0:300] + '...' if len(extract_summary(page.text())) > 300 else extract_summary(page.text()) 
            G.nodes[entity]['description'] = "The description of this exact entity was not found but this a search result that is similar " + search_page + ": " + description

   


    # Add edges to the graph with prior probabilities and cosine similarity as edge weight
    for kt, candidate_mappings in mapping_dict.items():
        for entity in candidate_mappings:
            if entity != kt:
                prior_prob = candidate_mappings[entity]['prior_prob']
                # Compute cosine similarity between document vector and entity vector
                # Fit the vectorizer to the document text
                vectorizer.fit([doc_text])
                doc_vector = vectorizer.transform([doc_text]).toarray()
                # Transform both the document and entity text using the fitted vectorizer
                entity_vector = vectorizer.transform([entity]).toarray()
                # Reshape vectors to have the same number of features
                if entity_vector.shape[1] < doc_vector.shape[1]:
                    entity_vector = np.pad(entity_vector, ((0, 0), (0, doc_vector.shape[1] - entity_vector.shape[1])), mode='constant')
                elif entity_vector.shape[1] > doc_vector.shape[1]:
                    doc_vector = np.pad(doc_vector, ((0, 0), (0, entity_vector.shape[1] - doc_vector.shape[1])), mode='constant')
                doc_entity_cos_sim = cosine_similarity(doc_vector, entity_vector)[0][0]
                print(doc_entity_cos_sim)
                # Combine prior probability and cosine similarity as edge weight
                edge_weight = prior_prob * doc_entity_cos_sim
                if G.has_edge(kt, entity):
                    # Update edge weight if it already exists
                    G[kt][entity]['edge_weight'] += edge_weight
                else:
                    # Create a new edge if it does not exist
                    G.add_edge(kt, entity, weight=edge_weight)
                    G[kt][entity]['prior_prob'] = prior_prob
                    G[kt][entity]['cos_sim'] = doc_entity_cos_sim

    return G






def extract_summary(text):
    expr = re.compile("(?:^|}})([^{{\}}]+)(?:\{{|$)")
    content =  "".join(expr.findall(text)[0:])
    expr = re.compile("'''[\s\S]*$")
    return "".join(expr.findall(content)[0:])




# So the total time complexity of this part of the function is O(N^2*K). K is the number of keyterms, N is the number of unique entities found across all the keyterms


    


import collections



def compute_semantic( model, term1, term2):
    # Compute cosine similarity between the vector representations of term1 and term2
    
    try:
        similarity = model.wv.similarity(term1, term2)
        
        return similarity
    except:
        return 0.0

def cluster_keyterms(keyterms,input_doc):
    # Train Word2Vec model on the keyphrases
    doc = nlp(input_doc)
    sentences = []
    for sent in doc.sents:
        words = []
        for token in sent:
            if token.is_stop or not token.is_alpha:
                continue
            words.append(token.text)
        sentences.append(words)

    # Train a Word2Vec model on the sentences
    model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)
    # print(model.wv.key_to_index)
    # Compute semantic relatedness between all pairs of keyphrases
    n = len(keyterms)
    similarities = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            similarities[i][j] = compute_semantic(model, keyterms[i], keyterms[j])
            similarities[j][i] = similarities[i][j]
    
    # Cluster keyphrases using affinity propagation
    clustering = AffinityPropagation().fit(similarities)
    labels = clustering.labels_
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(keyterms[i])
    clusters = dict(collections.OrderedDict(sorted(clusters.items())))
    return clusters


def viewKG(G):
    G = nx.read_graphml("knowledge_graph.graphml")
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=14, font_family='Arial')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_family='Arial')
    nx.draw_networkx_edge_labels(G, pos, font_size=12, font_family='Arial', edge_labels={(u, v): G.nodes[u]['description'] + " - " + G.nodes[v]['description'] for (u, v) in G.edges()})
    plt.axis('off')
    plt.show()

    

# if __name__ == '__main__':
#     multiprocessing.freeze_support()
#     text = filter_keyterms("transcripts/transcript1/cosmosdb.txt")
#     print(text)
#     text_corpus = open("transcripts/transcript1/cosmosdb.txt").read()
#     G = construct_knowledge_graph_parallel(text,doc_text=text_corpus)
# #     # text = filter_phrases("transcripts/transcript2/From Python Script to Python app.txt")
# #     # G = construct_knowledge_graph_parallel(text,G=G, doc_text=text_corpus)
# #     # text = filter_phrases("transcripts/transcript3/Course Work 1 - Part 2.txt")
# #     # G = construct_knowledge_graph_parallel(text,G=G, doc_text=text_corpus)
# #     # text = filter_phrases("transcripts/transcript4/Data in the Cloud.txt")
# #     # G = construct_knowledge_graph_parallel(text,G=G, doc_text=text_corpus)
# #     # text = filter_phrases("transcripts/transcript5/iaac.txt")
# #     # G = construct_knowledge_graph_parallel(text,G=G, doc_text=text_corpus)
# #     # text = filter_phrases("transcripts/transcript6/Starting-PHP.txt")
# #     # G = construct_knowledge_graph_parallel(text,G=G, doc_text=text_corpus)
# #     # text = filter_phrases("transcripts/transcript7/WithSecure.txt")
# #     # G = construct_knowledge_graph_parallel(text,G=G, doc_text=text_corpus)
# #     # text = filter_phrases("transcripts/transcript8/Vulnerability-Fixing.txt")
# #     # G = construct_knowledge_graph_parallel(text,G=G, doc_text=text_corpus)
# #     # text = filter_phrases("transcripts/transcript9/robpress.txt")
# #     # G = construct_knowledge_graph_parallel(text,G=G, doc_text=text_corpus)
# #     # text = filter_phrases("transcripts/transcript10/ToolsandTechniques.txt")
# #     # G = construct_knowledge_graph_parallel(text,G=G, doc_text=text_corpus)
#     nx.write_graphml(G, "knowledge_graph.graphml")







# Knowledge graph is constructed from all transcripts and then saved as graphml
def clusters(text_corpus):
    keyterms = filter_keyterms(text_corpus)
    text = open(text_corpus).read()
    clusters = cluster_keyterms(keyterms,text)
    # print(dict(collections.OrderedDict(sorted(clusters.items()))))
    return clusters



# Run functions from here

# Define function to construct h-hop keyterm graph

from multiprocessing import Pool
import networkx as nx
from gensim.models import Word2Vec

def build_h_hop(cluster, kg, h, threshold,input_doc):
    # Initialize the keyterm graph
    keyterm_graph = nx.Graph()
    list = []
    for node in cluster.values():
        list.append(node)
    clusterKT =  set([item for sublist in list for item in sublist])
    
    # Add anchor nodes to the keyterm graph
    anchor_nodes = [node for node in kg.nodes() if node in clusterKT]
    
    keyterm_graph.add_nodes_from(anchor_nodes)
    count1 = 0
    # Apply Breadth-first search to extract paths of length no longer than h
    for anchor_node in anchor_nodes:
        print(keyterm_graph.edges())
        count1 = count1 + 1

        print(anchor_node + " : " + str(count1) + " : " + str(len(anchor_nodes)))
        bfs_paths = nx.bfs_tree(kg, anchor_node, depth_limit=h).edges()
        print(len(bfs_paths))
        count = 0
        
        # Remove paths that are less related with anchor nodes
        with Pool() as p:
            semantic_relatedness = p.starmap_async(
                compute_semantic_relatedness, [(anchor_nodes, [node for node in path if node not in anchor_nodes], input_doc) for path in bfs_paths]
            ).get()
        
        for i, path in enumerate(bfs_paths):
            count = count + 1
            print("path" + " : " + str(count) )
            print(path)
            if semantic_relatedness[i] > threshold: #This value needs to be changed
                continue
            
            print("continue")
            keyterm_graph.add_edges_from([path])

    # Add edges between anchor nodes that occur in the same window
    keyterm_graphs = []
    for clust in cluster.values():
        keyterm_grapht = build_keyterm_graph(clust, 5, input_doc)
        keyterm_graphs.append(keyterm_grapht)
    combined = combine_keyterm_graphs(keyterm_graphs)
    keyterm_graph.add_edges_from(combined.edges())

    return keyterm_graph
import spacy
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


nlp = spacy.load('en_core_web_sm')
def compute_semantic_relatedness(anchor_nodes, expanded_nodes, input_doc):
    # Tokenize the input document into phrases , 
    # better to train the model with sentences insteada
    doc = nlp(input_doc)
    phrases = []
    for chunk in doc.noun_chunks:  
        phrases.append(chunk.text)
    
    # Convert the phrases to TaggedDocuments with unique tags
    tagged_docs = []
    for i, phrase in enumerate(phrases): 
        tagged_doc = TaggedDocument(words=phrase.split(), tags=[i])
        tagged_docs.append(tagged_doc)
    
    # Train a Doc2Vec model on the tagged documents
    model = Doc2Vec(tagged_docs, vector_size=300, window=3, min_count=1, epochs=50)

    # Find the vectors for the anchor nodes and expanded nodes
    anchor_vectors = []
    for node in anchor_nodes:
        vector = model.infer_vector(node.split())
        anchor_vectors.append(vector)
    
    expanded_vectors = []
    for node in expanded_nodes:
        vector = model.infer_vector(node.split())
        expanded_vectors.append(vector)
    # Compute the cosine similarity between the anchor and expanded vectors
    similarities = []
    for anchor_vector in anchor_vectors:
        for expanded_vector in expanded_vectors:
            similarities.append(cosine_similarity(np.squeeze(anchor_vector), np.squeeze(expanded_vector)))
    
    return sum(similarities) / len(similarities)



def build_keyterm_graph(cluster, window_size, input_doc):
    """
    Build a keyterm graph for a cluster of terms.
    
    Args:
    - cluster (dict): a dictionary where the keys are the anchor nodes in the cluster and the values are the corresponding keyterms
    - kg (nx.Graph): the knowledge graph
    - window_size (int): the maximum number of words between two keyterms for an edge to be added
    - input_doc (str): the input document as a string
    
    Returns:
    - keyterm_graph (nx.Graph): the keyterm graph for the cluster
    """
    # Initialize the keyterm graph
    keyterm_graph = nx.Graph()
    
    # Add anchor nodes to the keyterm graph
    anchor_nodes = cluster
    keyterm_graph.add_nodes_from(anchor_nodes)
    
    # Add edges between anchor nodes based on window size
    for i, node1 in enumerate(anchor_nodes):
        for j in range(i+1, len(anchor_nodes)):
            keyterm1 = cluster[i]
            keyterm2 = cluster[j]
            if are_in_same_window(keyterm1, keyterm2, input_doc, window_size):
                keyterm_graph.add_edge(keyterm1, keyterm2)
    
    return keyterm_graph


def combine_keyterm_graphs(keyterm_graphs):

    # Combine keyterm graphs together

    combined_graph = nx.Graph()
    for graph in keyterm_graphs:
        combined_graph.add_edges_from(graph.edges())
    
    return combined_graph




def cosine_similarity(u, v):
    return u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v))

def are_in_same_window(keyterm1, keyterm2, input_doc, window_size):
    
    # Find the indices of the keyterms in the input document
    indices1 = [i for i, x in enumerate(input_doc.split()) if x == keyterm1]
    indices2 = [i for i, x in enumerate(input_doc.split()) if x == keyterm2]
    
    # Check if any of the indices is within the window of the other
    for index1 in indices1:
        for index2 in indices2:
            if abs(index1 - index2) <= window_size:
                return True
    
    return False






# if __name__ == '__main__':
#     multiprocessing.freeze_support()
#     cluster = clusters("transcripts/transcript1/cosmosdb.txt")
#     print(cluster)
#     input_doc = open("transcripts/transcript1/cosmosdb.txt").read()
#     KG = nx.read_graphml("knowledge_graph.graphml")
#     h_hop = build_h_hop(cluster, KG,2,0.5, input_doc)
#     nx.write_graphml(h_hop,"h_hop.graphml")
from collections import Counter


import networkx as nx
import numpy as np

def personalized_page_rank(h_hop_graph, keyterms, damping_factor=0.85, max_iter=100, tol=1e-6):
    """
    Computes the personalized page rank for each node in the graph based on the given keyterms and their frequencies.

    Parameters:
        h_hop_graph (nx.Graph): A networkx graph representing the h-hop graph.
        keyterms (dict): A dictionary mapping node ids to their corresponding term frequencies.
        damping_factor (float): The damping factor, usually set to 0.85.
        max_iter (int): The maximum number of iterations.
        tol (float): The tolerance for convergence.

    Returns:
        dict: A dictionary mapping node ids to their corresponding personalized page rank scores.
    """
    # Create a mapping from node ids to their indices in the adjacency matrix.
    node_ids = list(h_hop_graph.nodes())
    node_indices = {node_id: i for i, node_id in enumerate(node_ids)}

    # Compute the jump probabilities based on the given keyterms.
    num_nodes = len(node_ids)
    jump_probs = np.zeros(num_nodes)
    for node_id, term_freq in keyterms.items():
        if node_id in node_indices:
            jump_probs[node_indices[node_id]] = term_freq / sum(keyterms.values())

    # Compute the weight for each predicate based on its inverse document frequency.
    edge_counts = {}
    for node1, node2 in h_hop_graph.edges():
        edge_counts[node2] = edge_counts.get(node2, 0) + 1
    num_edges = sum(edge_counts.values())
    pred_weights = {node2: np.log(num_edges / count) for node2, count in edge_counts.items()}

    # Initialize the transition matrix based on the weighted graph.
    trans_matrix = np.zeros((num_nodes, num_nodes))
    for i, node_id in enumerate(node_ids):
        out_degree = len(h_hop_graph[node_id])
        for neighbor in h_hop_graph[node_id]:
            if neighbor in node_indices:
                j = node_indices[neighbor]
                trans_matrix[j, i] = pred_weights.get(neighbor, 0) / out_degree

    # Initialize the page rank scores.
    page_rank = np.ones(num_nodes) / num_nodes

    # Iterate until convergence or until reaching the maximum number of iterations.
    for i in range(max_iter):
        old_page_rank = page_rank.copy()
        page_rank = (1 - damping_factor) * jump_probs + damping_factor * np.matmul(trans_matrix, page_rank)
        diff = np.linalg.norm(page_rank - old_page_rank)
        if diff < tol:
            break

    # Map the page rank scores back to the original node ids.
    return {node_id: score for node_id, score in zip(node_ids, page_rank)}





from nltk import word_tokenize, pos_tag, sent_tokenize

def extract_keyphrases(ppr_scores, input_document, k):
    # Generate candidate phrases
    sentences = sent_tokenize(input_document)
    print(" ".join(sentences).strip("\n\n"))
    # Tokenize the words and tag their parts of speech
    document = []
    for sentence in sentences:
        tagged = pos_tag(word_tokenize(sentence))
        # print(document)
        document.append(tagged)
    
    candidates = []
    for sentence in document:
        # print(sentence)
        for i, (word, pos) in enumerate(sentence):
            # print(word)
            if pos in ['NN', 'NNS', 'NNP', 'NNPS']:
                candidate = [word]
                for j in range(i+1, len(sentence)):
                    next_word, next_pos = sentence[j]
                    if next_pos in ['JJ', 'CD']:
                        candidate.append(next_word)
                    elif next_pos in ['NN', 'NNS', 'NNP', 'NNPS']:
                        candidate.append(next_word)
                    else:
                        break
                if len(candidate) > 1:
                    candidates.append(tuple(candidate))

    # Calculate the score for each candidate phrase
    phrase_scores = {}
    keyterms = set(ppr_scores.keys())
    for phrase in candidates:
        print("Phrase: " + str(phrase))
        score = sum([ppr_scores[keyterm] for keyterm in keyterms if keyterm in phrase])
        print("Score: " + str(score))
        frequency = sum([1 for sentence in document for word, _ in sentence if word == phrase[0]])
        print("Frequency: " + str(frequency))
        first_occurrence = " ".join(" ".join(sentences).split()).index(" ".join(phrase))
        print("First occurences: " + str(first_occurrence))
        if first_occurrence:
            phrase_scores[phrase] = (frequency / first_occurrence )* score
    # Select the top-K phrases with the largest scores as the keyphrases
    keyphrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [keyphrase[0] for keyphrase in keyphrases]



if __name__ == '__main__':
    h_hop = nx.read_graphml("h_hop.graphml")
    input_doc = open("transcripts/transcript1/cosmosdb.txt").read()
    words = nltk.word_tokenize(input_doc)

    # Remove stop words and stem the remaining words.
    stopwords = set(nltk.corpus.stopwords.words("english"))
    terms = [word.lower() for word in words if word.lower() not in stopwords]

    # Count the frequency of each term.
    term_freqs = Counter(terms)

    # Construct the keyterms dictionary mapping node ids to their corresponding term frequencies.
    keyterms = {term: freq for term, freq in term_freqs.items()}
    ppr_scores = personalized_page_rank(h_hop,keyterms)
    

# Tokenize the words and tag their parts of speech
    text = input_doc.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    print(extract_keyphrases(ppr_scores,text,10))

#Add parralization and fix the big KG example 
# I have to construct knowledge graph in the paper they just use dbpedia
# Need to make sure my keywords are being chosen correct at the start as I don't know if the paper has multiple keywords, have to use doc2vec instead because of this
# Do final checks to make sure this flow is right and performs as the one proposed in the paper, and also comment the code so that it makes sense with potential time complexity
# Flow = have a universal keyword selection mechanism this can be changed as wanted, and how I used one specified in paper (need to check)
# Constuct clusters of these keywords
# Construct a KG from these keywords using semantics relatedness
# Construct a H hop keyterm graph from the KG and the clustering of keyterms
# Use pagerank to go throught the H hop and finalize the keywords

#Something I overcame was how long it took to run need to show what I did

# There are several potential optimizations that can be made to improve the speed of the algorithm:

# 1. Use a more efficient keyword extraction algorithm: The current implementation uses a brute-force approach to extract candidate phrases, which can be slow for large amounts of text. There are more efficient algorithms, such as RAKE (Rapid Automatic Keyword Extraction), that can be used to speed up this step.

# 2. Use a more efficient graph representation: The current implementation uses a NetworkX graph to represent the knowledge graph, which can be slow for large graphs. One alternative is to use a graph database, such as Neo4j, which is optimized for storing and querying large graphs.

# 3. Use caching to avoid redundant computations: The current implementation recalculates the TF-IDF scores and candidate mappings every time the algorithm is run. By using caching to store these results, the algorithm can avoid redundant computations and run faster.

# 4. Parallelize the algorithm: The current implementation is single-threaded and does not take advantage of multi-core CPUs. By parallelizing the algorithm, it can be run faster on machines with multiple cores.

# 5. Use a more efficient Wikipedia API client: The current implementation uses the `wikipediaapi` library to query Wikipedia for entity descriptions. There are other Wikipedia API clients, such as `wikipedia` and `mwclient`, that may be faster and more efficient.

# 6. More effecient porgramming language

# 7. Get Ethernet, Use HPC and free up RAM on computer, potentially use VMware

# 8. Allcating more resources in python

# Only need to combine KGs if they are very simialr as the main issue with this study is the KGs are not big enough

# 9. DBpedia was too slow and also need too specific of a query, but worth looking into for the more enhanced results, perhaps the way they construct in the paper is with a lot less nodes like my original KG so need to look into thi 
# These are just a few potential optimizations that can be made to speed up the algorithm. Depending on the specific use case and constraints, there may be other optimizations that are more appropriate.