import string
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag, sent_tokenize, WordNetLemmatizer
from collections import defaultdict, Counter
import collections
import networkx as nx
import matplotlib.pyplot as plt
import mwclient
import re
from sklearn.cluster import AffinityPropagation
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import multiprocessing
import numpy as np
from multiprocessing import Pool
import time
import spacy
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


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


# Step 2: Select Keyterms


def extract_keyterms(tokens):
    # extracts all the nouns that are not stop words from the input tokens and returns them as a list of keyterms
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
    # Filter keyterms that occur more than five times
    keyterm_freq = defaultdict(int)
    for keyterm in keyterms:
        keyterm_freq[keyterm] += 1
    filtered_keyterms = [keyterm for keyterm in keyterms if keyterm_freq[keyterm] > 5] # hyperparameter
    filtered_keyterms = list(set(filtered_keyterms))
    return filtered_keyterms


# Step 3: Keyterm Clustering (Using affinity propagation)

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
    model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4) # Hyper_parameters
   
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

def clusters(text_corpus):
    # Helper function
    keyterms = filter_keyterms(text_corpus)
    text = open(text_corpus).read()
    clusters = cluster_keyterms(keyterms,text)
    return clusters



# Step 4: Keyterm Linking (Construct knowledge graph)

def process_keyterm(kt, mapping_dict, KG_size=50):
    # Method to calculate Prior probabilties for each key-term and it's mappings
    site = mwclient.Site('en.wikipedia.org')
    vectorizers = {}
    candidate_mappings = {}
    total_count = 0  # Total occurrence count of keyterm in linked candidate phrases
    # Get entity candidates for each keyterm
    page = site.pages[kt]
    if page.exists:
        # Add the keyterm as a node in the mapping dictionary
        if kt not in mapping_dict:
            mapping_dict[kt] = {}
        count = 0
        for link in page.links():
            if count < KG_size: # Limited to 50 as KG becomes too big
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
            time.sleep(1.5)  # Add a delay of 1.5 seconds between requests, this stops the mwclient from making too many requests and creating an error
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

                
        #Repeated for terms that don't have a direct wikipedia page
    else:
        # Add the keyterm as a node in the mapping dictionary
        if kt not in mapping_dict:
            mapping_dict[kt] = {}
        count = 0
        for link in site.search(kt):
            if count < KG_size:
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
            time.sleep(1.5) 
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
    # Process to build KG , add descriptions to the keyterm nodes and then compute the cosine similarities of the keyterm nodes
    if not G:
        G = nx.Graph()
    site = mwclient.Site('en.wikipedia.org')

    # Build a mapping dictionary for keyterms
    # Uses parralelization to speed process up 
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
        # Get entity description from Wikipedia, this section isn't really needed but helps for human interpretation as probabilites have already been calculated. The descriptions of the keyterms
        # are never actually used it is only the descriptions of the mapping entities used to calculate prior probabilities
        page = site.pages[entity]
        if page.exists:
            # If the page leads to a redirect this redirects it to that page
            if page.redirect:
                target_title = page.redirects_to().name
                page = site.pages[target_title]
                description = extract_summary(page.text())[0:300] + '...' if len(extract_summary(page.text())) > 300 else extract_summary(page.text()) # This value can be changed depending on how log you want description to be
                G.nodes[entity]['description'] = "This is not the direct entity description as it was redirected to " + target_title + ": " +  description
            else:
                # This fetches the description of the direct wikipedia page
                description = extract_summary(page.text())[0:300] + '...' if len(extract_summary(page.text())) > 300 else extract_summary(page.text()) # Hyper parameter
                G.nodes[entity]['description'] = description
        # If page doesn't exist redirects to the nearest search result description
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
    # This function cleans the wikipedia page description so that it gives useful summary data
    # Gets all text after any {{}} that appear
    expr = re.compile("(?:^|}})([^{{\}}]+)(?:\{{|$)")
    content =  "".join(expr.findall(text)[0:])
    # Gets all text after any ''' that appear
    expr = re.compile("'''[\s\S]*$")
    return "".join(expr.findall(content)[0:])



# total time complexity of this stage is :  O(N^2*K). K is the number of keyterms, N is the number of unique entities found across all the keyterms


# Function to visualize the KG's produced

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



# Step 5: Keyterm Graph Construction (Building of H-hop graph)


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
        bfs_paths = nx.bfs_tree(kg, anchor_node, depth_limit=h).edges() # This step is only really necessary if the candidate nodes were expanded aswell, which is not done in the paper, this would be very computationally expensive but potentially yield good results
        print(len(bfs_paths)) # depth_limit is a hyper parameter
        count = 0
        
        # Remove paths that are less related with anchor nodes
        with Pool() as p:
            semantic_relatedness = p.starmap_async(
                compute_semantic_relatedness, [(anchor_nodes, [node for node in path if node not in anchor_nodes], input_doc) for path in bfs_paths]
            ).get()
        
        for i, path in enumerate(bfs_paths):
            count = count + 1
            if semantic_relatedness[i] < threshold: # Includes paths above semantic threshold, hyper parameter
                continue
            
            keyterm_graph.add_edges_from([path])

    # Add edges between anchor nodes that occur in the same window
    keyterm_graphs = []
    for clust in cluster.values():
        keyterm_grapht = build_keyterm_graph(clust, 5, input_doc)
        keyterm_graphs.append(keyterm_grapht)
    combined = combine_keyterm_graphs(keyterm_graphs)
    keyterm_graph.add_edges_from(combined.edges())

    return keyterm_graph

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
    
    # Train a Doc2Vec model on the tagged documents, doc2vec was used instead of word2vec so that the infer_vector function can be used, and doc2vec is more powerful generally for this type of task
    model = Doc2Vec(tagged_docs, vector_size=300, window=3, min_count=1, epochs=50) # Hyper parameters

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
            similarities.append(cosine_similarities(np.squeeze(anchor_vector), np.squeeze(expanded_vector)))
    
    return sum(similarities) / len(similarities)


def build_keyterm_graph(cluster, window_size, input_doc):
    """
    Builds a keyterm graph for a cluster of terms.
    
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

    # Combine keyterm graphs together, generated for each cluster

    combined_graph = nx.Graph()
    for graph in keyterm_graphs:
        combined_graph.add_edges_from(graph.edges())
    
    return combined_graph


def cosine_similarities(u, v):
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

# The time complexity of this step is : O(N * (|E| * |V|)^h), where N is the number of anchor nodes, |E| is the number of edges in the knowledge graph, |V| is the number of vertices in the knowledge graph, and h is the length of the path.
# But because in this case h is always one the time complexity can be simiplified to O(N * |E| * |V|)


# Step 6: Keyterm Ranking (Personalized page rank)

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



# Step 7: keyphrase generation

def extract_keyphrases(ppr_scores, input_document, k):
    # Generate candidate phrases
    sentences = sent_tokenize(input_document)
    # Tokenize the words and tag their parts of speech
    document = []
    for sentence in sentences:
        tagged = pos_tag(word_tokenize(sentence))
        document.append(tagged)
    
    candidates = []
    for sentence in document:
        for i, (word, pos) in enumerate(sentence):
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



# Step 9: Pipeline


# Helper function

def keyphraseExtraction(source,kg_name): 
    cluster = clusters(source)
    input_doc = open(source).read()
    KG = nx.read_graphml( kg_name + ".graphml")
    h_hop = build_h_hop(cluster, KG,2,0.5, input_doc)
    nx.write_graphml(h_hop, kg_name + "_h_hop.graphml")

    input_doc = open(source).read()
    words = nltk.word_tokenize(input_doc)

    # Remove stop words 
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
    keyphrases = extract_keyphrases(ppr_scores,text,20)
    with open(kg_name + ".txt","w") as f:
          f.write(str(keyphrases))
    




if __name__ == '__main__':
    multiprocessing.freeze_support()
    text = filter_keyterms("transcripts/transcript1/cosmosdb.txt")
    text_corpus = open("transcripts/transcript1/cosmosdb.txt").read()
    G = construct_knowledge_graph_parallel(text,doc_text=text_corpus)
    nx.write_graphml(G, "cosmosDB.graphml")
    keyphraseExtraction("transcripts/transcript1/cosmosdb.txt","cosmosDB")

    text = filter_keyterms("transcripts/transcript2/From Python Script to Python app.txt")
    text_corpus = open("transcripts/transcript2/From Python Script to Python app.txt").read()
    G = construct_knowledge_graph_parallel(text, doc_text=text_corpus)
    nx.write_graphml(G, "pythonScript.graphml")
    keyphraseExtraction("transcripts/transcript2/From Python Script to Python app.txt","pythonScript")

    text = filter_keyterms("transcripts/transcript3/Course Work 1 - Part 2.txt")
    text_corpus = open("transcripts/transcript3/Course Work 1 - Part 2.txt").read()
    G = construct_knowledge_graph_parallel(text, doc_text=text_corpus)
    nx.write_graphml(G, "Coursework.graphml")
    keyphraseExtraction("transcripts/transcript3/Course Work 1 - Part 2.txt","Coursework")

    text = filter_keyterms("transcripts/transcript4/Data in the Cloud.txt")
    text_corpus = open("transcripts/transcript4/Data in the Cloud.txt").read()
    G = construct_knowledge_graph_parallel(text, doc_text=text_corpus)
    nx.write_graphml(G, "DataInCloud.graphml")
    keyphraseExtraction("transcripts/transcript4/Data in the Cloud.txt","DataInCloud")

    text = filter_keyterms("transcripts/transcript5/iaac.txt")
    text_corpus = open("transcripts/transcript5/iaac.txt").read()
    G = construct_knowledge_graph_parallel(text, doc_text=text_corpus)
    nx.write_graphml(G, "IaaC.graphml")
    keyphraseExtraction("transcripts/transcript5/iaac.txt","IaaC")

    text = filter_keyterms("transcripts/transcript6/Starting-PHP.txt")
    text_corpus = open("transcripts/transcript6/Starting-PHP.txt").read()
    G = construct_knowledge_graph_parallel(text, doc_text=text_corpus)
    nx.write_graphml(G, "startingPHP.graphml")
    keyphraseExtraction("transcripts/transcript6/Starting-PHP.txt","startingPHP")

    text = filter_keyterms("transcripts/transcript7/WithSecure.txt")
    text_corpus = open("transcripts/transcript7/WithSecure.txt").read()
    G = construct_knowledge_graph_parallel(text, doc_text=text_corpus)
    nx.write_graphml(G, "WithSecure.graphml")
    keyphraseExtraction("transcripts/transcript7/WithSecure.txt","WithSecure")

    text = filter_keyterms("transcripts/transcript8/Vulnerability-Fixing.txt")
    text_corpus = open("transcripts/transcript8/Vulnerability-Fixing.txt").read()
    G = construct_knowledge_graph_parallel(text, doc_text=text_corpus)
    nx.write_graphml(G, "VulnFix.graphml")
    keyphraseExtraction("transcripts/transcript8/Vulnerability-Fixing.txt","VulnFix")

    text = filter_keyterms("transcripts/transcript9/robpress.txt")
    text_corpus = open("transcripts/transcript9/robpress.txt").read()
    G = construct_knowledge_graph_parallel(text, doc_text=text_corpus)
    nx.write_graphml(G, "Robpress.graphml")
    keyphraseExtraction("transcripts/transcript9/robpress.txt","Robpress")

    text = filter_keyterms("transcripts/transcript10/ToolsandTechniques.txt")
    text_corpus = open("transcripts/transcript10/ToolsandTechniques.txt").read()
    G = construct_knowledge_graph_parallel(text, doc_text=text_corpus)
    nx.write_graphml(G, "ToolsAndTechniques.graphml")
    keyphraseExtraction("transcripts/transcript10/ToolsandTechniques","ToolsAndTechniques")



# Hyper_parameters:

# Filter_keyterms(): amount of times word has to appear

# Cluster_keyterms(): Word2vec parameters

# Process_keyterm(KG_size): how many nodes a keyterm can be linked to

# construct_knowledge_graph_parallel(): the size of the description for the keyterm nodes

# build_h_hop(h,threshold)

# compute_semantic_relatedness(): Doc2vec parameters

# build_keyterm_graph(window_size)

# personalized_page_rank(damping_factor=0.85, max_iter=100, tol=1e-6)

# extract_keyphrases(k)




# The only steps where the time complexity plays a big role are steps 4 and 5

# There are several potential optimizations that can be made to improve the speed of the algorithm:

# 1. Keyword extraction using RAKE or more optimized algorithms (This wasn't done in the paper)

# 2. Using Neo4J for larger graphs

# 3. Use caching to avoid redundant computations

# 4. Parallelize more of the algorithm

# 5. More effecient porgramming language

# 6. Faster internt for wikiepedia queries 

# 7. High Peformance Computing could potentially be used

# 8. Allcating more resources in python

# 9. Queries were made to wikipedia instead of DBpedia as it was a lot faster but DBpedia may provide better descriptions for the nodes
