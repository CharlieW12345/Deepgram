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
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return lemmatized_tokens

def extract_noun_phrases(tokens):
    grammar = "NP: {<JJ>*<NN.*>+}"
    chunker = nltk.RegexpParser(grammar)
    tagged_tokens = nltk.pos_tag(tokens)
    tree = chunker.parse(tagged_tokens)
    candidate_phrases = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            phrase = ' '.join([token for token, tag in subtree.leaves()])
            candidate_phrases.append(phrase)
    return candidate_phrases

def filter_phrases(text):
    text = open(text).read()
    phrase = preprocess(text)
    phrases = extract_noun_phrases(phrase)
    # Filter phrases that occur more than once and have at least two words
    phrase_freq = defaultdict(int)
    for phrase in phrases:
        phrase_freq[phrase] += 1
    filtered_phrases = [phrase for phrase in phrases if phrase_freq[phrase] > 1 and len(phrase.split()) > 1]
    return filtered_phrases


# Step 4: Construct knowledge graph



def process_keyterm(kt, mapping_dict):
    site = mwclient.Site('en.wikipedia.org')
    candidate_mappings = {}
    # Get entity candidates for each keyterm
    page = site.pages[kt]
    if page.exists:
        for link in page.links():
            candidate = link.page_title
            if candidate != kt:
                if candidate in candidate_mappings:
                    candidate_mappings[candidate] += 1
                else:
                    candidate_mappings[candidate] = 1
        # Calculate prior probability for each candidate entity
        total_count = sum(candidate_mappings.values())
        for candidate, count in candidate_mappings.items():
            prior_prob = count / total_count
            if candidate in mapping_dict:
                mapping_dict[candidate][kt] = prior_prob
            else:
                mapping_dict[candidate] = {kt: prior_prob}

def construct_knowledge_graph_parallel(keyterms, num_processes=None, G=None):
    if not G:
        G = nx.Graph()
    site = mwclient.Site('en.wikipedia.org')

    # Build a mapping dictionary for keyterms
    mapping_dict = multiprocessing.Manager().dict()
    processes = []
    for kt in keyterms:
        p = multiprocessing.Process(target=process_keyterm, args=(kt, mapping_dict))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    # Add nodes to the graph
    for entity in mapping_dict:
        G.add_node(entity)
        # Get entity description from Wikipedia
        page = site.pages[entity]
        if page.exists:
            if page.redirect:
                target_title = page.redirects_to().name
                page = site.pages[target_title]
                description = extract_summary(page.text())[0:300] + '...' if len(extract_summary(page.text())) > 300 else extract_summary(page.text()) # can experiment with this value
                G.nodes[entity]['description'] = description
            else:
                description = extract_summary(page.text())[0:300] + '...' if len(extract_summary(page.text())) > 300 else extract_summary(page.text()) 
                G.nodes[entity]['description'] = description

        else:
            G.nodes[entity]['description'] = 'No description available.'
        
    # Add edges to the graph
    for i, entity1 in enumerate(mapping_dict):
        for j in range(i+1, len(mapping_dict)):
            entity2 = list(mapping_dict.keys())[j]
            edge_weight = 0
            for kt in mapping_dict[entity1]:
                if kt in mapping_dict[entity2]:
                    edge_weight += mapping_dict[entity1][kt] * mapping_dict[entity2][kt]
            if edge_weight > 0:
                G.add_edge(entity1, entity2, weight=edge_weight)

    return G




def extract_summary(text):
    expr = re.compile("(?:^|}})([^{{\}}]+)(?:\{{|$)")
    content =  "".join(expr.findall(text)[0:])
    expr = re.compile("'''[\s\S]*$")
    return "".join(expr.findall(content)[0:])




# So the total time complexity of this part of the function is O(N^2*K). K is the number of keyterms, N is the number of unique entities found across all the keyterms


    

def cluster_keyphrases(keyphrases):
    # Convert keyphrases to tf-idf vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(keyphrases)
    # Cluster keyphrases using affinity propagation
    clustering = AffinityPropagation(damping=0.9, max_iter=1000).fit(X)
    labels = clustering.labels_
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(keyphrases[i])
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
#     text = filter_phrases("transcripts/transcript1/cosmosdb.txt")
#     G = construct_knowledge_graph_parallel(text)
#     text = filter_phrases("transcripts/transcript2/From Python Script to Python app.txt")
#     G = construct_knowledge_graph_parallel(text,G)
#     text = filter_phrases("transcripts/transcript3/Course Work 1 - Part 2.txt")
#     G = construct_knowledge_graph_parallel(text,G)
#     text = filter_phrases("transcripts/transcript4/Data in the Cloud.txt")
#     G = construct_knowledge_graph_parallel(text,G)
#     text = filter_phrases("transcripts/transcript5/iaac.txt")
#     G = construct_knowledge_graph_parallel(text,G)
#     text = filter_phrases("transcripts/transcript6/Starting-PHP.txt")
#     G = construct_knowledge_graph_parallel(text,G)
#     text = filter_phrases("transcripts/transcript7/WithSecure.txt")
#     G = construct_knowledge_graph_parallel(text,G)
#     text = filter_phrases("transcripts/transcript8/Vulnerability-Fixing.txt")
#     G = construct_knowledge_graph_parallel(text,G)
#     text = filter_phrases("transcripts/transcript9/robpress.txt")
#     G = construct_knowledge_graph_parallel(text,G)
#     text = filter_phrases("transcripts/transcript10/ToolsandTechniques.txt")
#     G = construct_knowledge_graph_parallel(text,G)
#     nx.write_graphml(G, "knowledge_graph.graphml")







# Knowledge graph is constructed from all transcripts and then saved as graphml

def clusters(text_corpus):
    keyphrases = filter_phrases(text_corpus)
    clusters = cluster_keyphrases(keyphrases)
    # print(dict(collections.OrderedDict(sorted(clusters.items()))))
    return clusters


# Run functions from here

# Define function to construct h-hop keyterm graph

def construct_h_hop_keyterm_graph(KG, cluster, h):
    # Initialize graph
    G = nx.Graph()
    KG = nx.from_dict_of_lists(KG)
    # Add keyterms as nodes to the graph
    for keyterm in cluster:
        if keyterm in KG:
            G.add_node(keyterm)

    # Add h-hop neighbors to the graph
    for i in range(h):
        for keyterm in G.nodes:
            if keyterm in KG:
                neighbors = KG.neighbors(keyterm)
                for neighbor in neighbors:
                    if neighbor not in G.nodes and neighbor in KG:
                        G.add_node(neighbor)
                    G.add_edge(keyterm, neighbor)

    # Calculate edge weights using shared neighbor weighting
    for u, v in G.edges:
        shared_neighbors = list(nx.common_neighbors(KG, u, v))
        weight = sum([1/KG.degree(neighbor) for neighbor in shared_neighbors])
        G.edges[u, v]['weight'] = weight

    return G



def H_hop():
    cluster = clusters("transcripts/transcript1/cosmosdb.txt")
    print(cluster)
    KG = nx.read_graphml("knowledge_graph.graphml")
    h_hop = construct_h_hop_keyterm_graph(cluster, KG, 2)
    for node, attrs in h_hop.nodes(data=True):
        print(node, attrs)

H_hop()

def PageRank(G, h, alpha, num_keywords):
    # Construct the h-hop keyterm graph
    keyterms = []
    for node in G.nodes():
        if 'keyterm' in G.nodes[node] and G.nodes[node]['keyterm']:
            keyterms.append(node)
    h_hop_graph = construct_h_hop_keyterm_graph(G, h, keyterms)
    
    # Compute personalized PageRank
    node_weights = dict.fromkeys(h_hop_graph.nodes(), 0)
    for keyterm in keyterms:
        node_weights[keyterm] = 1
    node_weights = nx.pagerank(h_hop_graph, alpha=alpha, personalization=node_weights, max_iter=100)

    # Rank nodes by weight and return the top num_keywords
    ranked_nodes = sorted(node_weights.items(), key=lambda x: x[1], reverse=True)[:num_keywords]
    keywords = [node for node, weight in ranked_nodes]

    return keywords

def extract_candidate_phrases(processed):
    
    # Extract noun phrases using part-of-speech tagging
    grammar = r"""
        NP: {<DT|PP\$>?<JJ>*<NN>+}
            {<NNP>+}
            """
    cp = nltk.RegexpParser(grammar)
    tagged_tokens = nltk.pos_tag(processed)
    tree = cp.parse(tagged_tokens)
    
    # Get candidate phrases from the parse tree
    candidate_phrases = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            candidate_phrases.append(' '.join([word for word, tag in subtree.leaves()]))
    
    return candidate_phrases

def filter_phrases(phrases):
    # Filter phrases that occur more than once and have at least two words
    phrase_freq = defaultdict(int)
    for phrase in phrases:
        phrase_freq[phrase] += 1
    filtered_phrases = [phrase for phrase in phrases if phrase_freq[phrase] > 1 and len(phrase.split()) > 1]
    return filtered_phrases

#The value of K is determined empirically by evaluating the performance of the keyphrase extraction algorithm on a development set. Perhaps change this so it can have multiple words aswell
def extract_keywords(keyphrases, num_keywords):
    # Extract candidate phrases
    candidate_phrases = extract_candidate_phrases(keyphrases)
    
    # Filter candidate phrases
    filtered_phrases = filter_phrases(candidate_phrases)
    
    # Rank candidate phrases by personalized PageRank scores
    ranked_phrases = []
    for phrase in filtered_phrases:
        if phrase in keyphrases:
            score = keyphrases[phrase]
        else:
            score = 0.0
        ranked_phrases.append((phrase, score))
    ranked_phrases.sort(key=lambda x: x[1], reverse=True)

    # Select top K phrases
    selected_phrases = [phrase[0] for phrase in ranked_phrases[:num_keywords]]

    return selected_phrases


# I have to construct knowledge graph in the paper they just use dbpedia

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

# 9. DBpedia was too slow, but worth looking into for the more enhanced results, perhaps the way they construct in the paper is with a lot less nodes like my original KG so need to look into thi 
# These are just a few potential optimizations that can be made to speed up the algorithm. Depending on the specific use case and constraints, there may be other optimizations that are more appropriate.