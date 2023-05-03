import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt


# Step 1: Collect data
text_corpus = "transcripts/transcript1/CosmosDB.txt"


# Step 2: Preprocess data
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
    # Join tokens back into a string
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

# Step 3: Extract candidate phrases
def extract_candidate_phrases(text):
    grammar = "NP: {<JJ>*<NN.*>+}"
    chunker = nltk.RegexpParser(grammar)
    tagged_tokens = nltk.pos_tag(text)
    tree = chunker.parse(tagged_tokens)
    candidate_phrases = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            phrase = ' '.join([token for token, tag in subtree.leaves()])
            candidate_phrases.append(phrase)
    print(candidate_phrases)
    return candidate_phrases
    

# Step 4: Filter phrases
def filter_phrases(phrases):
    phrase_freq = defaultdict(int)
    for phrase in phrases:
        phrase_freq[phrase] += 1
    filtered_phrases = [phrase for phrase in phrases if phrase_freq[phrase] > 1 and len(phrase) > 1]
    return filtered_phrases

# Step 5: Construct knowledge graph
def construct_knowledge_graph(phrases):
    G = nx.Graph()
    node_url_dict = {}
    for phrase in phrases:
        G.add_node(phrase)
        node_url_dict[phrase] = f"https://en.wikipedia.org/wiki/{phrase.replace(' ', '_')}"
    for i in range(len(phrases)):
        for j in range(i+1, len(phrases)):
            w1, w2 = sorted([phrases[i], phrases[j]])
            if w1 != w2:
                G.add_edge(w1, w2)
    # Add tooltips with links to corresponding pages for each node
    for node in G.nodes():
        node_url = node_url_dict[node]
        G.nodes[node]['tooltip'] = f'<a href="{node_url}" target="_blank">{node}</a>'
    return G



# Run the program
def KG(text_corpus):
    text = open(text_corpus).read()
    tokens = preprocess(text)
    phrases = extract_candidate_phrases(tokens)
    filtered_phrases = filter_phrases(phrases)
    G = construct_knowledge_graph(filtered_phrases)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.axis('off')
    plt.show()
KG(text_corpus)

# I have to construct knowledge graph in the paper they just use dbpedia