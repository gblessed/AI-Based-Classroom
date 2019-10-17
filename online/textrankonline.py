from collections import OrderedDict
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import names
import re
from nltk.stem.wordnet import WordNetLemmatizer
import speech_recognition as sr
import requests 
  
URL = "http://192.168.137.1:8090/"
  

r = sr.Recognizer()
mic = sr.Microphone(device_index=2)
#audiofile = sr.AudioFile('paper.wav')
x = ''
lem = WordNetLemmatizer()

stop_words =   set(stopwords.words('english'))
new_words = ['using', 'show', 'large', 'also', 'one', 'two', 'new', 'previously', 'shown']
stop_words = stop_words.union(new_words)


d = 0.85 # damping coefficient, usually is .85
min_diff = 1e-5 # convergence threshold
steps = 10 # iteration steps
nodeweight = None
#gett 
namelist = [w.lower() for w in names.words('male.txt')]
for w in names.words('female.txt'):
    namelist.append(w.lower())


def get_token_pairs(window_size, sentences):
    #    """Build token_pairs from windows in sentences"""
    token_pairs = list()
    for i, word in enumerate(sentences):
        #print(i, word)
        for j in range(i+1, i+window_size):
            if j >= len(sentences):
                break
            pair = (word, sentences[j])
            if pair not in token_pairs:
                token_pairs.append(pair)
    return token_pairs

def get_vocab(sentences):
    """Get all tokens"""
    vocab = OrderedDict()
    i = 0
    for word in sentences:
        if word not in vocab:
            vocab[word] = i
            i += 1
    return vocab


def get_matrix(vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1 # assign an initial weight of 1 to all edges.
            
        # Get Symmeric matrix
        g = symmetrize(g)
        
        # Normalize matrix by column
        norm = np.sum(g, axis=0) # we get the number of links that each node is connected to
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in normv, this is to divide be by the number of vertices that each node is connected to.
        
        return g_norm
    
def symmetrize(a):
        return a + a.T - np.diag(a.diagonal())


def get_keywords(number=10):
    """Print top number keywords"""
    parameters = dict()
    node_weight = OrderedDict(sorted(nodeweight.items(), key=lambda t: t[1], reverse=True))
    for i, (key, value) in enumerate(node_weight.items()):
        print(key + ' - ' + str(value))
        parameters[key] = value
        if i > number:
            break
    return parameters

def speechrecognition():
    print('BEGIN TO SPEAK....')
    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(mic)
    print('FINSISHED LISTENING ....PROCESSING NOW/.............')
    try:
        statement = str(r.recognize_google(audio))
        global x; x += ' '+statement
    except:
        print('some error occured')
    print(x)
    return x
    
        
'''f = open('hardware.txt','rU')
raw = f.readlines()'''
tokens = []

while True:
    tokens += nltk.word_tokenize(speechrecognition())

    #text = nltk.Text(tokens)
    edited = []

    for word in tokens:
        word = word.lower()
        text = re.sub('[^a-zA-Z0-9]', '', word)
        if text != '' and text not in stop_words not in namelist:#ANNONYMIZATION
            edited.append(lem.lemmatize(text))#LEMMATIZATION
            
    #print(edited)
    token_pairs= get_token_pairs(window_size = 10, sentences =edited)
    vocab = get_vocab(edited)
    g = get_matrix(vocab, token_pairs)
    pr = np.array([1] * len(vocab))

    previous_pr = 0
    for epoch in range(steps):
        pr = (1-d) + d * np.dot(g, pr)
        if abs(previous_pr - sum(pr))  < min_diff:
            break
        else:
            previous_pr = sum(pr)

    node_weight = dict()

    for word, index in vocab.items():
            node_weight[word] = pr[index]
            
    nodeweight = node_weight
    parameters = get_keywords()
    # defining a params dict for the parameters to be sent to the API 
    PARAMS = {'model_identifier':'example_support_vector_machine',
              'ml_pipeline_identifier' : 'ai_based_classroom',
               'features' : parameters
              } 
      
    # sending get request and saving the response as response object 
    rep = requests.post(url = URL, json = PARAMS) 
      
    # extracting data in json format 
    data = rep.json()
    print(data)
        


