# ====================================================================================================================================
# Module imports
# ====================================================================================================================================

import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
nltk.download('averaged_perceptron_tagger')
from scipy.linalg import norm
from scipy.spatial.distance import cosine
from matplotlib import pyplot as plt
import re
import pickle
from googletrans import Translator
from os import listdir, mkdir, chdir, getcwd
from os.path import join
import requests

from pywebio.input import input_group, input, select, checkbox, radio, actions, textarea
from pywebio.output import put_text, put_table, put_image, output, set_scope, remove, clear, get_scope, use_scope, put_markdown
from pywebio.output import put_processbar, set_processbar, put_html

# ====================================================================================================================================
# Setup
# ====================================================================================================================================

HOME = getcwd()

# download support files from github if not yet available

URL_FAA_pr = "https://github.com/fabio-a-oliveira/NLP_Regulations/blob/main/Regulations/FAR_Part121_nodes.xlsx?raw=true"
URL_ANAC_pr = "https://github.com/fabio-a-oliveira/NLP_Regulations/blob/main/Regulations/RBAC121_nodes.xlsx?raw=true"
URL_TOKENIZER_pr = "https://github.com/fabio-a-oliveira/NLP_Regulations/blob/main/models/tokenizer___2021-04-16_16-04-53.h5?raw=true"
URL_EMBEDDING_pr = "https://github.com/fabio-a-oliveira/NLP_Regulations/blob/main/models/embedding_layer___2021-04-16_16-04-53.h5?raw=true"
URL_FAA_cl = "https://github.com/fabio-a-oliveira/NLP_Regulations/blob/main/Regulations/FAR_Part121_nodes_labelled.xlsx?raw=true"
URL_ANAC_cl = "https://github.com/fabio-a-oliveira/NLP_Regulations/blob/main/Regulations/RBAC121_nodes_en_labelled.xlsx?raw=true"
URL_TOKENIZER_cl = "https://github.com/fabio-a-oliveira/NLP_Regulations/blob/main/models/tokenizer___2021-04-27_00-01-32.h5?raw=true"
URL_MODEL_cl = "https://github.com/fabio-a-oliveira/NLP_Regulations/blob/main/models/model___2021-04-27_00-01-32___GRU_stack_Softmax.h5?raw=true"
 
URLs = [URL_FAA_pr, URL_ANAC_pr, URL_TOKENIZER_pr, URL_EMBEDDING_pr, URL_FAA_cl, URL_ANAC_cl, URL_TOKENIZER_cl, URL_MODEL_cl]
filenames = ['FAA_pr.xlsx', 'ANAC_pr.xlsx', 'TK_pr.h5', 'EMB_pr.h5', 'FAA_cl.xlsx', 'ANAC_cl.xlsx', 'TK_cl.h5', 'MD_cl.h5']

chdir(HOME)
if 'models' not in listdir():
    mkdir('models')
    chdir('models')
    for n in range(len(URLs)):
        r = requests.get(URLs[n])
        with open(filenames[n], 'wb') as file:
            file.write(r.content)

chdir(HOME)

# filepaths for support files

filepath_FAA_pr = join(HOME, 'models', filenames[0])
filepath_ANAC_pr = join(HOME, 'models', filenames[1])
filepath_TK_pr = join(HOME, 'models', filenames[2])
filepath_EMB_pr = join(HOME, 'models', filenames[3])
filepath_FAA_cl = join(HOME, 'models', filenames[4])
filepath_ANAC_cl = join(HOME, 'models', filenames[5])
filepath_TK_cl = join(HOME, 'models', filenames[6])
filepath_MD_cl = join(HOME, 'models', filenames[7])

# Variable definitions for comparison/projection

df_FAA_pr = pd.read_excel(filepath_FAA_pr)[['title','requirement']]
df_ANAC_pr = pd.read_excel(filepath_ANAC_pr)[['title','requirement']]
translator = Translator()

with open(filepath_TK_pr, 'rb') as file:
    tokenizer_pr = pickle.load(file)

with open(filepath_EMB_pr, 'rb') as file:
    embedding_layer = pickle.load(file)

# Variable definitions for classification

df_FAA_cl = pd.read_excel(filepath_FAA_cl)[['title','requirement','label','tag']]
df_ANAC_cl = pd.read_excel(filepath_ANAC_cl)[['title','requirement','label','tag']]
translator = Translator()

with open(filepath_TK_cl, 'rb') as file:
    tokenizer_cl = pickle.load(file)
    
model = tf.keras.models.load_model(filepath_MD_cl)

sequence_length = 200

# ====================================================================================================================================
# Helper functions
# ====================================================================================================================================

def relevant_words(excerpt):
    tokens = re.findall('[a-zA-Z]{3,}', excerpt)
    words, allowed_pos = [], ['NN','NNS','NNP','NNPS','JJ','RB','VB','VBG','VBN','VBP','VBZ','VBD']
    for item in nltk.pos_tag(tokens):
        if item[0] not in words and item[1] in allowed_pos:
            words.append(item[0])
    return words

def sorted_keywords(words, tokenizer):
    w,f = [], []

    for word in words:
        if word in tokenizer.word_index.keys():
            w.append(word)
            f.append(tokenizer.word_counts[word])

    sorting_indices = np.array(f).argsort()
    return np.array(w)[sorting_indices]

def jaccard_mod(string1, string2, tokenizer, embedding_layer, max_words = None):
    # with max_words != None, repetitions are ignored, only the frequency in the original dictionary is used
    # with max_words == None, repetitions are somewhat accounted for (not very thoughtfully..)
    
    cutoff = .5
    
    if max_words == None:
        tokens1 = tokenizer.texts_to_sequences(relevant_words(string1))
        tokens2 = tokenizer.texts_to_sequences(relevant_words(string2))
    else:
        tokens1 = sorted_keywords(relevant_words(string1), tokenizer)
        tokens1 = tokenizer.texts_to_sequences(tokens1[:max_words])
        tokens2 = sorted_keywords(relevant_words(string2), tokenizer)
        tokens2 = tokenizer.texts_to_sequences(tokens2[:max_words])
        
    if len(tokens1) < 2 or len(tokens2) < 2:
        return 0
    else:
        embedding1 = embedding_layer(np.array(tokens1)).numpy().squeeze()
        norms1 = norm(embedding1, axis=1)
        embedding1 = embedding1[norms1 != 0]
        norms1 = norms1[norms1 != 0].reshape([-1,1])

        embedding2 = embedding_layer(np.array(tokens2)).numpy().squeeze()
        norms2 = norm(embedding2, axis=1)
        embedding2 = embedding2[norms2 != 0]
        norms2 = norms2[norms2 != 0].reshape([-1,1])

        cosine_similarity = np.matmul(embedding1, embedding2.T) / norms1 / norms2.T
        tril = np.ones_like(cosine_similarity) * (np.tril(cosine_similarity) != 0).astype(int)

        intersection = np.sum((cosine_similarity * cosine_similarity) > cutoff)
        union = cosine_similarity.shape[0] + cosine_similarity.shape[1] - intersection

        return intersection/union
    
    
# ====================================================================================================================================
# main()    
# ====================================================================================================================================

def main():

    # meta for page preview
    URL = 'https://github.com/fabio-a-oliveira/PyWebIO_NLP_demo/raw/main/images/requirement_comparison_small.png'
    put_html('<meta property="og:image" content="' + URL + '">')
    
    # header
    put_markdown('## NLP applied to aviation regulations')
    put_markdown('### Demonstrations using 14 CFR Part 121 and RBAC 121')
    put_text('This series of demos showcases the use of some Natural Language Processing (NLP) techniques to aviation regulations. Choose your favorite and enjoy!')
    put_markdown('---')
    
    for _ in range(1000):
        choose_demonstration()
    
# ====================================================================================================================================
# choose_demonstration()
# ====================================================================================================================================

@use_scope('demo', clear=True)
def choose_demonstration():
       
    # Select between comparison and classification
    
    type_of_problem = radio('What type of application are you interested in?', 
                            ['Requirements comparison', 'Requirements classification'])
    
    # Select between available demonstrations
    
    available_demos = ['Choose ANAC requirement and find corresponding FAA requirement',
                       'Provide custom input and find corresponding FAA requirement',
                       'Show me how this works',
                       'Choose FAA requirement and classify it',
                       'Choose ANAC requirement and classify it',
                       'Provide custom input and classify it',
                       'Show me how this works']
    
    if type_of_problem == 'Requirements comparison':
        short_description('comparison')
        type_of_demo = radio('What would you like to see?', available_demos[:3])  
        
    elif type_of_problem == 'Requirements classification':
        short_description('classification')
        type_of_demo = radio('What would you like to see?', available_demos[3:])
        
    # Respond to each demo selection
    
    if type_of_demo == available_demos[0]:
        find_FAA_from_ANAC(df_ANAC_pr, df_FAA_pr, tokenizer_pr, embedding_layer)
        
    elif type_of_demo == available_demos[1]:
        find_FAA_from_input(df_FAA_pr, tokenizer_pr, embedding_layer)
        
    elif type_of_demo == available_demos[2] and type_of_problem == 'Requirements comparison':
        about_comparison()
        
    elif type_of_demo == available_demos[3]:
        classify_FAA(df_FAA_cl, tokenizer_cl, model)
        
    elif type_of_demo == available_demos[4]:
        classify_ANAC(df_ANAC_cl, tokenizer_cl, model)
        
    elif type_of_demo == available_demos[5]:
        classify_from_input(tokenizer_cl, model)
        
    elif type_of_demo == available_demos[6] and type_of_problem == 'Requirements classification':
        about_classification()
    
    actions('Start over?', ['confirm'])

# ====================================================================================================================================
# find_FAA_from_ANAC()    
# ====================================================================================================================================
    
@use_scope('demo', clear=True)
def find_FAA_from_ANAC(df_ANAC, df_FAA, tokenizer, embedding_layer):
        
    random_prompt = "I don't care, just choose at random!"
    
    selection = select("Which requirement do you want to compare?",
                       [random_prompt] + df_ANAC.title.to_list())
    
    if selection == random_prompt:
        EXAMPLE_NUMBER = np.random.randint(0,df_ANAC.shape[0])
        req = df_ANAC.requirement[EXAMPLE_NUMBER].replace('\n',' ')
        title = df_ANAC.title[EXAMPLE_NUMBER]
    else:
        req = df_ANAC.requirement.loc[df_ANAC.title == selection].values[0]
        req = str(req).replace('\n', ' ')
        title = selection
    
    put_markdown('### Selected requirement:')    
    
    put_table([['ANAC RBAC ' + title]], header = ['Requirement:'])
    put_table([[req]], header = ['Original text:'])
    
    translation = Translator().translate(req, dest='en', src='pt').text
    
    put_table([[translation]], header = ['Translation:'])
    
    set_scope('wait')
    put_text('Looking for similar requirements in FAA 14-CFR Part 121...', scope = 'wait')
    
    similarity = []
    put_processbar('bar',init=0, auto_close=True)
    
    for n, ref in enumerate(df_FAA.requirement):
        similarity.append(jaccard_mod(translation, ref, tokenizer, embedding_layer))
        set_processbar('bar', value = (n+1) / df_FAA.shape[0])
        
    match_index = np.array(similarity).argmax()
    
    clear('wait')
    remove('wait')
    
    put_markdown('### Matching reference:')
    
    put_table([['FAR' + df_FAA.title[match_index], np.round(np.max(similarity),3)]], 
              header = ['Match:', 'Similarity Index:'])
    put_table([[df_FAA.requirement[match_index]]], header = ['Requirement:'])
    
    plt.figure(figsize = (15,5));
    plt.hist(np.array(similarity), 50);
    plt.title('Histogram - similarity translation vs reference');
    plt.savefig(join('images', 'temp.png'))
    #plt.show();
        
    put_image(open(join('images', 'temp.png'), 'rb').read(), width = '100%')
    
# ====================================================================================================================================
# find_FAA_from_input()    
# ====================================================================================================================================
    
@use_scope('demo', clear=True)
def find_FAA_from_input(df_FAA, tokenizer, embedding_layer):
        
    placeholder = "Type or paste your requirement here"
    help_text = "PRO TIP: use any language you want, I'll auto-detect and translate to English"
    
    req = textarea("What requirement do you want to compare?",
                   placeholder = placeholder,
                   help_text = help_text)
    
    put_markdown('### Selected requirement:')    
    put_table([[req]], header = ['Original text:'])
    
    translation = Translator().translate(req, dest='en')
    src_language = translation.src
    translation = translation.text
    
    if src_language != 'en':
        put_table([[translation]], header = ["Translation (from '{}' to 'en')".format(src_language)])
    
    set_scope('wait')
    put_text('Looking for similar requirements in FAA 14-CFR Part 121...', scope = 'wait')
    
    similarity = []
    put_processbar('bar',init=0, auto_close=True)
    
    for n, ref in enumerate(df_FAA.requirement):
        similarity.append(jaccard_mod(translation, ref, tokenizer, embedding_layer))
        set_processbar('bar', value = (n+1) / df_FAA.shape[0])
        
    match_index = np.array(similarity).argmax()
    
    clear('wait')
    remove('wait')
    
    put_markdown('### Matching reference:')
    
    put_table([['FAR' + df_FAA.title[match_index], np.round(np.max(similarity),3)]], 
              header = ['Match:', 'Similarity Index:'])
    put_table([[df_FAA.requirement[match_index]]], header = ['Requirement:'])
    
    plt.figure(figsize = (15,5));
    plt.hist(np.array(similarity), 50);
    plt.title('Histogram - similarity translation vs reference');
    plt.savefig(join('images', 'temp.png'))
    #plt.show();
        
    put_image(open(join('images', 'temp.png'), 'rb').read(), width = '100%')    
    
# ====================================================================================================================================
# about_comparison()
# ====================================================================================================================================
    
@use_scope('demo', clear=True)
def about_comparison():
    put_image(open(join('images', 'requirement_comparison.png'), 'rb').read(), width = '100%')
    
# ====================================================================================================================================
# classify_FAA()    
# ====================================================================================================================================

@use_scope('demo', clear=True)    
def classify_FAA(df, tokenizer, model):

    random_prompt = "I don't care, just choose at random!"
    
    selection = select("Which requirement do you want to classify?",
                       [random_prompt] + df.title.to_list())
    
    if selection == random_prompt:
        EXAMPLE_NUMBER = np.random.randint(0,df.shape[0])
        req = df.requirement[EXAMPLE_NUMBER].replace('\n',' ')
        tag = df.tag[EXAMPLE_NUMBER]
        title = df.title[EXAMPLE_NUMBER]
    else:
        req = df.requirement.loc[df.title == selection].values[0]
        req = str(req).replace('\n', ' ')
        title = selection
        tag = df.tag.loc[df.title == selection].values[0]

    put_markdown('### Selected requirement:')    
    
    put_table([['FAA 14 CFR ' + title, tag]], header = ['Requirement:', 'Correct classification:'])
    put_table([[req]], header = ['Original text:'])
    
    tokens = tokenizer.texts_to_sequences([req])
    padded_tokens = tf.keras.preprocessing.sequence.pad_sequences(tokens, maxlen = sequence_length)
    X = tf.constant(np.array(padded_tokens))
    prediction = model.predict(X)
    predicted_prob = prediction.max()
    predicted_label = prediction.argmax()
    predicted_tag = ['Title Only', 'Aircraft', 'Operator'][predicted_label]
    
    put_table([[predicted_tag, str(np.round(100*predicted_prob,3)) + '%']], header = ['Predicted Classification:', 'Level of Confidence'])

# ====================================================================================================================================
# classify_ANAC()    
# ====================================================================================================================================
  
@use_scope('demo', clear=True)
def classify_ANAC(df, tokenizer, model):

    random_prompt = "I don't care, just choose at random!"
    
    selection = select("Which requirement do you want to classify?",
                       [random_prompt] + df.title.to_list())
    
    if selection == random_prompt:
        EXAMPLE_NUMBER = np.random.randint(0,df.shape[0])
        req = df.requirement[EXAMPLE_NUMBER].replace('\n',' ')
        title = df.title[EXAMPLE_NUMBER]
        tag = df.tag[EXAMPLE_NUMBER]
    else:
        req = df.requirement.loc[df.title == selection].values[0]
        tag = df.tag.loc[df.title == selection].values[0]
        req = str(req).replace('\n', ' ')
        title = selection

    put_markdown('### Selected requirement:')    
    
    put_table([['ANAC RBAC ' + title, tag]], header = ['Requirement:', 'Correct classification:'])
    put_table([[req]], header = ['Original text:'])
    
    translation = Translator().translate(req, src = 'pt', dest = 'en').text
    put_table([[translation]], header = ['Translation:'])
    
    tokens = tokenizer.texts_to_sequences([translation])
    padded_tokens = tf.keras.preprocessing.sequence.pad_sequences(tokens, maxlen = sequence_length, 
                                                                  truncating='pre', padding='pre')
    X = tf.constant(np.array(padded_tokens))
    prediction = model.predict(X)
    predicted_prob = prediction.max()
    predicted_label = prediction.argmax()
    predicted_tag = ['Title Only', 'Aircraft', 'Operator'][predicted_label]
    
    put_table([[predicted_tag, str(np.round(100*predicted_prob,3)) + '%']], header = ['Predicted Classification:', 'Level of Confidence'])

# ====================================================================================================================================
# classify_from_input()
# ====================================================================================================================================

@use_scope('demo', clear=True)
def classify_from_input(tokenizer, model):
    
    placeholder = "Type or paste your requirement here"
    help_text = "PRO TIP: use any language you want, I'll auto-detect and translate to English"
    
    req = textarea("What requirement do you want to classify?",
                   placeholder = placeholder,
                   help_text = help_text)
    
    put_markdown('### Selected requirement:')    
    put_table([[req]], header = ['Original text:'])
    
    translation = Translator().translate(req, dest='en')
    src_language = translation.src
    translation = translation.text
    
    if src_language != 'en':
        put_table([[translation]], header = ["Translation (from '{}' to 'en')".format(src_language)])
        
    tokens = tokenizer.texts_to_sequences([translation])
    padded_tokens = tf.keras.preprocessing.sequence.pad_sequences(tokens, maxlen = sequence_length)
    X = tf.constant(np.array(padded_tokens))
    prediction = model.predict(X)
    predicted_prob = prediction.max()
    predicted_label = prediction.argmax()
    predicted_tag = ['Title Only', 'Aircraft', 'Operator'][predicted_label]
    
    put_table([[predicted_tag, str(np.round(100*predicted_prob,3)) + '%']], header = ['Predicted Classification:', 'Level of Confidence'])

# ====================================================================================================================================
# about_classification()
# ====================================================================================================================================

@use_scope('demo', clear=True)
def about_classification():
    put_image(open(join('images', 'requirement_classification.png'), 'rb').read(), width = '100%')
    
# ====================================================================================================================================
# short_description()
# ====================================================================================================================================

def short_description(type = 'classification'):
    
    if type == 'classification':
        
        put_markdown(r'''
                     ### Requirement classification
                      This is what happens in this demonstration:
                      * You will select a requirement
                      * Your selection will be pre-processed according to a sequence of steps:
                          1. _Tokenization_: identification of individual words
                          2. _Padding_: truncated or filled with zeros to make a sequence of 200 tokens (words)
                          3. _Embedding_: translation of each word to a semantic vector representation with 300 dimensions
                      * The resulting sequence is fed to a _Recurrent Neural Network_ that outputs the probability of each class
                      * The class with the highest probability is selected
                      ---
                      ''', lstrip = True)
        
    if type == 'comparison':   
        
        put_markdown(r'''
                     ### Requirement comparison
                     This is what happens in this demonstration:
                     * You will select a requirement
                     * Your selection will be pre-processed according to a sequence of steps:
                         1. _Translation_: automatic translation to English (if required)
                         2. _Tokenization_: identification of individual words
                         3. _POS filter_: removal of irrelevant parts-of-speech
                         4. _Sorting_: remaining words are sorted according to their _Inverse Document Frequency (IDF)_
                         5. _Embedding_: translation of each word to a semantic vector representation with 300 dimensions
                     * Each FAA requirement is processed similarly and results are compared to find the best match considering the proportion of words that have similar vector representations
                     * The FAA requirement with the largest semantic overlap is selected
                     ---
                    ''', lstrip = True)
    
