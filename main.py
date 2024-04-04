"""
GROUP 1: Anunay Amrit, Angelica Cabato, Pranav Vijay Chand, Riya Chapatwala, Sai Satya Jagannadh Doddipatla, Nhat Ho

Dr. Shah

CPSC 535: Advanced Algorithms (Spring 2024)

"""

import requests
from bs4 import BeautifulSoup
import nltk
import string
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from time import perf_counter_ns

"""Core Text Search Algorithms"""
##################### Rabin-Karp ########################

"""PROVIDED CODE"""
def rabin_karp(text, pattern, d, q):
    n = len(text)
    m = len(pattern)
    h = pow(d, m-1) % q
    p = 0
    t = 0
    result = []

    #Preprocessing: Calculation of hash value for the pattern
    for i in range(m):
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q

    for s in range(n - m + 1):
        if p == t:
            if text[s:s+m] == pattern:
                result.append(s)

        if s < n-m:
            t = (d*(t-ord(text[s])*h) + ord(text[s+m])) % q

            if t<0:
                t = t+q
    return result

"""EDITED CODE FOR URL INPUT"""
def rabin_karp_url(url, patterns):
    text = extract_text(url)

    rk_matches = {}
    rk_start_time = perf_counter_ns()

    for pattern in patterns:
        rk_cur_matches = rabin_karp(text, pattern, d=256, q=101)
        rk_matches[pattern] = rk_cur_matches

    rk_end_time = perf_counter_ns()

    rk_execution_time = rk_end_time - rk_start_time

    return rk_matches, rk_execution_time

#########################################################

##################### Suffix Tree #######################
"""PROVIDED CODE"""
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True


class SuffixTree:
    def __init__(self, text):
        self.text = text
        self.trie = Trie()
        self.build()

    def build(self):
        for i in range(len(self.text)):
            self.trie.insert(self.text[i:])

    def display(self, node=None, prefix=''):
        node = node or self.trie.root
        if not node.children:
            print(prefix)
        else:
            for char, child in node.children.items():
                self.display(child, prefix + char)

    def search(self, pattern):
        node = self.trie.root
        if pattern not in node.children:
            return False
        else:
            return True

"""EDITED CODE FOR URL INPUT"""


def suffix_tree_url(url, patterns):
    text = extract_text(url)
    tokens = word_tokenize(text)
    st_occurrences = {}
    st_start_time = perf_counter_ns()

    text_tree = SuffixTree(tokens)

    for pattern in patterns:
        st_cur_occurrences = text_tree.search(pattern)
        st_occurrences[pattern] = st_cur_occurrences

    st_end_time = perf_counter_ns()

    st_execution_time = st_end_time - st_start_time

    return st_occurrences, st_execution_time

#########################################################

##################### Suffix Array ######################
"""PROVIDED CODE"""
def construct_suffix_array(text):
    suffixes = [(text[i:], i) for i in range(len(text))]
    suffixes.sort(key=lambda x: x[0])
    suffix_array = [item[1] for item in suffixes]
    return suffix_array

"""EDITED CODE FOR URL INPUT"""


def search_suffix_array(pattern, text, suffix_array):
    n = len(text)
    m = len(pattern)

    # Initialize left and right indicies
    left = 0
    right = n - 1

    # preform binary search
    while left <= right:
        middle = left + (right - left) // 2
        substring = text[suffix_array[middle]:suffix_array[middle] + m]

        if substring == pattern:
            return suffix_array[middle]

        if substring < pattern:
            left = middle + 1
        else:
            right = middle - 1
    return None

def suffix_array_url(url, patterns):
    text = extract_text(url)
    tokens = word_tokenize(text)
    sa_matches = {}
    sa_start_time = perf_counter_ns()

    text_suffix_array = construct_suffix_array(text)

    for pattern in patterns:
        sa_cur_matches = search_suffix_array(pattern, text, text_suffix_array)
        if sa_cur_matches != None:
            sa_matches[pattern] = sa_cur_matches

    sa_end_time = perf_counter_ns()

    sa_execution_time = sa_end_time - sa_start_time

    return sa_matches, sa_execution_time

#########################################################

############### Naive String Matching ###################

"""PROVIDED CODE"""
def naive_string_matcher(text, pattern):

    n = len(text)
    m = len(pattern)
    matches = []

    for s in range(n - m + 1):
        print(text[s:s+m])
        if text[s:s+m] == pattern:
            matches.append(s)

    return matches

"""EDITED CODE FOR URL INPUT"""

def naive_string_matcher_url(url, patterns):
    text = extract_text(url)

    nsm_matches = {}
    nsm_start_time = perf_counter_ns()

    for pattern in patterns:
        nsm_cur_matches = kmp_search(text, pattern)
        nsm_matches[pattern] = nsm_cur_matches

    nsm_end_time = perf_counter_ns()

    nsm_execution_time = nsm_end_time - nsm_start_time    # in nanoseconds
    return nsm_matches, nsm_execution_time

#########################################################

#################### KMP algorithm ######################

"""PROVIDED CODE"""
def compute_prefix_function(pattern):
    m = len(pattern)
    pi = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = pi[j-1]
        if pattern[i] == pattern[j]:
            j += 1
        pi[i] = j
    return pi


def kmp_search(text, pattern):
    n = len(text)
    m = len(pattern)
    pi = compute_prefix_function(pattern)
    j = 0  # Number of characters matched
    occurrences = []

    for i in range(n):
        while j > 0 and text[i] != pattern[j]:
            j = pi[j-1]  # Fall back in the pattern
        if text[i] == pattern[j]:
            j += 1  # Match next character
        if j == m:  # A match is found
            occurrences.append(i - m + 1)
            j = pi[j-1]  # Prepare for the next possible match

    return occurrences

"""EDITED CODE FOR URL INPUT"""
def kmp_search_url(url, patterns):
    text = extract_text(url)
    """calculates the occurrences of the patterns in text"""

    kmp_occurrences = {}
    kmp_start_time = perf_counter_ns()

    for pattern in patterns:
        kmp_cur_occurrences = kmp_search(text, pattern)
        kmp_occurrences[pattern] = kmp_cur_occurrences

    kmp_end_time = perf_counter_ns()
    kmp_execution_time = kmp_end_time - kmp_start_time    # in nanoseconds

    return kmp_occurrences, kmp_execution_time



#########################################################

""" Return the top 10 key words in text"""

def get_keywords(url):
    text = extract_text(url)

    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform([text])

    # Convert the matrix to a DataFrame and sort the keywords by their frequency
    counts = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out()).sum().sort_values(ascending=False)

    # Get the top 10 keywords
    #keywords = counts.head(10)

    # Converting dataframe to lists
    keywords_list = counts.index.tolist()[:10]
    frequency_list = counts.values.tolist()[:10]

    # combining lists into a dictionary
    keywords_dict = dict(zip(keywords_list, frequency_list))

    return keywords_dict


""" Create Word Cloud"""

def create_wordcloud(keywords):
    wordcloud = WordCloud(background_color='white', colormap='Paired', width=800, height=500).generate_from_frequencies(keywords)

    plt.figure(figsize=(15, 8))

    # display wordcloud
    plt.imshow(wordcloud)
    plt.show()

    return wordcloud

"""Preforms natural language preprocessing to extract text from URL"""

def extract_text(url):
    webpage = requests.get(url)
    soup = BeautifulSoup(webpage.content, 'html.parser')
    text = soup.get_text()

    # convert to lowercase
    text = text.lower()

    # remove punctuation
    punctuation_remover = str.maketrans('', '', string.punctuation)
    text = text.translate(punctuation_remover)

    # remove whitespace
    text = " ".join(text.split())

    # remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = word_tokenize(text)
    updated_text = [word for word in tokens if word not in stop_words]

    # join tokens back into a string
    separator = ' '  # The separator is a space by default
    text = separator.join(updated_text)

    # this text is now tokenized and cleaned
    return(text)

def main():
    nltk.download('stopwords')
    nltk.download('punkt')
    #url = input("Enter the URL:\n")
    url = "https://en.wikipedia.org/wiki/California_State_University,_Fullerton" # user inputs using text field
    selected_algo = ["Rabin-Karp", "Suffix Tree", "Suffix Array", "Naive String Matching", "KMP"]   # user selects using drop down menu

    """ Running Keywords and wordcloud functions that will always be displayed """
    # Get top 10 keywords
    keywords = get_keywords(url)
    print("Top 10 Keywords Based on Frequency")
    print(keywords, "\n")

    # get only the keys for function inputs
    keywords_tokens = []
    for key in keywords.keys():
        keywords_tokens.append(key)

    """Display wordcloud"""
    create_wordcloud(keywords)

    """ Running all functions to store values """

    all_execution_times = []
    rk_matches, rk_execution_time = rabin_karp_url(url, keywords_tokens)
    all_execution_times.append(rk_execution_time)

    st_occurrences, st_execution_time = suffix_tree_url(url, keywords_tokens)
    all_execution_times.append(st_execution_time)

    sa_matches, sa_execution_time = suffix_array_url(url, keywords_tokens)
    all_execution_times.append(sa_execution_time)

    nsm_matches, nsm_execution_time = naive_string_matcher_url(url, keywords_tokens)
    all_execution_times.append(nsm_execution_time)

    kmp_matches, kmp_execution_time = kmp_search_url(url, keywords_tokens)
    all_execution_times.append(kmp_execution_time)

    """ Display Bar Graph Comparing Execution Times """
    all_algorithms = ["Rabin-Karp", "Suffix Tree", "Suffix Array", "Naive String Matching", "KMP"]


    plt.bar(all_algorithms, all_execution_times, color=['green', 'orange', 'blue', 'purple', 'red'])
    plt.xlabel('Algorithm')
    plt.ylabel('Execution Time (ns)')
    plt.tick_params(axis='y', pad=10)
    plt.title('Comparing Algorithm Efficiency')
    plt.show()

    """"Loop through selected algorithms"""
    # switches input handling based on selected algorithm
    for algo in selected_algo:
        if algo == "Rabin-Karp":
            print(f"Rabin-Karp - Execution Time is {rk_execution_time} ns")
            print(f"Keyword Indexes using Rabin-Karp: {rk_matches}\n")


        elif algo == "Suffix Tree":
            print(f"Suffix Tree - Execution Time is {st_execution_time} ns")
            print(f"Verifying all keywords are in the Suffix Tree built using the text: {st_occurrences}\n")


        elif algo == "Suffix Array":
            print(f"Suffix Array - Execution Time is {sa_execution_time} ns")
            print(f"Keyword Indexes using using Suffix Array: {sa_matches}\n")

        elif algo == "Naive String Matching":
            print(f"Naive String Matching - Execution Time is {nsm_execution_time} ns")
            print(f"Keyword Indexes using Naive String Matching: {nsm_matches}\n")

        elif algo == "KMP":
            print(f"KMP - Execution Time is {kmp_execution_time} ns")
            print(f"Keyword Indexes of Keywords using KMP: {kmp_matches}\n")


    """Code for searching for a user-inputted word in the text"""

    input_keyword = "orange"    # user inputs via text field
    text = extract_text(url)
    text_length = len(text)
    top_keyword = list(keywords.keys())[0]
    top_keyword_occurrence = keywords[top_keyword]
    percent_of_top_keyword = ((top_keyword_occurrence / text_length) * 100)
    temp = "{:.2f}".format(percent_of_top_keyword)
    percent_of_top_keyword = float(temp)

    # using kmp since it is fastest out of multiple tests
    input_keyword_matches = kmp_search(text, input_keyword)
    if input_keyword_matches is None:
        print(f'"{input_keyword}" not found in text. Not a suitable keyword.')
    else:
        percent_of_text = ((len(input_keyword_matches)/text_length) * 100)
        temp = "{:.2f}".format(percent_of_text)
        percent_of_text = float(temp)

        print(f'The inputted term "{input_keyword}" appears {len(input_keyword_matches)} times in the text. This is {percent_of_text}% of the entire text. (For Reference: Top keyword "{top_keyword}" is {percent_of_top_keyword}% of the entire text.)')



    #############################################################################################################


if __name__ == '__main__':
    main()





