"""
GROUP 1: Anunay Amrit, Angelica Cabato, Pranav Vijay Chand, Riya Chapatwala, Sai Satya Jagannadh Doddipatla, Nhat Ho

Dr. Shah

CPSC 535: Advanced Algorithms (Spring 2024)

"""
import nltk
import requests
from bs4 import BeautifulSoup
import nltk
import string
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud



"""Core Text Search Algorithms"""
##################### Rabin-Karp ########################

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

#########################################################

##################### Suffix Tree #######################
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


#########################################################

##################### Suffix Array ######################

def construct_suffix_array(text):
    suffixes = [(text[i:], i) for i in range(len(text))]
    suffixes.sort(key=lambda x: x[0])
    suffix_array = [item[1] for item in suffixes]
    return suffix_array

#########################################################

############### Naive String Matching ###################

""" Provided Function """
def naive_string_matcher(text, pattern):

    n = len(text)
    m = len(pattern)
    matches = []

    for s in range(n - m + 1):
        print(text[s:s+m])
        if text[s:s+m] == pattern:
            matches.append(s)

    return matches

""" Edited Function for URL input """
def naive_string_matcher_url(url, pattern):
    text = extract_text(url)
    return naive_string_matcher(text, pattern)

#########################################################

#################### KMP algorithm ######################

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

""" Helper Functions """

def extract_text(url):
    """Preforms natural language preprocessing to extract text from URL"""
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
    #url = input("Enter the URL:\n")
    url = "https://www.fullerton.edu/ecs/cs/about/"
    #matches = naive_string_matcher_url(url, "computer")
    #print(matches)

    """Return top 10 keywords"""
    keywords = get_keywords(url)
    print("Keyword Suggestions Based on Frequency")
    print(keywords)

    """Create wordcloud"""
    create_wordcloud(keywords)




    # selected_algo = input("Please enter the corresponding number to select the algorithm you would like to use.\n"
    #                       "1 - Rabin Karp\n"
    #                       "2 - Suffix Tree\n"
    #                       "3 - Suffix Array\n"
    #                       "4 - Naive String Matching\n"
    #                       "5 - KMP Matching\n"
    #                       "6 - EXIT PROGRAM\n")
    #
    # # switches input handling based on selected algorithm
    # while selected_algo != "6":
    #     if selected_algo == "1":
    #         """Rabin Karp"""
    #
    #         d = 256
    #         q = 101
    #
    #         # User Input
    #         text = input("Enter the text: ")
    #         pattern = input("Enter the pattern: ")
    #
    #         # searching for the pattern provided by the user
    #         matches = rabin_karp(text, pattern, d, q)
    #
    #         # display results
    #         if matches:
    #             print(f"Pattern found at positions: {matches}")
    #         else:
    #             print("Pattern not found in the text.")
    #
    #     elif selected_algo == "2":
    #         """Suffix Tree"""
    #
    #         # Get user input
    #         text = input("Enter the text: ")
    #
    #         # Construct and display suffix tree
    #         suffix_tree = SuffixTree(text)
    #         suffix_tree.display()
    #
    #     elif selected_algo == "3":
    #         """Suffix Array"""
    #
    #         # Get user input
    #         text = input("Enter the text: ")
    #
    #         # Construct and print suffix array
    #         suffix_array = construct_suffix_array(text)
    #         print("Suffix Array:", suffix_array)
    #
    #     elif selected_algo == "4":
    #         """Naive String Matching"""
    #
    #         text = input("Enter the text: ")
    #         pattern = input("Enter the pattern: ")
    #
    #         matches = naive_string_matcher(text, pattern)
    #
    #         if matches:
    #             print(f"Pattern found at positions: {matches}")
    #         else:
    #             print("Pattern not found in the text.")
    #
    #     elif selected_algo == "5":
    #         """KMP Matching"""
    #
    #         # Get user input
    #         text = input("Enter the text: ")
    #         pattern = input("Enter the pattern: ")
    #
    #         # Find and display occurrences
    #         occurrences = kmp_search(text, pattern)
    #         if occurrences:
    #             print(f'Pattern found at indices: {occurrences}')
    #         else:
    #             print('Pattern not found in the text.')
    #
    #
    #     ## Re-ask intial prompt to select algorithm
    #     print("\n")
    #     selected_algo = input("Please enter the corresponding number to select the algorithm you would like to use.\n"
    #                           "1 - Rabin Karp\n"
    #                           "2 - Suffix Tree\n"
    #                           "3 - Suffix Array\n"
    #                           "4 - Naive String Matching\n"
    #                           "5 - KMP Matching\n"
    #                           "6 - EXIT PROGRAM\n")
    #
    # if selected_algo == "6":
    #     """EXIT Program"""
    #     print("You have successfully exited the program! Have a great day!")
    #     exit()


if __name__ == '__main__':
    main()





