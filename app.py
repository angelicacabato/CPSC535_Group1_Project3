"""
GROUP 1: Anunay Amrit, Angelica Cabato, Pranav Vijay Chand, Riya Chapatwala, Sai Satya Jagannadh Doddipatla, Nhat Ho

Dr. Shah

CPSC 535: Advanced Algorithms (Spring 2024)

"""

import requests
import streamlit as st
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import nltk

from main import extract_text, get_keywords, rabin_karp_url, suffix_tree_url, suffix_array_url, \
    naive_string_matcher_url, kmp_search_url, kmp_search

def main():
    nltk.download('stopwords')
    nltk.download('punkt')
    st.header("SEO Keyword Tracker and Analyzer")
    st.subheader("CPSC 535 (Spring 2024) - Group 1")

    valid_url = False
    valid_keyword = False

    url = st.text_input('Please enter a URL')
    if url:
        # check is URL is valid
        try:
            response = requests.get(url, timeout=10)
            # Status codes 200 indicates reachable url
            if response.status_code == 200:
                print("The website is reachable.")
                valid_url = True
            else:
                print("The website is not reachable. Status Code:", response.status_code)
        except requests.ConnectionError:
            print("Failed to connect to inputted URL.")
        except requests.exceptions.MissingSchema:
            print(f"Error: The URL is invalid because it lacks a schema (http or https).")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

        if valid_url is True:
            st.write('The URL you inputted it is', url, '.')
        else:
            st.write('ERROR: The URL you inputted it is not valid/reachable. Please input a valid URL.')

    algo = st.selectbox(
        'Please Select Text Search Algorithm',
        ("Rabin-Karp", "Suffix Tree", "Suffix Array", "Naive String Matching", "KMP", "ALL ALGORITHMS"), index=None,
        placeholder="REQUIRED")
    if algo:
        st.write(f'The Text Search Algorithm you selected is **{algo}**.')

    inputted_keyword = st.text_input(
        'Would you like to search and analyze a keyword from the URL? If so, enter the word below!',
        placeholder="OPTIONAL")
    if inputted_keyword:
        if len(inputted_keyword.split()) > 1:
            st.write("You entered more than one word. Please only enter a single word. Try again.")
        else:
            valid_keyword = True
            st.write(f'The keyword you entered is **{inputted_keyword}**.')

    run_analysis = st.button("Run Keyword Analysis")

    # places a divider between inputs and outputs
    st.divider()

    # if button is pressed, url is inputted and valid, and algo is selected, run analysis
    if run_analysis and url and algo and valid_url:
        # Get top 10 keywords
        keywords = get_keywords(url)
        st.subheader("Top 10 Keywords Based on Frequency")
        st.dataframe(keywords)

        # get only the keys for function inputs
        keywords_tokens = []
        for key in keywords.keys():
            keywords_tokens.append(key)

        # Display wordcloud
        st.subheader("Keywords WordCloud")
        wordcloud = WordCloud(background_color='white', colormap='Paired', width=800,
                              height=500).generate_from_frequencies(keywords)

        plt.figure(figsize=(15, 8))

        # display wordcloud
        plt.imshow(wordcloud)
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)  # disable warnings
        st.pyplot()

        # Loop through selected algorithms
        # switches input handling based on selected algorithm
        if algo == "Rabin-Karp":
            rk_matches, rk_execution_time = rabin_karp_url(url, keywords_tokens)
            st.subheader("Keyword Search Using Rabin-Karp")
            st.write(f"Rabin-Karp - Execution Time is **{rk_execution_time}** ns")
            st.write(f"Keyword indexes using Rabin-Karp:")
            st.json(rk_matches, expanded=False)
            # place a divider between algorithms
            st.divider()


        elif algo == "Suffix Tree":
            st_occurrences, st_execution_time = suffix_tree_url(url, keywords_tokens)
            sa_matches, sa_execution_time = suffix_array_url(url, keywords_tokens)
            st.subheader("Keyword Search Using Suffix Tree")
            st.write(f"Suffix Tree - Execution Time is **{st_execution_time}** ns")
            st.write(f"Verifying keywords using Suffix Tree:")
            st.json(st_occurrences, expanded=False)
            st.write(
                f"Since Suffix Trees are used to verify patterns, here are the first keyword indexes using a Suffix "
                f"Array")
            st.json(sa_matches, expanded=False)
            # place a divider between algorithms
            st.divider()

        elif algo == "Suffix Array":
            sa_matches, sa_execution_time = suffix_array_url(url, keywords_tokens)
            st.subheader("Keyword Search Using Suffix Array")
            st.write(f"Suffix Array - Execution Time is **{sa_execution_time}** ns")
            st.write(f"First Keyword indexes using using Suffix Array:")
            st.json(sa_matches, expanded=False)
            # place a divider between algorithms
            st.divider()

        elif algo == "Naive String Matching":
            nsm_matches, nsm_execution_time = naive_string_matcher_url(url, keywords_tokens)
            st.subheader("Keyword Search Using Naive String Matching")
            st.write(f"Naive String Matching - Execution Time is **{nsm_execution_time}** ns")
            st.write(f"Keyword indexes using Naive String Matching:")
            st.json(nsm_matches, expanded=False)
            # place a divider between algorithms
            st.divider()

        elif algo == "KMP":
            kmp_matches, kmp_execution_time = kmp_search_url(url, keywords_tokens)
            st.subheader("Keyword Search Using KMP")
            st.write(f"KMP - Execution Time is **{kmp_execution_time}** ns")
            st.write(f"Keyword indexes of Keywords using KMP:")
            st.json(kmp_matches, expanded=False)
            # place a divider between algorithms
            st.divider()

        elif algo == "ALL ALGORITHMS":
            # Running all functions to store values
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

            # Display Bar Graph Comparing Execution Times
            st.subheader('Graph Comparing Algorithm Efficiency')
            all_algorithms = ["Rabin-Karp", "Suffix Tree", "Suffix Array", "Naive String Matching", "KMP"]

            plt.bar(all_algorithms, all_execution_times, color=['green', 'orange', 'blue', 'purple', 'red'])
            plt.xlabel('Algorithm')
            plt.ylabel('Execution Time (ns)')
            plt.tick_params(axis='y', pad=10)
            plt.title('Comparing Algorithm Efficiency')
            plt.show()
            st.set_option('deprecation.showPyplotGlobalUse', False)  # disable warnings
            st.pyplot()

            st.divider()
            st.subheader("Keyword Search Using All Algorithms")

            # Rabin-Karp
            st.subheader("Keyword Search Using Rabin-Karp")
            st.write(f"Rabin-Karp - Execution Time is **{rk_execution_time}** ns")
            st.write(f"Keyword indexes using Rabin-Karp:")
            st.json(rk_matches, expanded=False)
            st.divider()

            # Suffix Tree
            st.subheader("Keyword Search Using Suffix Tree")
            st.write(f"Suffix Tree - Execution Time is **{st_execution_time}** ns")
            st.write(f"Verifying keywords using Suffix Tree:")
            st.json(st_occurrences, expanded=False)
            st.divider()

            # Suffix Array
            st.subheader("Keyword Search Using Suffix Array")
            st.write(f"Suffix Array - Execution Time is **{sa_execution_time}** ns")
            st.write(f"FIrst Keyword indexes using using Suffix Array:")
            st.json(sa_matches, expanded=False)
            st.divider()

            # Naive String Matching
            st.subheader("Keyword Search Using Naive String Matching")
            st.write(f"Naive String Matching - Execution Time is **{rk_execution_time}** ns")
            st.write(f"Keyword indexes using Naive String Matching:")
            st.json(nsm_matches, expanded=False)
            st.divider()

            # KMP
            st.subheader("Keyword Search Using KMP")
            st.write(f"KMP - Execution Time is **{kmp_execution_time}** ns")
            st.write(f"Keyword indexes of Keywords using KMP:")
            st.json(kmp_matches, expanded=False)
            st.divider()

        # if inputted Keyword is provided and valid - search for word in text
        if inputted_keyword and valid_keyword:
            # Search for Keyword
            st.subheader("Inputted Keyword Search and Analysis")
            text = extract_text(url)
            text_length = len(text)
            top_keyword = list(keywords.keys())[0]
            top_keyword_occurrence = keywords[top_keyword]
            percent_of_top_keyword = ((top_keyword_occurrence / text_length) * 100)
            temp = "{:.2f}".format(percent_of_top_keyword)
            percent_of_top_keyword = float(temp)

            # Using Suffix Tree to verify that keyword is in text
            inputted_keyword_lower = inputted_keyword.lower()     # convert to lowercase
            token_keyword = [inputted_keyword_lower, "placeholder"]  # need to convert to list for function to work properly
            verify_keyword = (suffix_tree_url(url, token_keyword))[0]  # only getting the matches, not execution time
            if verify_keyword[inputted_keyword_lower] == True:
                st.write(f"Keyword **{inputted_keyword}** found in URL content!")

                # if pattern is found, use kmp to find occurrences in text
                input_keyword_matches = kmp_search(text, inputted_keyword_lower)
                if input_keyword_matches is None:
                    st.write(f'"**{inputted_keyword}**" not found in text. Not a suitable keyword.')
                else:
                    percent_of_text = ((len(input_keyword_matches) / text_length) * 100)
                    temp = "{:.2f}".format(percent_of_text)
                    percent_of_text = float(temp)

                    num_keywords = len(input_keyword_matches)  # variable to store number of keyword matches

                    if inputted_keyword_lower in keywords:
                        num_keywords = keywords.get(inputted_keyword_lower)
                        percent_of_text = ((num_keywords / text_length) * 100)
                        temp = "{:.2f}".format(percent_of_text)
                        percent_of_text = float(temp)

                        # return keyword data
                        st.write(
                            f'The inputted term "**{inputted_keyword}**" appears **{num_keywords}** times in the text. It is already within the top 10 keywords! This keyword is **{percent_of_text}%** of the entire text.')
                        st.json(input_keyword_matches, expanded=False)

                    else:
                        # return keyword data
                        st.write(
                            f'The inputted term "**{inputted_keyword}**" appears **{num_keywords}** times in the text. This is **{percent_of_text}%** of the entire text. (For Reference: Top keyword "**{top_keyword}**" is **{percent_of_top_keyword}%** of the entire text.)')
                        st.write(f'Below are the indexes where "**{inputted_keyword}**" is found in the text.')
                        st.json(input_keyword_matches, expanded=False)

            else:
                st.write(f'"{inputted_keyword}" not found in text. Not a suitable keyword.')
        else:
            if inputted_keyword and valid_keyword is False:
                st.subheader("Inputted Keyword Search and Analysis")
                st.write("ERROR: You entered more than one word. Please only enter a single word. Try again.")

    # if button is pressed or url is not inputted, or algo is not selected, prompt user to enter required information
    else:
        "Cannot Proceed. Must enter a valid/reachable URL and select an algorithm. Thank you!"


if __name__ == '__main__':
    main()
