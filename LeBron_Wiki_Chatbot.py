# Install necessary libraries
#pip install nltk
#pip install newspaper3k

import random
import string
import nltk
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Download the NLTK 'punkt' dataset
nltk.download('punkt', quiet=True)

# Fetch information about LeBron James from Wikipedia
article = Article('https://en.wikipedia.org/wiki/LeBron_James')
article.download()
article.parse()
article.nlp()
info_lebron = article.text

# Store the text in 'info_lebron' variable
text = info_lebron

# Tokenize the text into sentences
sentence_list = nltk.sent_tokenize(text)

# Define a function to respond to greetings
def greeting_response(text):
    text = text.lower()
    bot_greetings = ['hello :D', 'hola :D', 'hi', 'hola :)', 'hey', 'Wassup']
    user_greetings = ['hi', 'hello', 'hey', 'hola', 'greetings', 'wassup', 'greetings']

    for word in text.split():
        if word in user_greetings:
            return random.choice(bot_greetings)

# Define a function to sort a list of indices
def index_sort(list_var):
    length = len(list_var)
    list_index = list(range(0, length))
    x = list_var
    for i in range(length):
        for j in range(length):
            if x[list_index[i]] > x[list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp
    return list_index

# Define a function for the chatbot's response
def bot_response(user_input, tfidf_vectorizer, tfidf_matrix):
    user_input = user_input.lower()
    sentence_list.append(user_input)
    bot_response = ''
    tfidf_user_input = tfidf_vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(tfidf_user_input, tfidf_matrix)
    similarity_scores_list = similarity_scores.flatten()
    index = index_sort(similarity_scores_list)
    index = index[1:]
    response_flag = 0
    j = 0

    for i in range(len(index)):
        if similarity_scores_list[index[i]] > 0.0:
            bot_response = bot_response + ' ' + sentence_list[index[i]]
            response_flag = 1
            j = j + 1
        if j > 2:
            break

    if response_flag == 0:
        bot_response = bot_response + ' ' + "Sorry, I don't understand"

    sentence_list.remove(user_input)
    return bot_response

# Define the main bot function
def bot():
    print('Bot: Hi! I am a chat bot designed to answer all your queries about LeBron James. Whenever you want to finish our conversation just type "bye"')
    exit_list = ['bye', 'good bye', 'see you', 'adios', 'adiós']
    exit_list_bot = ['Take care', 'Bye :D', 'It was nice meeting you!']
    groserias = ['fuck you', 'bitch', 'fuck', 'ass', 'bitch ass']
    groserias_bot = ['Language', 'Be respectful', '>:(']

    # Create TF-IDF vectorizer and matrix
    tfidf_vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentence_list)

    while True:
        user_input = input()
        if user_input.lower() in exit_list:
            print('Bot:', random.choice(exit_list_bot))
            break
        elif user_input.lower() in groserias:
            print('Bot: ' + random.choice(groserias_bot) + '. I will come back when you are more respectful')
            break
        else:
            if greeting_response(user_input) is not None:
                print('Bot: ' + greeting_response(user_input))
            else:
                print('Bot: ' + bot_response(user_input, tfidf_vectorizer, tfidf_matrix))

bot()