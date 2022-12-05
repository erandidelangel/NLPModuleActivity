# -*- coding: utf-8 -*-

# Libraries
    # Necessary in order for the file to run succesfully
    
    # Sentiment Analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
    # NER
    # pip install flair
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter
import os
    # Translator Comparison
    # pip install googletrans
    # pip install keras
from googletrans import Translator
from transformers import pipeline
from nltk.translate.bleu_score import sentence_bleu


# Functions

    # Sentiment Analysis
def sentiment_analysis(txt_name): # Completes all tasks in Activity 1
    
    with open(txt_name) as f:
        contents = f.readlines() # Read the txt file

    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer() # Import the analyzer

    results = []
    for i in contents:
        pre_sentiment = sid.polarity_scores(i) # Get the sentiment classification
        NEG = pre_sentiment['neg']
        POS = pre_sentiment['pos']
        if NEG >= POS: # Classify only in POSITIVE or NEGATIVE
            sentiment = 'NEGATIVE'
        else:
            sentiment = 'POSITIVE'
        results.append(sentiment)
    print(results) # Return all the results
    return results


    # NER Model
def access_ner_data(path, train, test, dev, PERCENT_OF_DATASET_TO_TRAIN): # Obtain the necessary file for Activity 2
    
    columns = {0 : 'text', 1 : 'ner'} # Columns found in the files
    corpus: Corpus = ColumnCorpus(path, columns,
                                  train_file = train,
                                  test_file = test,
                                  dev_file = dev) 
    percentage_in_decimal = PERCENT_OF_DATASET_TO_TRAIN / 100 
    corpus = corpus.downsample(percentage_in_decimal) # Use only a percentage of the samples
    return corpus

def train_ner_model(corpus, path): # Train further the NER model
    # Make tag dictionary from the corpus
    tag_dictionary = corpus.make_label_dictionary(label_type='ner') 
    
    embedding_types = [
        # GloVe embeddings
        WordEmbeddings('glove'),
        # Contextual string embeddings, forward
        FlairEmbeddings('news-forward'),
        # Contextual string embeddings, backward
        FlairEmbeddings('news-backward'),
        ]
    embeddings = StackedEmbeddings(embeddings=embedding_types)
    
    tagger = SequenceTagger(hidden_size=256,             
                      embeddings=embeddings, 
                      tag_dictionary=tag_dictionary,
                      tag_type='ner') 
    
    trainer = ModelTrainer(tagger, corpus)
    training = trainer.train(path, max_epochs=7, 
                             monitor_train = True, monitor_test = True) # Fit and train the model
    return training
    
def get_graphic(path): # Loss and F1 Score Graphs
    plotter = Plotter()
    directory = os.path.join(path, "loss.tsv") # Obtain values from 'loss.tsv' previously generated file
    plotter.plot_training_curves(directory) # Plot the graph
  
def implement_ner_model(path, train_file, test_file, dev_file, PERCENT_OF_DATASET_TO_TRAIN): # Use all the functions from Activity 2 and return the final graph
    corpus = access_ner_data(path, train_file, test_file, dev_file, PERCENT_OF_DATASET_TO_TRAIN)
    training = train_ner_model(corpus, path)
    get_graphic(path)
    
   
    # Translator Comparison
def access_data(original_text, translated_text): # Read files
    
    with open(original_text, encoding='utf-8') as split_original_text:
        original_content = split_original_text.readlines()
        
    with open(translated_text, encoding='utf-8') as split_translated_text:
        translated_content = split_translated_text.readlines()
        
    split_original_text = original_content[0:100] # Use only the first 100 lines
    split_translated_text = translated_content[0:100]
    return split_original_text, split_translated_text # Return the reduced texts


def translate(split_original_text): # Use Translators to translate the text
    # Google Translator
    translator = Translator()
    results = translator.translate(split_original_text, src='es', dest='en')
    google_results = []
    for i in results:
        google_results.append(i.text)
        
    # Helsinki Translator
    helsinki_results = []
    model_checkpoint = "Helsinki-NLP/opus-mt-es-en"
    translator = pipeline("translation", model=model_checkpoint)
    for i in translator(split_original_text):
        helsinki_results.append(i['translation_text'])
    return google_results, helsinki_results # Returns translated results

def translator_comparison(split_translated_text, google_results, helsinki_results): # Get BLEU Score for Translators performance
    ref = []
    h_test = []
    g_test = []
   
    # Split data
    for i in split_translated_text:
        ref.append(i.split())
    for i in helsinki_results:
        h_test.append(i.split())
    for i in google_results:
        g_test.append(i.split())
    
    # BLEU Scores
    google_score = 0
    helsinki_score = 0
    
    for i in range(0, len(ref)):
        # Google BLEU Score
        score = sentence_bleu(ref[i], g_test[i])
        google_score += score
        # Helsinki BLEU Score
        score = sentence_bleu(ref[i], h_test[i])
        helsinki_score += score
    
    google_score = google_score / len(g_test)
    helsinki_score = helsinki_score / len(h_test)
    return google_score, helsinki_score # Returns BLEU Scores

def get_translator_comparison_results(original_text, translated_text): # Use all functions to complete Activity 3
    split_original_text, split_translated_text = access_data(original_text, translated_text)
    google_results, helsinki_results = translate(split_original_text)
    google_score, helsinki_score = translator_comparison(split_translated_text, google_results, helsinki_results)
    print('GOOGLE_TRANSLATOR: {}'.format(google_score))
    print('HELSINKI_TRANSLATOR: {}'.format(helsinki_score)) # Prints BLEU Scores
    
    
# Only manipulate the next code to custom your path, etc.
if __name__ == '__main__':
    # PATH where all files are includings this .py file
    path = r"C:\Users\erand\Documents\NLP_Module"
    
    # Activity 1: Sentiment Analysis
    sentiment_analysis(r"C:\Users\erand\Documents\NLP_Module\tiny_movie_reviews_dataset.txt")
    
    # Activity 2: NER Model
    implement_ner_model(path, 'train.txt', 'test.txt', 'dev.txt', 0.1)
    
    # Activity 3: Translator Comparison
    get_translator_comparison_results(r"C:\Users\erand\Documents\NLP_Module\europarl-v7.es-en.es", r"C:\Users\erand\Documents\NLP_Module\europarl-v7.es-en.en")
    
    