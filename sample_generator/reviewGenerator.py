import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from gensim import corpora
import string

class reviewGenerator(object):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.dictionary = None
    
    def get_training_sample(self, sample_per_review):
        review_data= pd.read_csv(self.file_path)
        review_data.dropna(axis = 0, how ='any',inplace=True) 
        review_data['Text'] = review_data['Text'].apply(self.clean_text)
        review_data['Num_words_text'] = review_data['Text'].apply(lambda x:len(str(x).split())) 

        mask = (review_data['Num_words_text'] < 100) & (review_data['Num_words_text'] >=20)
        df_short_reviews = review_data[mask]
        df_sampled = df_short_reviews.groupby('Score').apply(lambda x: x.sample(n=sample_per_review, random_state=0)).reset_index(drop = True)
        df_sampled['Text']=df_sampled['Text'].apply(self.remove_stopwords)
        text_list=df_sampled['Text'].tolist()
        tokenized_reviews = self.lemmatization(text_list)

        self.dictionary = corpora.Dictionary(tokenized_reviews)
        doc_term_matrix = [self.dictionary.doc2bow(rev) for rev in tokenized_reviews]

        return (self.dictionary, doc_term_matrix, tokenized_reviews)

    def get_testing_sample(self, sample_per_review):
        if(self.dictionary == None):
            print("Dictionary is not initialized. Please call get_training_sample first.")
            exit(1)
        review_data= pd.read_csv(self.file_path,)
        review_data.dropna(axis = 0, how ='any',inplace=True) 
        df_sampled = review_data.groupby('Score').apply(lambda x: x.sample(n=sample_per_review)).reset_index(drop = True)
        test_sampled = df_sampled['Text'].to_frame()
        df_sampled['Text'] = df_sampled['Text'].apply(self.clean_text)
        # review_data['Num_words_text'] = review_data['Text'].apply(lambda x:len(str(x).split())) 

        # mask = (review_data['Num_words_text'] < 100) & (review_data['Num_words_text'] >=20)
        # df_sampled = review_data[mask]
        df_sampled['Text']=df_sampled['Text'].apply(self.remove_stopwords)
        text_list=df_sampled['Text'].tolist()
        tokenized_reviews = self.lemmatization(text_list)

        doc_term_matrix = [self.dictionary.doc2bow(rev) for rev in tokenized_reviews]

        return (doc_term_matrix, test_sampled)


    def clean_text(self, text): 
        delete_dict = {sp_character: '' for sp_character in string.punctuation} 
        delete_dict[' '] = ' ' 
        table = str.maketrans(delete_dict)
        text1 = text.translate(table)
        textArr= text1.split()
        text2 = ' '.join([w for w in textArr if ( not w.isdigit() and  ( not w.isdigit() and len(w)>3))]) 
        return text2.lower()

    def lemmatization(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): 
        nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])
        output = []
        for sent in texts:
                doc = nlp(sent) 
                output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags ])
        return output

    def remove_stopwords(self, text):
        stop_words = stopwords.words('english')
        textArr = text.split(' ')
        rem_text = " ".join([i for i in textArr if i not in stop_words])
        return rem_text

