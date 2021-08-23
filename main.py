# Copyright (c) Adil Bin Bhutto. All rights reserved.

import warnings
import os
import sys
import gensim

from sample_generator.reviewGenerator import reviewGenerator
from model_generator.ldaGenerator import ldaGenerator

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

## Configuration
sample_per_review_train = 10000 
sample_per_review_test = 100 
min_topic = 2
step = 1
max_topic = 42 
## End Configuration

def topic_modelling(argv):

    print("Starting topic modeling from data: ", argv[0])

    InputGenerator = reviewGenerator(file_path = argv[0])
    LdaGenerator = ldaGenerator()

    dictionary, word_embeddings, texts = InputGenerator.get_training_sample(sample_per_review=sample_per_review_train)

    # Training the LDA model and getting the best model based on coherence score
    ldaModel, total_topics, topics = LdaGenerator.fit(dictionary = dictionary, 
                                                      corpus = word_embeddings,
                                                      texts = texts, 
                                                      begin = min_topic, 
                                                      end = max_topic, 
                                                      step = step)
    LdaGenerator.plot_coherence_score(filepath="./results/coherence_vs_topics.pdf")
    
    # Writing all the topics along with words that contribute to the specific topic number  
    with open("./results/topics.txt", 'w') as f:
        f.write("Optimal number of topics: " + str(total_topics) + "\n\n")
        for idx, topic in topics:
            f.write("Topic: {} \nWords: {}\n".format(idx, topic))
    
    # Predicting few probable new instances based on the returned LDA model
    word_embeddings, df_raw_text = InputGenerator.get_testing_sample(sample_per_review=sample_per_review_test)
    prediction_list = LdaGenerator.predict(corpus = word_embeddings)
    df_raw_text['Topic'] = prediction_list
    df_raw_text.to_csv("./results/prediction.csv")

    ldaModel.save("./results/trained_model/model.lda")



if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print("Invalid parameters! Please use: python main.py ./filePath")
        exit(1)
    topic_modelling(sys.argv[1:])
