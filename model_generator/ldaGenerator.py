import gensim
import matplotlib.pyplot as plt

class ldaGenerator(object):
    def __init__(self):
        super().__init__()
        self.topics_range = None
        self.model_list = [] 
        self.coherence_values = []
        self.optimal_model = None
        self.optimal_topic_nums = None

    def fit(self, dictionary, corpus, texts, begin=2, end=25, step = 1):
        self.topics_range = range(begin, end, step)
        for num_topics in self.topics_range:
            model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
            self.model_list.append(model)
            coherencemodel = gensim.models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            self.coherence_values.append(coherencemodel.get_coherence())
        optimal_index = self.get_optimal_index()
        self.optimal_model = self.model_list[optimal_index]
        self.optimal_topic_nums = self.topics_range[optimal_index]
        topics = self.optimal_model.print_topics(-1)
        return self.optimal_model, self.optimal_topic_nums, topics
    
    def predict(self, corpus):
        if(self.optimal_model == None):
            print("Model is not trained")
            exit(1)
        result = []
        for item in corpus:
            temp = list(self.optimal_model[[item]])[0]
            topics = sorted(temp, key=lambda x:x[1], reverse=True)
            result.append(topics[0][0])
        return result

    def plot_coherence_score(self, filepath="./coherence_plot.pdf"):
        plt.plot(self.topics_range, self.coherence_values, marker="o")
        plt.xlabel("Number of topics")
        plt.ylabel("Coherence score")
        plt.grid()
        plt.savefig(filepath)

    def get_optimal_index(self):
        max_value = max(self.coherence_values)
        max_index = self.coherence_values.index(max_value)
        return max_index 

