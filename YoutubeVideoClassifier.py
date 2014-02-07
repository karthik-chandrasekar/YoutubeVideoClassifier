#!/usr/bin/env python
#coding=utf-8
import sys, codecs, json, os, ConfigParser, logging
from sklearn.feature_extraction.text import TfidfVectorizer
from SPARQLWrapper import SPARQLWrapper, JSON
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
from nltk.stem import PorterStemmer

class Utility:
    ''' Utilities to be used by all classes''' 
   
    def __init__(self):
  
        self.config = ConfigParser.ConfigParser()
        self.config.read("YoutubeVideoClassifier.config")
        self.movies_file_name = self.config.get('GLOBAL', 'movies_file')
        self.actors_file_name = self.config.get('GLOBAL', 'actors_file')
        self.tvshows_file_name = self.config.get('GLOBAL', 'tvshows_file')
        self.test_file_name = self.config.get('GLOBAL', 'test_file')
        self.logging_file_name = self.config.get('GLOBAL', 'log_file')

        self.input_dir = self.config.get('GLOBAL', 'input_dir')
        self.output_dir = self.config.get('GLOBAL', 'output_dir')
        cur_dir = os.getcwd()
 
        self.input_dir = os.path.join(cur_dir, self.input_dir)
        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)

        self.output_dir = os.path.join(cur_dir, self.output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.movies_file = os.path.join(self.input_dir, self.movies_file_name)
        self.actors_file = os.path.join(self.input_dir, self.actors_file_name)
        self.tvshows_file = os.path.join(self.input_dir, self.tvshows_file_name)
        self.test_file = os.path.join(self.input_dir, self.test_file_name) 
        self.logging_file = os.path.join(self.output_dir, self.logging_file_name)        

        logging.basicConfig(filename=self.logging_file, level=logging.INFO)
        logging.info("Initialized logging")
    
class DataSetCollector(Utility):
    ''' Fetch data from dbpedia and store in file'''
 
    def __init__(self):
        Utility.__init__(self)

        self.sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        self.getFilms = self.config.get('QUERY', 'getFilmsQuery')
        self.getTvShows = self.config.get('QUERY', 'getTvShowsQuery')
        self.getActors = self.config.get('QUERY', 'getActorsQuery')

    def run_main(self):
        self.collectFilms()
        self.collectActors()
        self.collectTvShows()

    def collectFilms(self): 
        self.movies_file_fd = codecs.open(self.movies_file, 'w', 'utf-8')
        self.sparql.setQuery(self.getFilms)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        
        films = results.get('results')
        films = films and films.get('bindings')
        if not films:return
        for result in films:
            try:
                movie_name = result.get('movie').get('value')
                movie_name = movie_name and movie_name.strip("http://dbpedia.org/resource/")
                if not movie_name:continue
                self.movies_file_fd.write("%s\n" % (movie_name))
            except:
                logging.info("Exception while parsing movie data") 
                continue
        self.movies_file_fd.close()

    def collectActors(self):
        self.actors_file_fd = codecs.open(self.actors_file, 'w', 'utf-8')
        self.sparql.setQuery(self.getActors)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        
        actors = results.get('results')
        actors = actors and actors.get('bindings')
        for result in actors:
            try:
                actor_name = result.get('actor').get('value')
                actor_name = actor_name and actor_name.strip("http://dbpedia.org/resource/")
                if not actor_name:continue
                self.actors_file_fd.write("%s\n" % (actor_name))
            except:
                logging.info("Exception while parsing actors data") 
                continue
        self.actors_file_fd.close()

    def collectTvShows(self):
        self.tvshows_file_fd = codecs.open(self.tvshows_file, 'w', 'utf-8')
        self.sparql.setQuery(self.getTvShows)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        
        tvshows = results.get('results')
        tvshows = tvshows and tvshows.get('bindings')
        for result in tvshows:
            try:
                tvshow_name = result.get('tvshow').get('value')
                tvshow_name = tvshow_name and tvshow_name.strip("http://dbpedia.org/resource/")
                if not tvshow_name:continue
                self.tvshows_file_fd.write("%s\n" % (tvshow_name)) 
            except:
                logging.info("Exception while parsing tvshow data")
                continue
        self.tvshows_file_fd.close()        

class YoutubeVideoClassifier(Utility):
    ''' Use the collected data as training set and classify test data''' 
  
    def __init__(self):
        Utility.__init__(self)
        self.nb_output_file_name = self.config.get('GLOBAL', 'nb_output_file')
        self.svm_output_file_name = self.config.get('GLOBAL', 'svm_output_file')
        self.nb_output = os.path.join(self.output_dir, self.nb_output_file_name)
        self.svm_output = os.path.join(self.output_dir, self.svm_output_file_name)        
    
        self.train_features = []
        self.stopwords_set = set(stopwords.words('english'))    

    def run_main(self):
        self.pre_processing()
        self.feature_extraction()
        self.classification()
        self.testing()

    def pre_processing(self):
        self.load_data()

    def load_data(self):
        self.load_movies()
        self.load_actors()
        self.load_tvshows()
        self.load_test_data()

    def load_movies(self):
        self.movies_list = []
        movies_fd = codecs.open(self.movies_file)
        
        for movie in movies_fd.readlines():
            if not movie: continue
            self.movies_list.append(movie)           
        movies_fd.close()   
 
    def load_actors(self):
        self.actors_list = []
        actors_fd = codecs.open(self.actors_file)

        for actor in actors_fd.readlines():
            if not actor: continue
            self.actors_list.append(actor)
        actors_fd.close()

    def load_tvshows(self):
        self.tvshows_list = []
        tvshows_fd = codecs.open(self.tvshows_file)

        for tvshow in tvshows_fd.readlines():
            if not tvshow:continue
            self.tvshows_list.append(tvshow)
        tvshows_fd.close()
    
    def load_test_data(self):
        json_data = open(self.test_file)
        self.test_data = json.load(json_data)

    def feature_selection(self, features_list):
        selected_features = []

        for feat in features_list:
            if feat and feat.strip() and feat.lower() not in self.stopwords_set:
                selected_features.append((feat.strip().lower(), True))        
        return dict(selected_features)


    def feature_extraction(self):
        for item in self.tvshows_list:
            if not item:continue
            selected_features = self.feature_selection(item.replace("_"," ").split(" "))
            self.train_features.append((selected_features, 'tvshow'))

        for item in self.movies_list:
            if not item: continue
            selected_features = self.feature_selection(item.replace("_"," ").split(" "))
            self.train_features.append((selected_features, 'movie'))
            
        for item in self.actors_list:
            if not item: continue
            selected_features = self.feature_selection(item.replace("_"," ").split(" ")) 
            self.train_features.append((selected_features, 'celebrity'))    

    def classification(self):

        #Training NB Classifier
        self.nb_classifier = NaiveBayesClassifier.train(self.train_features)         
        
        #Training SVM classifier
        self.svm_classifier = SklearnClassifier(LinearSVC()) 
        self.svm_classifier.train(self.train_features)

    def testing(self):
        nb_fd = codecs.open(self.nb_output, 'w', 'utf-8')
        svm_fd = codecs.open(self.svm_output, 'w', 'utf-8')

        for instance in self.test_data:
            try:
                if not instance:continue
                test_features = instance.get('title').split(" ")
                test_features.extend(instance.get('description').split(" "))
                selected_features = self.feature_selection(test_features)

                label = self.nb_classifier.classify(selected_features)
                nb_fd.write("%s\n" % (label))

                label = self.svm_classifier.classify(selected_features)
                svm_fd.write("%s\n" % (label))
            except:
                logging.info("Exception in test data ")       
                continue
 
        nb_fd.close()
        svm_fd.close()


class RelatedVideoGenerator(Utility):
    ''' Related video suggestions based on jaccard similarity and vector similarity'''
    
    def __init__(self):
        Utility.__init__(self)
        
        self.related_tfidf_file_name = self.config.get('GLOBAL', 'tfidf_related_output')
        self.related_jaccard_file_name = self.config.get('GLOBAL', 'jaccard_related_output')
       
        self.related_tfidf = os.path.join(self.output_dir, self.related_tfidf_file_name)
        self.related_jaccard = os.path.join(self.output_dir, self.related_jaccard_file_name)
 
        self.stopwords_set = set(stopwords.words('english'))    
        self.stemmer = PorterStemmer()
        
        self.test_data = []
        self.features_set_list = []
        self.features_string_list = []
        
    def run_main(self):
        self.load_data()
        self.select_features()
        self.find_related_jaccard()
        self.find_related_tfidf()

    def load_data(self):
        json_data = open(self.test_file)
        self.test_data = json.load(json_data)
     
    def select_features(self):
        for instance in self.test_data:
            try:
                feature = instance.get('title') + " " + instance.get('description')
                feature  = feature.split(" ")
                feature = [self.stemmer.stem(feat.lower().strip()) for feat in feature if feat and feat.lower().strip() not in self.stopwords_set]
                feature_string = " ".join(feature)
                self.features_set_list.append(set(feature))
                self.features_string_list.append(feature_string)
            except:
                logging.info("Exception in test data")
                continue

    def find_related_jaccard(self):
        related_fd = codecs.open(self.related_jaccard, 'w', 'utf-8')
        for index, feature in enumerate(self.features_set_list):
            related = self.get_relevant_entry(feature, index)
            related_fd.write("%s\t%s\n%s\n%s\n%s\n%s\n\n" % (index, related, self.features_set_list[index], self.features_set_list[related[0]], self.features_set_list[related[1]], self.features_set_list[related[2]]))
        related_fd.close()
 
    def get_relevant_entry(self, feature, index):
        relevant_value = []
        for ind, feat in enumerate(self.features_set_list):
            if ind == index:relevant_value.append(0);continue
            relevant_value.append(len(feat.intersection(feature))/float(len(feat.union(feature))))
        return self.get_similar(relevant_value)

    def find_related_tfidf(self):
        related_fd = codecs.open(self.related_tfidf, 'w', 'utf-8')
        vect = TfidfVectorizer(min_df=1)
        tfidf = vect.fit_transform(self.features_string_list)
        array = (tfidf * tfidf.T).A   
        array_list = array.tolist()
        
        for i,entry in enumerate(array_list):
            entry[i] = 0
            related = self.get_similar(entry)
            related_fd.write("%s\t%s\n%s\n%s\n%s\n%s\n\n" % (i, related, self.features_string_list[i], self.features_string_list[related[0]], self.features_string_list[related[1]], self.features_string_list[related[2]]))
        related_fd.close()

    def get_similar(self, entry):
        if not entry:return []
        if len(entry) <3 : return entry
        result = sorted(range(len(entry)), key=lambda i:entry[i], reverse=True) 
        return result[:3]
    
if __name__ == "__main__":

    mode = int(sys.argv[1])
   
    if mode == 0:
        data_obj = DataSetCollector()
        data_obj.run_main() 

    elif mode == 1:
        y_obj = YoutubeVideoClassifier()
        y_obj.run_main()

    elif mode == 2:
        r_obj = RelatedVideoGenerator()
        r_obj.run_main()

    else:
        print "Please enter the appropriate mode"
