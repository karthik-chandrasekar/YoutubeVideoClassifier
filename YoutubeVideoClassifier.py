#!/usr/bin/env python
#coding=utf-8
import sys, codecs
import operator, string, re, sys
import nltk, os, logging, ConfigParser
import nltk.classify.util
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from SPARQLWrapper import SPARQLWrapper, JSON
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC


class DataSetCollector:
    
    def __init__(self):
        self.sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        self.getFilms = """ SELECT ?movie WHERE { ?movie a dbpedia-owl:Film}"""
        self.getTvShows = """ SELECT ?tvshow WHERE { ?tvshow a dbpedia-owl:TelevisionShow}"""
        self.getActors = """ SELECT ?movie ?actor WHERE {?movie a dbpedia-owl:Film; dbpedia-owl:starring ?actor}"""
        self.movie_names_file = codecs.open('movies_name', 'w', 'utf-8')
        self.actors_names_file = codecs.open('actors_name', 'w', 'utf-8')
        self.tvshows_names_file = codecs.open('tvshows_name', 'w', 'utf-8')



    def run_main(self):
        self.collectFilms()
        self.collectActors()
        self.collectTvShows()

    def collectFilms(self): 
        self.sparql.setQuery(self.getFilms)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        
        for result in results.get('results').get('bindings'):
            movie_name = result.get('movie').get('value')
            movie_name = movie_name and movie_name.strip("http://dbpedia.org/resource/")
            if not movie_name:continue
            self.movie_names_file.write("%s\n" % (movie_name)) 
        self.movie_names_file.close()

    def collectActors(self):
        self.sparql.setQuery(self.getActors)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        
        for result in results.get('results').get('bindings'):
            actor_name = result.get('actor').get('value')
            actor_name = actor_name and actor_name.strip("http://dbpedia.org/resource/")
            if not actor_name:continue
            self.actors_names_file.write("%s\n" % (actor_name)) 
        self.actors_names_file.close()

    def collectTvShows(self):
        self.sparql.setQuery(self.getTvShows)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        
        for result in results.get('results').get('bindings'):
            tvshow_name = result.get('tvshow').get('value')
            tvshow_name = tvshow_name and tvshow_name.strip("http://dbpedia.org/resource/")
            if not tvshow_name:continue
            self.tvshows_names_file.write("%s\n" % (tvshow_name)) 

        self.tvshows_names_file.close()        

class YoutubeVideoClassifier:
    def __init__(self):
        self.movie_file = 'movies_name'
        self.actors_file = 'actors_name'
        self.tvshows_file = 'tvshows_name'
        self.test_file = 'CodeAssignmentDataSet.json'
        
        self.nb_output = 'NB_results'
        self.svm_output = 'SVM_results'
            
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
        movies_fd = codecs.open(self.movie_file)
        
        for movie in movies_fd.readlines():
            if not movie: continue
            self.movies_list.append(movie)           
 
    
    def load_actors(self):
        self.actors_list = []
        actors_fd = codecs.open(self.actors_file)

        for actor in actors_fd.readlines():
            if not actor: continue
            self.actors_list.append(actor)

    def load_tvshows(self):
        self.tvshows_list = []
        tvshows_fd = codecs.open(self.tvshows_file)

        for tvshow in tvshows_fd.readlines():
            if not tvshow:continue
            self.tvshows_list.append(tvshow)

    def load_test_data(self):
        self.test_instances_list = []
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
            selected_features = self.feature_selection([item.replace("_"," ")])
            self.train_features.append((selected_features, 'tvshow'))

        for item in self.movies_list:
            if not item: continue
            selected_features = self.feature_selection([item.replace("_"," ")])
            self.train_features.append((selected_features, 'movies'))
            
        for item in self.actors_list:
            if not item: continue
            selected_features = self.feature_selection([item.replace("_"," ")]) 
            self.train_features.append((selected_features, 'celebrity'))    

    def classification(self):

        #Training NB Classifier
        self.nb_classifier = NaiveBayesClassifier.train(self.train_features)         
        
        #Training SVM classifier
        self.svm_classifier = SklearnClassifier(LinearSVC()) 
        self.svm_classifier.train(self.train_features)

    def testing(self):
        self.test_instances_list = []
        nb_fd = codecs.open(self.nb_output, 'w', 'utf-8')
        svm_fd = codecs.open(self.svm_output, 'w', 'utf-8')

        for instance in self.test_data:
            if not instance:continue
            test_features = instance.get('title').split(" ")
            test_features.extend(instance.get('description').split(" "))
            selected_features = self.feature_selection(test_features)

            label = self.nb_classifier.classify(selected_features)
            nb_fd.write("%s\n" % (label))

            label = self.svm_classifier.classify(selected_features)
            svm_fd.write("%s\n" % (label))
        
        nb_fd.close()
        svm_fd.close()


class RelatedVideo:
    def __init__(self):
        self.test_file = 'CodeAssignmentDataSet.json'
        self.test_data = []
        self.selected_features = []

    def run_main(self):
        self.load_data()
        self.select_features()
        self.find_related()

    def load_data(self):
        self.test_instances_list = []
        json_data = open(self.test_file)
        self.test_data = json.load(json_data)
     
    def select_features(self):
        for instance in self.test_data:
            self.selected_features.append(instance.get('title') + instance.get('description'))
 
    def find_related(self):
        vect = TfidfVectorizer(min_df=1)
        tfidf = vect.fit_transform(self.selected_features)
        array = (tfidf * tfidf.T).A   
        array_list = array.tolist()

        for i,entry in enumerate(array_list):
            related = self.get_3_similar(entry)
            print "%s\t%s\n" % (i, related)

    def get_3_similar(self, entry):
        if not entry:return []
        if len(entry) <3 : return entry
        result = sorted(range(len(entry)), key=lambda i:entry[i]) 
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
        r_obj = RelatedVideo()
        r_obj.run_main()

        print "Please enter the appropriate mode"
