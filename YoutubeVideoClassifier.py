import sys, codecs
from SPARQLWrapper import SPARQLWrapper, JSON

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

    def collectActors(self):
        self.sparql.setQuery(self.getActors)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        
        for result in results.get('results').get('bindings'):
            actor_name = result.get('actor').get('value')
            actor_name = actor_name and actor_name.strip("http://dbpedia.org/resource/")
            if not actor_name:continue
            self.actors_names_file.write("%s\n" % (actor_name)) 

    def collectTvShows(self):
        self.sparql.setQuery(self.getTvShows)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        
        for result in results.get('results').get('bindings'):
            tvshow_name = result.get('tvshow').get('value')
            tvshow_name = tvshow_name and tvshow_name.strip("http://dbpedia.org/resource/")
            if not tvshow_name:continue
            self.tvshows_names_file.write("%s\n" % (tvshow_name)) 


class YoutubeVideoClassifier:
    def __init__(self):
        pass

    def run_main(self):
        pass


if __name__ == "__main__":

    mode = int(sys.argv[1])
   
    if mode == 0:
        data_obj = DataSetCollector()
        data_obj.run_main() 

    elif mode == 1:
        y_obj = YoutubeVideoClassifier()
        y_obj.run_main()

    else:
        print "Please enter the appropriate mode"
