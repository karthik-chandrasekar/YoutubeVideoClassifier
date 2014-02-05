import sys, codecs
from SPARQLWrapper import SPARQLWrapper, JSON

class DataSetCollector:
    
    def __init__(self):
        self.sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        self.getFilmQuery = """ SELECT DISTINCT ?film_title ?film_abstract
        WHERE {
        ?film_title rdf:type <http://dbpedia.org/ontology/Film> .
        ?film_title rdfs:comment ?film_abstract 
        } """

        self.getMovieActorQuery = """ SELECT ?movie ?actor
        WHERE {
             ?movie a dbpedia-owl:Film ;
                  dbpedia-owl:starring ?actor
        }"""

        self.film_file = codecs.open('film_dump', 'w', 'utf-8')
        self.movie_actor_file = codecs.open('movie_actor_dump', 'w', 'utf-8')


    def run_main(self):
        self.run_film_query()
        self.run_movie_actor_query()

    def run_film_query(self):
        self.sparql.setQuery(self.getFilmQuery)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()

        for result in results.get('results').get('bindings'):
            lang = result.get('film_abstract').get('xml:lang')
            if lang != 'en':
                continue
            desc = result.get('film_abstract').get('value')
            title = result.get('film_title').get('value')
            self.film_file.write("%s\t%s\n" % (title, desc))
 

    def run_movie_actor_query(self):
        self.sparql.setQuery(self.getMovieActorQuery)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()



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
