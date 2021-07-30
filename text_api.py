from flask import Flask, jsonify, request
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from text_storage import read_from_json, fetch_courses_from_db
from text_transform import process_text

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

api = Api(
    app,
    version='1.0',
    title='NLP API',
    description='Text NLP processing API'
)

CORS(app)

# https://sanjayasubedi.com.np/nlp/nlp-with-python-topic-modeling/
# example /api/topics/20/15
@api.route('/api/topics/<int:topics_num>/<int:top_words_num>', methods=['GET'])
@api.doc(params={
    'topics_num': 'Number of topics to model',
    'top_words_num': 'Number of most influencing words to display per topic'
})
class Topics(Resource):
    def get(self, topics_num, top_words_num):
        processed_text = list(
            map(lambda obj: obj['description'], read_from_json()))

        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(processed_text)

        nmf = NMF(n_components=topics_num, init='random',
                  random_state=0, max_iter=1000)
        nmf.fit(features)

        # list of unique words found by the vectorizer
        feature_names = vectorizer.get_feature_names()

        topics = []
        for i, topic_vec in enumerate(nmf.components_):
            topic = []
            # topic_vec.argsort() produces a new array
            # in which word_index with the least score is the
            # first array element and word_index with highest
            # score is the last array element. Then using a
            # fancy indexing [-1: -top_words_num-1:-1], we are
            # slicing the array from its end in such a way that
            # top `top_words_num` word_index with highest scores
            # are returned in desceding order
            for fid in topic_vec.argsort()[-1:-top_words_num-1:-1]:
                topic.append(feature_names[fid])
            topics.append(topic)
        return jsonify(topics)

parser = api.parser()
parser.add_argument('text', type=str, location='json',
                    help='Text to find neighbors of')

@api.route('/api/neighbors/<int:neighbors_num>', methods=['POST'])
@api.doc(params={
    'neighbors_num': 'Number of neighbors to find',
})
class Neighbors(Resource):
    @api.expect(parser)
    def post(self, neighbors_num):
        text = request.get_json()['text']
        data = read_from_json()

        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(
            list(map(lambda obj: obj['description'], data)))

        knn = NearestNeighbors(n_neighbors=neighbors_num, metric='cosine')
        knn.fit(features)

        neighbors = []
        D, I = knn.kneighbors(
            vectorizer.transform(
                process_text([text])
            )
        )
        for dist, index in zip(D[0], I[0]):
            neighbors.append({'id': data[index]['id'], 'dist': dist})

        return jsonify(neighbors)

@api.route('/api/clusters/<int:clusters_num>', methods=['GET'])
@api.doc(params={
    'clusters_num': 'Number of clusters to model'
})
class Clusters(Resource):
    def get(self, clusters_num):
        data = read_from_json()

        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(
            list(map(lambda obj: obj['description'], data)))

        kmeans = KMeans(n_clusters=clusters_num, random_state=0)
        kmeans.fit(features)

        pca = PCA(n_components=2, random_state=0)
        reduced_features = pca.fit_transform(features.toarray())

        res = []
        courses = fetch_courses_from_db()
        for label, course, location in zip(kmeans.labels_, courses, reduced_features.tolist()):
            res.append({
                'x': location[0] * 10,
                'y': location[1] * 10,
                'cluster': int(label),
                'data': {
                    'title': course['title'],
                    'url': course['url']
                }
            })

        return jsonify(res)

@api.route('/api/clusters/scores', methods=['POST'])
class ClustersScores(Resource):
    @api.expect(api.model('NumbersToCheck', {
        'numbers': fields.List(fields.Integer)
    }))
    def post(self):
        data = read_from_json()
        numbers = request.get_json()['numbers']

        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(
            list(map(lambda obj: obj['description'], data)))

        scores = {}
        for k in numbers:
            kmeans = KMeans(n_clusters=k).fit(features)
            labels = kmeans.labels_
            scores[k] = silhouette_score(features, labels, metric='euclidean')

        return jsonify(scores)

def main():
    app.run(debug=True, port=3000)

if __name__ == '__main__':
    main()