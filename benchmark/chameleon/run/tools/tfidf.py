from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tfidf(query, document):
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([query, document])
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        similarity_score = similarity_matrix[0][0] 
        if similarity_score>0.1:
            return {"match": 1}
        else:
            return {"match": 0}
    except Exception as e:
        return "Error: "+str(e)

if __name__ == "__main__":
    query = "Fastener-Free Idler Pulley"
    document = "Belt-driven systems, such as a front-end accessory drive for an internal-combustion engine, often include one or more idler pulleys. It is typical for these idler pulleys to be retained by a bolt or bolt and washer. To obviate the bolt and washer, a fastener-less system has a shaft with a circumferential groove and a bearing with an inner race that has a circumferential ridge. An idler pulley is mounted on the bearing. The ridge engages with the groove of the shaft to retain the idler pulley/bearing on the shaft. Additionally, either a key inserted into keyways or stakes are used to avoid relative rotation of the inner race of the bearing with respect to the shaft."
    print(tfidf(query, document))

