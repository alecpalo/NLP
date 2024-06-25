import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

from nltk.util import ngrams
import string


# Sample text data
document = """
Abstract
A coffee brewing system and method that includes a brew chamber that holds a brew solution during a brew cycle and dispenses the brew solution; a water system that dispenses water into the brew chamber; a content sensing system that measures the brew solution contents added to the brew chamber; a temperature control system with a heating element and a temperature sensor; at least one recirculating processing loop with a particle monitor system, wherein the recirculating processing loop circulates brew solution extracted from the brew chamber; and a control system that is communicatively coupled to the content sensing system, the temperature control system and the particle monitor system during a brew cycle, wherein the control system controls a brew cycle based on a selected a specified taste profile.

Description
The following description of the embodiments of the invention is not intended to limit the invention to these embodiments but rather is to enable a person skilled in the art to make and use this invention.
1. Overview
A system and method for controlling the brew process of a coffee maker functions to enable an enhanced level of control over the coffee brewing. The system and method can involve control over grind size, filter size, water temperature, brewing temperature, coffee-to-water ratio, and/or dissolved solid value. The system and method preferably employ automated control over the various variables. The system and method can be applied to consistently brewing customized coffee across a variety of taste profiles. For example, coffee can be customized by strength and extraction. The automated control can be used in producing a particular brewing process, which may be selected from a menu or any suitable option. The automated control can additionally be used in adapting a brew process to one or more user preferences. For example, the system and method could be employed in learning and executing a brewing process that is customized for a particular user. Additionally, a learned taste profile of a user could be translated across different coffee options such as bean or roast variations. A taste profile preferably characterizes preferences across various coffee types and options. A taste profile can be used to determine a brew process configuration, which characterizes how one brew cycle is executed by the coffee maker. In one exemplary usage scenario, after a taste profile is created for a user, that user could select a new type of bean to try and the coffee maker device will prepare a cup of coffee using that bean customized based on the bean and the user preferences. In another application, a set of user test profiles can be used to generate a brew process for a set of users such that people sharing a carafe of coffee have the coffee brewed in a style that may be more enjoyable to the whole group rather than just an individual. The system and method can additionally include an operating mode wherein a set of multiple brew processes can be performed for a single setup, which can function to enable a tasting flight of coffee or per cup customization.
The system and method may be implemented with sensed feedback within the coffee maker system. Alternative implementations may utilize open loop implementations of a coffee maker system that do not include sensors or as many sensors. In an open loop implementation, control of the coffee maker system can be performed based on predefined expectations. An open loop implementation may utilize more manual settings to define how the coffee maker system performs a particular brew cycle. Fewer or no sensors can lower the production cost of the device and still provide an intelligent brewing process. In one variation, the taste profile and/or a brew process configuration can be manually entered using one or more inputs. Preferably, a sensor feedback implementation system and an open loop implementation can be integrated within a connected platform such that brewing control intelligence can be shared between different models of coffee makers. For example, a premium coffee maker may be able to use integrated sensors to control how a new type of coffee blend is brewed. The control configurations determined using active feedback may be shared through a connected platform such that an open loop system could execute a brew using control settings learned by one or more premium coffee makers as shown in FIG. 4 .
As a first potential benefit, the system and method may offer greater flexibility and control when making coffee. The system and method may facilitate trying and using new coffee types by accepting coffee beans and/or coffee grounds as opposed to pre-packaged pods. It would be appreciated that the system and methods could be used with pre-packaged pods as a coffee source. In addition to being compatible with a wider variety of coffee bean sources, the system and method can adapt to different coffee types. More specifically, the system and method can dynamically alter the brewing process for a new coffee type or utilize coffee brew predictions based on information on the coffee type or properties.
As another potential benefit, the system and method can control one or more properties during the brew process so that coffee is produced in a controlled and repeatable manner. The system and method preferably addresses coffee brewing from a parameterized perspective. In particular, the system and method may target particular dissolved particle values in produced coffee. The system and method could additionally or alternatively target brewing time, water and/or brew solution temperature, water-to-coffee ratio, coffee grind size, and/or other properties.
As another potential benefit, the system and method can offer a connected personalized experience for users. The system and method can learn the taste preferences of a user and apply that in future brews. Applications of a user taste profile can include adjusting the brew process configuration used with a new coffee type, mixing taste preferences when brewing for multiple people, targeting different tastes based on the current situation, and other usages. Personalized brewing can additionally benefit from multi-user data. Data analysis of multiple users can be used to improve the experience of individual users.
The system and method described herein are described as being applied to a primary application of coffee, but the described system and method may be applied to cold brews, herbal teas, teas, and/or other suitable drinks and solutions. In one example, a cold brew implementation can forgo heating elements for other temperature control mechanisms. In some variations, a cold brew device may be designed to be stored in a refrigerator to provide the temperature regulation. In a tea maker embodiment, parameters such as tea type, filter size, water temperature, brewing temperature, tea-to-water ratio, dissolved solid value can be regulated to adapt the brewing of tea to a predefined steeping process and/or user taste-profile.

Claims
We claim:

1. A coffee brewing system comprising: a brew chamber that holds a brew solution during a brew cycle and dispenses the brew solution; a water system that is integrated to dispense water into the brew chamber; a content sensing system that measures the brew solution contents added to the brew chamber; a temperature control system with a heating element and a temperature sensor, wherein the heating element of the temperature control system directly heats liquid in the water system; at least one recirculating processing loop with a particle monitor system, wherein the recirculating processing loop circulates brew solution extracted from the brew chamber, wherein the recirculating processing loop comprises a subsection that is thermally coupled to the water system such that the heating element indirectly heats brew solution circulated through the processing loop; and a control system that is communicatively coupled to the content sensing system, the temperature control system and the particle monitor system during a brew cycle, wherein the control system controls a brew cycle based on a selected a specified taste profile.

2. The system of claim 1, wherein the specified taste profile is selected from a set of taste profiles with each taste profile associated with a distinct user.

3. The system of claim 1, further comprising a user application that collects user feedback on dispensed coffee, wherein the user feedback is used in part to augment a brew process configuration of a second brew cycle of the coffee maker.
4. The system of claim 3, wherein the user feedback is used in combination with a selected bean type to determine the brew process configuration used by the control system during the second brew cycle.

5. The system of claim 1, further comprising a set of manual controls that define the taste profile settings referenced by the control system.

6. The system of claim 1, further comprising a coffee grinding system with a grind outlet positioned to deliver coffee grounds to the brew chamber, wherein the grind size and quantity of produced coffee grounds is controlled by the control system.

7. The system of claim 1, wherein the control system includes a calibration mode, wherein the heating effect of the temperature control system is calibrated and accounted for in directing control of the temperature control system.

8. The system of claim 1, further comprising a tasting flight system that can be removably added to a brew chamber while the control system operates in a tasting flight mode; wherein the tasting flight system comprises at least a chamber divider segmenting the brew chamber into multiple sub-chambers and a chamber selection system through which the control system can individually control the brew cycle of each sub-chamber.

9. A coffee brewing system comprising: a brew chamber that holds a brew solution during a brew cycle and dispenses the brew solution; a water system is integrated to dispense water into the brew chamber; a content sensing system that measures the brew solution contents added to the brew chamber; a temperature control system with a heating element and a temperature sensor; at least one recirculating processing loop with a particle monitor system, wherein the recirculating processing loop circulates brew solution extracted from the brew chamber; a control system that is communicatively coupled to the content sensing system, the temperature control system and the particle monitor system during a brew cycle, wherein the control system controls a brew cycle based on a selected a specified taste profile; and a tasting flight system that can be removably added to a brew chamber while the control system operates in a tasting flight mode, the tasting flight system comprising: a chamber divider segmenting the brew chamber into multiple sub-chambers and a chamber selection system through which the control system can individually control the brew cycle of each sub-chamber.

10. The system of claim 9, wherein the specified taste profile is selected from a set of taste profiles with each taste profile associated with a distinct user.

11. The system of claim 9, further comprising a user application that collects user feedback on dispensed coffee, wherein the user feedback is used in part to augment a brew process configuration of a second brew cycle of the coffee maker.

12. The system of claim 11, wherein the user feedback is used in combination with a selected bean type to determine the brew process configuration used by the control system during the second brew cycle.

13. The system of claim 9, further comprising a set of manual controls that define the taste profile settings referenced by the control system.

14. The system of claim 9, further comprising a coffee grinding system with a grind outlet positioned to deliver coffee grounds to the brew chamber, wherein the grind size and quantity of produced coffee grounds is controlled by the control system.

15. The system of claim 9, wherein the heating element of the temperature control system directly heats liquid in the water system; and wherein the processing loop comprises a subsection that is thermally coupled to the water system such that the heating element indirectly heats brew solution circulated through the processing loop.

16. The system of claim 9, wherein the control system includes a calibration mode, wherein the heating effect of the temperature control system is calibrated and accounted for in directing control of the temperature control system.

brews coffee in the morning.

coffee.
"""
keywords = ["Has beeper to alert user when done.", "Has an electronic screen for control", "brews coffee", "brews milk", "Has a timer function.", "Has a temperature control system.", "can run and jump"]
# Assume these are the ground truth labels for the relevance of keywords (1 for relevant, 0 for not relevant)
ground_truth_labels = [0, 1, 1, 0, 0, 1, 0]

# Load GloVe embeddings from file
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Path to the GloVe embeddings file
glove_file_path = 'glove/vectors.txt'  # Adjust the file path as needed

# Load the embeddings
glove_embeddings = load_glove_embeddings(glove_file_path)

def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Generate n-grams from tokens
def generate_ngrams(tokens, ngram_range):
    ngrams_list = []
    for n in range(ngram_range[0], ngram_range[1] + 1):
        ngrams_list.extend([' '.join(gram) for gram in ngrams(tokens, n)])
    return ngrams_list

# Create TF-IDF weighted embeddings
def tfidf_weighted_embeddings(text, embeddings, tfidf_vectorizer, ngram_range=(1, 3)):
    tokens = preprocess(text)
    ngrams_list = generate_ngrams(tokens, ngram_range)
    tfidf_matrix = tfidf_vectorizer.transform([' '.join(ngrams_list)])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    weighted_vectors = []
    for word in ngrams_list:
        if word in embeddings and word in feature_names:
            index = feature_names.tolist().index(word)
            weight = tfidf_matrix[0, index]
            weighted_vectors.append(weight * embeddings[word])
    if weighted_vectors:
        return np.mean(weighted_vectors, axis=0)
    else:
        return np.zeros(len(next(iter(embeddings.values()))))

# Prepare TF-IDF vectorizer
corpus = [document] + keywords
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
tfidf_vectorizer.fit(corpus)

# Calculate TF-IDF weighted embeddings for the document
doc_vector = tfidf_weighted_embeddings(document, glove_embeddings, tfidf_vectorizer, ngram_range=(1, 3))

# Pearson Correlation function
def pearson_correlation(vec1, vec2):
    if np.std(vec1) == 0 or np.std(vec2) == 0:
        return 0  # Handle zero standard deviation cases
    return np.corrcoef(vec1, vec2)[0, 1]

# Function to get relevance scores for keywords/phrases
def get_keyword_relevance(keywords, embeddings, doc_vector):
    relevance_scores = {}
    for keyword in keywords:
        keyword_vector = tfidf_weighted_embeddings(keyword, embeddings, tfidf_vectorizer, ngram_range=(1, 3))

        # Calculate Cosine Similarity
        cosine_sim = cosine_similarity([doc_vector], [keyword_vector])[0][0]

        # Calculate Pearson Correlation
        pearson_corr = pearson_correlation(doc_vector, keyword_vector)

        # Combine Cosine Similarity and Pearson Correlation
        alpha = 0.5  # Weight for cosine similarity
        beta = 0.5   # Weight for Pearson correlation
        hybrid_similarity = alpha * cosine_sim + beta * pearson_corr

        relevance_scores[keyword] = hybrid_similarity
    return relevance_scores

# Get relevance scores for the given keywords
relevance_scores = get_keyword_relevance(keywords, glove_embeddings, doc_vector)

# Convert relevance scores to binary labels based on a threshold
threshold = 0.5
predicted_labels = [1 if score >= threshold else 0 for score in relevance_scores.values()]

# Calculate evaluation metrics
precision, recall, f1, _ = precision_recall_fscore_support(ground_truth_labels, predicted_labels, average='binary')
average_precision = average_precision_score(ground_truth_labels, list(relevance_scores.values()))

# Print the relevance scores
print("Relevance scores for the given keywords/phrases:")
for keyword, score in relevance_scores.items():
    print(f"{keyword}: {score:.2f}")

# Print evaluation metrics
print(f"\nPrecision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Average Precision: {average_precision:.2f}")

# Optional: Print the full document vector for debugging
print("\nDocument Vector:")
print(doc_vector)


