import nltk
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

conversation = '''
A: You look tired.
B: Yeah, I've been working so much overtime lately.
A: Really? How come?
B: My boss gave me a big project. I had to have it finished by this morning. It was so difficult.
A: You shouldn't work so hard.
B: I know, but hard work pays off. You know.
A: What do you mean?
B: Maybe now I'll get that promotion I was hoping for.
'''

# 문장 토큰화
sentences = sent_tokenize(conversation)

# 불용어 제거 및 어간 추출
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# 전처리
def preprocess(sentence):
    tokens = word_tokenize(sentence.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return tokens

sentences_tokenized = [preprocess(sentence) for sentence in sentences]

word_set = set(word for sentence in sentences_tokenized for word in sentence)
word_to_idx = {word: i for i, word in enumerate(word_set)}

word_vectors = np.zeros((len(word_set), len(word_set)))
for sentence in sentences_tokenized:
    for i in range(len(sentence)):
        for j in range(i+1, len(sentence)):
            word_i_idx, word_j_idx = word_to_idx[sentence[i]], word_to_idx[sentence[j]]
            word_vectors[word_i_idx, word_j_idx] += 1
            word_vectors[word_j_idx, word_i_idx] += 1

sentence_vectors = np.zeros((len(sentences), len(word_set)))
for i, sentence in enumerate(sentences_tokenized):
    for word in sentence:
        sentence_vectors[i, word_to_idx[word]] += 1

similarity_matrix = cosine_similarity(sentence_vectors)

# summarization
def summarization(similarity_matrix, d=0.85, max_iter=100):
    scores = np.ones(len(sentences))
    for i in range(max_iter):
        scores = (1-d) + d*np.dot(similarity_matrix.T, scores)
    return scores

scores = summarization(similarity_matrix)
ranked_sentences = sorted(((score, i) for i, score in enumerate(scores)), reverse=True)

# 요약문 출력 (대화문 중 핵심이 되는 대화 2문장 출력)
num_sentences = 2
summary_sentences = sorted(ranked_sentences[:num_sentences])
summary = ' '.join([sentences[i] for _, i in summary_sentences])
print(summary)