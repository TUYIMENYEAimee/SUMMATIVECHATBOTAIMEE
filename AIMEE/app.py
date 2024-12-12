from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from questions_answers import questions_answers
import nltk
nltk.download('punkt')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chatbot.html')

@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_question = request.form['question']
    answer = find_best_match(user_question)
    return render_template('chatbot.html', question=user_question, answer=answer)

def find_best_match(user_question):
    questions = list(questions_answers.keys())
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    vectors = tfidf_vectorizer.fit_transform(questions + [user_question])
    cosine_similarities = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    best_match_index = cosine_similarities.argmax()
    if cosine_similarities[best_match_index] > 0.5:
        return questions_answers[questions[best_match_index]]
    return "Sorry, I don't understand that question."

if __name__ == '__main__':
    app.run(debug=True)
