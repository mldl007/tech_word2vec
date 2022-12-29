from flask import Flask, render_template, request
from flask_cors import cross_origin, CORS
from gensim.models import KeyedVectors
import os


app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)


@app.route("/")
@cross_origin()
def index():
    """
    displays the index.html page
    """
    return render_template("index.html")


def return_output_table(vocab):
    html = f'''<br><p style="font-weight: bold;">Input word: {vocab}</p>
    <table style= "border: 1px solid black; border-collapse: collapse;">
    <tr style="background: black; color: white;">
    <td>Word: tech_w2v</td><td>Similarity score: tech_w2v</td>
    <td>Word: google_w2v</td><td>Similarity score: google_w2v</td>
    </tr>'''
    try:
        w2v = KeyedVectors.load(os.path.join(".", "w2v", "tech_w2v.bin"),
                                mmap='r')
        google_w2v = KeyedVectors.load(os.path.join(".", "google_word2vec", "google_w2v_100k.bin"),
                                       mmap='r')
        similar_words = w2v.wv.most_similar([vocab])
        google_similar_words = google_w2v.most_similar([vocab])
        n = len(similar_words)
        for i in range(n):
            if (i+1) % 2 == 0:
                html += f'<tr style="background: lightgray;">' \
                        f'<td>{similar_words[i][0]}</td><td>{similar_words[i][1]}</td>' \
                        f'<td>{google_similar_words[i][0]}</td><td>{google_similar_words[i][1]}</td></tr>'
            else:
                html += f'<tr>' \
                        f'<td>{similar_words[i][0]}</td><td>{similar_words[i][1]}</td>' \
                        f'<td>{google_similar_words[i][0]}</td><td>{google_similar_words[i][1]}</td></tr>'
        html += "</table>"
    except Exception as e:
        n = len(similar_words)
        for i in range(n):
            if (i + 1) % 2 == 0:
                html += f'<tr style="background: lightgray;">' \
                        f'<td>{similar_words[i][0]}</td><td>{similar_words[i][1]}</td>' \
                        f'</tr>'
            else:
                html += f'<tr>' \
                        f'<td>{similar_words[i][0]}</td><td>{similar_words[i][1]}</td>' \
                        f'</tr>'
        html += "</table>"
        # html = f'<p style="color: red;">Error: {e}</p>'
    return html


@app.route("/", methods=["POST"])
@cross_origin()
def form_prediction():
    """
    Returns the API response based on the inputs filled in the form.
    """
    try:
        vocab = request.form["vocab"]
        result = return_output_table(vocab)
    except Exception as e:
        result = f'<p style="color: red;">Error: {e}</p>'
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
