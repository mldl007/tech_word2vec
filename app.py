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
    <td>Word</td><td>Similarity score</td></tr>'''
    try:
        w2v = KeyedVectors.load(os.path.join(".", "w2v", "tech_w2v.bin"),
                                mmap='r')
        similar_words = w2v.wv.most_similar([vocab])
        n = 1
        for word, score in similar_words:
            if n % 2 == 0:
                html += f'<tr style="background: lightgray;">' \
                        f'<td>{word}</td><td>{score}</td></tr>'
            else:
                html += f'<tr><td>{word}</td><td>{score}</td></tr>'
            n += 1
        html += "</table>"
    except Exception as e:
        html = f'<p style="color: red;">Error: {e}</p>'
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
