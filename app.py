from flask import Flask, render_template, request
from mnist_predictor import math_symbol_predictor


app = Flask(__name__)


math = math_symbol_predictor()


# Base endpoint to perform prediction.
@app.route('/', methods=['POST'])
def make_prediction():
    if request.form['predictor'] == 'mnist':
        prediction = math.predict(request)
        return render_template('index.html', prediction=prediction, generated_text=None, tab_to_show='mnist')





@app.route('/', methods=['GET'])
def load():
    return render_template('index.html', prediction=None, generated_text=None, tab_to_show='mnist')


if __name__ == '__main__':
    app.run(debug=True)