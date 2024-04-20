from flask import Flask, request,render_template
import spell as sp
import error_correction_model as corr
import detection as md

app = Flask(__name__)

# routes
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/spell',methods=['POST','GET'])
def spell():
    if request.method=='POST':
        text = request.form['text']
        corrected_text = sp.correct_spellings_kannada_hunspell(text)
        return render_template('result.html',corrected_text=corrected_text)
@app.route('/grammar',methods=['POST','GET'])
def grammar():
    if request.method == 'POST':
        text = request.form['text']
        corrected_text = corr.grammar_correct(text)
        
        return render_template('result.html',corrected_text=corrected_text)
@app.route('/error-detect', methods=['POST','GET'])
def detection():
    if request.method == 'POST':
        text = request.form['text']
        corrected_text = md.detect(text)
        return render_template('result.html',corrected_text=corrected_text)

# python main
if __name__ == "__main__":
    app.run(debug=True)
