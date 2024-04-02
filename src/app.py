from flask import Flask, request,render_template
from model import SpellCheckerModule

app = Flask(__name__)
spell_checker_module = SpellCheckerModule()

# routes
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/spell',methods=['POST','GET'])
def spell():
    if request.method=='POST':
        text = request.form['text']
        corrected_text = spell_checker_module.correct_spellings_kannada_hunspell(text)
        return render_template('result.html',corrected_text=corrected_text)
# @app.route('/grammar',methods=['POST','GET'])
# def grammar():
#     if request.method == 'POST':
#         text = request.form['gtext']
#         corrected_text = grammar_checker_module.correct_spellings_kannada_hunspell(text)
#         return render_template('index.html',corrected_text=corrected_text)
#     return render_template('index.html',corrected_file_text=corrected_file_text,corrected_file_grammar=corrected_file_grammar)

# python main
if __name__ == "__main__":
    app.run(debug=True)