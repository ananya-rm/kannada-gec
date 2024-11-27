from flask import Flask,request,render_template, send_file, redirect, url_for
import spell as sp
import error_correction_model as corr
import detection as md
import os
import re

def write_to_file(text, original_file_path):
    # Get the directory and filename from the original file path
    directory, filename = os.path.split(original_file_path)
    # Add "_new" to the filename
    new_filename = filename.split('.')[0] + '_new.' + filename.split('.')[1]
    # Construct the new file path
    new_file_path = os.path.join(directory, new_filename)
    # Write the text to the new file
    with open(new_file_path, 'w', encoding="utf-8") as f:
        f.write(text)
    return new_file_path

app = Flask(__name__)

# routes
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/spell',methods=['POST','GET'])
def spell():
    if request.method=='POST':
        file = request.files['file']
        if file:
            file_text = file.read().decode("utf-8")
            processed_file_text = sp.correct_spellings_kannada_hunspell(file_text)
            processed_file_path = write_to_file(processed_file_text, file.filename)
            return send_file(processed_file_path, as_attachment=True)
        text = request.form['text']
        if text:
            corrected_text = sp.correct_spellings_kannada_hunspell(text)
            return render_template('result.html',corrected_text=corrected_text)
        

@app.route('/error-detect', methods=['POST','GET'])
def detection():
    if request.method == 'POST':
        text = request.form['text']
        corrected_text = md.detect(text)
        return render_template('result.html',corrected_text=corrected_text)
       
@app.route('/grammar',methods=['POST','GET'])
def grammar():
    if request.method == 'POST':
        text = request.form['text']
        corrected_text = corr.grammar_correct(text)
        
        return render_template('result.html',corrected_text=corrected_text)
    

def save_word_info(word, count, gender, filename="C:/Users/anany/Desktop/fyp/src/sub.txt"):

    kannada_pattern = re.compile(r'[\u0C80-\u0CFF]+')

    # Check if the text contains Kannada characters
    b = bool(re.search(kannada_pattern, word))

    if(b != True):
        return render_template('result.html',corrected_text="Please enter text in Kannada")
    # Open the file in append mode
    with open(filename, 'a', encoding='utf-8') as file:
        # Format the word info with tab separation
        line = f"{word}\t{gender}\t{count}\tT\n"
        file.write(line)
        append_word_to_dic(word)

def append_word_to_dic(word, dic_filename="C:/Users/anany/Desktop/fyp/src/kn.dic"):
    # Open the dictionary file in append mode
    with open(dic_filename, 'a', encoding='utf-8') as dic_file:
        dic_file.write(f"{word}\n")

@app.route('/lexicon', methods=['GET', 'POST'])
def lexicon():
    if request.method == 'POST':
        word = request.form['word']
        count = request.form['count']
        gender = request.form['gender']
        
        # Save the word information to the file
        save_word_info(word, count, gender)
        
        return redirect(url_for('lexicon'))
    
    return render_template('index.html')
    

    


# python main
if __name__ == "__main__":
    app.run(debug=True)
