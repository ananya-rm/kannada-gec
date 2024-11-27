import re
from hunspell import Hunspell

def correct_spellings_kannada_hunspell(text, dict_path="C:/Users/anany/Desktop/project/kn"):

    kannada_pattern = re.compile(r'[\u0C80-\u0CFF]+')

    # Check if the text contains Kannada characters
    b = bool(re.search(kannada_pattern, text))

    if(b != True):
        return "Please enter text in Kannada"

    # Initialize Hunspell with the Kannada dictionary
    kannada_dict_path = "C:/Users/anany/Desktop/fyp/src/kn"
    kannada_affix_path = "C:/Users/anany/Desktop/fyp/src/kn"
    kannada_spell_checker = Hunspell(kannada_affix_path, kannada_dict_path)

    # Tokenize the text into words
    words = text.split()

    kannada_spell_checker = Hunspell(dict_path , dict_path + '.dic')

    # Tokenize the text into words
    words = text.split()

    corrected_text = []

    for word in words:
        # Check if the word is misspelled
        if not kannada_spell_checker.spell(word):
            # If misspelled, suggest corrections
            suggestions = kannada_spell_checker.suggest(word)
            if suggestions:
                # If suggestions available, take the first suggestion
                corrected_word = suggestions[0]
                # Append the corrected word in brackets
                corrected_text.append(f"{word} [{corrected_word}]")
            else:
                # If no suggestions, keep the original word
                corrected_text.append(word)
        else:
            # If spelled correctly, keep the original word
            corrected_text.append(word)

    # Join the corrected words back into a sentence
    return ' '.join(corrected_text)


kannada_text = "ನನ್ನು ಹೊಗುತ್ತೇನೆ "
corrected_text = correct_spellings_kannada_hunspell(kannada_text)
# print("Original: ", kannada_text)
# print("Corrected:", corrected_text)

with open("C:/Users/anany/Desktop/spell.txt",'w',encoding='utf-8') as file:
    file.write(corrected_text)
