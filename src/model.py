from hunspell import Hunspell

class SpellCheckerModule:
    def __init__(self):
        pass

    def correct_spellings_kannada_hunspell(text, dict_path="C:/Users/anany/Desktop/project/kn"):
         # Initialize Hunspell with the Kannada dictionary
        kannada_dict_path = "C:/Users/anany/Desktop/dictionary/kn"
        kannada_affix_path = "C:/Users/anany/Desktop/dictionary/kn"
        kannada_spell_checker = Hunspell(kannada_affix_path, kannada_dict_path)

        # Tokenize the text into words
        words = text.split()

        corrected_words = []

        for word in words:
        # Check if the word is misspelled and suggest corrections
            if not kannada_spell_checker.spell(word):
                suggestions = kannada_spell_checker.suggest(word)
                if suggestions:
                    corrected_word = suggestions[0]
                    corrected_words.append(corrected_word)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

    # Join the corrected words back into a sentence
        corrected_text = ' '.join(corrected_words)

        with open("C:/Users/anany/Desktop/spell.txt",'w',encoding='utf-8') as file:
            file.write(corrected_text)

        return corrected_text
    
if __name__  == "__main__":
    obj = SpellCheckerModule()
    
    
