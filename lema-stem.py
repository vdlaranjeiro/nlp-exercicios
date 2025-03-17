# Realizar lematização e stemização de um texto em inglês
# Um caso de textos diferentes que retornem a mesma saída através das duas técnicas
# Saída: vetor ordenado de lemas e stems

from nltk.tokenize import ToktokTokenizer
from nltk.stem import PorterStemmer
import stanza

sample_text_1 = "The students conducted tests and submitted reports after the classes"
sample_text_2 = "The student conduct test and submit report after the class"

# Tokenização
tokenizer = ToktokTokenizer()
tokenized_sample_1 = tokenizer.tokenize(sample_text_1)
tokenized_sample_2 = tokenizer.tokenize(sample_text_2)

# Stemização
porter_stemmer = PorterStemmer()

stems_1 = []
for word in tokenized_sample_1:
    stem = porter_stemmer.stem(word)
    stems_1.append(stem)

stems_2 = []
for word in tokenized_sample_2:
    stem = porter_stemmer.stem(word)
    stems_2.append(stem)

# Lematização
stanza_nlp = stanza.Pipeline('en')

lemmas_1 = []
for sentence in stanza_nlp(sample_text_1).sentences:
    for word in sentence.words:
        lemmas_1.append(word.lemma)

lemmas_2 = []
for sentence in stanza_nlp(sample_text_2).sentences:
    for word in sentence.words:
        lemmas_2.append(word.lemma)

# Resultados
print("\n--- Resultados da Stemização ---")
print(f"Stems from Sample 1: {stems_1}")
print(f"Stems from Sample 2: {stems_2}")

print("\n--- Resultados da Lematização ---")
print(f"Lemmas from Sample 1: {lemmas_1}")
print(f"Lemmas from Sample 2: {lemmas_2}")

lemas_and_stems = stems_1 + stems_2 + lemmas_1 + lemmas_2
lemas_and_stems.sort()
print("\n--- Vetor Ordenado de Lemas e Stems ---")
print(lemas_and_stems)
