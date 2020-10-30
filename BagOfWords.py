
import nltk

sentence1 = "This is a sample sentence, to be replaced by an abstract"
sentence2 = "This is a different sample sentence!"

#step 1: create a vocabulary of known words

vocab = nltk.word_tokenize(sentence1) + nltk.word_tokenize(sentence2)
#tokens now has the words from both sentences, but it puts punctuation into separate words

vocab = list(dict.fromkeys(vocab))

#vocab has the set of unique tokens from both sentences
vocabSize = len(vocab)

#generate a vector of sentence1
#for later - update to use NUMPY to generate list of 0s instead
vector1 = [0] * vocabSize

#tokenize sentence1
tokens1 = nltk.word_tokenize(sentence1)

i = 0
while i < vocabSize:
	vector1[i] = tokens1.count(vocab[i])
	i = i+1

print(vocab)
print(tokens1)
print(vector1)