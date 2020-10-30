
import nltk
import csv

vocab = []
with open('dataset/train_first_three_no_header.csv') as train_file:
	reader = csv.reader(train_file, delimiter=',')
	for row in reader:
		vocab.extend(nltk.word_tokenize(f'{row[2]}'))

vocab = list(dict.fromkeys(vocab))

#Removed some punctuation (there is a more sophisticated way to do this! Need a tokenizer which will remove punctuation and common words)
vocab = [i for i in vocab if (i != "!" and i != "," and i != "." and i != "(" and i != ")" and i != ";" and i != ":")]

with open('vocab.txt','w') as f:
	f.write(','.join(vocab))

vocabSize = len(vocab)

#generate a vector for each abstract 
with open('dataset/train_first_three_no_header.csv') as train_file:
	reader = csv.reader(train_file, delimiter=',')
	for row in reader:
		vector = [0] * vocabSize
		tokens = nltk.word_tokenize(f'{row[2]}')
		i = 0
		while i < vocabSize:
			vector[i] = tokens.count(vocab[i])
			i = i+1
		print(vector)

#print(vocab)
#print(vocabSize)
