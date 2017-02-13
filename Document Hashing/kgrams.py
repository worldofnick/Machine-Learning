from nltk import ngrams

def word_kgram(text, k):
	grams = ngrams(text.split(), k)
	gram_set = set()
	for gram in grams:
		gram_set.add(' '.join(gram))
	return gram_set

def character_kgram(text, k):
	grams = ngrams(list(text), k)
	gram_set = set()
	for gram in grams:
		gram_set.add(''.join(gram))
	return gram_set

def jaccard_similarity(a, b):
	denom = a.union(b)
	if len(denom) == 0:
		return 0
	return len(a.intersection(b)) / float(len(denom))

# Q1 (a)
for i in range(1, 5):
	path = '/Users/nickporter/Library/Mobile Documents/com~apple~CloudDocs/Spring 2017/CS 5140/CS-5140/hw2/D%s.txt' % i
	data = open(path, 'r').read()
	data_char_2 = character_kgram(data, 2)
	data_char_3 = character_kgram(data, 3)
	data_word_2 = word_kgram(data, 2) 
	print 'D%s.txt' % i
	print 'Character 2-gram %s' % len(data_char_2)
	print 'Character 3-gram %s' % len(data_char_3)
	print 'Word 2-gram %s' % len(data_word_2)

	# (b)
	for j in range(1, 5):
		path = '/Users/nickporter/Library/Mobile Documents/com~apple~CloudDocs/Spring 2017/CS 5140/CS-5140/hw2/D%s.txt' % j
		data = open(path, 'r').read()
		data1_char_2 = character_kgram(data, 2)
		data1_char_3 = character_kgram(data, 3)
		data1_word_2 = word_kgram(data, 2) 
		
		x = min(i, j)
		y = max(i, j)

		#print 'Char 2-gram JS(D%s, D%s) = %.2f' % (x, y, jaccard_similarity(data_char_2, data1_char_2))
		#print 'Char 3-gram JS(D%s, D%s) = %.2f' % (x, y, jaccard_similarity(data_char_3, data1_char_3))
		print 'Word 2-gram JS(D%s, D%s) = %.2f' % (x, y, jaccard_similarity(data_word_2, data1_word_2))

	print '-----------------------------------------------------------------------------'


