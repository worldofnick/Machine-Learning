

def misra_gries(stream, k):
	C = [0]*k
	L = [None]*k

	for item in stream:
		
		if item in L:
			C[L.index(item)] += 1
		elif 0 in C:
			j = C.index(0)
			L[j] = item
			C[j] = 1
		else:
			C = [x - 1 for x in C]

	return C, L


s1_text = open("/Users/nickporter/Library/Mobile Documents/com~apple~CloudDocs/Spring 2017/CS 5140/CS-5140/hw4/S2.txt", "r")
lines = s1_text.read()
s1 = list(lines)
C, L = misra_gries(s1, 9)

print C
print L