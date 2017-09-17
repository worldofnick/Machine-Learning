# 1. Is their first name longer than their last name?
def firstLongerThanLast(first, last):
	return len(first) > len(last)

# 2. Do they have a middle name?

# 3. Does their first name start and end with the same letter? (ie "Ada")
def endsWithFirstLetter(first):
	first = first.lower()
	return first[0] == first[-1]

# 4. Does their first name come alphabetically before their last name? (ie "Dan Klein" because "d" comes before "k")
def firstBeforeLast(first, last):
	first = first.lower()
	last = last.lower()
	return first > last

# 5. Is the second letter of their first name a vowel (a,e,i,o,u)?
def secondLetterIsVowel(first):
	first = first.lower()
	if len(first) > 1:
		return first[1] in 'aeiou'
	return False

# 6. Is the number of letters in their last name even?
def lastNameEvenLength(last):
	return len(last) % 2 == 0

# 7. Is the first letter in the last name a vowel?
def firstLetterIsVowel(last):
	first = last.lower()
	if len(last) > 1:
		return first[0] in 'aeiou'
	return False

# 8. Is the first name only an initial?
def firstIsOnlyInitial(first):
	first.replace(".", "")
	return len(first) == 1

# 9. Is the first letter of the middle name a vowel?
def firstLetterIsVowelMiddle(middle):
	first = middle.lower()
	if len(middle) > 1:
		return middle[0] in 'aeiou'
	return False

# 10. Is the entire length of the full name even?
def lengthEven(full_name):
	return len(full_name) % 2 == 0

# 11. Does their last name start and end with the same letter? (ie "Ada")
def endsWithFirstLetterLast(last):
	last = last.lower()
	return last[0] == last[-1]

# 12. Beginning of first name and last name match
def matchFirstAndLast(first, last):
	return first[0] == last[-1]

# 13. The first and last name both end with a vowel
def firstAndLastEndWithVowel(first, last):
	return first[-1] in 'aeiou' and last[-1] in 'aeiou'

# 14. Last name over 8 characters
def longLastName(last):
	return len(last) > 8

# 15. First name under 5 characters
def shortFirstName(first):
	return 5 > len(first)

# 16. Both the first name and the middle name are only an initial
def firstAndMiddleAreInitial(first, middle):
	return len(first) <= 2 and len(middle) <= 2

# 17. First and last have same length
def firstAndLastLength(first, last):
	return len(first) == len(last)

# 18. Last is longer than both first and middle cobined
def lastNameWins(first, middle, last):
	return len(last) > len(first) + len(middle)

# 19.
def shortFullName(full_name):
	return len(full_name) <= 8

# Given a string it returns an array of 1's or 0's for features
def featureize(full_name):
	tokens = full_name.split()
	has_middle_name = len(tokens) == 3

	first_name = tokens[0]
	if has_middle_name:
		middle_name = tokens[1]
		last_name = tokens[-1]
	else:
		last_name = tokens[1]
		middle_name = ''

	features = [firstLongerThanLast(first_name, last_name), has_middle_name, endsWithFirstLetter(first_name),
				firstBeforeLast(first_name ,last_name), secondLetterIsVowel(first_name), lastNameEvenLength(last_name),
				firstLetterIsVowel(last_name), firstIsOnlyInitial(first_name), firstLetterIsVowelMiddle(middle_name),
				lengthEven(full_name), endsWithFirstLetterLast(last_name), matchFirstAndLast(first_name, last_name),
				firstAndLastEndWithVowel(first_name, last_name), longLastName(last_name), shortFirstName(first_name),
				firstAndMiddleAreInitial(first_name, middle_name), firstAndLastLength(first_name, last_name),
				lastNameWins(first_name, middle_name, last_name), shortFullName(full_name)]
	features = map(lambda x: 1 if x else 0, features)
	return features