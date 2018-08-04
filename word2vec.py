import numpy as np
import sys


def load_and_normalize_vectors(filename, encoding=None):
	with open(filename, "r", encoding=encoding) as f:
		words = []
		vectors = []
		for line in f:
			line = line.split()
			words.append(line[0])
			vectors.append(line[1:])

		vectors = np.array(vectors).astype(np.float)
		words = np.array(words)
		# Normalize
		#row_den = np.sum(vectors ** 2, axis=1)
		#vectors = vectors / row_sums[:, np.newaxis]

		return vectors, words


def main(argv):
	words_filename = argv[1]
	contexts_filename = argv[2]
	W, words = load_and_normalize_vectors(filename=words_filename, encoding="utf-8")
	C, contexts = load_and_normalize_vectors(filename=contexts_filename, encoding="utf-8")
	# W and words are numpy arrays.
	w2i = {w: i for i, w in enumerate(words)}
	#i2c = {i: c for i, c in enumerate(contexts)}

	words_to_compare = ["car", "bus", "hospital", "hotel", "gun", "bomb", "horse", "fox", "table", "bowl", "guitar", "piano"]
	for w in words_to_compare:
		print("Analyzing word: {}".format(w))
		word_vec = W[w2i[w]]  # get the dog vector

		sims = W.dot(word_vec)  # compute similarities
		most_similar_ids = sims.argsort()[-2:-21:-1]
		sim_words = words[most_similar_ids]
		print("\nMost similar words:")
		print(sim_words)

		print("\nMost similar contexts:")
		sims = C.dot(word_vec)  # compute similarities
		most_similar_ids = sims.argsort()[-2:-11:-1]
		sim_contexts = contexts[most_similar_ids]
		print(sim_contexts)


if __name__ == "__main__":
	main(sys.argv)

