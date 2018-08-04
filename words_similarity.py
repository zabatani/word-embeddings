from collections import defaultdict, Counter
from math import log
import sys


class WordsSim:
	def __init__(self, input_filename, min_word_freq, min_feature_freq, min_co_freq, co_type, smooth=1.0):
		self.input_filename = input_filename
		self.min_word_freq = min_word_freq
		self.min_feature_freq = min_feature_freq
		self.min_co_freq = min_co_freq
		self.co_type = co_type  # Possible values: "sentence", "window", "dependency"
		self.smooth = smooth
		self.w2i, self.words_count = self.corpus_initial()
		if self.co_type == "dependency":
			self.feature_matrix = self.build_dependency_matrix()
		else:
			self.feature_matrix = self.build_count_matrix()
		self.PMI_matrix = self.build_PMI_matrix()

	def corpus_initial(self):
		f = open(self.input_filename, "r", encoding="utf8")
		words_ctr = 0
		w2i = {}
		words_count = Counter()     # counts how many times each word appears
		for line in f.readlines():
			fields = line.split("\t")
			if len(fields) < 2:
				continue
			lemma_word = fields[2]
			if lemma_word not in w2i:
				w2i[lemma_word] = words_ctr
				words_ctr += 1
			words_count[w2i[lemma_word]] += 1
		f.close()
		return w2i, words_count

	def build_dependency_matrix(self):
		counts = defaultdict(Counter)
		feature_counter = Counter()
		with open(self.input_filename, "r", encoding="utf8") as f:
			sentence = []
			for line in f:
				if line == "\n":
					self.analyze_sentence_features(sentence, counts, feature_counter)
					sentence = []
					continue
				fields = line.split("\t")
				sentence.append(fields)

		# Filtering
		for counter in counts.values():
			for feature, count in counter.copy().items():
				if count < self.min_co_freq or feature_counter[feature] < self.min_feature_freq:
					counter.pop(feature)

		return counts

	def analyze_sentence_features(self, sentence, counts, feature_count):
		for fields in sentence:
			if fields[6] == 0:  # ROOT edge
				continue

			if fields[3] == "IN":  # Prepositions are being handled separately TODO: what if preposition has no children?
				continue

			target_word = self.w2i[fields[2]]
			head_word_pos = sentence[int(fields[6])-1][3]

			if head_word_pos != "IN":
				head_word = self.w2i[sentence[int(fields[6]) - 1][2]]
				label = fields[7]
			else:
				head_word = self.w2i[sentence[int(sentence[int(fields[6]) - 1][6])-1][2]]
				label = sentence[int(fields[6]) - 1][7] + "_" + sentence[int(fields[6])-1][2]
				print(sentence[int(fields[6])-1][2])

			# Forward Edge Feature Addition
			if self.words_count[target_word] > self.min_word_freq:
				feature = (head_word, 0, label)  # 0 means Parent edge
				counts[target_word][feature] += 1

				feature_count[feature] += 1

			# Backward Edge Feature Addition
			if self.words_count[head_word] > self.min_word_freq:
				feature = (target_word, 1, label)  # 1 means Child edge
				counts[head_word][feature] += 1

				feature_count[feature] += 1

	def build_count_matrix(self):  # this is for the first type - same sentence
		counts_matrix = {}
		for word_num in self.words_count:
			if self.words_count[word_num] > self.min_word_freq:
				counts_matrix[word_num] = Counter()
		f = open(self.input_filename, "r", encoding="utf8")
		sentence = []
		for line in f.readlines():
			if line == "\n":
				for i in range(len(sentence)):
					word = sentence[i]
					word_num = self.w2i[word]
					if word_num in counts_matrix:
						counts_matrix[word_num] = self.update_context_vector(counts_matrix[word_num], sentence, i)
				sentence = []
			else:
				fields = line.split("\t")
				if not self.is_function_word(fields):
					sentence.append(fields[2])
		f.close()
		for word_num in counts_matrix:
			for context_num in list(counts_matrix[word_num]):
				if counts_matrix[word_num][context_num] < self.min_co_freq:
					del counts_matrix[word_num][context_num]
		return counts_matrix

	def update_context_vector(self, word_vector, sentence, index):
		if self.co_type == "sentence":
			context_window = sentence
		elif self.co_type == "window":
			context_window = sentence[max(index-2, 0):min(index+3, len(sentence))]
		for word in context_window:
			if self.words_count[self.w2i[word]] < self.min_feature_freq:
				continue
			word_vector[self.w2i[word]] += 1
		return word_vector

	def build_PMI_matrix(self):
		PMI_mat = {}
		row_sums_counter = Counter()
		column_sums_counter = Counter()
		for word_id in self.feature_matrix:
			row_sums_counter[word_id] = sum(self.feature_matrix[word_id].values())
			for feature in self.feature_matrix[word_id]:
				column_sums_counter[feature] += self.feature_matrix[word_id][feature]
		sum_all_entries = sum(row_sums_counter.values())
		for word_id in self.feature_matrix:
			PMI_mat[word_id] = {}
			p_word = (row_sums_counter[word_id] ** self.smooth) / float(sum(x ** self.smooth for x in row_sums_counter.values()))
			for feature in self.feature_matrix[word_id]:
				p_context = column_sums_counter[feature] / float(sum_all_entries)
				p_word_context = self.feature_matrix[word_id][feature] / float(sum_all_entries)
				PMI_mat[word_id][feature] = log(p_word_context / (p_word * p_context))

		# Normalize matrix
		for w in PMI_mat:
			row_sum = 0
			for c in PMI_mat[w]:
				row_sum += PMI_mat[w][c] ** 2
			row_sum = row_sum ** 0.5
			for c in PMI_mat[w]:
				PMI_mat[w][c] /= row_sum
		return PMI_mat

	def compute_similarity(self, target_word):
		target = self.w2i[target_word]
		sim_dic = defaultdict(float)
		for c in self.PMI_matrix[target]:
			for w in self.PMI_matrix:
				if w == target:
					continue
				if c in self.PMI_matrix[w]:
					sim_dic[w] += self.PMI_matrix[w][c] * self.PMI_matrix[target][c]
		return sim_dic

	def is_function_word(self, fields):
		if fields[3] in ["DT", "IN", "CC", "TO", "PRP", "PRP$", ",", "RB", "WRB", "WP", "WP$", "MD", ".", "(", ")",
						 "''", "POS", "WDT", ":", "``"]:
			return True
		if fields[2] in ["be", "have", "do", "get", "[", "]"]:
			return True
		return False

	def eval(self, words, output_filename):
		output_filename = "{}_type={}_smooth={}".format(output_filename, self.co_type, self.smooth)
		output_file = open(output_filename, "w")
		for word in words:
			output_file.write(word + "\n")
			sim_dic = self.compute_similarity(word)
			sim_dic_sorted = sorted(sim_dic.items(), key=lambda x: x[1], reverse=True)[:20]
			for item in sim_dic_sorted:
				for w in self.w2i:
					if item[0] == self.w2i[w]:
						output_file.write(w + ": " + str(item[1]) + "\n")
			output_file.write("\n")
		output_file.close()


def main(argv):
	# input_filename = argv[1]
	# output_filename = argv[2]
	# min_word_freq = int(argv[3])
	# min_feature_freq = int(argv[4])
	# min_co_freq = int(argv[5])
	# co_type = argv[6]
	# smooth = float(argv[7])
	input_filename = "data/wikipedia.sample.trees.lemmatized"
	output_filename = "output/test"
	min_word_freq = 100
	min_feature_freq = 20
	min_co_freq = 5
	co_type = "dependency"
	smooth = 0.75
	word_sim = WordsSim(input_filename, min_word_freq=min_word_freq, min_feature_freq=min_feature_freq, min_co_freq=min_co_freq, co_type=co_type, smooth=smooth)
	words = ["car", "bus", "hospital", "hotel", "gun", "bomb", "horse", "fox", "table", "bowl", "guitar", "piano"]
	word_sim.eval(words, output_filename)


if __name__ == "__main__":
	main(sys.argv)
