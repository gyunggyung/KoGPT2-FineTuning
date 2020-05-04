from torch.utils.data import Dataset
from kogpt2.utils import download, tokenizer, get_tokenizer
from gluonnlp.data import SentencepieceTokenizer
import gluonnlp
import numpy as np


def sentencePieceTokenizer():
	tok_path = get_tokenizer()
	sentencepieceTokenizer = SentencepieceTokenizer(tok_path)
	return sentencepieceTokenizer


def koGPT2Vocab():
	cachedir = '~/kogpt2/'

	# download vocab
	vocab_info = tokenizer
	vocab_path = download(vocab_info['url'],
						vocab_info['fname'],
						vocab_info['chksum'],
						cachedir=cachedir)

	koGPT2_vocab = gluonnlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
															 mask_token=None,
															 sep_token=None,
															 cls_token=None,
															 unknown_token='<unk>',
															 padding_token='<pad>',
															 bos_token='<s>',
															 eos_token='</s>')
	return koGPT2_vocab

def toString(list):
	if not list:
		return ''
	result = ''

	for i in list:
		result = result + i
	return result

class Read_Dataset(Dataset):
	"""web novel dataset"""

	def __init__(self, file_path,vocab,tokenizer):
		self.file_path = file_path
		self.data =[]
		self.vocab =vocab
		self.tokenizer = tokenizer
		file = open(self.file_path, 'r', encoding='utf-8')

		lines = file.read()
		lines = lines.split("<|endoftext|>")

		datasets = []

		for i, line in enumerate(lines):
			line = tokenizer(line)
			while (1):
				if len(line) > 1020:
					datasets.append(line[:1020])
					line = line[:1020]
				else:
					datasets.append(line)
					break

		#now = ""
		#for i, line in enumerate(lines):
		# 	if i % 20 == 0 and i != 0:
		# 		datasets.append(now)
		# 		now = ""
		# 	now = now + "\n" + line

		for line in datasets:
			if not line:
				break
			if len(line) < 3:
				continue

			toeknized_line = tokenizer(line[:-1])

			### 여기부터는 그대로
			index_of_words = [vocab[vocab.bos_token],] + vocab[toeknized_line]+ [vocab[vocab.eos_token]]
			self.data.append(index_of_words)

		print(np.shape(self.data))

		file.close()

	def __len__(self):
		return len(self.data)

	def __getitem__(self,index):
		item = self.data[index]
		return item