from torch.utils.data import Dataset
from kogpt2.utils import download, tokenizer, get_tokenizer
from gluonnlp.data import SentencepieceTokenizer
import gluonnlp
import numpy as np
import pandas as pd

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

		df = pd.read_csv(self.file_path)

		datasets = []
		for _, row in df.iterrows():
			datasets.append([row["lyrics"], row["genre"], row["score"]])
			
		print("tokenizer ending")
		for line in datasets:
			if not line[0]:
				break
			if len(line[0]) < 3:
				continue
			toeknized_line = tokenizer(line[0][:-1])

			index_of_words = [vocab[vocab.bos_token], ] + vocab[toeknized_line] + [vocab[vocab.eos_token]]

			if len(index_of_words) > 1024:
				continue
			elif len(index_of_words) < 100:
				continue

			#print(len(index_of_words))	
			#print(line)
			self.data.append([index_of_words, line[1], line[2]])

		print(np.shape(self.data))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		item = self.data[index]
		return item