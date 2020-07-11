from gluonnlp.data import SentencepieceTokenizer
import gluonnlp
import pandas as pd
from kogpt2.utils import download, tokenizer, get_tokenizer

pytorch_kogpt2 = {
	'url':
	'checkpoint/pytorch_kogpt2_676e9bcfa7.params',
	'fname': 'pytorch_kogpt2_676e9bcfa7.params',
	'chksum': '676e9bcfa7'
}

kogpt2_config = {
	"initializer_range": 0.02,
	"layer_norm_epsilon": 1e-05,
	"n_ctx": 1024,
	"n_embd": 768,
	"n_head": 12,
	"n_layer": 12,
	"n_positions": 1024,
	"vocab_size": 50000
}

def auto_enter(text):
	text = (text.replace("   ", "\n"))
	text = text.split("\n")

	text = [t.lstrip() for t in text if t != '']
	return "\n\n".join(text)

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

def tt(st):
	try:
		len(st)
	except:
		return 0
	print(vocab[tokenizer(st)])
	return len(vocab[tokenizer(st)])

if __name__ == "__main__":
	cachedir = '~/kogpt2/'

	vocab_info = tokenizer
	vocab_path = download(vocab_info['url'],
						  vocab_info['fname'],
						  vocab_info['chksum'],
						  cachedir=cachedir)

	vocab_b_obj = gluonnlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
															  mask_token=None,
															  sep_token=None,
															  cls_token=None,
															  unknown_token='<unk>',
															  padding_token='<pad>',
															  bos_token='<s>',
															  eos_token='</s>')

	vocab = vocab_b_obj

	df = pd.read_csv("dataset/read_lyrics_dataset.csv")
	df["lens"] = df["lyrics"].map(tt)
	df.to_csv("use_df.csv", index=False)

