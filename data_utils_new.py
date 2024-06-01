import json
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class ABSAAugDataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []

        for i in range(0, len(lines), 4):
            text1_left, _, text1_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            polabel = lines[i + 2].strip()
            text2_left, _, text2_right = [s2.lower().strip() for s2 in lines[i + 3].partition("$T$")]

            text1_indices = tokenizer.text_to_sequence(text1_left + " " + aspect + " " + text1_right)
            text2_indices = tokenizer.text_to_sequence(text2_left + " " + aspect + " " + text2_right)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            aspect_len = np.sum(aspect_indices != 0)
            polarity = int(polarity) + 1
            polabel = int(polabel) + 1

            text1_len = np.sum(text1_indices != 0)
            text2_len = np.sum(text2_indices != 0)
            concat_text1_bert_indices = tokenizer.text_to_sequence(
                '[CLS] ' + text1_left + " " + aspect + " " + text1_right + ' [SEP] ' + aspect + " [SEP]")
            concat_text1_segments_indices = [0] * (text1_len + 2) + [1] * (aspect_len + 1)
            concat_text1_segments_indices = pad_and_truncate(concat_text1_segments_indices, tokenizer.max_seq_len)

            concat_text2_bert_indices = tokenizer.text_to_sequence(
                '[CLS] ' + text2_left + " " + aspect + " " + text2_right + ' [SEP] ' + aspect + " [SEP]")
            concat_text2_segments_indices = [0] * (text2_len + 2) + [1] * (aspect_len + 1)
            concat_text2_segments_indices = pad_and_truncate(concat_text2_segments_indices, tokenizer.max_seq_len)

            data = {
                'concat_bert_indices': concat_text1_bert_indices,
                'concat_segments_indices': concat_text1_segments_indices,
                'concat_text2_bert_indices': concat_text2_bert_indices,
                'concat_text2_segments_indices': concat_text2_segments_indices,
                'text1_indices': text1_indices,
                'text2_indices': text2_indices,
                'aspect_indices': aspect_indices,
                'polarity': polarity,
                'text1': lines[i],
                'text2': lines[i + 3],
                'aspect': aspect,
                'polabel': polabel,
            }
            all_data.append(data)

        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADataset(Dataset):

    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')

        lines = fin.readlines()

        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            polabel = lines[i + 2].strip()

            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            aspect_len = np.sum(aspect_indices != 0)
            polarity = int(polarity) + 1
            polabel = int(polabel) + 1

            text_len = np.sum(text_indices != 0)
            concat_bert_indices = tokenizer.text_to_sequence(
                '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
            concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)
            attention_mask = [1] * (text_len + 2 + aspect_len + 1)
            attention_mask = pad_and_truncate(attention_mask, tokenizer.max_seq_len)

            data = {
                'concat_bert_indices': concat_bert_indices,
                'concat_segments_indices': concat_segments_indices,
                'attention_mask': attention_mask,
                'text_indices': text_indices,
                'aspect_indices': aspect_indices,
                'polarity': polarity,
                'text': lines[i],
                'aspect': aspect,
                'polabel': polabel,
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSALLMDataset(Dataset):
    def __init__(self, fname, augname, tokenizer, example_num=5):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        augfin = open(augname, 'r', encoding='utf-8')
        lines = fin.readlines()
        aug_json = augfin.readlines()
        fin.close()
        augfin.close()

        all_data = []
        polarity_id_dict = {"negative": -1, "positive": 1, "neutral": 0}

        for i in range(0, len(lines), 4):
            aug_label = []
            text1_left, _, text1_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()  # 方面
            polarity = lines[i + 2].strip()  # 极性
            polabel = lines[i + 2].strip()  # 极性

            aug_dict = json.loads(aug_json[i // 4])
            aug_pos_sentence = [s.lower().strip() for s in aug_dict["generate_positive"]]
            aug_neg_sentence = [s.lower().strip() for s in aug_dict["generate_negative"][:example_num]]
            aug_pos_aspect = [a.lower().strip() for a in aug_dict["positive_aspect"]]
            aug_neg_aspect = [a.lower().strip() for a in aug_dict["negative_aspect"][:example_num]]

            aug_label.append(polarity_id_dict[aug_dict["positive_polarity"][0]] + 1)
            for label in aug_dict["negative_polarity"][:example_num]:
                aug_label.append(polarity_id_dict[label] + 1)
            pos_label = [aug_label[0]]
            neg_label = aug_label[1:]


            if len(aug_neg_sentence) != example_num:
                print(len(aug_neg_sentence))
                assert 0 == 1

            text1_indices = tokenizer.text_to_sequence(text1_left + " " + aspect + " " + text1_right)

            aspect_indices = tokenizer.text_to_sequence(aspect)
            aspect_len = np.sum(aspect_indices != 0)
            polarity = int(polarity) + 1
            polabel = int(polabel) + 1

            text1_len = np.sum(text1_indices != 0)

            concat_text1_bert_indices = tokenizer.text_to_sequence(
                '[CLS] ' + text1_left + " " + aspect + " " + text1_right + ' [SEP] ' + aspect + " [SEP]")
            concat_text1_segments_indices = [0] * (text1_len + 2) + [1] * (aspect_len + 1)
            concat_text1_segments_indices = pad_and_truncate(concat_text1_segments_indices, tokenizer.max_seq_len)
            concat_attention_mask = [1] * (text1_len + 2 + aspect_len + 1)
            concat_attention_mask = pad_and_truncate(concat_attention_mask, tokenizer.max_seq_len)

            pos_sentence_indices_list, neg_sentences_indices_list = [], []
            pos_sentence_seg_list, neg_sentences_seg_list = [], []
            neg_attention_mask_list = []
            for sentence, aspect in zip(aug_pos_sentence, aug_pos_aspect):  # 对比积极句子
                sentence_indices = tokenizer.text_to_sequence(sentence)
                sentence_len = np.sum(sentence_indices != 0)

                aspect_indices = tokenizer.text_to_sequence(aspect)
                aspect_len = np.sum(aspect_indices != 0)

                aug_pos_text_bert_indices = tokenizer.text_to_sequence(
                    '[CLS] ' + sentence + ' [SEP] ' + aspect + " [SEP]")

                pos_text_segments_indices = [0] * (sentence_len + 2) + [1] * (aspect_len + 1)
                pos_text_segments_indices = pad_and_truncate(pos_text_segments_indices, tokenizer.max_seq_len)
                pos_attention_mask = [1] * (sentence_len + 2 + aspect_len + 1)
                pos_attention_mask = pad_and_truncate(pos_attention_mask, tokenizer.max_seq_len)

            for sentence, aspect in zip(aug_neg_sentence, aug_neg_aspect):  # 对比消极句子
                sentence_indices = tokenizer.text_to_sequence(sentence)
                sentence_len = np.sum(sentence_indices != 0)

                aspect_indices = tokenizer.text_to_sequence(aspect)
                aspect_len = np.sum(aspect_indices != 0)

                aug_neg_text_bert_indices = tokenizer.text_to_sequence(
                    '[CLS] ' + sentence + ' [SEP] ' + aspect + " [SEP]")
                neg_sentences_indices_list.append(aug_neg_text_bert_indices)

                neg_text_segments_indices = [0] * (sentence_len + 2) + [1] * (aspect_len + 1)
                neg_text_segments_indices = pad_and_truncate(neg_text_segments_indices, tokenizer.max_seq_len)
                neg_sentences_seg_list.append(neg_text_segments_indices)

                neg_attention_mask = [1] * (sentence_len + 2 + aspect_len + 1)
                neg_attention_mask = pad_and_truncate(neg_attention_mask, tokenizer.max_seq_len)
                neg_attention_mask_list.append(neg_attention_mask)

            neg_sentences_seg_list = np.stack(neg_sentences_seg_list)
            neg_attention_mask_list = np.stack(neg_attention_mask_list)
            neg_sentences_indices_list = np.stack(neg_sentences_indices_list)
            data = {
                'concat_bert_indices': concat_text1_bert_indices,
                'concat_segments_indices': concat_text1_segments_indices,
                'concat_attention_mask':concat_attention_mask,
                'pos_sentence_indices_list': aug_pos_text_bert_indices,
                'pos_sentence_seg_list': pos_text_segments_indices,
                'pos_attention_mask': pos_attention_mask,
                'neg_sentences_indices_list': neg_sentences_indices_list,
                'neg_text_segments_indices': neg_sentences_seg_list,
                'neg_attention_mask_list':neg_attention_mask_list,
                'text1_indices': text1_indices,
                'aspect_indices': aspect_indices,
                'polarity': polarity,
                'text1': lines[i],
                'text2': lines[i + 3],
                'aspect': aspect,
                'polabel': polabel,
                'aug_label': np.array(aug_label),
                'pos_label': np.array(pos_label),  #正样本标签
                'neg_label': np.array(neg_label) #负样本标签
            }
            all_data.append(data)

        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
