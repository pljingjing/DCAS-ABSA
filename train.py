import logging
import argparse
import math
import os
import sys
import random
import numpy

from criterion_new import CL_sentiment, LC_InfoNCE
from sklearn import metrics
from time import strftime, localtime
from pytorch_pretrained_bert import BertModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils_new import Tokenizer4Bert, ABSADataset, ABSALLMDataset
from models.bert_lc import BERT_LC_ASPECT2

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        if 'bert' in opt.model_name:
            print("Loading BERT")
            tokenizer = Tokenizer4Bert(opt.max_seq_len,
                                       opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(
                "./" + opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(
                opt.device)
            print("End for loading BERT")

        self.trainset = ABSALLMDataset(opt.dataset_file['train'], opt.aug_files, tokenizer, opt.example_num)
        print("Train size:", len(self.trainset))
        self.testset = ABSADataset(opt.dataset_file['test'],
                                   tokenizer)
        print("Test size:", len(self.testset))

        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info(
            '> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, contrastiveloss_InfoNCE, optimizer, train_data_loader,
               val_data_loader, alpha, beta):
        max_val_acc = 0
        max_val_acc_f1 = 0

        max_val_f1 = 0
        max_val_f1_acc = 0

        max_val_epoch = 0
        global_step = 0
        path = None

        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0  # 总损失值
            self.model.train()

            for i_batch, batch in enumerate(
                    train_data_loader):
                global_step += 1

                optimizer.zero_grad()
                inputs_text = [batch['concat_bert_indices'].to(self.opt.device),
                               batch['concat_segments_indices'].to(self.opt.device),
                               batch["concat_attention_mask"].to(self.opt.device)]
                inputs_pos_augment = [batch['pos_sentence_indices_list'].to(self.opt.device),
                                      batch['pos_sentence_seg_list'].to(self.opt.device),
                                      batch["pos_attention_mask"].to(self.opt.device)]
                inputs_neg_augment = [batch['neg_sentences_indices_list'].to(self.opt.device),
                                      batch['neg_text_segments_indices'].to(self.opt.device),
                                      batch["neg_attention_mask_list"].to(self.opt.device)]

                logits, pooled_output, pos_pooled_output, neg_pooled_output, aspect_embedding, pos_aspect_embedding, neg_aspect_embedding = self.model(
                    inputs_text,
                    inputs_pos_augment,
                    inputs_neg_augment)

                targets = batch['polarity'].to(self.opt.device)

                loss1 = criterion(logits, targets)
                loss2 = contrastiveloss_InfoNCE(pooled_output, pos_pooled_output, neg_pooled_output)

                loss3 = contrastiveloss_InfoNCE(aspect_embedding, pos_aspect_embedding, neg_aspect_embedding)

                loss = loss1 + alpha * loss2 + beta * loss3

                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(logits, -1) == targets).sum().item()
                n_total += len(logits)
                loss_total += loss.item() * len(logits)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total

                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_acc_f1 = val_f1
                max_val_epoch = i_epoch
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_type_{2}_acc_{3}'.format(self.opt.model_name, self.opt.dataset,
                                                                        self.opt.type, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))

            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
                max_val_f1_acc = val_acc

            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break
        return path, max_val_acc, max_val_acc_f1, max_val_f1, max_val_f1_acc

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)[0]
                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)
                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        return acc, f1

    def _evaluate_acc_f1_Test(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        self.model.eval()
        List_label = []
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)
                n_correct += (torch.argmax(t_outputs[0], -1) == t_targets).sum().item()
                n_total += len(t_outputs[0])

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs[0]
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs[0]), dim=0)

                for i in range(len(t_outputs[0])):
                    label = torch.argmax(t_outputs[0][i], -1).cpu().data.numpy()
                    hiddenState = t_outputs[1][i].cpu().data.numpy().tolist()
                    Str = str(label) + "\t" + str(hiddenState) + "\n"
                    List_label.append(Str)
        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')

        return acc, f1

    def run_Test(self):
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        self.model.load_state_dict(torch.load('./state_dict/' + self.opt.testfname))
        test_acc, test_f1 = self._evaluate_acc_f1_Test(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))

    def run(self):
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        criterion = nn.CrossEntropyLoss()
        contrastiveloss_InfoNCE = LC_InfoNCE(self.opt)
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_path, max_val_acc, max_val_acc_f1, max_val_f1, max_val_f1_acc = self._train(
            criterion, contrastiveloss_InfoNCE, optimizer, train_data_loader,
            val_data_loader, self.opt.alpha, self.opt.beta)

        max_val_acc = round(max_val_acc, 4)
        max_val_acc_f1 = round(max_val_acc_f1, 4)
        max_val_f1_acc = round(max_val_f1_acc, 4)
        max_val_f1 = round(max_val_f1, 4)

        print(self.opt.model_name, self.opt.dataset_file['train'], "Test accMAX：", max_val_acc, max_val_acc_f1)
        print(self.opt.model_name, self.opt.dataset_file['train'], "Test f1MAX：", max_val_f1_acc, max_val_f1)
        print(str(self.opt.seed) + "_" + str(self.opt.lr))

        self.model.load_state_dict(torch.load(best_model_path))
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
        return test_acc, test_f1


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_lc', type=str)
    parser.add_argument('--dataset', default='cl_aug_bt_laptop14', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=50, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=128, type=int, help='try 16, 32, 64, 128 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='./bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--is_test', default=0, type=int)
    parser.add_argument('--type', default="normal", type=str)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=126, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float,
                        help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--SRD', default=3, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')
    parser.add_argument('--testfname', default=0, type=str)
    parser.add_argument('--temperatureP', default=0.03, type=float)
    parser.add_argument('--temperatureY', default=0.1, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--beta', default=0.4, type=float)
    parser.add_argument('--gama', default=0.4, type=float)
    parser.add_argument('--example_num', default=5, type=int)

    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'bert_lc': BERT_LC_ASPECT2
    }
    dataset_files = {
        "cl_aug_bt_acl2014": {
            'train': './datasets/Aug_Data/Aug_acl14/twitter14_bt_aug.txt',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        "cl_aug_bt_res2014": {
            'train': './datasets/Aug_Data/Aug_res14/res14_bt_aug.txt',
            'test': './datasets/semeval14/restaurant_test.raw'
        },
        "cl_aug_bt_laptop14": {
            'train': './datasets/Aug_Data/Aug_laptop14/laptop14_bt_aug.txt',
            'test': './datasets/semeval14/laptop_test.raw'
        },
        "cl_aug_mams": {
            'train': './datasets/Aug_Data/Aug_mams/train_aug.raw',
            'test': './datasets/Aug_Data/Aug_mams/test.raw'
        },

    }
    aug_files = {
        "cl_aug_bt_laptop14": './datasets/chatgpt_data/laptop2/laptop_train_llm.json',
        "cl_aug_bt_res2014": './datasets/chatgpt_data//restaurant//restaurant_train_generate_predict_llm.json',
        "cl_aug_bt_acl2014": './datasets/chatgpt_data//twitter//twitter_train_generate_predict_llm.json',
        "cl_aug_mams": './datasets/chatgpt_data//MAMS/train_generate_predict_llm.json',
    }

    input_colses = {
        'bert_lc': ['concat_bert_indices', 'concat_segments_indices', "attention_mask"],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax,
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.aug_files = aug_files[opt.dataset]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    acc_f1_list = []

    if opt.is_test == 0:
        if not os.path.exists('log'):
            os.mkdir('log')

        log_file = './log/{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
        logger.addHandler(logging.FileHandler(log_file))

        ins = Instructor(opt)
        acc, f1 = ins.run()
        acc_f1_list.append([acc, f1])
    else:
        print("Model Testing-----")
        ins = Instructor(opt)
        ins.run_Test()

    print(acc_f1_list)


if __name__ == '__main__':
    main()
