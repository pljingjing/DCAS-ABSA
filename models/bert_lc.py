import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F
from torch.distributions.normal import Normal




class BERT_LC_ASPECT2(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_LC_ASPECT2, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim * 2, opt.polarities_dim)
        self.aspect_dense = nn.Linear(opt.bert_dim, opt.bert_dim)


    def forward(self, inputs, pos_inputs=None, neg_inputs=None):
        if pos_inputs == None and neg_inputs == None:
            text_bert_indices, bert_segments_ids,text_attention_mask = inputs
            text_embed, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids,attention_mask=text_attention_mask,
                                                  output_all_encoded_layers=False)

            aspect_embedding = torch.sum(text_embed * bert_segments_ids.unsqueeze(-1), dim=1)
            aspect_embedding = self.aspect_dense(aspect_embedding)

            pooled_output = self.dropout(pooled_output)
            aspect_embedding = self.dropout(aspect_embedding)

            all_pooled_output = torch.cat([pooled_output, aspect_embedding], dim=-1)

            logits = self.dense(all_pooled_output)
            return logits, pooled_output

        text_bert_indices, bert_segments_ids, text_attention_mask = inputs
        pos_text_bert_indices, pos_bert_segments_ids, pos_attention_mask = pos_inputs
        neg_text_bert_indices, neg_bert_segments_ids, neg_attention_mask = neg_inputs

        example_num = neg_inputs[0].shape[1]

        text_embed, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids,
                                              output_all_encoded_layers=False)

        pos_text_embed, pos_pooled_output = self.bert(pos_text_bert_indices, token_type_ids=pos_bert_segments_ids,
                                                      output_all_encoded_layers=False)

        neg_text_embed, neg_pooled_output = self.bert(
            neg_text_bert_indices.view(-1, neg_text_bert_indices.shape[-1]),
            token_type_ids=neg_bert_segments_ids.view(-1,neg_bert_segments_ids.shape[-1]),
            output_all_encoded_layers=False)

        pooled_output = self.dropout(pooled_output)
        pos_pooled_output = self.dropout(pos_pooled_output)
        neg_pooled_output = self.dropout(neg_pooled_output)

        aspect_embedding = torch.sum(text_embed * bert_segments_ids.unsqueeze(-1), dim=1)
        pos_aspect_embedding = torch.sum(pos_text_embed * pos_bert_segments_ids.unsqueeze(-1), dim=1)
        neg_aspect_embedding = torch.sum(
            neg_text_embed * neg_bert_segments_ids.view(-1, neg_bert_segments_ids.shape[-1]).unsqueeze(-1), dim=1)

        aspect_embedding = self.aspect_dense(aspect_embedding)
        pos_aspect_embedding = self.aspect_dense(pos_aspect_embedding)
        neg_aspect_embedding = self.aspect_dense(neg_aspect_embedding)

        aspect_embedding = self.dropout(aspect_embedding)
        pos_aspect_embedding = self.dropout(pos_aspect_embedding)
        neg_aspect_embedding = self.dropout(neg_aspect_embedding)

        all_pooled_output = torch.cat([pooled_output, aspect_embedding], dim=-1)
        logits = self.dense(all_pooled_output)
        return logits, pooled_output, pos_pooled_output, neg_pooled_output.view(text_embed.shape[0], example_num,
                                                                                768), aspect_embedding, pos_aspect_embedding, neg_aspect_embedding.view(
            text_embed.shape[0], example_num, 768)






