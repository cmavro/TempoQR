import math
import torch
from torch import nn
import numpy as np
from tcomplex import TComplEx
from transformers import RobertaModel
from transformers import BertModel
from transformers import DistilBertModel


import pdb
# training data: questions
# model:
# 1. tkbc model embeddings (may or may not be frozen)
# 2. question sentence embeddings (may or may not be frozen)
# 3. linear layer to project question embeddings (unfrozen)
# 4. transformer that takes these embeddings (unfrozen) (cats them along a dimension, also takes a mask)
# 5. average output embeddings of transformer or take last token embedding?
# 6. linear projection of this embedding to tkbc embedding dimension
# 7. score with all possible entities/times and sigmoid
# 8. BCE loss (multiple correct possible)


class QA_baseline(nn.Module):
	def __init__(self, tkbc_model, args):
		super().__init__()
		self.tkbc_embedding_dim = tkbc_model.embeddings[0].weight.shape[1]
		self.sentence_embedding_dim = 768 # hardwired from roberta
        
		if args.model =='bert':
			self.pretrained_weights = 'bert-base-uncased'
			self.lm_model = BertModel.from_pretrained(self.pretrained_weights)
		elif args.model =='roberta':
			self.pretrained_weights = 'roberta-base'
			self.lm_model = RobertaModel.from_pretrained(self.pretrained_weights)
		else:
			self.pretrained_weights = 'distilbert-base-uncased'
			self.lm_model = DistilBertModel.from_pretrained(self.pretrained_weights)

		if args.lm_frozen == 1:
			print('Freezing LM params')
			for param in self.lm_model.parameters():
				param.requires_grad = False
		else:
			print('Unfrozen LM params')

		# creating combined embedding of time and entities (entities come first)
		self.tkbc_model = tkbc_model
		num_entities = tkbc_model.embeddings[0].weight.shape[0]
		num_times = tkbc_model.embeddings[2].weight.shape[0]
		ent_emb_matrix = tkbc_model.embeddings[0].weight.data
		time_emb_matrix = tkbc_model.embeddings[2].weight.data
		full_embed_matrix = torch.cat([ent_emb_matrix, time_emb_matrix], dim=0)
		self.entity_time_embedding = nn.Embedding(num_entities + num_times, self.tkbc_embedding_dim)
		self.entity_time_embedding.weight.data.copy_(full_embed_matrix)
		self.num_entities = num_entities
		self.num_times = num_times

		if args.frozen == 1:
				print('Freezing entity/time embeddings')
				self.entity_time_embedding.weight.requires_grad = False
				for param in self.tkbc_model.parameters():
					param.requires_grad = False
		else:
				print('Unfrozen entity/time embeddings')
		# print('Random starting embedding')
		self.linear = nn.Linear(768, self.tkbc_embedding_dim) # to project question embedding

		self.linear1 = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)
		self.linear2 = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)

		self.loss = nn.CrossEntropyLoss(reduction='mean')

		self.dropout = torch.nn.Dropout(0.3)
		self.bn1 = torch.nn.BatchNorm1d(self.tkbc_embedding_dim)
		self.bn2 = torch.nn.BatchNorm1d(self.tkbc_embedding_dim)
		return

	def getQuestionEmbedding(self, question_tokenized, attention_mask):
		lm_last_hidden_states = self.lm_model(question_tokenized, attention_mask=attention_mask)[0]
		states = lm_last_hidden_states.transpose(1,0)
		cls_embedding = states[0]
		question_embedding = cls_embedding
		# question_embedding = torch.mean(roberta_last_hidden_states, dim=1)
		return question_embedding

    
        
        
        
class QA_lm(QA_baseline):
	def __init__(self, tkbc_model, args):
		super().__init__(tkbc_model, args)
		self.final_linear = nn.Linear(4*self.tkbc_embedding_dim, self.tkbc_embedding_dim)
		return

    
	def forward(self, a):
		question_tokenized = a[0].cuda()
		question_attention_mask = a[1].cuda()
		heads = a[2].cuda()
		tails = a[3].cuda()
		times = a[4].cuda()

		head_embedding = self.entity_time_embedding(heads)
		tail_embedding = self.entity_time_embedding(tails)
		time_embedding = self.entity_time_embedding(times)
		question_embedding = self.getQuestionEmbedding(question_tokenized, question_attention_mask)
		relation_embedding = self.linear(question_embedding)

		output = self.final_linear(torch.cat((head_embedding, relation_embedding, tail_embedding, time_embedding), dim=-1))
		scores = torch.matmul(output, self.entity_time_embedding.weight.data.T)
        
		return scores
        
class QA_embedkgqa(QA_baseline):
	def __init__(self, tkbc_model, args):
		super().__init__(tkbc_model, args)
		return

	def score(self, head_embedding, relation_embedding):
		lhs = head_embedding
		rel = relation_embedding
		right = torch.cat((self.entity_embedding.weight, self.time_embedding.weight), dim=0)
		lhs = lhs[:, :self.rank], lhs[:, self.rank:]
		rel = rel[:, :self.rank], rel[:, self.rank:]
		right = right[:, :self.rank], right[:, self.rank:]
		return (lhs[0] * rel[0] - lhs[1] * rel[1]) @ right[0].transpose(0, 1) + (lhs[0] * rel[1] + lhs[1] * rel[0]) @ right[1].transpose(0, 1)
        
	def forward(self, a):
		question_tokenized = a[0].cuda()
		question_attention_mask = a[1].cuda()
		heads = a[2].cuda()

		head_embedding = self.entity_embedding(heads)
		question_embedding = self.getQuestionEmbedding(question_tokenized, question_attention_mask)
		relation_embedding = self.linear(question_embedding)
		relation_embedding1 = self.dropout(self.bn1(self.linear1(relation_embedding)))
		scores = self.score(head_embedding, relation_embedding1)
		# exit(0)
		# scores = torch.cat((scores_entity, scores_time), dim=1)
		return scores
        
        
class QA_cronkgqa(QA_baseline):
	def __init__(self, tkbc_model, args):
		super().__init__(tkbc_model, args)
		self.supervision = args.supervision
		return


	def infer_time(self, head_embedding, tail_embedding, relation_embedding):
		lhs = head_embedding
		rhs = tail_embedding
		rel = relation_embedding

		time = self.tkbc_model.embeddings[2].weight  # + self.tkbc_model.lin2(self.tkbc_model.time_embedding.weight)
		# time = self.entity_time_embedding.weight

		lhs = lhs[:, :self.tkbc_model.rank], lhs[:, self.tkbc_model.rank:]
		rel = rel[:, :self.tkbc_model.rank], rel[:, self.tkbc_model.rank:]
		rhs = rhs[:, :self.tkbc_model.rank], rhs[:, self.tkbc_model.rank:]
		time = time[:, :self.tkbc_model.rank], time[:, self.tkbc_model.rank:]

		return torch.cat([
					(lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
					lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]),
					(lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
					lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1])], dim=-1
		)
		# scoring function from TComplEx
	def score_time(self, head_embedding, tail_embedding, relation_embedding):
		lhs = head_embedding
		rhs = tail_embedding
		rel = relation_embedding

		time = self.tkbc_model.embeddings[2].weight
		# time = self.entity_time_embedding.weight

		lhs = lhs[:, :self.tkbc_model.rank], lhs[:, self.tkbc_model.rank:]
		rel = rel[:, :self.tkbc_model.rank], rel[:, self.tkbc_model.rank:]
		rhs = rhs[:, :self.tkbc_model.rank], rhs[:, self.tkbc_model.rank:]
		time = time[:, :self.tkbc_model.rank], time[:, self.tkbc_model.rank:]

		return (
					(lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
					lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
					(lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
					lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
		)

	def score_entity(self, head_embedding, tail_embedding,relation_embedding, time_embedding):
		lhs = head_embedding[:, :self.tkbc_model.rank], head_embedding[:, self.tkbc_model.rank:]
		rel = relation_embedding
		time = time_embedding

		rel = rel[:, :self.tkbc_model.rank], rel[:, self.tkbc_model.rank:]
		time = time[:, :self.tkbc_model.rank], time[:, self.tkbc_model.rank:]

		right = self.tkbc_model.embeddings[0].weight
		# right = self.entity_time_embedding.weight
		right = right[:, :self.tkbc_model.rank], right[:, self.tkbc_model.rank:]

		rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
		full_rel = rt[0] - rt[3], rt[1] + rt[2]

		return (
							(lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
							(lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
					)

	def forward(self, a):
		question_tokenized = a[0].cuda()
		question_attention_mask = a[1].cuda()
		heads = a[2].cuda()
		tails = a[3].cuda()
		times = a[4].cuda()

		head_embedding = self.entity_time_embedding(heads)
		tail_embedding = self.entity_time_embedding(tails)
		time_embedding = self.entity_time_embedding(times)
		question_embedding = self.getQuestionEmbedding(question_tokenized, question_attention_mask)
		relation_embedding = self.linear(question_embedding)
		if self.supervision == 'soft':
			t1_emb = self.infer_time(head_embedding, tail_embedding, relation_embedding)
			t2_emb = self.infer_time(tail_embedding, head_embedding, relation_embedding)
			time_embedding = (time_embedding + t1_emb + t2_emb)/3 #just the mean
		relation_embedding1 = self.dropout(self.bn1(self.linear1(relation_embedding)))
		relation_embedding2 = self.dropout(self.bn2(self.linear2(relation_embedding)))
		scores_time = self.score_time(head_embedding, tail_embedding, relation_embedding1)
		scores_entity = self.score_entity(head_embedding, tail_embedding,relation_embedding2, time_embedding)

		scores = torch.cat((scores_entity, scores_time), dim=1)
		return scores

