import math
import torch
from torch import nn
import numpy as np
from tcomplex import TComplEx
from transformers import DistilBertModel
from torch.nn import LayerNorm

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

class QA_TempoQR(nn.Module):
	def __init__(self, tkbc_model, args):
		super().__init__()
		self.model = args.model
		self.supervision = args.supervision
		self.extra_entities = args.extra_entities
		self.fuse = args.fuse
		self.tkbc_embedding_dim = tkbc_model.embeddings[0].weight.shape[1]
		self.sentence_embedding_dim = 768  # hardwired from 
        
		self.pretrained_weights = 'distilbert-base-uncased'
		self.lm_model = DistilBertModel.from_pretrained(self.pretrained_weights)
		if args.lm_frozen == 1:
				print('Freezing LM params')
				for param in self.lm_model.parameters():
					param.requires_grad = False
		else:
				print('Unfrozen LM params')

        
		# transformer
		self.transformer_dim = self.tkbc_embedding_dim  # keeping same so no need to project embeddings
		self.nhead = 8
		self.num_layers = 6
		self.transformer_dropout = 0.1
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_dim, nhead=self.nhead,
																		dropout=self.transformer_dropout)
		encoder_norm = LayerNorm(self.transformer_dim)
		self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers,
																			norm=encoder_norm)
		self.project_sentence_to_transformer_dim = nn.Linear(self.sentence_embedding_dim, self.transformer_dim)
		self.project_entity = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)
		
		#TKG embeddings
		self.tkbc_model = tkbc_model
		num_entities = tkbc_model.embeddings[0].weight.shape[0]
		num_times = tkbc_model.embeddings[2].weight.shape[0]
		ent_emb_matrix = tkbc_model.embeddings[0].weight.data
		time_emb_matrix = tkbc_model.embeddings[2].weight.data
        
        
		full_embed_matrix = torch.cat([ent_emb_matrix, time_emb_matrix], dim=0)
		# +1 is for padding idx
		self.entity_time_embedding = nn.Embedding(num_entities + num_times + 1,
																self.tkbc_embedding_dim,
																padding_idx=num_entities + num_times)
		self.entity_time_embedding.weight.data[:-1, :].copy_(full_embed_matrix)
        

		if args.frozen == 1:
				print('Freezing entity/time embeddings')
				self.entity_time_embedding.weight.requires_grad = False
				for param in self.tkbc_model.parameters():
					param.requires_grad = False
		else:
				print('Unfrozen entity/time embeddings')

		# position embedding for transformer
		self.max_seq_length = 100  # randomly defining max length of tokens for question
		self.position_embedding = nn.Embedding(self.max_seq_length, self.tkbc_embedding_dim)
		# print('Random starting embedding')
		self.loss = nn.CrossEntropyLoss(reduction='mean')
		self.layer_norm = nn.LayerNorm(self.transformer_dim)
        

		self.linear = nn.Linear(768, self.tkbc_embedding_dim)  # to project question embedding
		self.linearT = nn.Linear(768, self.tkbc_embedding_dim)  # to project question embedding
		self.lin_cat = nn.Linear(3*self.transformer_dim, self.transformer_dim)

		self.linear1 = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)
		self.linear2 = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)
        
		self.dropout = torch.nn.Dropout(0.3)
		self.bn1 = torch.nn.BatchNorm1d(self.tkbc_embedding_dim)
		self.bn2 = torch.nn.BatchNorm1d(self.tkbc_embedding_dim)
        
		return


         
    
	def invert_binary_tensor(self, tensor):
		ones_tensor = torch.ones(tensor.shape, dtype=torch.float32).cuda()
		inverted = ones_tensor - tensor
		return inverted


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

	def score_entity(self, head_embedding, tail_embedding, relation_embedding, time_embedding):

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
		#Tokenized questions, where entities are masked from the sentence to have TKG embeddings
		question_tokenized = a[0].cuda()
		question_attention_mask = a[1].cuda()
		entities_times_padded = a[2].cuda()
		entity_mask_padded = a[3].cuda()
        
		#Annotated entities/timestamps
		heads = a[4].cuda()
		tails = a[5].cuda()
		times = a[6].cuda()
        
		#t1 and t2 by Hard Supervision
		t1 = a[7].cuda()
		t2 = a[8].cuda()
        
		#One extra entity for new before & after question type
		tails2 = a[9].cuda()
        
		#TKG embeddings
		head_embedding = self.entity_time_embedding(heads)
		tail_embedding = self.entity_time_embedding(tails)
		tail_embedding2 = self.entity_time_embedding(tails2)
		time_embedding = self.entity_time_embedding(times)
        
		#Hard Supervision
		t1_emb = self.tkbc_model.embeddings[2](t1) 
		t2_emb = self.tkbc_model.embeddings[2](t2) 

        
        
		#entity embeddings to replace in sentence
		entity_time_embedding = self.entity_time_embedding(entities_times_padded)
        
		#context-aware step
		outputs = self.lm_model(question_tokenized, attention_mask=question_attention_mask)
		last_hidden_states = outputs.last_hidden_state
        
		if self.supervision == 'soft':
		#infer time embeddings for t1 and t2 with Soft Supervision
			lm_last_hidden_states = outputs[0]
			states = lm_last_hidden_states.transpose(1,0)
			cls_embedding = states[0]
			cls_embedding = self.linearT(cls_embedding)
		    
			t1_emb = self.infer_time(head_embedding, tail_embedding, cls_embedding)
			t2_emb = self.infer_time(tail_embedding, head_embedding, cls_embedding)

			if self.extra_entities == True and not self.training:   #this is for the created before & after questions - we expect one more entity
				t3_emb = self.infer_time(tail_embedding2, head_embedding, cls_embedding)
				t2_emb = t2_emb+t3_emb #for simplicity, can be omitted and treated as t3_emb

		#entity-aware step
		question_embedding = self.project_sentence_to_transformer_dim(last_hidden_states)
		entity_mask = entity_mask_padded.unsqueeze(-1).expand(question_embedding.shape)
		masked_question_embedding = question_embedding * entity_mask  # set entity positions 0

		entity_time_embedding_projected = self.project_entity(entity_time_embedding)
        
        
        
		#time-aware step 
		if self.model == 'tempoqr':
			time_pos_embeddings1 = t1_emb.unsqueeze(0).transpose(0,1)
			time_pos_embeddings1 = time_pos_embeddings1.expand(entity_time_embedding_projected.shape)
		    
			time_pos_embeddings2 = t2_emb.unsqueeze(0).transpose(0,1)
			time_pos_embeddings2 = time_pos_embeddings2.expand(entity_time_embedding_projected.shape)
			if self.fuse == 'cat':
				entity_time_embedding_projected = self.lin_cat(torch.cat((entity_time_embedding_projected, time_pos_embeddings1,time_pos_embeddings2), dim=-1))
			else:
				entity_time_embedding_projected = entity_time_embedding_projected + time_pos_embeddings1 + time_pos_embeddings2
        
		# Transformer information fusion layer
		masked_entity_time_embedding = entity_time_embedding_projected * self.invert_binary_tensor(entity_mask)
        
        
		combined_embed = masked_question_embedding + masked_entity_time_embedding
		# also need to add position embedding
		sequence_length = combined_embed.shape[1]
		v = np.arange(0, sequence_length, dtype=np.long)
		indices_for_position_embedding = torch.from_numpy(v).cuda()
		position_embedding = self.position_embedding(indices_for_position_embedding)
		position_embedding = position_embedding.unsqueeze(0).expand(combined_embed.shape)
        

		combined_embed = combined_embed + position_embedding 

		combined_embed = self.layer_norm(combined_embed)
		combined_embed = torch.transpose(combined_embed, 0, 1)

		mask2 = ~(question_attention_mask.bool()).cuda()

		output = self.transformer_encoder(combined_embed, src_key_padding_mask=mask2)
        
		# Answer Predictions
		if self.model == 'cronkgqa': #no transformer, go back and get cls_embedding
			relation_embedding = cls_embedding
		else:
			relation_embedding = output[0]# self.linear(output[0]) #cls token embedding
		relation_embedding1 = self.dropout(self.bn1(self.linear1(relation_embedding)))
		relation_embedding2 = self.dropout(self.bn1(self.linear2(relation_embedding)))


        
		scores_time = self.score_time(head_embedding, tail_embedding, relation_embedding1)

		if self.model == 'cronkgqa' or (self.model == 'entityqr' and self.supervision != 'none'): #supervision for cronkgqa and entityqr
			time_embedding = (time_embedding + t1_emb + t2_emb)/3   #just take the mean
		scores_entity1 = self.score_entity(head_embedding, tail_embedding, relation_embedding2, time_embedding)
		scores_entity2 = self.score_entity(tail_embedding, head_embedding, relation_embedding2, time_embedding)
		scores_entity = torch.maximum(scores_entity1, scores_entity2)
        
		scores = torch.cat((scores_entity, scores_time), dim=1)
        
		return scores  


