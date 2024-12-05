# naive token-level classification model (including bert-based and roberta-based)

import torch
import transformers
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel, RobertaForTokenClassification, \
	RobertaConfig
from transformers import BertTokenizer, BertModel, BertForTokenClassification, BertConfig
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class RobertaNER(RobertaForTokenClassification):
	config_class = RobertaConfig
	base_model_prefix = "roberta"
	
	def __init__(self, config, dataset_label_nums, multi_gpus=False):
		super().__init__(config)
		self.roberta = RobertaModel(config)
		self.label_encoder = RobertaModel(config)
		self.label_context = {"location": "location", "corporation":"corporation", "person":"person", "group":"group", "creative_work":"creative work", "product":"product"}
		self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
		
		self.index_context = {
			"B": "Begin Flag",
			"I": "Inside Flag",
			"E": "End Flag",
			"S": "Single Flag"
		}
		self.dataset_label_nums = dataset_label_nums
		self.multi_gpus = multi_gpus
		#self.label_embedding = torch.nn.Embedding(13,768)
		
		
		# self.tokenizer = tokenizer
		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		self.linear = torch.nn.Linear(768 * 2, 768)
		# self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels-1)
		# used in training dataset
		# self.classifier1 = torch.nn.Linear(config.hidden_size, dataset_label_nums[0])
		# used in testing dataset (if from different domains)
		# self.classifier2 = torch.nn.Linear(config.hidden_size, dataset_label_nums[1])
		self.classifiers = torch.nn.ModuleList([torch.nn.Linear(config.hidden_size, x) for x in dataset_label_nums])
		self.background = torch.nn.Parameter(torch.zeros(1) - 2., requires_grad=True)
		
		self.init_weights()
	
	def build_label_representation(self):
		tag2id = {'O': 0, 'B-location': 1, 'I-location': 2, 'B-corporation': 3, 'I-corporation': 4, 'B-person': 5, 'I-person': 6, 'B-group': 7, 'I-group': 8, 'B-creative_work': 9, 'I-creative_work': 10, 'B-product': 11, 'I-product': 12}
		labels = []
		for k, v in tag2id.items():
			if k.split('-')[-1] != 'O':
				idx, label = k.split('-')[0], k.split('-')[-1]
				label = self.label_context[label]
				labels.append(label +" "+ self.index_context[idx])
			else:
				labels.append("other")
		'''
		mutul(a,b) a和b维度是否一致的问题
		A.shape =（b,m,n)；B.shape = (b,n,k)
		torch.matmul(A,B) 结果shape为(b,m,k)
		'''
		
		tag_max_len = max([len(l) for l in labels])
		tag_embeddings = []
		for label in labels:
			input_ids = self.tokenizer.encode_plus(label, return_tensors='pt', padding='max_length', max_length=tag_max_len)
			outputs = self.label_encoder(**input_ids.to(torch.device('cuda:0')))
			pooler_output = outputs[1]
			tag_embeddings.append(pooler_output)
		label_embeddings = torch.stack(tag_embeddings, dim=0)
		label_embeddings = label_embeddings.squeeze(1)
		return label_embeddings
	
	def mixture_of_knowledge(self, sequence_output, label_representation, ents):
		
		# label_embeddings = torch.nn.Embedding.from_pretrained(torch.cat([label_outputs, cat_tensor.to(torch.device('cuda:0'))],0))
		
		#label_emds = torch.nn.functional.embedding(ents, label_representation)
		#batch_size, feature_dim = ents.shape
		#index = ents.flatten()
		#tmp = torch.index_select(label_representation, 0, index)
		#ent_embs = tmp.view(batch_size, feature_dim, -1)
		
		ent_embs = torch.nn.functional.embedding(ents, label_representation)
		
		weight = torch.sigmoid(self.linear(torch.cat([sequence_output, ent_embs], -1)))
		outputs = sequence_output * weight + ent_embs * (1 - weight)
		# weight = torch.sigmoid(self.linear(torch.cat([sequence_output, label_emds], -1)))
		# outputs = sequence_output * weight + label_emds * (1 - weight)
		return outputs
	
	def forward(self, input_ids,
	            attention_mask=None,
	            dataset=0,
	            position_ids=None,
	            head_mask=None,
	            inputs_embeds=None,
	            labels=None,
	            output_logits=False,
	            ents=None):
		
		outputs = self.roberta(
			input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)
		
		sequence_output = outputs[0]
		sequence_output = self.dropout(sequence_output)
		# 4,128,768
		
		batch_size, max_len, feat_dim = sequence_output.shape
		label_representation = self.build_label_representation().to('cuda:0')
		tag_lens, hidden_size = label_representation.shape
		label_embeddings = label_representation.expand(batch_size, tag_lens, hidden_size)
		label_embeddings = label_embeddings.transpose(2, 1)
		
		hidden_out = self.mixture_of_knowledge(sequence_output, label_representation, ents)
		
		logits = torch.matmul(hidden_out, label_embeddings)
		#softmax_embedding = torch.nn.Softmax(dim=-1)(logits)
		#outputs = torch.argmax(softmax_embedding, dim=-1)
		#matrix_embeddings = torch.matmul(sequence_output, label_embeddings)
		# 4,128,13
		#logits = self.classifiers[dataset](sequence_output)
		#4,128,13
		# logits = torch.cat((self.background.unsqueeze(0).unsqueeze(0).repeat(batch_size, max_len, 1), logits), dim=2)
		outputs = torch.argmax(logits, dim=2)
		#4,128
		
		if labels is not None:
			
			loss_fct = CrossEntropyLoss()
			# loss = loss_fct(matrix_embeddings.view(-1, len(tag2id)), labels.view(-1))
			
			loss = loss_fct(logits.view(-1, self.dataset_label_nums[dataset]), labels.view(-1))
			if output_logits:
				return loss, outputs, logits
			else:
				return loss, outputs
		else:
			return outputs
	
	def forward_unsup(self, input_ids,
	                  attention_mask=None,
	                  dataset=0,
	                  position_ids=None,
	                  head_mask=None,
	                  inputs_embeds=None,
	                  t_prob=None,
	                  output_logits=False):
		
		outputs = self.roberta(
			input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)
		
		sequence_output = outputs[0]
		batch_size, max_len, feat_dim = sequence_output.shape
		sequence_output = self.dropout(sequence_output)
		logits = self.classifiers[dataset](sequence_output)
		# logits = torch.cat((self.background.unsqueeze(0).unsqueeze(0).repeat(batch_size, max_len, 1), logits), dim=2)
		outputs = torch.argmax(logits, dim=2)
		sel_idx = torch.tensor(
			[j + i * len(x) for i, x in enumerate(attention_mask) for j in range(len(x)) if x[j] == 1]).cuda()
		# print(f"shape: {sel_idx.shape}")
		# print(sel_idx)
		log_pred_prob = torch.log(F.softmax(logits.view(-1, self.dataset_label_nums[dataset]), dim=-1))
		log_pred_prob = torch.index_select(log_pred_prob, 0, sel_idx)
		t_prob = F.softmax(t_prob.view(-1, self.dataset_label_nums[dataset]), dim=-1)
		t_prob = torch.index_select(t_prob, 0, sel_idx)
		
		kl_criterion = torch.nn.KLDivLoss()
		loss = kl_criterion(log_pred_prob, t_prob)
		if output_logits:
			return loss, outputs, logits
		else:
			return loss, outputs


class BertNER(BertForTokenClassification):
	config_class = BertConfig
	
	def __init__(self, config, dataset_label_nums, multi_gpus=False):
		super().__init__(config)
		self.bert = BertModel(config)
		self.dataset_label_nums = dataset_label_nums
		self.multi_gpus = multi_gpus
		# self.tokenizer = tokenizer
		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		# self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels-1)
		# self.classifier1 = torch.nn.Linear(config.hidden_size, dataset_label_nums[0])
		# self.classifier2 = torch.nn.Linear(config.hidden_size, dataset_label_nums[1])
		# self.classifiers = [self.classifier1, self.classifier2]
		self.classifiers = torch.nn.ModuleList([torch.nn.Linear(config.hidden_size, x) for x in dataset_label_nums])
		self.background = torch.nn.Parameter(torch.zeros(1) - 2., requires_grad=True)
		
		self.init_weights()
	
	def forward(self, input_ids,
	            attention_mask=None,
	            dataset=0,
	            position_ids=None,
	            head_mask=None,
	            inputs_embeds=None,
	            labels=None):
		
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)
		
		sequence_output = outputs[0]
		batch_size, max_len, feat_dim = sequence_output.shape
		sequence_output = self.dropout(sequence_output)
		logits = self.classifiers[dataset](sequence_output)
		# logits = torch.cat((self.background.unsqueeze(0).unsqueeze(0).repeat(batch_size, max_len, 1), logits), dim=2)
		outputs = torch.argmax(logits, dim=2)
		
		if labels is not None:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits.view(-1, self.dataset_label_nums[dataset]), labels.view(-1))
			return loss, outputs
		else:
			return outputs
