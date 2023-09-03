import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
import torch.nn.functional as F

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import MLPLayers

class MLPS(ContextRecommender):
    r"""MLP_Student is a context-based recommendation model.
    It distill the fused knowledge from CF and content teachers.
    """

    def __init__(self, config, dataset):
        super(MLPS, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]
        self.temper_con = config['distill_temperature']
        self.temper_cf = config['distill_temperature']
        self.cf_w = config['cf_weight']
        self.con_w = 1-config['cf_weight']

        size_list = [
            self.embedding_size * self.num_feature_field
        ] + self.mlp_hidden_size
        self.mlp_layers = MLPLayers(size_list, self.dropout_prob)
        self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()
        self.bce_loss = nn.BCELoss()
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, interaction):
        deep_all_embeddings = self.concat_embed_input_fields(interaction) 
        batch_size = deep_all_embeddings.shape[0]
        deep_output = self.deep_predict_layer(
            self.mlp_layers(deep_all_embeddings.view(batch_size, -1))
        )
        return deep_output.squeeze(-1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))

    def distill_soft(self, interaction, soft_con, soft_cf):
        output = self.forward(interaction)
        soft_label_con = soft_con(interaction)
        soft_label_cf = soft_cf(interaction)
        soft_label = self.con_w*soft_label_con+self.cf_w*soft_label_cf
        soft_loss = self.bce_loss(self.sigmoid(output/self.temper_cf), soft_label)
        return  soft_loss 

