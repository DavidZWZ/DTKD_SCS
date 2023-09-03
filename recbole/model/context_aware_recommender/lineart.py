import torch.nn as nn
from torch.nn.init import xavier_normal_
from recbole.model.abstract_recommender import ContextRecommender


class LinearT(ContextRecommender):
    r"""LinearT is a context-based recommendation model.
    It aims to a train linear model of the form :math:`y = w^Tx + b`
    """

    def __init__(self, config, dataset):
        super(LinearT, self).__init__(config, dataset)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.temper = config["distill_temperature"]
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, interaction):
        output = self.first_order_linear(interaction)
        return output.squeeze(-1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(self.sigmoid(output), label)

    def predict(self, interaction):
        return self.forward(interaction)

    def calculate_soft(self, interaction):
        output = self.forward(interaction)
        return self.sigmoid(output/self.temper)
