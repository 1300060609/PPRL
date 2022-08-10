# -*- coding: utf-8 -*-


"""Implementation of the PPRL model."""

import torch
import logging
from dataclasses import dataclass
from typing import Dict

from base import BaseModule

__all__ = [
    'PPRL',
    'PPRLConfig',
]

log = logging.getLogger(__name__)


@dataclass
class PPRLConfig:
    lp_norm: str

    @classmethod
    def from_dict(cls, config: Dict) -> 'PPRLConfig':
        """Generate an instance from a dictionary."""
        return cls(
            lp_norm=config['lp_norm'],
        )


class PPRL(BaseModule):
    model_name = 'PPRL'
    margin_ranking_loss_size_average: bool = True

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.final_configuration = config
        config = PPRLConfig.from_dict(config)

        # Embeddings
        self.l_p_norm_entities = config.lp_norm
        self._initialize()

    def _initialize(self):
        entity_embeddings_init_bound = 1
        torch.nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-entity_embeddings_init_bound,
            b=entity_embeddings_init_bound,
        )

    def _compute_loss(self, pos_scores, neg_scores):
        # y = np.repeat([-1], repeats=pos_scores.shape[0])
        # y = torch.tensor(y, dtype=torch.float, device=self.device)
        pos_score = torch.mean(pos_scores, 0)

        y = torch.tensor([-1], device=self.device)
        # Scores for the positive and negative triples
        # pos_scores = torch.tensor(pos_scores, dtype=torch.float, device=self.device)
        # neg_scores = torch.tensor(neg_scores, dtype=torch.float, device=self.device)
        # print(pos_scores,neg_scores,y)
        # print(pos_score,neg_score)
        if neg_scores != None:
            neg_score = torch.mean(neg_scores, 0)
            loss = self.criterion(pos_score, -neg_score, y)
        else:
            loss = self.criterion(pos_score, torch.tensor([0]), y)
        # print(loss)
        return loss

    def _compute_scores(self, h_embs, t_embs):
        """
        :param h_embs: embeddings of head entities of dimension batchsize x embedding_dim
        :param t_embs: embeddings of tail entities of dimension batchsize x embedding_dim
        :return: Tensor of dimension batch_size containing the scores for each batch element
        """
        epsilon = 1e-30
        return -torch.log(torch.sigmoid(torch.sum(h_embs * t_embs, dim=1).view(-1, 1)) + epsilon)

    def predict(self, pairs):
        heads = pairs[:, 0]
        tails = pairs[:, 1]

        head_embs = self.entity_embeddings(heads).view(-1, self.embedding_dim)
        tail_embs = self.entity_embeddings(tails).view(-1, self.embedding_dim)

        scores = self._compute_scores(h_embs=head_embs, t_embs=tail_embs)
        return scores.detach().cpu().numpy()

    def forward(self, batch_positives, batch_negatives):
        # Normalize embeddings of entities
        norms = torch.norm(self.entity_embeddings.weight, p=self.l_p_norm_entities, dim=1).data
        self.entity_embeddings.weight.data = self.entity_embeddings.weight.data.div(
            norms.view(self.num_entities, 1).expand_as(
                self.entity_embeddings.weight))  # Vector normalization, component divided by LP norm

        pos_heads = batch_positives[:, 0]
        pos_tails = batch_positives[:, 1]

        pos_h_embs = self.entity_embeddings(pos_heads).view(-1, self.embedding_dim)
        pos_t_embs = self.entity_embeddings(pos_tails).view(-1, self.embedding_dim)

        # if torch.isnan(pos_h_embs[0][0]):
        #     print(pos_h_embs,pos_t_embs)
        pos_scores = self._compute_scores(h_embs=pos_h_embs, t_embs=pos_t_embs)

        if batch_negatives != None:

            neg_heads = batch_negatives[:, 0]
            neg_tails = batch_negatives[:, 1]

            neg_h_embs = self.entity_embeddings(neg_heads).view(-1, self.embedding_dim)
            neg_t_embs = self.entity_embeddings(neg_tails).view(-1, self.embedding_dim)

            neg_scores = self._compute_scores(h_embs=neg_h_embs, t_embs=-neg_t_embs)

            loss = self._compute_loss(pos_scores=pos_scores, neg_scores=neg_scores)
        else:
            loss = self._compute_loss(pos_scores=pos_scores, neg_scores=None)
        return loss


if __name__ == '__main__':
    print(PPRL.hyper_params)
