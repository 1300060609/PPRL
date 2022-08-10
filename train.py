import torch
from model import PPRL
from typing import Union, List, Mapping
import timeit
import logging
from dataclasses import dataclass
import random
from constants import *
import sys, os

__all__ = [
    'TrainPPRL',
    'TrainPPRLConfig',
]

log = logging.getLogger(__name__)


# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                     datefmt='%a, %d %b %Y %H:%M:%S')


@dataclass
class TrainPPRLConfig:
    input_dir: str
    margin_loss: float
    embedding_dim: int
    lp_norm: float
    K: int
    learning_rate: float
    T: Union[int, str]
    device_name: str
    prior_weight: float

    @classmethod
    def from_dict(cls, configures: Dict) -> 'TrainPPRLConfig':
        return cls(input_dir=configures['input_dir'],
                   margin_loss=configures['margin_loss'],
                   embedding_dim=configures['embedding_dim'],
                   lp_norm=configures['lp_norm'],
                   K=configures['K'],
                   learning_rate=configures['learning_rate'],
                   T=configures['T'],
                   device_name=configures['preferred_device'],
                   prior_weight=configures['prior_weight'])


class TrainPPRL:
    """Skip with negative sampling_ Gram model training data, return the trained word vector and its weight
    :param configures: Dict
    """

    def __init__(self, configures: Dict) -> None:
        train_conf = TrainPPRLConfig.from_dict(configures)
        self.prior_weight = train_conf.prior_weight
        self.Eca, self.Ega = self._get_edges(train_conf.input_dir)
        self.K = train_conf.K
        self.learning_rate = train_conf.learning_rate
        self.T = train_conf.T

        self.vocab = self._get_vocab()
        # print('vocab', self.vocab)
        configures['num_entities'] = len(self.vocab)
        configures['num_relations'] = 0
        self.device_name = (
            'cuda:0'
            if torch.cuda.is_available() and configures[PREFERRED_DEVICE] == GPU else
            CPU
        )
        self.device = torch.device(self.device_name)
        self.model = PPRL(configures).to(self.device)
        # self.model=torch.nn.DataParallel(PPRL(configures)).module.fc()
        # self.model.to(self.device)

    def _get_edges(self, input_dir):
        Eca = []
        with open(os.path.join(input_dir, 'sample_context.txt'), 'r') as f1:
            for line in f1:
                l = line.strip().split(',')
                Eca.append([l[0], l[1], eval(l[2])])
        Ega = []
        if self.prior_weight:
            with open(os.path.join(input_dir, 'protein_context.txt'), 'r') as f2:
                for line in f2:
                    l = line.strip().split(',')
                    Ega.append([l[0], l[1], eval(l[2])])
        return Eca, Ega

    def _get_vocab(self):
        vocab = {}
        if self.prior_weight:
            total_edges = self.Eca + self.Ega
        else:
            total_edges = self.Eca

        i = 0
        for tri in total_edges:
            oth = tri[0]
            if oth not in vocab:
                vocab[oth] = i
                i += 1
        for tri in total_edges:
            con = tri[1]
            if con not in vocab:
                vocab[con] = i
                i += 1
        return vocab  # vocabulary:dict,Each word corresponds to an index value -- word:index

    def _get_edge_pool(self, edges):
        edge_weights = []
        for edge in edges:
            edge_weights.append(np.abs(edge[2]))
        edge_weights = torch.Tensor(edge_weights)
        edge_pool = torch.multinomial(edge_weights, self.T, replacement=True).to(self.device)
        return list(edge_pool)

    def _get_context_pool(self, context_w):
        contexts = []
        values = []
        for i in context_w:
            contexts.append(i[0])
            values.append(i[1])
        pool = torch.multinomial(torch.tensor(values), self.T * self.K * 2, replacement=True)
        return contexts, list(pool)

    def run(self):
        edge_pool_eca = self._get_edge_pool(self.Eca)
        if self.prior_weight:
            edge_pool_ega = self._get_edge_pool(self.Ega)
        context_w_eca = self._get_contextprotein_value(
            self.Eca)  # Obtain the probability of each context protein. Here, the 3/4 power of frequency is used as the construction basis
        contexts_eca, context_pool_eca = self._get_context_pool(context_w_eca)

        optimizer = torch.optim.SGD(PPRL.parameters(self.model), lr=self.learning_rate)
        log.debug(f'****running model on {self.device}****')
        # start_training = timeit.default_timer()
        if self.T == 'infer':
            self.T = max(len(self.Eca), len(self.Ega)) * 3
        loop = 1
        init_time = timeit.default_timer()
        while loop <= self.T:
            # sys.stdout.write('\rCurrent progress%d update%d remaining' % (loop, self.T-loop))
            # sys.stdout.flush()
            start_training = timeit.default_timer()
            # sample a positive edge from Eca and K negative edges form the noise distribution Pca and update
            # the samples's representations and the context-proteins' representations
            eca = self.Eca[edge_pool_eca.pop()]  # Target binary
            eca_id = (self.vocab[eca[0]], self.vocab[eca[1]])
            eca_weight = eca[2]
            batch_pos_eca, batch_neg_eca = self._get_train_batch(eca_id, contexts_eca, context_pool_eca)
            # batch_pos_eca, batch_neg_eca=torch.tensor([[278, 524]]),torch.tensor([[ 278,  795],[ 278, 5072],[ 278, 5168],[ 278, 5470],[ 278,  810]])
            self._update_parameters(optimizer, batch_pos_eca, batch_neg_eca, 1 * eca_weight / np.abs(eca_weight))

            # sample a positive edge from Ega and K negative edges form the noise distribution Pga and update
            # the proteins's representations and the context-proteins' representations
            if self.prior_weight:
                ega = self.Ega[edge_pool_ega.pop()]  # Target binary
                ega_weight = ega[2]
                batch_pos_ega = torch.tensor([[self.vocab[ega[0]], self.vocab[ega[1]]]], device=self.device)
                self._update_parameters(optimizer, batch_pos_ega, None,
                                        self.prior_weight * ega_weight / np.abs(ega_weight))

            stop_training = timeit.default_timer()
            sys.stdout.write(
                '\rThe current progress is updated for the %d time, and the remaining %d time is %f seconds. It is estimated that it will take %f hours' % (
                    loop, self.T - loop, stop_training - start_training,
                    (stop_training - init_time) / loop * (self.T - loop) / 3600))
            sys.stdout.flush()
            loop += 1

        # log.debug("training took %.2fs seconds", stop_training - start_training)
        # print(self.entity_embeddings(torch.tensor([0,1])))
        # print(self.entity_embeddings(torch.tensor([0,1])).size())
        # print('\n')
        return self.model

    def _get_contextprotein_value(self, edges):
        res = {}
        for tri in edges:
            con = tri[1]
            val = np.abs(tri[2])
            k = res.setdefault(con, 0)
            res[con] = k + val
        for i in res:
            res[i] = res[i] ** .75
        return [[self.vocab[i], res[i]] for i in res]

    def _get_train_batch(self, edge, contexts, context_pool):
        i = edge[0]  # target sample id
        j = edge[1]  # target context id
        neg_sample_indices_e = self._get_neg_samples(j, contexts, context_pool)

        batch_pos = torch.tensor([[i, j]], device=self.device)
        batch_neg = torch.tensor([[i, wo] for wo in neg_sample_indices_e], device=self.device)
        return batch_pos, batch_neg

    def random_pick(self, item_ps: List):
        """
            :param item_ps:like [['protein1', 0.1065], ['protein2', 0.3164], ['protein3', 0.226], ['protein4', 0.3511]]
        """
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for item_p in item_ps:
            item = item_p[0]
            probability = item_p[1]
            cumulative_probability += probability
            if x < cumulative_probability:
                return item

    def _get_edges_p(self, edges):
        s = np.sum(list(map(lambda x: x[2], edges)))
        edges_p = [[(e[0], e[1]), e[2] / s] for e in edges]
        edges_p.sort(key=lambda x: x[1], reverse=True)
        return edges_p

    def _edge_sampleing(self, edges_p):
        return self.random_pick(edges_p)

    def _get_neg_samples(self, pos_index, contexts, context_pool):
        """
                Obtain negative sampling samples by randomly selecting index values in the auxiliary array
                :param pos_ Index: index of the target sample in the glossary
                :param context_ p: list of [context: probability]
                : return: an array of indexes of negative samples in the vocabulary
                """
        neg_samples = []
        while len(neg_samples) < self.K:
            neg_sample_index = contexts[context_pool.pop()]
            if neg_sample_index == pos_index:  # The negative sample cannot be the same as the target sample
                continue
            neg_samples.append(neg_sample_index)
        return neg_samples

    def _update_parameters(self, optimizer, batch_pos, batch_neg, weight):
        # Update word vector and weight according to formula
        # print(batch_pos)
        optimizer.zero_grad()
        loss = PPRL.forward(self.model, batch_pos, batch_neg) * weight
        # print(batch_pos,loss)
        loss.backward()
        optimizer.step()
        # print('step')


def _make_results(trained_model,
                  loss_per_epoch,
                  entity_to_embedding: Mapping[str, np.ndarray],
                  relation_to_embedding: Mapping[str, np.ndarray],
                  eval_summary,
                  entity_to_id,
                  rel_to_id,
                  params) -> Dict:
    results = OrderedDict()
    results[TRAINED_MODEL] = trained_model
    results[LOSSES] = loss_per_epoch
    results[ENTITY_TO_EMBEDDING]: Mapping[str, np.ndarray] = entity_to_embedding
    results[RELATION_TO_EMBEDDING]: Mapping[str, np.ndarray] = relation_to_embedding
    results[EVAL_SUMMARY] = eval_summary
    results[ENTITY_TO_ID] = entity_to_id
    results[RELATION_TO_ID] = rel_to_id
    results[FINAL_CONFIGURATION] = params
    return results


if __name__ == '__main__':
    Eca = [('sample1', 'protein1', 1), ('sample1', 'protein2', 2), ('sample2', 'protein2', 3),
           ('sample3', 'protein3', 4)]
    Ega = [['aa', 'protein1', 1.1], ['bb', 'protein2', 2.3], ['sdf', 'protein2', 2.4], ['fds', 'protein3', 3],
           ['dfff', 'protein4', 5.4]]
    config = {'margin_loss': -0.62, 'embedding_dim': 10, 'lp_norm': 2, 'Eca': Eca, 'Ega': Ega,
              'T': 100, 'K': 5, 'learning_rate': .1}
    train = TrainPPRL(config)
    model = TrainPPRL.run(train)
