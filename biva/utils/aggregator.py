from collections import defaultdict
from numbers import Number
from typing import *

import torch

CPU = 'cpu'


class Aggregator():
    """
    A data structure to store and summarize diagnostics data
    This summarize data using moving average across the first dimension (batch dimension)
    This keeps the data on GPU until it is summarized
    """

    def __init__(self):
        self.initialize()

    def initialize(self):
        """initialize aggregator"""
        self.data = defaultdict(dict)
        self.count = defaultdict(dict)

    def update(self, new_data: Dict[str, Dict[str, Union[Number, torch.Tensor]]]):
        """moving average update : s_n+k = s_n + 1/n+k (x_n+k - k * s_n)"""
        for k, v in new_data.items():
            for k_, v_ in v.items():
                if v_ is not None:

                    sum, count = self._sum_and_count(v_)

                    if k_ not in self.data[k].keys():
                        self.count[k][k_] = count
                        self.data[k][k_] = sum / count
                    else:
                        self.count[k][k_] += count
                        self.data[k][k_] += (sum - count * self.data[k][k_]) / self.count[k][k_]

    @staticmethod
    def _sum_and_count(v: Union[Number, torch.Tensor]) -> Tuple[Union[Number, torch.Tensor]]:
        """compute sum and count"""
        if isinstance(v, torch.Tensor):
            if isinstance(v, torch.LongTensor):
                v = v.float()
            count = v.size(0) if v.dim() > 0 else 1
            sum = torch.sum(v, 0)
            count = count * torch.ones_like(sum)
            return sum, count
        else:
            return v, 1

    def summarize(self) -> Dict[str, Dict[str, Union[Number, torch.Tensor]]]:
        """
        move data to CPU and return as a dict of dict
        :return: summary a dict of dict with leaf equals to the mean of the series
        """
        summary = defaultdict(dict)
        if len(self.data):
            # iterate through aggregator and move to cpu
            for k, v in self.data.items():
                summary[k] = {}
                for k_, v_ in v.items():
                    if isinstance(v_, torch.Tensor):
                        v_ = v_.to(CPU)  # move to CPU and implicitly checks that the value is of one element
                    summary[k][k_] = v_
            # add the count to the summary
            N_iters = next(iter(next(iter(self.count.values())).values()))
            summary['general']['iters'] = N_iters
        return summary
