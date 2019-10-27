from collections import defaultdict
from typing import *

import torch


class DataCollector(defaultdict):
    def __init__(self):
        super().__init__(list)

    def extend(self, data: Dict[str, List[Optional[torch.Tensor]]]) -> None:
        """Append new data item"""
        for key, d in data.items():
            self[key] += d

    def sort(self) -> Dict[str, List[Optional[torch.Tensor]]]:
        """sort data and return"""
        for key, d in self.items():
            d = d[::-1]
            self[key] = [t for t in d if t is not None]

        return self
