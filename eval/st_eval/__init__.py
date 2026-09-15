"""Evaluation harness for background-removal models.

* ``st_eval.synth``   : deterministic synthetic benchmark of hard Iranian product categories
* ``st_eval.metrics`` : mask IoU, boundary F, alpha errors, thin-structure recall
* ``st_eval.bench``   : runs any registered model over the set and sweeps batch sizes
* ``st_eval.report``  : renders eval/results.md from run JSON
"""

__version__ = "0.1.0"
