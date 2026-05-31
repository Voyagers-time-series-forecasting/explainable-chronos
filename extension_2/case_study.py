import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from extension_2.dialogue import DialogueSystem
from extension_2.evaluation import _make_eval_scenario, COVARIATE_NAMES

history, covariates = _make_eval_scenario()
system = DialogueSystem(history=history, covariates=covariates, horizon=14)

turns = [
    "What if there were no marketing spend?",
    "How confident are you in this forecast?",
    "Show me the next three weeks",
    "What would have happened if sales had been higher last month?",
]

with open("extension_2/eval_results/case_study_ext2.txt", "w") as f:
    for q in turns:
        r = system.query(q)
        f.write(r.summary())
        f.write("\n")

print("Saved to extension_2/eval_results/case_study_ext2.txt")