# Next Steps

- Semi-Supervised learning 
|- Enable pseudo-labelling on ground-truth labels ? 
|- Expand coverage: add GADBench and PyGOD wrappers to the collaborative loop and test them in notebooks.
- Graph handling
|- Transductive set-up (immediate)
|- Inductive set-up : we erase the nodes and edges of the test_set from the graph during the training | we can do pseudo-inductive learning by erasing the test nodes and their edges from the graph, then the training nodes won't get influenced by test nodes through message passing, it has to be super clearly formulated if in the report
- Hyperparameters: tune anomaly_threshold, patience, and max_chapters per dataset; add configs under configs/detectors/.
|- Robustness: add unit tests for predict_scores(use_train=True/False) across all wrappers.
- Recurrent model: wire up a post-hoc ensemble (RecurrentModel) using embeddings collected during cotrain().
|- CI/testing: add light tests that train tiny models for 1–2 epochs to catch interface regressions.
- Docs: write a nice latex doc 
|- Visual interface

