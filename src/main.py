from base.py import CoLearning
from ModelWrapper.py import get_model_detector_dict()
from DataWrapper.py import load_data

parser()
model_detector_dict = get_model_detector_dict()

for dataset in Datasets:
    data = load_data(dataset)
    for models_combo in model_combinaisons:
        for t in range(trials):
            time_cost().start()

            load_train_configs(dataset, models_combo)
            loaded_models = load_models()
            set_seed(t)
            Collab = CoLearner(loaded_models, 
                               data, 
                               colearning_strategy)
            results = Collab.train()
            results.analyse()

            time_cost().stop()
            results.log_time(time_cost())

results.save()
            

