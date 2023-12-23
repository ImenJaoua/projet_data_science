import os
from comet_ml import Experiment
from dotenv import load_dotenv
load_dotenv()


def start_experiment(workspace = "", project_name = ""):
    project_name = project_name or os.getenv('PROJECT_NAME')
    workspace = workspace or os.getenv('WORKSPACE')
    return Experiment(
        api_key=os.getenv('COMET_API_KEY'),
        project_name=project_name,
        workspace=workspace
    )

def add_metrics(exp: Experiment, accuracy, roc_auc, precision, recall):
    exp.log_metric('Accuracy', accuracy)
    exp.log_metric('Roc-auc', roc_auc)
    exp.log_metric('Precision', precision)
    exp.log_metric('Recall', recall)
    return exp