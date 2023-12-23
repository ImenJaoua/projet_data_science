import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
import io

def plots(y_valid, y_prob, model_name, exp=None):

    if len(y_prob.shape) == 2:
        if 1 in y_prob.shape:
            y_prob = y_prob.reshape(-1)
        else:
            y_prob = y_prob[:,1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.suptitle(f'Performance Evaluation of: {model_name}', fontsize=16)


    # Plot ROC curve and calculate AUC
    fpr, tpr, thresholds = roc_curve(y_valid, y_prob)
    roc_auc = roc_auc_score(y_valid, y_prob)

    axes[0,0].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    axes[0,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0,0].set_xlim([0.0, 1.0])
    axes[0,0].set_ylim([0.0, 1.05])
    axes[0,0].set_xlabel('False Positive Rate')
    axes[0,0].set_ylabel('True Positive Rate')
    axes[0,0].set_title('Receiver Operating Characteristic (ROC)')
    axes[0,0].legend(loc="lower right")

    

    axes[0,1].plot([0, 1], [0, 1], 'k:', label="Perfectly calibrated")
    prob_true, prob_pred = calibration_curve(y_valid, y_prob, n_bins=10)
    axes[0,1].plot(prob_pred, prob_true, 's-', label="%s" % ('clf_distance',))
    axes[0,1].set_ylabel('Fraction of positives')
    axes[0,1].set_xlabel('Mean predicted probability')
    axes[0,1].set_title('Calibration Plot')

    # Calculate and plot the rate of goals and cumulative proportion of goals
    sorted_indices = np.argsort(y_prob)
    sorted_goals = y_valid[sorted_indices]
    predicted_probs = y_prob[sorted_indices]

    n_bins = 20
    bins = np.linspace(0, 1, n_bins + 1)
    midpoints = (bins[:-1] + bins[1:]) / 2

    goal_rates = []
    for i in range(n_bins):
        start_idx = int(i * len(predicted_probs) / n_bins)
        end_idx = int((i + 1) * len(predicted_probs) / n_bins)
        
        goals = sum(sorted_goals[start_idx:end_idx])
        total_shots = end_idx - start_idx
        
        goal_rate = 100 * goals / total_shots
        goal_rates.append(goal_rate)

    axes[1,0].plot(midpoints*100, goal_rates, linestyle='-')
    axes[1,0].set_xlim([100,0])
    axes[1,0].set_ylim([0,100])
    axes[1,0].set_xlabel('Centile of Probability')
    axes[1,0].set_ylabel('Rate of Goals')
    axes[1,0].set_title('Rate of Goals vs. Centile of Probability')

    cumulative_goals = np.cumsum(sorted_goals[::-1])
    tot_goals = y_valid.sum()

    axes[1,1].plot(np.arange(len(y_prob), 0, -1) * 100 / len(y_prob), cumulative_goals * 100 / tot_goals, linestyle='-')
    axes[1,1].set_xlim([100,0])
    axes[1,1].set_ylim([0,100])
    axes[1,1].set_xlabel('Centile of Probability')
    axes[1,1].set_ylabel('Cumulative Proportion of Goals')
    axes[1,1].set_title('Cumulative Proportion of Goals vs. Centile of Probability')

    if exp:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        exp.log_image(buf, name='performance.png')
        buf.close()
    else:
        plt.show()

def ratio_by_percentile(y_true, y_score):

    sorted_indices = np.argsort(y_score)
    predicted_probs = y_score[sorted_indices]
    actual_outcomes = y_true[sorted_indices]

    n_bins = 20
    bins = np.linspace(0, 1, n_bins + 1)
    midpoints = (bins[:-1] + bins[1:]) / 2

    goal_rates = []
    for i in range(n_bins):
        start_idx = int(i * len(predicted_probs) / n_bins)
        end_idx = int((i + 1) * len(predicted_probs) / n_bins)
        
        goals = sum(actual_outcomes[start_idx:end_idx])
        total_shots = end_idx - start_idx
        
        goal_rate = 100 * goals / total_shots
        goal_rates.append(goal_rate)

    plt.plot(midpoints*100, goal_rates, linestyle='-')
    plt.xlim([100,0])
    plt.ylim([0,100])
    plt.xlabel('Shot Probability Model Percentile')
    plt.ylabel('Goal Rate')
    plt.title('Goal Rate')
    plt.grid(True)
    plt.show()

def proportion_by_percentile(y_true, y_score):

    sorted_indices = np.argsort(y_score)
    sorted_indices = sorted_indices[::-1]
    predicted_probs = y_score[sorted_indices]
    actual_outcomes = y_true[sorted_indices]

    n_bins = 100
    bins = np.linspace(1, 0, n_bins + 1)
    midpoints = (bins[:-1] + bins[1:]) / 2

    tot_goals = y_true.sum()
    goal_rates = []
    goals = 0
    for i in range(n_bins):
        start_idx = int(i * len(predicted_probs) / n_bins)
        end_idx = int((i + 1) * len(predicted_probs) / n_bins)
        
        goals += sum(actual_outcomes[start_idx:end_idx])
        
        goal_rate = goals * 100 / tot_goals
        goal_rates.append(goal_rate)

    plt.plot(midpoints*100, goal_rates, linestyle='-')
    plt.xlim([100,0])
    plt.ylim([0,100])
    plt.xlabel('Shot Probability Model Percentile')
    plt.ylabel('Proportion')
    plt.title('Cumulative % of goals')
    plt.grid(True)
    plt.show()


def p_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(fpr, tpr)

    # Plotting ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")