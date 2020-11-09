import argparse
import scikitplot as skplt
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import numpy as np


parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--vid_path', type=str, default="./videos/")
parser.add_argument('--res_path', type=str, default="./results/")
parser.add_argument('--model', type=str, default="tsn")

args = parser.parse_args()

# Read annotations
results = pd.read_table(args.res_path + args.model + "_results.csv", sep=",")

# Plot loss
plt.plot(results['loss'])
plt.ylabel("Loss (" + args.model + ")")
plt.xlabel("Segment number")
plt.grid(True)
plt.show()

# Plot confussion matrix
skplt.metrics.plot_confusion_matrix(results['gt_verb_classes'], results['pred_verb_classes'],
                                    normalize=True, title="Verbs - Normalized Confusion Matrix (" + args.model + ")")
skplt.metrics.plot_confusion_matrix(results['gt_noun_classes'], results['pred_noun_classes'],
                                    normalize=True, title="Nouns - Normalized Confusion Matrix (" + args.model + ")")

# Generate actions
pred_actions = []
gt_actions = []
for idx, verb in enumerate(results['pred_verb_classes']):
    pred_actions.append(results['pred_verb_classes'][idx] + " " + results['pred_noun_classes'][idx])
    gt_actions.append(results['gt_verb_classes'][idx] + " " + results['gt_noun_classes'][idx])

skplt.metrics.plot_confusion_matrix(gt_actions, pred_actions,
                                    normalize=True, title="Actions - Normalized Confusion Matrix (" + args.model + ")")
plt.show()

# Top-1 Accuracy
print("Verb Top-1 Accuracy: " + str(sklearn.metrics.accuracy_score(results['gt_verb_indices'], results['pred_verb_indices'])))
print("Noun Top-1 Accuracy: " + str(sklearn.metrics.accuracy_score(results['gt_noun_indices'], results['pred_noun_indices'])))
print("Action Top-1 Accuracy: " + str(sklearn.metrics.accuracy_score(gt_actions, pred_actions)))

print("Verb Avg. Class Precision: " + str(sklearn.metrics.precision_score(results['gt_verb_indices'], results['pred_verb_indices'], average='weighted')))
print("Noun Avg. Class Precision: " + str(sklearn.metrics.precision_score(results['gt_noun_indices'], results['pred_noun_indices'], average='weighted')))
print("Action Avg. Class Precision: " + str(sklearn.metrics.precision_score(gt_actions, pred_actions, average='weighted')))

print("Verb Avg. Class Recall: " + str(sklearn.metrics.recall_score(results['gt_verb_indices'], results['pred_verb_indices'], average='weighted')))
print("Noun Avg. Class Recall: " + str(sklearn.metrics.recall_score(results['gt_noun_indices'], results['pred_noun_indices'], average='weighted')))
print("Action Avg. Class Recall: " + str(sklearn.metrics.recall_score(gt_actions, pred_actions, average='weighted')))

# Dataset histograms
labels, counts = np.unique(np.array(results['gt_verb_classes']), return_counts=True)
plt.bar(labels, counts, align='center', color='green')
plt.gca().set_xticks(labels)
plt.ylabel("Frequency")
plt.title("Test dataset histogram (Verbs)")
plt.show()

labels, counts = np.unique(np.array(results['gt_noun_classes']), return_counts=True)
plt.bar(labels, counts, align='center', color='orange')
plt.gca().set_xticks(labels)
plt.ylabel("Frequency")
plt.title("Test dataset histogram (Nouns)")
plt.show()

labels, counts = np.unique(np.array(gt_actions), return_counts=True)
plt.bar(labels, counts, align='center', color='blue')
plt.gca().set_xticks(labels)
plt.ylabel("Frequency")
plt.title("Test dataset histogram (Actions)")
plt.show()

# Latency and FPS
print("Avg. Data Loading Latency: " + str(np.array(results['lat_load']).mean()))
print("Avg. Inference Latency: " + str(np.array(results['lat_inference']).mean()))
print("Avg. FPS: " + str(8.0 / (np.array(results['lat_load']).mean() + np.array(results['lat_inference']).mean())))
