import argparse
from torchvision.transforms import Compose
import torch.nn.functional as F
from transforms import GroupScale, GroupCenterCrop, GroupOverSample, Stack, ToTorchFormatTensor, GroupNormalize
from task_utils.utils import *
import pandas as pd
import time
from archs.tsn_pl import TSN
from task_utils.datasets import KFCDataset


import torch.hub

parser = argparse.ArgumentParser(
    description="Action recognition training")
parser.add_argument('--model_ckpt', type=str, default="./lightning_logs/satis_ckpt/checkpoints/epoch=3.ckpt")
parser.add_argument('--dataset_csv', type=str, default="./videos/satis_task_b_dataset/val/val.csv")
# parser.add_argument('--dataset_csv', type=str, default="./videos/satis_task_b_dataset/train/train.csv")
parser.add_argument('--batch', default=1, type=int)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--segment_count', default=8, type=int)
parser.add_argument('--repo', type=str, default="epic-kitchens/action-models")
parser.add_argument('--base_model', type=str, default="resnet50")
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--num_epochs', default=30, type=int)
parser.add_argument('--validation_epochs', default=2, type=int)
parser.add_argument('--reset_model', default=1, type=int)

args = vars(parser.parse_args())

# Check if CUDA is available
if args['gpu_id']>=0 and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Model instance
model = TSN.load_from_checkpoint(args['model_ckpt'], hparams_file=os.path.join(os.path.dirname(os.path.dirname(args['model_ckpt'])), "hparams.yaml"))
print("INFO: Model loaded", args['model_ckpt'])

# Re-initialize model
if args['reset_model']:
    model.reset_model()
    print("INFO: Model has been reset")

# Read classes
nouns_classes = pd.read_table("./epic-kitchens-55-annotations/EPIC_noun_classes.csv", sep=",")['class_key']
verbs_classes = pd.read_table("./epic-kitchens-55-annotations/EPIC_verb_classes.csv", sep=",")['class_key']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loss criterion
criterion = torch.nn.CrossEntropyLoss().to(device)

# Move to GPU if available and set to evaluation
model.eval()
model.to(device)

# Val transform
cropping = GroupOverSample(model.tsn_model.input_size, model.tsn_model.scale_size)
val_transform = Compose([
    cropping,
    Stack(roll=args['base_model'] == args['base_model']),
    ToTorchFormatTensor(div=args['base_model'] != args['base_model']),
    GroupNormalize(model.tsn_model.input_mean, model.tsn_model.input_std),
])

# Datasets
val_dataset = KFCDataset(args['dataset_csv'], segment_count=args['segment_count'], transform=val_transform, debug=True)

pred_verb_indices = []
pred_noun_indices = []
pred_verb_classes = []
pred_noun_classes = []
gt_verb_indices = []
gt_noun_indices = []
gt_verb_classes = []
gt_noun_classes = []

d = {
    'pred_verb_indices': [],
    'pred_noun_indices': [],
    'pred_verb_classes': [],
    'pred_noun_classes': [],
    'gt_verb_indices': [],
    'gt_noun_indices': [],
    'gt_verb_classes': [],
    'gt_noun_classes': [],
    'lat_load': [],
    'lat_inference': [],
    'loss': []
}

for idx, batch in enumerate(val_dataset):
    # Extract 1 frame per snippet
    print("Processing video: ", val_dataset.kfc_anno['uid'][idx] + ".mp4")
    tic = time.clock()

    inputs, verb_targets, noun_targets = batch

    toc = time.clock()
    d['lat_load'].append(toc-tic)
    print("Data loading latency: " + str(toc-tic))

    # or just call the object to classify inputs in a single forward pass
    tic = time.clock()
    with torch.no_grad():
        verb_logits, noun_logits = model(inputs.to(model.device))
        verb_targets = (torch.ones(10, dtype=torch.long) * val_dataset.kfc_anno['verb_class'][idx]).to(device)
        noun_targets = (torch.ones(10, dtype=torch.long) * val_dataset.kfc_anno['noun_class'][idx]).to(device)
        print(verb_logits.shape)
        print(verb_targets.shape)
        loss = (criterion(verb_logits, verb_targets) + criterion(noun_logits, noun_targets)).cpu().detach().numpy()
        d['loss'].append(loss)

    verb_indices = torch.argmax(F.softmax(torch.mean(verb_logits, dim=0, keepdim=True), dim=1), dim=1).cpu().detach().numpy()
    noun_indices = torch.argmax(F.softmax(torch.mean(noun_logits, dim=0, keepdim=True), dim=1), dim=1).cpu().detach().numpy()
    toc = time.clock()
    d['lat_inference'].append(toc-tic)
    print("Inference latency: " + str(toc-tic))

    for j, verb in enumerate(verb_indices):
        print("ground truth:", val_dataset.kfc_anno['verb'][idx], val_dataset.kfc_anno['noun'][idx])
        print("predicted:", verbs_classes[verb_indices[j]], nouns_classes[noun_indices[j]])

        # Store outputs for further representation
        d['pred_verb_indices'].append(verb_indices[j])
        d['pred_noun_indices'].append(noun_indices[j])
        d['pred_verb_classes'].append(verbs_classes[verb_indices[j]])
        d['pred_noun_classes'].append(nouns_classes[noun_indices[j]])
        d['gt_verb_indices'].append(val_dataset.kfc_anno['verb_class'][idx])
        d['gt_noun_indices'].append(val_dataset.kfc_anno['noun_class'][idx])
        d['gt_verb_classes'].append(val_dataset.kfc_anno['verb'][idx])
        d['gt_noun_classes'].append(val_dataset.kfc_anno['noun'][idx])

# Store results in csv
df = pd.DataFrame(d, columns= ['pred_verb_indices', 'pred_noun_indices','pred_verb_classes', 'pred_noun_classes',
                               'gt_verb_indices', 'gt_noun_indices', 'gt_verb_classes', 'gt_noun_classes','lat_load','lat_inference','loss'])
df.to_csv(os.path.join(os.path.dirname(args['model_ckpt']), os.path.basename(args['dataset_csv']).split(sep='.')[0] + "_model_results.csv"), index=False, header=True)