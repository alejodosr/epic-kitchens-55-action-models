import argparse
from torchvision.transforms import Compose
import torch.nn.functional as F
from transforms import GroupScale, GroupCenterCrop, GroupOverSample, Stack, ToTorchFormatTensor, GroupNormalize
from task_utils.utils import *
import pandas as pd
import time

import torch.hub

parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--vid_path', type=str, default="./videos/")
parser.add_argument('--res_path', type=str, default="./results/")

args = parser.parse_args()

# Repo path
repo = 'epic-kitchens/action-models'

# Read classes
nouns_classes = pd.read_table("./epic-kitchens-55-annotations/EPIC_noun_classes.csv", sep=",")['class_key']
verbs_classes = pd.read_table("./epic-kitchens-55-annotations/EPIC_verb_classes.csv", sep=",")['class_key']

# Read annotations
kfc_anno = pd.read_table(args.vid_path + "kfc_test_annotations.csv", sep=",")

# Models load
class_counts = (125, 352)
segment_count = 8
base_model = 'resnet50'
tsn = torch.hub.load(repo, 'TSN', class_counts, segment_count, 'RGB',
                     base_model=base_model,
                     pretrained='epic-kitchens', force_reload=True)
trn = torch.hub.load(repo, 'TRN', class_counts, segment_count, 'RGB',
                     base_model=base_model,
                     pretrained='epic-kitchens')
mtrn = torch.hub.load(repo, 'MTRN', class_counts, segment_count, 'RGB',
                      base_model=base_model,
                      pretrained='epic-kitchens')
tsm = torch.hub.load(repo, 'TSM', class_counts, segment_count, 'RGB',
                     base_model=base_model,
                     pretrained='epic-kitchens')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = [tsn, trn, mtrn, tsm]
model_names = ["tsn", "trn", "mtrn", "tsm"]

# Loss criterion
criterion = torch.nn.CrossEntropyLoss().to(device)

# Show all entrypoints and their help strings
for entrypoint in torch.hub.list(repo):
    print(entrypoint)
    print(torch.hub.help(repo, entrypoint))

for j, model in enumerate(models):
    # Move to GPU if available and set to evaluation
    model.eval()
    model.to(device)

    # Define the transform
    batch_size = 1
    snippet_length = 1  # Number of frames composing the snippet, 1 for RGB, 5 for optical flow
    snippet_channels = 3  # Number of channels in a frame, 3 for RGB, 2 for optical flow
    height, width = 224, 224

    crop_count = 10

    if crop_count == 1:
        cropping = Compose([
            GroupScale(model.scale_size),
            GroupCenterCrop(model.input_size),
        ])
    elif crop_count == 10:
        cropping = GroupOverSample(model.input_size, model.scale_size)
    else:
        raise ValueError("Only 1 and 10 crop_count are supported while we got {}".format(crop_count))

    transform = Compose([
        cropping,
        Stack(roll=base_model == base_model),
        ToTorchFormatTensor(div=base_model != base_model),
        GroupNormalize(model.input_mean, model.input_std),
    ])

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

    for i in range(len(kfc_anno['video_id'])):
        # Extract 1 frame per snippet
        print("Processing video: ", kfc_anno['video_id'][i] + ".mp4")
        tic = time.clock()
        frames = extract_frames(args.vid_path + kfc_anno['video_id'][i] + ".mp4", segment_count)

        print("No. of video frames: " + str(len(frames)))

        # Tranform frames
        inputs = transform(frames).to(device)

        toc = time.clock()
        d['lat_load'].append(toc-tic)
        print("Data loading latency: " + str(toc-tic))

        # or just call the object to classify inputs in a single forward pass
        tic = time.clock()
        with torch.no_grad():
            verb_logits, noun_logits = model(inputs)
            verb_targets = (torch.ones(10, dtype=torch.long) * kfc_anno['verb_class'][i]).to(device)
            noun_targets = (torch.ones(10, dtype=torch.long) * kfc_anno['noun_class'][i]).to(device)
            loss = (criterion(verb_logits, verb_targets) + criterion(noun_logits, noun_targets)).cpu().detach().numpy()
            d['loss'].append(loss)

        verb_indices = torch.argmax(F.softmax(torch.mean(verb_logits, dim=0, keepdim=True), dim=1), dim=1).cpu().detach().numpy()
        noun_indices = torch.argmax(F.softmax(torch.mean(noun_logits, dim=0, keepdim=True), dim=1), dim=1).cpu().detach().numpy()
        toc = time.clock()
        d['lat_inference'].append(toc-tic)
        print("Inference latency: " + str(toc-tic))

        for idx, verb in enumerate(verb_indices):
            print("ground truth:", kfc_anno['verb'][i], kfc_anno['noun'][i])
            print("predicted:", verbs_classes[verb_indices[idx]], nouns_classes[noun_indices[idx]])

            # Store otuputs for further representation
            d['pred_verb_indices'].append(verb_indices[idx])
            d['pred_noun_indices'].append(noun_indices[idx])
            d['pred_verb_classes'].append(verbs_classes[verb_indices[idx]])
            d['pred_noun_classes'].append(nouns_classes[noun_indices[idx]])
            d['gt_verb_indices'].append(kfc_anno['verb_class'][i])
            d['gt_noun_indices'].append(kfc_anno['noun_class'][i])
            d['gt_verb_classes'].append(kfc_anno['verb'][i])
            d['gt_noun_classes'].append(kfc_anno['noun'][i])

    # Store results in csv
    df = pd.DataFrame(d, columns= ['pred_verb_indices', 'pred_noun_indices','pred_verb_classes', 'pred_noun_classes',
                                   'gt_verb_indices', 'gt_noun_indices', 'gt_verb_classes', 'gt_noun_classes','lat_load','lat_inference','loss'])
    df.to_csv(args.res_path + model_names[j] + "_results.csv", index=False, header=True)