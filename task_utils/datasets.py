import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from torch.utils.data import random_split
import re
import functools
import subprocess
import numpy as np
from PIL import Image


class KFCDataset(Dataset):

    def __init__(self, csv_file, segment_count=8, transform=None, debug=False):
        # Store transform
        self.transform = transform
        self.debug = debug

        # Store csv file locally
        self.csv_file = csv_file
        self.root_path = os.path.dirname(csv_file)

        # Read csv
        self.kfc_anno = pd.read_table(csv_file, sep=",")

        # Load dataset in RAM (preliminary implementation for reduced datasets)
        self.loaded_dataset = []
        for file in self.kfc_anno['uid']:
            self.loaded_dataset.append(self.extract_frames(os.path.join(self.root_path, file), segment_count))

    def __len__(self):
        return len(self.kfc_anno['uid'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get segment
        item = self.loaded_dataset[idx]

        return self.transform(item), self.kfc_anno['verb_class'][idx], self.kfc_anno['noun_class'][idx]


    def extract_frames(self, video_file, num_frames=8):
        if self.debug:
            print("DATASET: Reading video file", video_file)
        try:
            os.makedirs(os.path.join(os.getcwd(), 'frames'))
        except OSError:
            pass

        output = subprocess.Popen(['ffmpeg', '-i', video_file],
                                  stderr=subprocess.PIPE).communicate()
        # Search and parse 'Duration: 00:05:24.13,' from ffmpeg stderr.
        re_duration = re.compile('Duration: (.*?)\.')
        duration = re_duration.search(str(output[1])).groups()[0]

        seconds = functools.reduce(lambda x, y: x * 60 + y,
                                   map(int, duration.split(':')))
        rate = num_frames / float(seconds)

        output = subprocess.Popen(['ffmpeg', '-i', video_file,
                                   '-vf', 'fps={}'.format(rate),
                                   '-vframes', str(num_frames),
                                   '-loglevel', 'panic',
                                   'frames/%d.jpg']).communicate()
        frame_paths = sorted([os.path.join('frames', frame)
                              for frame in os.listdir('frames')])

        frames = self.load_frames(frame_paths)
        subprocess.call(['rm', '-rf', 'frames'])
        return frames

    def load_frames(self, frame_paths, num_frames=8):
        frames = [Image.open(frame).convert('RGB') for frame in frame_paths]
        if len(frames) >= num_frames:
            return frames[::int(np.ceil(len(frames) / float(num_frames)))]
        else:
            raise ValueError('Video must have at least {} frames'.format(num_frames))