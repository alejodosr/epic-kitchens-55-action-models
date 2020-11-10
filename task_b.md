## Task B: Action Recognition (Alejandro Rodriguez Ramos)

_Use ffmpeg to split the videos you’ve collected into 8 second segments and label them by putting them into folders with label names (put, take, move)._

  -  The selected videos have been [one](https://www.youtube.com/watch?v=ZUr3DxYyTqI&t), [two](https://www.youtube.com/watch?v=c-uBjf988yE&t) and [three](https://www.youtube.com/watch?v=wiAYDb73Dbo&t). They do not provide ego-motion as in the epic kitchens dataset.
  - Every video has been splitted into segments of 8 seconds each and resampled to constant 30 FPS (using `ffmpeg`)
  - The dataset has been annotated with verbs "put, take, move" and nouns "meat, bag, sauce, potato, tortilla". It is available [here](https://upm365-my.sharepoint.com/:u:/g/personal/alejandro_rramos_alumnos_upm_es/EXBXPixxHlRKvTmXoOsp7gUBuOhIWmlKIejCB51FputDwg?e=uFLXyI) and it is built on 102 videos segments of 8 seconds each (79 for training and 23 for validation).
  
  The format of the train/val annotations was included in a single `.csv` file as:
  
  ```
  uid,verb,noun,action,verb_class,noun_class
  ```

_Next, split the folders into test and train sets._

1. _Transfer learn on one of the models above and report accuracy and loss on train_val test_val sets. Report the hardware you use for this task as well._
  
  - The training/testing pipeline has been implemented in [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/stable/) and TSN architecture has been selected for fine tuning/transfer learning.  

2. _Run inference on the videos from Task1 (exclude ones you used for training) and compare the difference between epic-kitchen models and your own trained model!_
3. _(optional) how would you improve accuracy and loss?_
4. _(optional) how would you incorporate speech data?_

_*Using GitHub commits for performing tasks above is a big plus._

### Environment and approach

  ```
  conda version : 4.9.1
  python version : 3.8.5.final.0
  pytorch_lightning: 1.0.5
  torch: 1.5.0
  platform : linux-64
  OS: Ubuntu 18.04 Bionic Beaver
  GPU: GeForce GTX 950M (VRAM: 2GB)
  ```
 
  - The models have been executed in their RGB version.
  - The selected backbone has been _Resnet 50_.
  - For training 1 crop is used.
  - Models are validated using 10 crops (center and corner crops as well as their horizontal flips) for each clip. The scores from these are averaged pre-softmax to produce a single clip-level score (as reported in [epic kitchen evaluation paper](https://arxiv.org/pdf/1908.00867.pdf)).
  - Each segment is divided into 8 snippets of 1 frame each for both training and validation.
  - Batch size has been set to 1 for training and validation.
  - Frames from train/vale datasets were preloaded in RAM directly from video, using `ffmpeg` to extract snippet's frames. This is possible due to the reduced size of the datasets.
  - The script used to train was `train_pl_model.py`, to generate the validation results was `test_pl_model.py` and to generate the metrics was `generate_metrics.py`.
  - The training evaluation was carried out with [tensorboard](https://www.tensorflow.org/tensorboard?hl=es-419) integration.
