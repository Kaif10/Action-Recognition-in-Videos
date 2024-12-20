# Action-Recognition-in-Videos (Disseratation research)

This repo will serve as a playground where I investigate different approaches to solving the problem of action recognition in video.# and it takes inspiration from https://github.com/eriklindernoren/Action-Recognition

I will mainly use the UCF-101 dataset.
```
$ cd data/              
$ bash download_ucf101.sh     # Downloads the UCF-101 dataset (~7.2 GB)
$ unrar x UCF101.rar          # Unrars dataset
$ unzip ucfTrainTestlist.zip  # Unzip train / test split
$ python3 extract_frames.py   # Extracts frames from the video (~26.2 GB, go grab a coffee for this)
```
ConvLSTM
The only approach investigated so far. Enables action recognition in video by a bi-directional LSTM operating on frame embeddings extracted by a pre-trained ResNet-152 (ImageNet).

The model is composed of:

A convolutional feature extractor (ResNet-152) which provides a latent representation of video frames
A bi-directional LSTM classifier which based on the latent representation of the video predicts the activity depicted



```
$ python3 train.py  --dataset_path data/UCF-101-frames/ \
                    --split_path data/ucfTrainTestlist \
                    --num_epochs 200 \
                    --sequence_length 40 \
                    --img_dim 112 \
                    --latent_dim 512
```


```
$ python3 test_on_video.py  --video_path data/UCF-101/SoccerPenalty/v_SoccerPenalty_g01_c01.avi \
                            --checkpoint_model model_checkpoints/ConvLSTM_150.pth
```


The model reaches a classification accuracy of 99% accuracy on a randomly sampled test set, composed of 20% of the total amount of video sequences from UCF-101. 
