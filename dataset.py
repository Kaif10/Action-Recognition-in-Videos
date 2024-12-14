import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

class Dataset(Dataset):
    def __init__(self, dataset_path, split_path, split_number, input_shape, sequence_length, training):
        self.dataset_path = dataset_path
        self.training = training
        self.label_index = self._extract_label_mapping(split_path)
        self.sequences = self._extract_sequence_paths(dataset_path, split_path, split_number, training)
        self.sequence_length = sequence_length
        self.label_names = sorted(list(set([self._activity_from_path(seq_path) for seq_path in self.sequences])) )
        self.num_classes = len(self.label_names)
        self.transform = transforms.Compose([
            transforms.Resize(input_shape[-2:], Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def _extract_label_mapping(self, split_path="data/ucfTrainTestlist"):
        with open(os.path.join(split_path, "classInd.txt")) as file:
            lines = file.read().splitlines()
        label_mapping = {}
        for line in lines:
            if not line.strip():
                continue  # Skip empty lines
            label, action = line.split()
            label_mapping[action] = int(label) - 1
        return label_mapping

    def _extract_sequence_paths(self, dataset_path, split_path="data/ucfTrainTestlist", split_number=1, training=True):
        """ Extracts paths to sequences given the specified train / test split """
        assert split_number in [1, 2, 3], "Split number has to be one of {1, 2, 3}"
        fn = f"trainlist0{split_number}.txt" if training else f"testlist0{split_number}.txt"
        split_path = os.path.join(split_path, fn)
        with open(split_path) as file:
            lines = file.read().splitlines()
        sequence_paths = []
        for line in lines:
            seq_name = line.split(".avi")[0]
            sequence_paths.append(os.path.join(dataset_path, seq_name))
        return sequence_paths

    def _activity_from_path(self, path):
        """ Extracts activity name from filepath """
        path = path.replace("\\", "/")  # Normalize path separators
        return path.split("/")[-2]

    def _pad_to_length(self, sequence):
        if len(sequence) == 0:
            raise ValueError(f"No frames found in sequence for video in {self.dataset_path}")
        elif len(sequence) < self.sequence_length:
            # If too few frames, repeat the last frame
            sequence += [sequence[-1]] * (self.sequence_length - len(sequence))
        return sequence[:self.sequence_length]

    def __getitem__(self, index):
        sequence_path = self.sequences[index % len(self)]
        image_paths = sorted(glob.glob(f"{sequence_path}/*.jpg"))
        
        # Skip directories with no images
        if not image_paths:
            print(f"No images found in directory: {sequence_path}. Skipping this directory.")
            return self.__getitem__((index + 1) % len(self))  # Try the next index

        num_frames = len(image_paths)
        
        # Handle cases where there are fewer frames than required for a sequence
        if num_frames < self.sequence_length:
            print(f"Not enough frames in sequence: {sequence_path}. Expected {self.sequence_length}, but found {num_frames}. Padding with last frame.")
            # Pad the sequence with the last frame to meet the required sequence length
            image_paths = image_paths + [image_paths[-1]] * (self.sequence_length - num_frames)
            num_frames = self.sequence_length  # Set num_frames to the sequence length after padding

        # If training, apply random sampling for frames
        if self.training:
            sample_interval = max(1, num_frames // self.sequence_length)  # At least 1 frame per interval
            
            # Ensure that the range for np.random.randint is valid
            if num_frames - self.sequence_length * sample_interval + 1 <= 0:
                print(f"Not enough frames to sample. Skipping this sequence: {sequence_path}")
                return self.__getitem__((index + 1) % len(self))  # Skip this sequence and try the next one

            start_i = np.random.randint(0, num_frames - self.sequence_length * sample_interval + 1)
            flip = np.random.random() < 0.5
        else:
            # If testing, sample frames uniformly from the sequence
            start_i = 0
            sample_interval = 1  # Just take frames sequentially
            flip = False

        # Extract frames as tensors
        image_sequence = []
        for i in range(start_i, num_frames, sample_interval):
            if len(image_sequence) < self.sequence_length:
                image_tensor = self.transform(Image.open(image_paths[i]))
                if flip:
                    image_tensor = torch.flip(image_tensor, (-1,))  # Flip horizontally
                image_sequence.append(image_tensor)

        # Pad sequence to the required length
        while len(image_sequence) < self.sequence_length:
            image_sequence.append(image_sequence[-1])  # Repeat last frame if necessary

        image_sequence = torch.stack(image_sequence)
        target = self.label_index[self._activity_from_path(sequence_path)]
        return image_sequence, target

    def __len__(self):
        return len(self.sequences)
