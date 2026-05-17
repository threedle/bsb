"""
Model wrappers for feature extraction from large pre-trained vision models.
"""
from abc import abstractmethod, ABC

import numpy as np
import torch
from torch.nn import functional as F
import torchvision.transforms as T


class ModelWrapper(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, interpolation, images):
        pass

    @abstractmethod
    def patch_size(self):
        pass


class DINOWrapper(ModelWrapper):
    def __init__(self, device=None, small=False, target_size=224):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not small:
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').to(device)
        else:
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').to(device)
        self.model.eval()

        self.target_size = target_size
        
        self.image_transforms = T.Compose([
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @staticmethod
    def pad(images):
        # image is a tensor of shape [B, C, H, W]
        # padding is a tuple of (left, right, top, bottom)

        target_size = max(*images.shape[2:])
        
        assert (
            len(images.shape) == 4
            and images.shape[1] == 3
            ), f"the images must be with shape BCHW."
        
        # Pad
        h, w = images.shape[-2:]
        padh = target_size - h
        padw = target_size - w
        images = F.pad(images, (0, padw, 0, padh))

        return images
    
    def postprocess_features(self, features, input_size, original_size):
        """
        Remove padding and upscale features to the original image size.

        Arguments:
          features (torch.Tensor): Batched embedding features, in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        features = F.interpolate(features, (self.target_size, self.target_size), mode="bilinear", align_corners=False)
        features = features[..., :input_size[0], :input_size[1]]
        features = F.interpolate(features, original_size, mode="bilinear", align_corners=False)
        return features
    
    def forward(self, images):
        images = self.image_transforms(images)
        out = self.model.forward_features(images)
        features = out['x_norm_patchtokens']          
        N, num_patches, C = features.shape
        features = torch.permute(features, (0, 2, 1))
        features = features.view(N, C, int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))
        return features

    def patch_size(self):
        return 14
