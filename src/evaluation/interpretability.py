"""Model interpretability utilities using Grad-CAM."""

import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from typing import Optional


def generate_gradcam_overlay(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_category: Optional[int] = None,
    use_cuda: bool = False,
) -> np.ndarray:
    """
    Generates a Grad-CAM overlay on an image.

    Args:
        model (torch.nn.Module): The model to generate CAM for.
        target_layer (torch.nn.Module): The target convolutional layer.
        input_tensor (torch.Tensor): The input image tensor (B, C, H, W).
        target_category (int, optional): The target category for CAM.
                                         If None, the predicted category is used. Defaults to None.
        use_cuda (bool, optional): Whether to use CUDA. Defaults to False.

    Returns:
        np.ndarray: The image with the CAM overlay.
    """
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=use_cuda)

    # Get the CAM which is a numpy array of shape (N, H, W)
    grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)

    # Take the first CAM, and un-normalize the input image
    cam_image = grayscale_cam[0, :]
    rgb_img = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Normalize image to be in [0, 1] range for visualization
    rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))

    # Create the overlay
    visualization = show_cam_on_image(rgb_img, cam_image, use_rgb=True)

    return visualization
