import os
from random import randint
import numpy as np
import torch
from skimage import io, color
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
import pandas as pd
from numbers import Number
from typing import Container
from collections import defaultdict
from batchgenerators.utilities.file_and_folder_operations import *
from collections import OrderedDict
from torchvision.transforms import InterpolationMode
from einops import rearrange
import random
from utils.gradcam import SAMUSGradCAM
from models.model_dict import get_model
import matplotlib.pyplot as plt

def initialize_gradcam(model, opt):
    """Initialize GradCAM if visualization is enabled."""
        # Try different layer combinations for SAMUS
    possible_layers = [
            ['image_encoder.neck.3']
    ]
        
    model_layers = [name for name, _ in model.named_modules()]
        
    for target_layers in possible_layers:
        if all(layer in model_layers for layer in target_layers):
            print(f"Using GradCAM layers: {target_layers}")
            return SAMUSGradCAM(model, target_layers)
        
    return None
def visualize_gradcam_and_prompts(image, gradmask, pt, point_label, filename=""):
    """
    Visualize original image, GradCAM mask, and prompt points in one plot
    
    Args:
        image: Original image tensor (C, H, W) or numpy array
        gradmask: GradCAM binary mask (H, W)
        pt: Point coordinates array (N, 2)
        point_label: Point labels array (N,)
        filename: Optional filename for title
    """
    # Convert tensor to numpy if needed
    if hasattr(image, 'cpu'):
        img_np = image.cpu().numpy()
        if len(img_np.shape) == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
    else:
        img_np = image
    
    # Normalize image to 0-1 range
    if img_np.max() > 1:
        img_np = img_np / 255.0
    
    # Handle grayscale
    if len(img_np.shape) == 2 or img_np.shape[-1] == 1:
        img_np = np.stack([img_np.squeeze()]*3, axis=-1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_np, cmap='gray' if img_np.shape[-1] == 1 else None)
    axes[0].set_title(f'Original Image\n{filename}')
    axes[0].axis('off')
    
    # GradCAM mask
    axes[1].imshow(gradmask, cmap='hot', alpha=0.8)
    axes[1].imshow(img_np, alpha=0.3)
    axes[1].set_title('GradCAM Mask Overlay')
    axes[1].axis('off')
    
    # Image with prompt points
    axes[2].imshow(img_np)
    
    # Plot points with different colors for positive/negative
    for i, (point, label) in enumerate(zip(pt, point_label)):
        color = 'lime' if label == 1 else 'red'
        marker = 'o' if label == 1 else 'x'
        axes[2].scatter(point[0], point[1], c=color, s=100, marker=marker, 
                       edgecolors='black', linewidth=2)
    
    axes[2].set_title('Prompt Points\n(Green=Pos, Red=Neg)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(f"visualizations/vis_{filename.replace('.png', '')}.png", dpi=150, bbox_inches='tight')

    print(f"Visualization saved")
    
    # Try to show (might not work in some environments)
    try:
        plt.show()
    except:
        print("Display not available, check saved image instead")
    
    plt.close() 


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


def random_click(mask, class_id=1):
    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt = indices[np.random.randint(len(indices))]
    return pt[np.newaxis, :], [point_label]

def fixed_click(mask, class_id=1):
    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt = indices[len(indices)//2] 
    return pt[np.newaxis, :], [point_label]


def random_clicks(mask, class_id = 1, prompts_number=10):
    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt_index = np.random.randint(len(indices), size=prompts_number)
    pt = indices[pt_index]
    point_label = np.repeat(point_label, prompts_number)
    return pt, point_label

def pos_neg_clicks(mask, class_id=1, pos_prompt_number=1, neg_prompt_number=1):
    pos_indices = np.argwhere(mask == class_id)
    pos_indices[:, [0,1]] = pos_indices[:, [1,0]]
    
    # Get positive points or pad with zeros
    if len(pos_indices) > 0:
        pos_count = min(pos_prompt_number, len(pos_indices))
        pos_prompt_indices = np.random.choice(len(pos_indices), size=pos_count, replace=False)
        pos_prompt = pos_indices[pos_prompt_indices]
    else:
        pos_count = 0
        pos_prompt = np.zeros((0, 2), dtype=int)
    
    neg_indices = np.argwhere(mask != class_id)
    neg_indices[:, [0,1]] = neg_indices[:, [1,0]]
    
    # Get negative points or pad with zeros
    if len(neg_indices) > 0:
        neg_count = min(neg_prompt_number, len(neg_indices))
        neg_prompt_indices = np.random.choice(len(neg_indices), size=neg_count, replace=False)
        neg_prompt = neg_indices[neg_prompt_indices]
    else:
        neg_count = 0
        neg_prompt = np.zeros((0, 2), dtype=int)
    
    # CRITICAL: Always return FIXED SIZE arrays
    total_points = pos_prompt_number + neg_prompt_number
    pt = np.zeros((total_points, 2), dtype=int)
    point_label = np.zeros(total_points, dtype=int)
    
    # Fill in actual points
    if pos_count > 0:
        pt[:pos_count] = pos_prompt
        point_label[:pos_count] = 1
    if neg_count > 0:
        pt[pos_prompt_number:pos_prompt_number+neg_count] = neg_prompt
        point_label[pos_prompt_number:pos_prompt_number+neg_count] = 0
    
    return pt, point_label

def random_bbox(mask, class_id=1, img_size=256):
    # return box = np.array([x1, y1, x2, y2])
    indices = np.argwhere(mask == class_id) # Y X
    indices[:, [0,1]] = indices[:, [1,0]] # x, y
    if indices.shape[0] ==0:
        return np.array([-1, -1, img_size, img_size])

    minx = np.min(indices[:, 0])
    maxx = np.max(indices[:, 0])
    miny = np.min(indices[:, 1])
    maxy = np.max(indices[:, 1])

    classw_size = maxx-minx+1
    classh_size = maxy-miny+1

    shiftw = randint(int(0.95*classw_size), int(1.05*classw_size))
    shifth = randint(int(0.95*classh_size), int(1.05*classh_size))
    shiftx = randint(-int(0.05*classw_size), int(0.05*classw_size))
    shifty = randint(-int(0.05*classh_size), int(0.05*classh_size))

    new_centerx = (minx + maxx)//2 + shiftx
    new_centery = (miny + maxy)//2 + shifty

    minx = np.max([new_centerx-shiftw//2, 0])
    maxx = np.min([new_centerx+shiftw//2, img_size-1])
    miny = np.max([new_centery-shifth//2, 0])
    maxy = np.min([new_centery+shifth//2, img_size-1])

    return np.array([minx, miny, maxx, maxy])

def fixed_bbox(mask, class_id = 1, img_size=256):
    indices = np.argwhere(mask == class_id) # Y X (0, 1)
    indices[:, [0,1]] = indices[:, [1,0]]
    if indices.shape[0] ==0:
        return np.array([-1, -1, img_size, img_size])
    minx = np.min(indices[:, 0])
    maxx = np.max(indices[:, 0])
    miny = np.min(indices[:, 1])
    maxy = np.max(indices[:, 1])
    return np.array([minx, miny, maxx, maxy])

def random_clicks(mask, class_id = 1, prompts_number=10):
    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt_index = np.random.randint(len(indices), size=prompts_number)
    pt = indices[pt_index]
    point_label = np.repeat(point_label, prompts_number)
    return pt, point_label

def gradcam_low_activation_mask(cam_np, low_threshold=0.3):
    """
    Create binary mask from least activated regions of GradCAM.
    
    Args:
        cam_np: GradCAM numpy array (normalized 0-1)
        low_threshold: threshold below which regions are considered "least activated"
    
    Returns:
        binary_mask: binary mask where 1 = least activated regions
    """
    # Normalize CAM to 0-1 range
    cam_normalized = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
    
    # Create binary mask for least activated regions
    binary_mask = (cam_normalized < low_threshold).astype(np.uint8)
    
    return binary_mask

def mask_gradcam_clicks(mask, gradcam, class_id=1, pos_count=3, neg_count=3):
    """Generate positive clicks from mask and negative clicks from GradCAM low activation regions."""
    # Get positive clicks from mask
    pos_pts, pos_labels = random_clicks(mask, class_id, pos_count)
    
    # Get negative clicks from GradCAM low activation regions
    gradcam_mask = gradcam_low_activation_mask(gradcam)
    neg_pts, _ = random_clicks(gradcam_mask, class_id=1, prompts_number=neg_count)
    
    # Combine points and labels
    pts = np.vstack([pos_pts, neg_pts])
    labels = np.hstack([pos_labels, np.zeros(neg_count, dtype=int)])
    
    return pts, labels


def gradcam_to_binary_mask(cam, img_height, img_width):
    """Convert GradCAM to binary mask with morphological postprocessing."""
    # Hyperparameters 
    threshold = 0.94
    morph_kernel_size = 3
    apply_opening = True
    apply_closing = True
    
    # Convert CAM to grayscale and normalize
    if len(cam.shape) > 2:
        cam_gray = cv2.cvtColor(cam, cv2.COLOR_RGB2GRAY) if cam.shape[-1] == 3 else cam.squeeze()
    else:
        cam_gray = cam
    cam_normalized = (cam_gray - cam_gray.min()) / (cam_gray.max() - cam_gray.min())
    
    # RESIZE CAM to match original image dimensions
    cam_resized = cv2.resize(cam_normalized, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    
    # Apply threshold to create binary mask
    binary_mask = (cam_resized > threshold).astype(np.uint8)
    
    # Apply morphological operations (postprocessing)
    if apply_opening or apply_closing:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        if apply_opening:
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        if apply_closing:
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    return binary_mask


def generate_gradcam_mask(image, model, gradcam_obj, opt):
    """Generate binary mask from GradCAM for click function usage"""
    try:
        # Add batch dimension and move to device
        img_tensor = image.unsqueeze(0).to(opt.device)
        b, c, h, w = img_tensor.shape
        
        # Generate grid seed points for GradCAM
        grid_size = getattr(opt, 'grid_seed_count', 4)
        xs = torch.linspace(0, w - 1, grid_size)
        ys = torch.linspace(0, h - 1, grid_size)
        grid_coords = torch.cartesian_prod(xs, ys).flip(-1).to(dtype=torch.float32, device=opt.device)
        seed_coords = grid_coords.unsqueeze(0)
        seed_labels = torch.ones(grid_coords.size(0), dtype=torch.int, device=opt.device).unsqueeze(0)
        
        # Generate CAM
        cam = gradcam_obj.generate_cam(img_tensor, (seed_coords, seed_labels))

        
        # Ensure cam is properly converted to numpy
        if isinstance(cam, torch.Tensor):
            cam_np = cam.detach().cpu().numpy()
            
        else:
            cam_np = np.array(cam)
        
        # Handle any remaining squeeze operations
        cam_np = cam_np.squeeze()
        
        # Convert to binary mask
        binary_mask = gradcam_to_binary_mask(cam_np, h, w)
        return binary_mask,cam_np
    except Exception as e:
        print(f"GradCAM mask generation failed: {e}")
        # Return center-focused mask as fallback
        h, w = image.shape[-2:]
        fallback_mask = np.zeros((h, w), dtype=np.uint8)
        fallback_mask[h//4:3*h//4, w//4:3*w//4] = 1
        return fallback_mask


class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """

    def __init__(self, img_size=256, low_img_size=256, ori_size=256, crop=(32, 32), p_flip=0.0, p_rota=0.0, p_scale=0.0, p_gaussn=0.0, p_contr=0.0,
                 p_gama=0.0, p_distor=0.0, color_jitter_params=(0.1, 0.1, 0.1, 0.1), p_random_affine=0,
                 long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.img_size = img_size
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask
        self.low_img_size = low_img_size
        self.ori_size = ori_size

    def __call__(self, image, mask):
        #  gamma enhancement
        if np.random.rand() < self.p_gama:
            c = 1
            g = np.random.randint(10, 25) / 10.0
            # g = 2
            image = (np.power(image / 255, 1.0 / g) / c) * 255
            image = image.astype(np.uint8)
        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random horizontal flip
        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)
        # random rotation
        if np.random.rand() < self.p_rota:
            angle = T.RandomRotation.get_params((-30, 30))
            image, mask = F.rotate(image, angle), F.rotate(mask, angle)
        # random scale and center resize to the original size
        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(1, 1.3)
            new_h, new_w = int(self.img_size * scale), int(self.img_size * scale)
            image, mask = F.resize(image, (new_h, new_w), InterpolationMode.BILINEAR), F.resize(mask, (new_h, new_w), InterpolationMode.NEAREST)
            # image = F.center_crop(image, (self.img_size, self.img_size))
            # mask = F.center_crop(mask, (self.img_size, self.img_size))
            i, j, h, w = T.RandomCrop.get_params(image, (self.img_size, self.img_size))
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random add gaussian noise
        if np.random.rand() < self.p_gaussn:
            ns = np.random.randint(3, 15)
            noise = np.random.normal(loc=0, scale=1, size=(self.img_size, self.img_size)) * ns
            noise = noise.astype(int)
            image = np.array(image) + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = F.to_pil_image(image.astype('uint8'))
        # random change the contrast
        if np.random.rand() < self.p_contr:
            contr_tf = T.ColorJitter(contrast=(0.8, 2.0))
            image = contr_tf(image)
        # random distortion
        if np.random.rand() < self.p_distortion:
            distortion = T.RandomAffine(0, None, None, (5, 30))
            image = distortion(image)
        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)
        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)
        # transforming to tensor
        image, mask = F.resize(image, (self.img_size, self.img_size), InterpolationMode.BILINEAR), F.resize(mask, (self.ori_size, self.ori_size), InterpolationMode.NEAREST)
        low_mask = F.resize(mask, (self.low_img_size, self.low_img_size), InterpolationMode.NEAREST)
        image = F.to_tensor(image)

        if not self.long_mask:
            mask = F.to_tensor(mask)
            low_mask = F.to_tensor(low_mask)
        else:
            mask = to_long_tensor(mask)
            low_mask = to_long_tensor(low_mask)
        return image, mask, low_mask


class ImageToImage2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
                |-- MainPatient
                    |-- train.txt
                    |-- val.txt
                    |-- text.txt 
                        {subtaski}/{imgname}
                    |-- class.json
                |-- subtask1
                    |-- img
                        |-- img001.png
                        |-- img002.png
                        |-- ...
                    |-- label
                        |-- img001.png
                        |-- img002.png
                        |-- ...
                |-- subtask2
                    |-- img
                        |-- img001.png
                        |-- img002.png
                        |-- ...
                    |-- label
                        |-- img001.png
                        |-- img002.png
                        |-- ... 
                |-- subtask...   

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str,model, split='train',modelname='SAMUS', joint_transform: Callable = None, img_size=256, prompt = "click", class_id=1,
                 one_hot_mask: int = False,opt=None) -> None:
        self.dataset_path = dataset_path
        self.one_hot_mask = one_hot_mask
        self.split = split
        self.modelname=modelname
        self.model=model
        self.opt=opt
        id_list_file = os.path.join(dataset_path, 'MainPatient/{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.prompt = prompt
        self.img_size = img_size
        self.class_id = class_id
        self.class_dict_file = os.path.join(dataset_path, 'MainPatient/class.json')
        with open(self.class_dict_file, 'r') as load_f:
            self.class_dict = json.load(load_f)
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))
        if self.modelname=='GradSAMUS' or self.modelname=='PNGradSAMUS':
            self.gradcam_obj = initialize_gradcam(model, opt)


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        if "test" in self.split:
            sub_path, filename = id_.split('/')[0], id_.split('/')[1]
            # class_id0, sub_path, filename = id_.split('/')[0], id_.split('/')[1], id_.split('/')[2]
            # self.class_id = int(class_id0)
        else:
            class_id0, sub_path, filename = id_.split('/')[0], id_.split('/')[1], id_.split('/')[2]
        img_path = os.path.join(os.path.join(self.dataset_path, sub_path), 'img')
        label_path = os.path.join(os.path.join(self.dataset_path, sub_path), 'label')
        image = cv2.imread(os.path.join(img_path, filename + '.png'), 0)
        mask = cv2.imread(os.path.join(label_path, filename + '.png'), 0)
        classes = self.class_dict[sub_path]
        if classes == 2:
            mask[mask > 1] = 1

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)  
        if self.joint_transform:
            image, mask, low_mask = self.joint_transform(image, mask)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

         # --------- make the point prompt -----------------
        if self.prompt == 'click':
            point_label = 1
            if 'train' in self.split:
                class_id = int(class_id0)
            elif 'val' in self.split:
                class_id = int(class_id0)
            else:
                class_id = self.class_id
                
            if 'train' in self.split:
                if self.modelname=='SAMUS':
                    pt, point_label = random_click(np.array(mask), class_id)
                elif self.modelname=='PNSAMUS':
                    pt,point_label = pos_neg_clicks(np.array(mask), class_id, pos_prompt_number=1, neg_prompt_number=1)
                elif self.modelname=='GradSAMUS':
                    gradmask = generate_gradcam_mask(image, self.model, self.gradcam_obj, self.opt)
                    pt, point_label = random_click(np.array(gradmask), class_id)
                elif self.modelname=='PNGradSAMUS':
                    gradmask = generate_gradcam_mask(image, self.model, self.gradcam_obj, self.opt)
                    pt,point_label = pos_neg_clicks(np.array(gradmask), class_id, pos_prompt_number=1, neg_prompt_number=1)

            elif 'val' in self.split: 
                if self.modelname=='SAMUS':
                    pt, point_label = fixed_click(np.array(mask), class_id)  # or random_click for consistency
                elif self.modelname=='PNSAMUS':
                    pt, point_label = pos_neg_clicks(np.array(mask), class_id, pos_prompt_number=1, neg_prompt_number=1)
                elif self.modelname=='GradSAMUS':
                    gradmask = generate_gradcam_mask(image, self.model, self.gradcam_obj, self.opt)
                    pt, point_label = fixed_click(np.array(gradmask), class_id)  # or random_click
                elif self.modelname=='PNGradSAMUS':
                    gradmask = generate_gradcam_mask(image, self.model, self.gradcam_obj, self.opt)
                    pt,point_label = pos_neg_clicks(np.array(gradmask), class_id, pos_prompt_number=1, neg_prompt_number=1)
                else:
                    pt, point_label = fixed_click(np.array(mask), class_id)  # fallback
            elif 'test' in self.split: 
                if self.modelname=='SAMUS':
                    pt, point_label = fixed_click(np.array(mask), class_id)  # or random_click for consistency
                    print("Samus point")
                elif self.modelname=='PNSAMUS':
                    pt, point_label = pos_neg_clicks(np.array(mask), class_id, pos_prompt_number=1, neg_prompt_number=1)
                    print("PN point")
                elif self.modelname=='GradSAMUS':
                    gradmask,cam = generate_gradcam_mask(image, self.model, self.gradcam_obj, self.opt)
                    # pt, point_label = random_clicks(np.array(gradmask), class_id,5)  # or random_click
                    # visualize_gradcam_and_prompts(image, gradmask, pt, point_label, filename)
                    pt, point_label=mask_gradcam_clicks(mask, cam, class_id=1, pos_count=5, neg_count=0)
                    print(pt,point_label)
                elif self.modelname=='PNGradSAMUS':
                    gradmask = generate_gradcam_mask(image, self.model, self.gradcam_obj, self.opt)
                    pt,point_label = pos_neg_clicks(np.array(gradmask), class_id, pos_prompt_number=1, neg_prompt_number=1)
                    print("pngrad point")
                else:
                    pt, point_label = random_click(np.array(mask), class_id)  # fallback
            else:
                pt, point_label = fixed_click(np.array(mask), class_id)
            mask[mask!=class_id] = 0
            mask[mask==class_id] = 1
            low_mask[low_mask!=class_id] = 0
            low_mask[low_mask==class_id] = 1
            point_labels = np.array(point_label)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        low_mask = low_mask.unsqueeze(0)
        mask = mask.unsqueeze(0)
        return {
            'image': image,
            'label': mask,
            'p_label': point_labels,
            'pt': pt,
            'low_mask':low_mask,
            'image_name': filename + '.png',
            'class_id': class_id,
            }


class Logger:
    def __init__(self, verbose=False):
        self.logs = defaultdict(list)
        self.verbose = verbose

    def log(self, logs):
        for key, value in logs.items():
            self.logs[key].append(value)

        if self.verbose:
            print(logs)

    def get_logs(self):
        return self.logs

    def to_csv(self, path):
        pd.DataFrame(self.logs).to_csv(path, index=None)


