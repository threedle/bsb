import copy
from collections import defaultdict
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
import torch
import torchvision
from modules.obj_loader import ObjLoader
from itertools import permutations, product
from pathlib import Path
from tqdm import tqdm
from torch.autograd import grad
from modules.utils import device, loadmesh, setcolor_mesh, load_state_dict, show_mask, show_anns
from torchvision import transforms
from modules.render import save_renders, Renderer
import matplotlib.pyplot as plt
from SAM_repo.segment_anything.utils.transforms import ResizeLongestSide
from SAM_repo.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from modules.sam import load_img_seg_model
from modules.dino import DINOWrapper
from modules.encoder import Encoder
from modules.dataset import EncoderDataset
from torch.utils.data import Dataset, DataLoader


def test_encoder(args, sam):
    render = Renderer(dim=(64, 64), radius=args.radius)
    render_high = Renderer(dim=(args.render_res, args.render_res), radius=args.radius)
   
    targetmesh = args.mesh  
    target_rgb = torch.zeros_like(args.mesh.vertices) 
    target_rgb[:] = torch.tensor(args.mesh_color).to(device)
    
    setcolor_mesh(targetmesh, target_rgb)
    targetmesh.face_attributes = targetmesh.face_attributes.float()

    # Generate a test view
    test_elev = np.array([args.test_elev_deg], dtype=np.float32)/180 * np.pi
    test_azim = np.array([args.test_azim_deg], dtype=np.float32)/360 * 2*np.pi
    target_rendered_images, elev, azim, mask_mesh_2d = render_high.render_views(targetmesh, num_views=1,
                                                                show=True,
                                                                center_elev=torch.tensor(test_elev[0:1]),
                                                                center_azim=torch.tensor(test_azim[0:1]),
                                                                random_views=False,
                                                                std=args.frontview_std,
                                                                return_views=True,
                                                                return_mask=True,
                                                                lighting=True,
                                                                background=torch.ones(3).to(device))

    save_renders(args.encoder_model_dir, 0, target_rendered_images, name='test_image.png')

    # Read the saved image
    image = cv2.imread(os.path.join(args.encoder_model_dir, 'test_image.png'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Check SAM encoded feature
    sam_feature_2D = sam_features_encoder(sam, image)
    
    # 3D-consistent features
    load_features = True
    if load_features:
        pred_f = torch.load(os.path.join(args.encoder_model_dir, 'pred_f.pth'))
    else:
        encoder_checkpoint = torch.load(os.path.join(args.encoder_model_dir, 'encoder_checkpoint.pth'))
        encoder = Encoder(args.depth, args.width, out_dim=args.n_classes, positional_encoding=args.positional_encoding,
                          sigma=args.sigma, network_verbose=args.network_verbose).to(device)
        encoder.load_state_dict(encoder_checkpoint['encoder_state_dict'])
        pred_f = encoder(args.mesh.vertices)

    # Color and render the learned 3D features at the same viewing angle
    sampled_mesh = args.mesh
    setcolor_mesh(sampled_mesh, pred_f)
    rendered_images, elev, azim, mask = render.render_views(sampled_mesh, num_views=1,
                                                            random_views = False,
                                                            center_azim=azim,
                                                            center_elev=elev,
                                                            std=args.frontview_std,
                                                            return_views=True,
                                                            return_features=True,
                                                            lighting=False,
                                                            background=torch.ones(256).to(device),
                                                            return_mask = True)
    
    # Calculate the loss
    mask_mesh = (rendered_images != 1.0)
    rendered_images[~mask_mesh] = sam_feature_2D[~mask_mesh] 
    loss = (sam_feature_2D[mask_mesh] - rendered_images[mask_mesh]).pow(2).sum()/mask_mesh.sum()
    print('test loss: %f' % loss)

    mask_generator = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=32,
                    pred_iou_thresh=0.90,
                    stability_score_thresh=0.92,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=150,  # Requires open-cv to run post-processing
                )
    
    mask_generator_org = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=32,
                    pred_iou_thresh=0.90,
                    stability_score_thresh=0.92,#
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=150,  # Requires open-cv to run post-processing
                )
   
    # Pass the 3D feature to SAM
    mask_generator.feature_3D = rendered_images
    
    # Pass the original 2D feature to SAM
    mask_generator_org.feature_3D = sam_feature_2D
    
    # Generate masks
    masks = mask_generator.generate(image)
    masks_org = mask_generator_org.generate(image)
    
    for i in range(len(masks)):
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        show_mask(masks[i]['segmentation'], plt.gca())
        plt.axis('off')
        plt.savefig(os.path.join(args.encoder_model_dir, 'mff_automasks_{}_test_{}.png'.format(args.mesh.name, i)))
        plt.close()

    for i in range(len(masks_org)):
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        show_mask(masks_org[i]['segmentation'], plt.gca())
        plt.axis('off')

        plt.savefig(os.path.join(args.encoder_model_dir, 'sam_automasks_{}_test_{}.png'.format(args.mesh.name, i)))
        plt.close()

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks, plt)
    plt.axis('off')

    plt.savefig(os.path.join(args.encoder_model_dir, 'mff_automasks_{}_test.png'.format(args.mesh.name)))
    plt.close()

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks, plt)
    plt.axis('off')

    plt.savefig(os.path.join(args.encoder_model_dir, 'sam_automasks_{}_test.png'.format(args.mesh.name)))
    plt.close()

    return mask_mesh_2d, len(masks)


def generate_random_views(args):
    # Constrain most sources of randomness
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.cuda.empty_cache()
    target_rgb = torch.zeros_like(args.mesh.vertices) 
    target_rgb[:] = torch.tensor(args.mesh_color)
    targetmesh = args.mesh  
    setcolor_mesh(targetmesh, target_rgb)
    targetmesh.face_attributes = targetmesh.face_attributes.float()

    elev_list, azim_list = [], []
    render_high = Renderer(dim=(args.render_res, args.render_res), radius = args.radius)

    # Make directory
    if not os.path.exists(os.path.join(args.encoder_data_dir, '{}/'.format(args.render_res))):
        os.makedirs(os.path.join(args.encoder_data_dir, '{}/'.format(args.render_res)))

    if not os.path.exists(os.path.join(args.encoder_data_dir, 'sam_f/{}/'.format(args.render_res))):
        os.makedirs(os.path.join(args.encoder_data_dir, 'sam_f/{}/'.format(args.render_res)))

    # Resume
    if os.path.exists(os.path.join(args.encoder_data_dir, '{}/random_viewing_angles.pt'.format(args.render_res))):
        data = torch.load(os.path.join(args.encoder_data_dir, '{}/random_viewing_angles.pt'.format( args.render_res)))
        elev_list = data['elev']
        azim_list = data['azim']
        sampled_mesh = args.mesh

    # Optimization loop
    for i in tqdm(len(elev_list)+np.arange(args.n_views-len(elev_list))):

        # Elevation: [-pi/2, pi/2]
        elev_rand = -np.pi/2 + torch.rand(1) * np.pi
        # Azimuth: [0, 2pi]
        azim_rand = torch.rand(1) * 2 * np.pi
        target_rendered_images, elev, azim = render_high.render_views(targetmesh, num_views=1,
                                                                    random_views = True,
                                                                    std=args.frontview_std,
                                                                    return_views=True,
                                                                    lighting=True,
                                                                    background=torch.tensor(args.background).to(device))
        elev_list.append(elev)
        azim_list.append(azim)
        
        save_renders(os.path.join(args.encoder_data_dir, '{}/'.format(args.render_res)), 0, target_rendered_images, name='target_save_{}.png'.format(int(i)))


def encode_random_views(args, model_type, model):
    print(f"Generate encoded image features for model %s" % model_type)

    save_dir = os.path.join(args.encoder_data_dir, '{}_f/{}/'.format(model_type, args.render_res))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = torch.load(os.path.join(args.encoder_data_dir, '{}/random_viewing_angles.pt'.format( args.render_res)))
    elev_list = data['elev']
    azim_list = data['azim']
    
    # loop over exsiting views
    for i in tqdm(np.arange(len(elev_list))):
        elev = elev_list[i]
        azim = azim_list[i]

        # Generate encoded image features
        save_path = os.path.join(save_dir, 'target_{}_f_{}.pt'.format(model_type, i))
        
        if not os.path.exists(save_path):     
            image = cv2.imread(os.path.join(args.encoder_data_dir, '{}/target_save_{}.png'.format(args.render_res, i)))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if model_type == 'sam':
                model_f = sam_features_encoder(model, image)
            elif model_type == 'dino':
                model_f = dino_features_encoder(model, image, args.target_image_size)
            else:
                raise ValueError("Unsupported model type: %s." % model_type)
            
            torch.save({'model_f':model_f, 'elev':elev, 'azim':azim}, save_path)


def sam_features_encoder(sam, image):
    # SAM encoder: transform and encode
    with torch.no_grad():
        transform = ResizeLongestSide(sam.image_encoder.img_size)
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=sam.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        original_size = image.shape[:2]
        input_size = tuple(input_image_torch.shape[-2:])
        input_image = sam.preprocess(input_image_torch)
        sam_features = sam.image_encoder(input_image)
        return sam_features


def preprocess_image(image, target_image_size):
    transform = ResizeLongestSide(target_image_size)
    input_image = transform.apply_image(image)
    
    input_image_torch = torch.as_tensor(input_image, device=device, dtype=torch.float32)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :] / 255.0
    input_size = input_image_torch.shape[-2:]

    input_image_torch_square = DINOWrapper.pad(input_image_torch)
    return input_image_torch_square, input_size


def dino_features_encoder(model, image, target_image_size):
    # DINO encoder: transform and encode
    with torch.no_grad():
        # original_size = image.shape[:2]
        input_image_square, input_size = preprocess_image(image, target_image_size)
        features = model(input_image_square)
        dino_features = model.postprocess_features(features, input_size=input_size, original_size=(64,64))
        return dino_features


def train_encoder(args, sam):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    render = Renderer(dim=(64, 64), radius=args.radius)
    vertices = args.mesh.vertices
    encoder = Encoder(args.depth, args.width, out_dim=args.n_classes, positional_encoding=args.positional_encoding,
                      sigma=args.sigma, network_verbose=args.network_verbose).to(device)
    optim = torch.optim.Adam(encoder.parameters(), args.learning_rate)

    losses = []
    feature_losses = defaultdict(list)
    torch.cuda.empty_cache()
    
    # Read viewing angles
    data = torch.load(os.path.join(args.encoder_data_dir, '{}/random_viewing_angles.pt'.format(args.render_res)))
    elev_list = data['elev']
    azim_list = data['azim']
    sampled_mesh = args.mesh

    dataset = EncoderDataset(args.encoder_data_dir, args)
    
    # create the DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    
    # Training loop
    for _ in range(args.num_epochs):
        for i, (batch_sam_f, batch_elevs, batch_azims) in enumerate(tqdm(dataloader)):
            optim.zero_grad()
            # predict highlight probabilities
            pred_f = encoder(vertices)
            
            # color and render mesh
            setcolor_mesh(sampled_mesh, pred_f)
            rendered_prob_views, elev, azim, mask = render.render_views(sampled_mesh, num_views=1,
                                                                                show=False,
                                                                                std=args.frontview_std,
                                                                                return_views=True,
                                                                                center_azim=batch_azims,
                                                                                center_elev=batch_elevs,
                                                                                return_features=True,
                                                                                lighting=False,
                                                                                background=torch.ones(args.n_classes).to(device),
                                                                                return_mask=True)
            batch_sam_f = batch_sam_f.squeeze(0)
            mask = (rendered_prob_views != 1.0)
            batch_sam_f[~mask] = 1.0
            rendered_prob_views[~mask] = 1.0
            loss = (batch_sam_f[mask] - rendered_prob_views[mask]).pow(2).sum()/mask.sum()
            loss.backward(retain_graph=True)
            optim.step()
            
            with torch.no_grad():
                losses.append(loss.item())
        
            # Report loss
            if i % 20 == 0: 
                print("Last 20 MSE score: {}".format(np.mean(losses[-20:])))
                current_lr = optim.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr}")

    if not os.path.exists(args.encoder_model_dir):
        os.makedirs(args.encoder_model_dir,exist_ok=True)
   
    torch.save({'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'losses': losses
                }, os.path.join(args.encoder_model_dir, 'encoder_checkpoint.pth'))   

    pred_f = encoder(args.mesh.vertices)
    torch.save(pred_f, os.path.join(args.encoder_model_dir, 'pred_f.pth'))


def save_loss(loss, dir, name=None):
    plt.figure()
    plt.plot(loss)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Ensure the directory exists
    os.makedirs(dir, exist_ok=True)

    # Save the figure
    if name is not None:
        plt.title(name+' over time')
        plt.savefig(os.path.join(dir, name+'.jpg'))
        plt.close()
    else:
        plt.title('Loss over time')
        plt.savefig(os.path.join(dir, 'loss.jpg'))
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--seed', type=int, default=0)
    
    # Mesh info
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--name', type=str, default='guitar')

    # Directory structure
    parser.add_argument('--encoder_data_dir', type=str, default='./data/guitar/encoder_data')
    parser.add_argument('--encoder_model_dir', type=str, default='./experiments/guitar/encoder_sam')

    # 2D segmentation model settings
    parser.add_argument('--sam_dir', type=str, default='./SAM_repo/model_checkpoints/')
    parser.add_argument('--sam_fname', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--sam_model_type', type=str, default='vit_h')
    
    # Feature extraction model
    parser.add_argument('--model_type', type=str, default='sam', choices=['sam', 'dino'])
    parser.add_argument('--target_image_size', type=int, default=224) # used only for dino
    
    # Render
    parser.add_argument('--background', nargs=3, type=float, default=[1., 1., 1.])
    parser.add_argument('--frontview_std', type=float, default=4)
    parser.add_argument('--render_res', type=int, default=224)
    parser.add_argument('--frontview_center', nargs=2, type=float, default=[0., 0.])
    parser.add_argument('--mesh_color', nargs=3, type=float, default=[2./3., 2./3., 2./3.])
    parser.add_argument('--test_elev_deg', type=int, default=0)
    parser.add_argument('--test_azim_deg', type=int, default=240)

    # Data
    parser.add_argument('--n_views', type=int, default=1000)
    parser.add_argument('--data_percentage', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=1)
    
    # Network
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--n_classes', type=int, default=256) 
    parser.add_argument('--positional_encoding', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=5.0)
    parser.add_argument('--radius', type=float, default=2.0)
    parser.add_argument('--network_verbose', type=int, default=1)

    # Optimization
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=3)

    # Flags for different stages of the pipeline
    parser.add_argument('--generate_random_views', type=int, default=0)
    parser.add_argument('--encode_random_views', type=int, default=0)
    parser.add_argument('--start_training', type=int, default=0)
    parser.add_argument('--test', type=int, default=0)

    args = parser.parse_args()

    # Load mesh object
    args.mesh = loadmesh(dir=args.obj_path, name=args.name, load_rings=True)

    # Load 2D model
    if args.model_type == 'sam':
        sam = load_img_seg_model(args.sam_dir, args.sam_fname, args.sam_model_type)
        model = sam
    elif args.model_type == 'dino':
        device_dino = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dino = DINOWrapper(device=device_dino, small=True, target_size=args.target_image_size)
        dino.to(device=device)
        model = dino
    else:
        raise ValueError("Unsupported model type: %s." % args.model_type)
    
    if args.generate_random_views == 1:
        generate_random_views(args)
    
    if args.encode_random_views == 1:
        encode_random_views(args, args.model_type, model)
    
    if args.start_training == 1:
        train_encoder(args, sam)
    
    if args.test == 1:
        test_encoder(args, sam)
