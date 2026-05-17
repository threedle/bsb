import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import os
import copy
import random
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from modules.utils import device, setcolor_mesh_batched, load_state_dict, find_red_green_pixels, show_mask, show_points, loadmesh, add_spheres, write_ply
from modules.dataset import DecoderDataset
from modules.render import save_renders, Renderer
from modules.click_attention import ClickAttention
from modules.decoder import Decoder
from modules.sam import load_img_seg_model
from modules.best_segmentation_buddies import BestSegmentationBuddies


def constrain_randomness(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_shape_decoder(args):
    print('train shape decoder')
    num_gpus = torch.cuda.device_count()
    print('number of avilable gpus: %d' % num_gpus)

    render_high = Renderer(dim=(args.render_res, args.render_res))

    # MLP Settings: predicting 3D probability mask
    attention = ClickAttention(256, args).to(device)
    mlp = Decoder(depth=args.depth, width=[512]+[256]*args.depth, out_dim=2, input_dim=512, positional_encoding=args.positional_encoding,
                            sigma=args.sigma).to(device)
    
    parameters = list(attention.parameters()) + list(mlp.parameters())
    optim = torch.optim.Adam(parameters, args.learning_rate)
    
    if args.use_data_parallel and num_gpus > 1:
        device_ids_list = list(range(num_gpus))
        attention = torch.nn.DataParallel(attention, device_ids=device_ids_list)
        mlp = torch.nn.DataParallel(mlp, device_ids=device_ids_list)
       
    # Create the dataset
    dataset = DecoderDataset(args.decoder_data_dir, args)

    # Initialize variables
    start_epoch = 0
    start_batch = 0
    save_path = os.path.join(args.save_dir, args.model_name)
    losses = []
    all_indices = list(range(len(dataset)))

    if args.continue_train:
        checkpoint = torch.load(save_path, map_location=device)
        attention.load_state_dict(checkpoint['attention_state_dict'])
        mlp.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch_idx']
        all_indices = checkpoint['shuffled_indices']
        losses = checkpoint['losses']

    # read learned 3D features
    input_tensor_A = load_features(args)

    for epoch in range(start_epoch, args.num_epochs):
         # If this is a new epoch, shuffle the indices
        if epoch != start_epoch or start_batch == 0:
            random.shuffle(all_indices)

        # SubsetRandomSampler handles the batching
        sampler = SubsetRandomSampler(all_indices[start_batch * args.batch_size:])
        
        # create the DataLoader
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

        # Sampling data using shuffled indices
        num_betches = len(dataloader)
        for batch_idx, (batch_data, batch_labels, batch_elevs, batch_azims) in enumerate(tqdm(dataloader)):
            print("batch_idx: %d" % (batch_idx + start_batch))
            batch_size = batch_labels.shape[0]
            
            # input_tensor_A: feature field, NxF
            # batch_labels: selected vertex indices. BxC

            feature_field_batch = input_tensor_A.unsqueeze(0).expand(batch_size, -1, -1)

            # click attentntion
            weighted_vals = attention(feature_field_batch, batch_labels)
        
            input_tensor = torch.cat((feature_field_batch, weighted_vals), dim=-1)
            
            # mlp
            prob_tensor = mlp(input_tensor)

            # set probability per face
            setcolor_mesh_batched(args.mesh, prob_tensor)
            args.mesh.face_attributes = args.mesh.face_attributes.float()

            # render
            rendered_prob_views, elev, azim, mask = render_high.render_views_batched(args.mesh, #num_views=1,
                                                                            show=False,
                                                                            std=args.frontview_std,
                                                                            return_views=True,
                                                                            center_azim=batch_azims,
                                                                            center_elev=batch_elevs,
                                                                            return_features=True,
                                                                            lighting=False,
                                                                            background=None,
                                                                            return_mask=True)
            
            # Calculate the loss
            loss = masked_bce_loss(rendered_prob_views, batch_data, mask)
            loss.backward(retain_graph=True)
            optim.step()
            optim.zero_grad()
            with torch.no_grad():
                losses.append(loss.item())
            torch.cuda.empty_cache()

            # Save checkpoint logic (every args.save_interval batches)
            batch_count = batch_idx + 1
            batch_count_tot = batch_idx + start_batch + 1
            if batch_count == num_betches or batch_count_tot % args.save_interval == 0:
                torch.save({
                    'attention_state_dict': attention.state_dict(),
                    'model_state_dict': mlp.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'epoch': epoch,
                    'batch_idx': batch_count_tot,
                    'shuffled_indices': all_indices,
                    'losses': losses,
                    'num_batch': round(dataset.num_batch)+1
                }, save_path)
                
                print('checkpoint saved for epoch number %d batch count total %d' % (epoch + 1, batch_count_tot))
        
        # save epoch check point
        if num_betches > 0:
            save_dir, fname = os.path.split(save_path)
            fbody, fext = fname.split(".")
            fname_epoch = ".".join(["%s_e%d" % (fbody, epoch + 1), fext])
            save_path_epoch = os.path.join(save_dir, fname_epoch)
            torch.save({
                        'attention_state_dict': attention.state_dict(),
                        'model_state_dict': mlp.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'epoch': epoch,
                        'batch_idx': batch_idx + start_batch + 1,
                        'shuffled_indices': all_indices,
                        'losses': losses,
                        'num_batch': round(dataset.num_batch)+1
                        }, save_path_epoch)
            print('checkpoint saved for epoch number %d' % (epoch + 1))
            
        start_batch = 0


# Append to the existing files or create them if they don't exist
def save_or_append(filename, data):
    if os.path.exists(filename):
        existing_data = torch.load(filename, map_location=device)
        if len(data.shape)<len(existing_data.shape):
            combined_data = torch.cat([existing_data, data.unsqueeze(0)])
        else:
            combined_data = torch.cat([existing_data, data])
        torch.save(combined_data, filename)
    else:
        torch.save(data, filename)


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


def load_features(args):
    # load features
    if torch.cuda.is_available():
        pred_f = torch.load(args.encoder_sam_f_path)
    else:
        pred_f = torch.load(args.encoder_sam_f_path, map_location=torch.device('cpu'))
    return pred_f


def masked_bce_loss(y_pred, y_true, mask):
    # Compute the raw BCE loss term-wise
    bce = F.binary_cross_entropy(y_pred, F.one_hot(y_true.long(), num_classes=2).squeeze(1).permute(0, 3, 1, 2).float(), reduction='none')
    mask = torch.cat((mask.unsqueeze(1), mask.unsqueeze(1)), dim=1)
    # Apply the mask
    masked_bce = bce * mask
    
    # Compute the mean of the masked BCE values
    loss = masked_bce.sum() / mask.sum()
    return loss


def test_shape_decoder(args, test_index):
    print('test shape decoder')
    
    num_gpus = torch.cuda.device_count()
    print('number of avilable gpus: %d' % num_gpus)

    # MLP Settings: predicting 3D probability mask
    if test_index is not None:
        attention = ClickAttention(256, args).to(device)
        mlp = Decoder(depth=args.depth, width=[512]+[256]*args.depth, out_dim=2, input_dim=512, positional_encoding=args.positional_encoding,
                                sigma=args.sigma).to(device)
            
        if args.use_data_parallel and num_gpus > 1:
            device_ids_list = list(range(num_gpus))
            attention = torch.nn.DataParallel(attention, device_ids=device_ids_list)
            mlp = torch.nn.DataParallel(mlp, device_ids=device_ids_list)
        
        save_path = os.path.join(args.save_dir, args.model_name)
        print('decoder checkpoint path: %s' % save_path)
    
        checkpoint = torch.load(save_path, map_location=device)
        load_state_dict(attention, checkpoint['attention_state_dict'])
        load_state_dict(mlp, checkpoint['model_state_dict'])

        # read learned 3D features
        pred_f = load_features(args)

        if len(test_index.shape) == 1:
            test_index = test_index.unsqueeze(0)
    
        batch_size = test_index.shape[0]
        
        feature_field_batch = pred_f.unsqueeze(0).expand(batch_size, -1, -1)
        
        # click attentntion
        weighted_vals = attention(feature_field_batch, test_index)
            
        input_tensor = torch.cat((feature_field_batch, weighted_vals), dim=-1)
                
        # mlp
        prob_tensor = mlp(input_tensor)
    else:
        prob_tensor = torch.ones([1, args.mesh.vertices.shape[0], 2], dtype=torch.float32).to(device)
        prob_tensor[0, :, 1] = 0.

    prob_tensor_np = prob_tensor.detach().cpu().numpy().astype(np.float32)
    prob_to_save = prob_tensor_np[0, :, 1]

    # save vertex probability
    if args.save_vertex_probability:            
        if test_index is not None:
            select_vertices_str = '_'.join(['v%d' % idx for idx in list(test_index[0])])
            save_name = 'vertex_probability_' + select_vertices_str + '.npy'
        else:
            save_name = 'vertex_probability_empty_segmentation.npy'
        
        save_path = os.path.join(args.save_dir, save_name)
        np.save(save_path, prob_to_save)
        print("vertex probability saved to %s" % save_path)
    
    # save colored mesh
    if test_index is not None:
        test_index_list = list(test_index[0].cpu())
    else:
        test_index_list = None
    prepare_colored_mesh(args.mesh, np.expand_dims(prob_to_save, 0), test_index_list, args)
    
    # save render views
    render_high = Renderer(dim=(args.render_res, args.render_res))
    test_azim_deg = [0, 60, 120, 180, 240, 300]
    test_elev_deg = [-60, -30, 0, 30, 60]
    render_views(args.mesh, test_index, prob_tensor, render_high, test_azim_deg, test_elev_deg, args)


def prepare_colored_mesh(mesh, prob_tensor, test_index, args):
    base_color = np.array([args.base_color], dtype=np.float32) / 255.
    seg_color = np.array([args.seg_color], dtype=np.float32) / 255.

    pos_color = np.array(args.pos_color, dtype=np.float32) / 255.
    neg_color = np.array(args.neg_color, dtype=np.float32) / 255.

    if args.show_seg:
        vertex_colors = prob_tensor.T * seg_color + (1. - prob_tensor.T) * base_color
    else:
        vertex_colors = np.ones_like(prob_tensor.T) * base_color
    
    mesh_o3d = mesh.export_to_open3d()
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    if test_index is None or not args.show_spheres:
        mesh_to_save = mesh_o3d
    else:
        sel_verts_idx = np.array(test_index, dtype=np.int32)
        sel_verts = mesh.vertices[np.abs(sel_verts_idx)].detach().cpu().numpy()

        col_verts = np.zeros_like(sel_verts)
        for i, v in enumerate(list(sel_verts_idx)):
            if v >= 0:
                col_verts[i] = pos_color
            else:
                col_verts[i] = neg_color

        mesh_to_save = add_spheres(copy.deepcopy(mesh_o3d), sel_verts, col_verts, save_mesh=False, save_path=None, sphere_radius=args.sphere_radius, sphere_resolution=10)

    verts = np.asarray(mesh_to_save.vertices)
    faces = np.asarray(mesh_to_save.triangles)
    colors = np.array(np.asarray(mesh_to_save.vertex_colors) * 255 + 0.5, dtype=np.int32)
        
    if test_index is None:
        selected_vertices_str = 'empty_segmentation'
    else:
        selected_vertices_str = '_'.join(['v%d' % idx for idx in test_index])
    
    if not args.show_spheres:
        save_name = 'colored_mesh_' + selected_vertices_str + '.ply'
    else:
        save_name = 'colored_mesh_with_clicks_' + selected_vertices_str + '.ply'
    save_path = os.path.join(args.save_dir, save_name)
    write_ply(save_path, verts, faces, colors)    
    print("mesh saved to %s" % save_path)

    return save_path


def render_views(mesh, test_index, prob_tensor, renderer, test_azim_deg, test_elev_deg, args):
    test_azim = np.array(test_azim_deg, dtype=np.float32)/360 * 2*np.pi
    test_elev = np.array(test_elev_deg, dtype=np.float32)/180 * np.pi

    # assuming your mesh.vertices is a torch tensor and on the correct device
    target_rgb = torch.zeros_like(mesh.vertices)  # initialize colors with zeros
    
    target_rgb[:] = torch.tensor([2./3., 2./3., 2./3.]).to(device)
    
    if test_index is not None:
        target_rgb[test_index[0, 0]] = torch.tensor([0., 255., 0.]).to(device)
        
        if test_index.shape[-1] > 1:
            if test_index[0, 1] >= 0:
                target_rgb[test_index[0, 1]] = torch.tensor([0., 255., 0.]).to(device)
            else:
                target_rgb[abs(test_index[0, 1])] = torch.tensor([255., 0., 0.]).to(device)

    azim_num = len(test_azim)
    elev_num = len(test_elev)
    _, axs = plt.subplots(elev_num, azim_num, figsize=(azim_num * 10, elev_num * 10))

    target_mesh = mesh
    for i in range(elev_num):
        for j in range(azim_num):
            setcolor_mesh_batched(target_mesh, target_rgb.unsqueeze(0))
            image_temp, elev, azim = renderer.render_views(mesh, num_views=1,
                                            show=args.show,
                                            std=args.frontview_std,
                                            random_views=False,
                                            center_elev=torch.tensor(test_elev[i:i+1]).to(device),
                                            center_azim= torch.tensor(test_azim[j:j+1]).to(device),
                                            lighting=True,
                                            return_views=True,
                                            background=torch.ones(3).to(device),
                                            return_mask=False)
            
            if test_index is not None:
                image_temp_for_find = image_temp.squeeze(0).permute(1, 2, 0)
                selected_vertex_r, selected_vertex_g = find_red_green_pixels(image_temp_for_find)

            setcolor_mesh_batched(mesh, prob_tensor)
            rendered_prob_views = renderer.render_views(mesh, num_views=1,
                                                        show=args.show,
                                                        std=args.frontview_std,
                                                        random_views=False,
                                                        center_elev=elev,
                                                        center_azim=azim,
                                                        lighting=False,
                                                        background=None,
                                                        return_mask=True)
                    
            save_iamge_name = 'temp_view.png'
            save_image_path = os.path.join(args.save_dir, save_iamge_name)
            save_renders(args.save_dir, 0, image_temp, name=save_iamge_name)
            image = cv2.imread(save_image_path)
            axs[i, j].imshow(image)

            axs[i, j].set_xticks([])
            axs[i, j].set_xticks([], minor=True)
            axs[i, j].set_yticks([])
            axs[i, j].set_yticks([], minor=True)

            os.remove(save_image_path)

            show_mask(rendered_prob_views[0][0][1].detach().cpu().numpy(), axs[i, j])
            
            if test_index is not None:
                if len(selected_vertex_r) != 0:
                    show_points(np.array([[selected_vertex_r[0][1].detach().cpu().numpy(), selected_vertex_r[0][0].detach().cpu().numpy()]]), 
                        np.array([0]), axs[i, j], marker_size=307)
                if len(selected_vertex_g) != 0:
                    show_points(np.array([[selected_vertex_g[0][1].detach().cpu().numpy(), selected_vertex_g[0][0].detach().cpu().numpy()]]), 
                        np.array([1]), axs[i, j], marker_size=307)    
        
    if test_index is not None:
        selected_vertices_str = '_'.join(['v%d' % idx for idx in list(test_index[0])])
    else:
        selected_vertices_str = 'empty_segmentation'
    
    save_name = 'rendered_views_' + selected_vertices_str + '.png'
    save_path = os.path.join(args.save_dir, save_name)
    
    plt.tight_layout()
    plt.show()  
    plt.savefig(save_path)
    plt.close()
    print("rendered views saved to %s" % save_path)

    return save_path


def image_shape_correspondence(args):
    # create image encoder directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
        
    # read image
    image = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)

    # load image segmentaiotn model
    sam = load_img_seg_model(args.sam_dir, args.sam_fname, args.sam_model_type)

    # find best segmentation buddy
    bsb = BestSegmentationBuddies(args, sam)

    if args.mode == 'image_seg':
        bsb.segment_image(image)
        return
        
    vert_idx, compute_seg = bsb.find_bsb(image)
    
    # segment the shape
    test_index = torch.tensor([vert_idx]) if compute_seg else None
    test_shape_decoder(args, test_index=test_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--seed', type=int, default=0)

    # directory structure
    parser.add_argument('--obj_path', type=str, default='./meshes/guitar.obj')
    parser.add_argument('--encoder_sam_f_path', type=str, default='./experiments/guitar/encoder_sam/pred_f.pth')
    parser.add_argument('--decoder_data_dir', type=str, default='./data/guitar/decoder_data')
    parser.add_argument('--model_name', type=str, default='decoder_checkpoint.pth')

    # training data setting
    parser.add_argument('--use_positive_click', type=int, default=0)
    parser.add_argument('--use_negative_click', type=int, default=0)

    # mesh + data info
    parser.add_argument('--name', type=str, default='guitar')
    parser.add_argument('--data_percentage', type=float, default=1.0)
    parser.add_argument('--views_per_vert', type=int, default=100)

    # render
    parser.add_argument('--background', nargs=3, type=float, default=[1., 1., 1.])
    parser.add_argument('--n_views', type=int, default=1)
    parser.add_argument('--frontview_std', type=float, default=4)
    parser.add_argument('--frontview_center', nargs=2, type=float, default=[3.14, 0.])
    parser.add_argument('--render_res', type=int, default=224)

    # attention
    parser.add_argument('--use_attention_q', type=int, default=1)
    parser.add_argument('--use_attention_k', type=int, default=1)
    parser.add_argument('--use_attention_v', type=int, default=1)
    parser.add_argument('--redsidual_attention', type=int, default=0)
    parser.add_argument('--scale_attention', type=int, default=1)
    
    # network
    parser.add_argument('--continue_train', type=int, default=0)
    parser.add_argument('--depth', type=int, default=14)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--n_classes', type=int, default=256)
    parser.add_argument('--positional_encoding', action='store_true')
    parser.add_argument('--sigma', type=float, default=5.0)

    # optimization
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--return_original', type=int, default=0)
    
    # parallel, multi-GPU training
    parser.add_argument('--use_data_parallel', type=int, default=0)
    
    # shape segmentation
    parser.add_argument('--select_vertices', nargs='+', type=int, default=[0])
    parser.add_argument('--save_vertex_probability', type=int, default=0)

    # visualization
    parser.add_argument('--show', type=int, default=0)
    parser.add_argument('--base_color', nargs=3, type=int, default=[180, 180, 180])
    parser.add_argument('--show_seg', type=int, default=1)
    parser.add_argument('--seg_color', nargs=3, type=int, default=[60, 160, 250])

    parser.add_argument('--show_spheres', type=int, default=0)
    parser.add_argument('--sphere_radius', type=float, default=0.025)
    parser.add_argument('--pos_color', nargs=3, type=int, default=[0, 255, 0])
    parser.add_argument('--neg_color', nargs=3, type=int, default=[255, 0, 0])

    # bsb configuration
    parser.add_argument('--img_path', type=str, default='images/guitar_01.png')
    parser.add_argument('--object_pixel', nargs=2, type=int, default=[270, 370], help='Object pixel (x,y) coordinates')
    parser.add_argument('--object_mask_idx', type=int, default=2)
    parser.add_argument('--show_object_click', type=int, default=1)
    parser.add_argument('--load_object_seg_mask', type=int, default=0)
    parser.add_argument('--load_pix_dino_features', type=int, default=0)
    parser.add_argument('--test_pixel', nargs=2, type=int, default=[270, 370], help='Test pixel (x,y) coordinates')
    parser.add_argument('--test_mask_idx', type=int, default=0)
    parser.add_argument('--show_test_click', type=int, default=1)
    parser.add_argument('--add_test_box', type=int, default=0)
    parser.add_argument('--test_box', nargs=4, type=int, default=[100, 150, 480, 650], help='Test bounding box coordinates for the test object (top left (x,y) and bottom right (x,y))')
    parser.add_argument('--show_test_box', type=int, default=1)
    parser.add_argument('--num_cand_verts', type=int, default=100)
    parser.add_argument('--batch_size_sim', type=int, default=10)
    parser.add_argument('--batch_size_iou', type=int, default=100)
    parser.add_argument('--mask_color', nargs=4, type=float, default=[30, 144, 255, 0.6])
    parser.add_argument('--compute_seg_for_non_bsb', type=int, default=0)
    parser.add_argument('--show_nn_pix', type=int, default=1)
    parser.add_argument('--mask_color_nn_pix', nargs=4, type=float, default=[205, 255, 5, 0.6])
    
    # bsb directory structure
    parser.add_argument('--save_dir', type=str, default='./experiments/guitar/bsb/')
    parser.add_argument('--encoder_dino_f_path', type=str, default='./experiments/guitar/encoder_dino/pred_f.pth')

    # 2D segmentation model settings
    parser.add_argument('--sam_dir', type=str, default='./SAM_repo/model_checkpoints/')
    parser.add_argument('--sam_fname', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--sam_model_type', type=str, default='vit_h')

    # mode
    parser.add_argument('--mode', type=str, default='bsb', choices=['train_shape_decoder', 'test_shape_decoder', 'image_seg', 'bsb'])

    args = parser.parse_args()
    
    # Load a 3D mesh
    if not args.mode == 'image_seg':
        args.mesh = loadmesh(dir=args.obj_path, name=args.name, load_rings=True)

    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
       
    # Constrain most sources of randomness
    constrain_randomness(args.seed)
    
    # Train shape decoder
    if args.mode == 'train_shape_decoder':
        train_shape_decoder(args)
    
    # Test shape decoder
    if args.mode == 'test_shape_decoder':
        test_shape_decoder(args, test_index=torch.tensor(args.select_vertices))
    
    # Run image-shape correspondence
    if args.mode == 'image_seg' or args.mode == 'bsb':
        image_shape_correspondence(args)
