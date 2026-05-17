import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from modules.utils import get_img_name, cosine_similarity_along_F, get_features_in_mask, set_features_in_mask, compute_iou, compute_iou_batch
from modules.dino import DINOWrapper
from modules.sam import show_image, show_mask, show_points, show_box
from SAM_repo.segment_anything import SamPredictor
from SAM_repo.segment_anything.utils.transforms import ResizeLongestSide

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BestSegmentationBuddies:
    """
    A class for finding pixel-vertex best segmentation buddies.
    """
    
    def __init__(self, args, sam):
        """
        Initialize the BestSegmentationBuddies class.
        """
        self.args = args
        self.sam = sam

        self.object_seg_mask = None
        self.pix_dino_features = None

    @staticmethod
    def test_seg_click(img_path, test_click, test_mask_idx, mask_color, show_test_click, save_dir, image, sam, external_name=None, save=True, new_fig=True, close_fig=True):
        predictor = SamPredictor(sam)
        predictor.set_image(image)

        mask_color_np = np.array(mask_color).astype(np.float32)
        mask_color_np[:3] = mask_color_np[:3] / 255.0
        
        test_click_np = np.expand_dims(test_click, axis=0)
        input_label_np = np.array([1])
                
        masks, scores, logits = predictor.predict(
            point_coords=test_click_np,
            point_labels=input_label_np,
            multimask_output=True
            )

        mask = masks[test_mask_idx]
                
        # visualization
        if save:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
        
            if new_fig:
                plt.figure(figsize=(5, 5))
            
            ax = plt.gca()
            show_image(image, ax)
            show_mask(mask, ax, color=mask_color_np)

            if show_test_click:
                show_points(test_click_np, input_label_np, ax, marker_size=80, linewidth=1)
                            
            img_name = get_img_name(img_path)
            if external_name is None:
                click_str = 'click_%d_%d_sam_mask_%d' % (test_click[0], test_click[1], test_mask_idx)
                save_name = '_'.join([img_name, click_str]) + '.png'
            else:
                save_name = external_name
            
            save_path = os.path.join(save_dir, save_name)

            title_str = 'click (x, y)=(%d, %d) sam mask %d' % (test_click[0], test_click[1], test_mask_idx)
            
            plt.tight_layout()
            plt.show()
            plt.title(title_str)
            plt.savefig(save_path)
            print('SAM mask saved to %s' % save_path)
            
            if close_fig:
                plt.close()

        return mask

    @staticmethod
    def test_seg_batch(test_clicks_batch, test_boxes_batch, test_mask_idx, image, sam):
        predictor = SamPredictor(sam)
        predictor.set_image(image)

        batch_size = test_clicks_batch.shape[0]        
        test_clicks_label_batch = torch.ones([batch_size, 1], dtype=torch.int32, device=device)

        transformed_test_clicks_batch = predictor.transform.apply_coords_torch(test_clicks_batch, image.shape[:2])

        if test_boxes_batch is not None:
            transformed_test_boxes_batch = predictor.transform.apply_boxes_torch(test_boxes_batch, image.shape[:2])
        else:
            transformed_test_boxes_batch = None
        
        masks, scores, logits = predictor.predict_torch(
            point_coords=transformed_test_clicks_batch,
            point_labels=test_clicks_label_batch,
            boxes=transformed_test_boxes_batch,
            multimask_output=True
            )
        
        masks_batch = masks[:, test_mask_idx, :, :]

        return masks_batch
    
    @staticmethod
    def test_seg_box(img_path, test_click, test_box, test_mask_idx, mask_color, show_test_click, show_test_box, save_dir, image, sam, external_name=None, save=True, new_fig=True, close_fig=True):
        predictor = SamPredictor(sam)
        predictor.set_image(image)
               
        test_click_np = np.expand_dims(test_click, axis=0)
        input_label_np = np.array([1])

        test_box_np = np.array(test_box)
        
        mask_color_np = np.array(mask_color).astype(np.float32)
        mask_color_np[:3] = mask_color_np[:3] / 255.0

        masks, scores, logits = predictor.predict(
            point_coords=test_click_np,
            point_labels=input_label_np,
            box=test_box_np,
            multimask_output=True
            )
        
        mask = masks[test_mask_idx]
        
        # visualization
        if save:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
        
            if new_fig:
                plt.figure(figsize=(5, 5))
            
            ax = plt.gca()
            show_image(image, ax)
            show_mask(mask, ax, color=mask_color_np)

            if show_test_click:
                show_points(test_click_np, input_label_np, ax, marker_size=80, linewidth=1)
            
            if show_test_box:
                box_points = np.reshape(test_box_np, (2, 2))
                input_labels = np.array([1, 1])
                show_box(test_box_np, ax)
                show_points(box_points, input_labels, ax, marker='s', marker_size=80, linewidth=1)

            img_name = get_img_name(img_path)
            if external_name is None:
                click_str = 'click_%d_%d' % (test_click[0], test_click[1])
                box_str = 'box_%d_%d_%d_%d_sam_mask_%d' % (test_box[0], test_box[1], test_box[2], test_box[3], test_mask_idx)
                save_name = '_'.join([img_name, click_str, box_str]) + '.png'
            else:
                save_name = external_name
                
            save_path = os.path.join(save_dir, save_name)

            title_str = 'click (x, y)=(%d, %d) box (x, y)=(%d, %d) (x, y)=(%d, %d) sam mask %d' % (test_click[0], test_click[1], test_box[0], test_box[1], test_box[2], test_box[3], test_mask_idx)
                    
            plt.tight_layout()
            plt.show()
            plt.title(title_str)
            plt.savefig(save_path)
            print('SAM mask saved to %s' % save_path)
                
            if close_fig:
                plt.close()

        return mask
    
    def segment_image(self, image):
        if not self.args.add_test_box:
            test_mask = self.test_seg_click(self.args.img_path, self.args.test_pixel, self.args.test_mask_idx, self.args.mask_color, self.args.show_test_click, self.args.save_dir, image, self.sam)
        else:
            test_mask = self.test_seg_box(self.args.img_path, self.args.test_pixel, self.args.test_box, self.args.test_mask_idx, self.args.mask_color, self.args.show_test_click, self.args.show_test_box, self.args.save_dir, image, self.sam)
        
        return test_mask
        
    def preprocess_image(self, image, target_image_size):
        transform = ResizeLongestSide(target_image_size)
        input_image = transform.apply_image(image)
        
        input_image_torch = torch.as_tensor(input_image, device=device, dtype=torch.float32)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :] / 255.0
        input_size = input_image_torch.shape[-2:]

        input_image_torch_square = DINOWrapper.pad(input_image_torch)
        
        return input_image_torch_square, input_size

    def get_dino_features(self, image, target_image_size=224):
        original_size = image.shape[:2]

        input_image_square, input_size = self.preprocess_image(image, target_image_size=target_image_size)
        model = DINOWrapper(device=device, small=True, target_size=target_image_size)
        features = model(input_image_square)
        pix_features = model.postprocess_features(features, input_size=input_size, original_size=original_size)
        pix_features = pix_features[0].permute(1, 2, 0).contiguous()

        return pix_features
        
    def prepare_data(self, image):
        img_name = get_img_name(self.args.img_path)

        # object segmentation mask
        click_str = 'click_%d_%d_sam_mask_%d' % (self.args.object_pixel[0], self.args.object_pixel[1], self.args.object_mask_idx)
        save_name = '_'.join([img_name, click_str]) + '.pth'
        save_path = os.path.join(self.args.save_dir, save_name)
        if not self.args.load_object_seg_mask:
            object_seg_mask = self.test_seg_click(self.args.img_path, self.args.object_pixel, self.args.object_mask_idx, self.args.mask_color, self.args.show_object_click, self.args.save_dir, image, self.sam)
            self.object_seg_mask = torch.as_tensor(object_seg_mask, device=device)
            torch.save(self.object_seg_mask, save_path)
        else:
            self.object_seg_mask = torch.load(save_path, map_location=device)

        # per pixel dino features 
        save_name = '_'.join([img_name, 'pix_dino_features']) + '.pth'
        save_path = os.path.join(self.args.save_dir, save_name)
        if not self.args.load_pix_dino_features:
            self.pix_dino_features = self.get_dino_features(image)
            torch.save(self.pix_dino_features, save_path)
        else:
            self.pix_dino_features = torch.load(save_path, map_location=device)
        
    def find_bsb(self, image):
        # prepare data
        self.prepare_data(image)
        
        vert_dino_features = torch.load(self.args.encoder_dino_f_path, map_location=device)
        vert_dino_features = vert_dino_features.unsqueeze(0).expand(1, -1, -1)

        # test click
        test_click_x = self.args.test_pixel[0]
        test_click_y = self.args.test_pixel[1]
        click_feature = torch.unsqueeze(self.pix_dino_features[test_click_y, test_click_x, :], dim=0)

        # test mask
        test_mask = self.segment_image(image)
        test_mask_batch = torch.tensor(test_mask, device=device).unsqueeze(0)

        # pixel to vertex similarity
        similarity_p_to_v = cosine_similarity_along_F(vert_dino_features, click_feature)
        similarity_p_to_v = torch.squeeze(similarity_p_to_v, dim=0)

        # vertex candidates
        similarity_p_to_v_idx_sort = torch.argsort(similarity_p_to_v, descending=True)
        cand_verts_idx = similarity_p_to_v_idx_sort[:self.args.num_cand_verts]

        cand_verts_is_bsb = torch.zeros(len(cand_verts_idx), dtype=torch.bool, device=device)
        cand_verts_nn_pix_x = torch.zeros(len(cand_verts_idx), dtype=torch.long, device=device)
        cand_verts_nn_pix_y = torch.zeros(len(cand_verts_idx), dtype=torch.long, device=device)

        pix_dino_features_in_mask = torch.unsqueeze(self.pix_dino_features[self.object_seg_mask, :], dim=0)

        # find best segmentation buddy vertex candidates
        num_cand_verts = len(cand_verts_idx)
        for i in range(0, num_cand_verts, self.args.batch_size_sim):
            batch_end = min(i + self.args.batch_size_sim, num_cand_verts)
            batch_size = batch_end - i

            print('Computing similarity for vertex candidates %d to %d out of %d' % (i+1, batch_end, num_cand_verts))

            # batch of candidates
            batch_cand_verts_idx = cand_verts_idx[i:batch_end]
            batch_cand_verts_features = vert_dino_features[0, batch_cand_verts_idx, :]  # BxF
            
            # find the most similar pixel in the object segmentation mask for each vertex candidate
            similarity_v_to_p = cosine_similarity_along_F(pix_dino_features_in_mask, batch_cand_verts_features)  # BxN
            similarity_v_to_p_max_idx = torch.argmax(similarity_v_to_p, dim=1)  # B

            idx_shift = torch.arange(batch_size) * similarity_v_to_p.shape[1]
            similarity_v_to_p_max_idx_shifted = similarity_v_to_p_max_idx + idx_shift.to(device)

            nn_pix_indicator_flat = torch.zeros_like(similarity_v_to_p, dtype=torch.bool).reshape(-1)
            nn_pix_indicator_flat[similarity_v_to_p_max_idx_shifted] = True
            nn_pix_indicator = nn_pix_indicator_flat.reshape(batch_size, -1)

            batch_nn_pix_img = set_features_in_mask(nn_pix_indicator.unsqueeze(-1), self.object_seg_mask.unsqueeze(0).expand(batch_size, -1, -1)).squeeze(-1)
            _, batch_cand_verts_nn_pix_y, batch_cand_verts_nn_pix_x = torch.where(batch_nn_pix_img)

            cand_verts_nn_pix_x[i:batch_end] = batch_cand_verts_nn_pix_x
            cand_verts_nn_pix_y[i:batch_end] = batch_cand_verts_nn_pix_y

            # ckeck if the most similar pixel falls within the test mask
            intersection_img_batch = torch.logical_and(batch_nn_pix_img, test_mask_batch)
            batch_is_bsb = intersection_img_batch.sum(dim=(1, 2)) > 0
            cand_verts_is_bsb[i:batch_end] = batch_is_bsb
                
        is_bsb = torch.any(cand_verts_is_bsb)
        if is_bsb:
            is_bsb_str = 'bsb'

            cand_bsb_verts_idx = cand_verts_idx[cand_verts_is_bsb]
            cand_bsb_verts_nn_pix_x = cand_verts_nn_pix_x[cand_verts_is_bsb]
            cand_bsb_verts_nn_pix_y = cand_verts_nn_pix_y[cand_verts_is_bsb]
        else:
            is_bsb_str = 'non_bsb'

            if self.args.compute_seg_for_non_bsb:
                cand_bsb_verts_idx = cand_verts_idx
                cand_bsb_verts_nn_pix_x = cand_verts_nn_pix_x
                cand_bsb_verts_nn_pix_y = cand_verts_nn_pix_y
            else:
                cand_bsb_verts_idx = torch.tensor([], dtype=torch.int64, device=device)
                cand_bsb_verts_nn_pix_x = torch.tensor([], dtype=torch.int64, device=device)
                cand_bsb_verts_nn_pix_y = torch.tensor([], dtype=torch.int64, device=device)          
                    
        # find the best segmentation buddy vertex according to Intesection over Union (IoU) between the test mask
        # and the segmentation mask for the pixel corresponding to the best segmentation buddy vertex candidates
        num_cand_bsb_verts = len(cand_bsb_verts_idx)
        cand_bsb_verts_iou = torch.zeros(cand_bsb_verts_idx.shape, dtype=torch.float32, device=device)

        # run over best segmentation buddy vertex candidates in batches
        for i in range(0, num_cand_bsb_verts, self.args.batch_size_iou):
            batch_end = min(i + self.args.batch_size_iou, num_cand_bsb_verts)
            batch_size = batch_end - i

            batch_cand_bsb_verts_nn_pix_x = cand_bsb_verts_nn_pix_x[i:batch_end]
            batch_cand_bsb_verts_nn_pix_y = cand_bsb_verts_nn_pix_y[i:batch_end]

            batch_cand_bsb_verts_nn_pix_point = torch.stack((batch_cand_bsb_verts_nn_pix_x, batch_cand_bsb_verts_nn_pix_y), dim=1)
            
            if self.args.batch_size_iou > 1:
                print('Computing IoU for %s vertex candidates %d to %d out of %d' % (is_bsb_str, i+1, batch_end, num_cand_bsb_verts))
                
                batch_test_boxes = None if not self.args.add_test_box else torch.tensor(self.args.test_box, device=device).unsqueeze(0).expand(batch_size, -1)
                batch_cand_bsb_verts_nn_pix_mask = self.test_seg_batch(batch_cand_bsb_verts_nn_pix_point.unsqueeze(1), batch_test_boxes, self.args.test_mask_idx, image, self.sam)
                
                batch_cand_bsb_verts_iou = compute_iou_batch(batch_cand_bsb_verts_nn_pix_mask, test_mask_batch)
                cand_bsb_verts_iou[i:batch_end] = batch_cand_bsb_verts_iou
            else:
                print('Computing IoU for %s vertex candidate %d out of %d' % (is_bsb_str, i+1, num_cand_bsb_verts))
                batch_cand_bsb_verts_nn_pix_point_np = batch_cand_bsb_verts_nn_pix_point[0].cpu().numpy()
                
                if not self.args.add_test_box:
                    nn_pix_mask = self.test_seg_click(self.args.img_path, batch_cand_bsb_verts_nn_pix_point_np, self.args.test_mask_idx, self.args.mask_color, self.args.show_test_click, self.args.save_dir, image, self.sam, save=False)
                else:
                    nn_pix_mask = self.test_seg_box(self.args.img_path, batch_cand_bsb_verts_nn_pix_point_np, self.args.test_box, self.args.test_mask_idx, self.args.mask_color, self.args.show_test_click, self.args.show_test_box, self.args.save_dir, image, self.sam, save=False)
                                
                batch_cand_bsb_verts_iou_np = compute_iou(nn_pix_mask, test_mask)
                cand_bsb_verts_iou[i] = torch.tensor(batch_cand_bsb_verts_iou_np, device=device)
                
        msg_str_list = ['Test image: %s' % self.args.img_path, 'Test click: (%d, %d)' % (test_click_x, test_click_y)]
        if self.args.add_test_box:
            box_str = 'Test box: top left (%d, %d) bottom right (%d, %d)' % (self.args.test_box[0], self.args.test_box[1], self.args.test_box[2], self.args.test_box[3])
            msg_str_list = msg_str_list + [box_str]
        
        msg_str_list = msg_str_list + ['Number of vertex candidates: %d' % num_cand_verts, 'SAM mask index: %d' % self.args.test_mask_idx]
        seg_str = ', '.join(msg_str_list)

        astrix_str = '*' * len(seg_str)

        compute_seg = is_bsb or (not is_bsb and self.args.compute_seg_for_non_bsb)
        if compute_seg:
            if is_bsb:
                bsb_cands_idx_sort = torch.argsort(cand_bsb_verts_iou, descending=True)
                bsb_idx = bsb_cands_idx_sort[0]
                sel_idx = bsb_idx
            else:
                rand_idx = torch.randint(0, num_cand_bsb_verts, (1,), device=device)[0]
                sel_idx = rand_idx
                
            bsb_vert_iou = cand_bsb_verts_iou[sel_idx]
            bsb_vert_nn_pix_x = cand_bsb_verts_nn_pix_x[sel_idx]
            bsb_vert_nn_pix_y = cand_bsb_verts_nn_pix_y[sel_idx]
            bsb_vert_idx = cand_bsb_verts_idx[sel_idx]
                                
            # visualize nearest pixel image segmentation for the best buddy vertex
            img_name = get_img_name(self.args.img_path)
            if not self.args.add_test_box:
                external_name = '%s_pix_%d_%d_%s_v%d_nn_pix_%d_%d_sam_mask_%d.png' % (img_name, test_click_x, test_click_y, is_bsb_str, bsb_vert_idx, bsb_vert_nn_pix_x, bsb_vert_nn_pix_y, self.args.test_mask_idx)
            else:
                box_str = 'box_%d_%d_%d_%d' % (self.args.test_box[0], self.args.test_box[1], self.args.test_box[2], self.args.test_box[3])
                external_name = '%s_pix_%d_%d_%s_%s_v%d_nn_pix_%d_%d_sam_mask_%d.png' % (img_name, test_click_x, test_click_y, box_str, is_bsb_str, bsb_vert_idx, bsb_vert_nn_pix_x, bsb_vert_nn_pix_y, self.args.test_mask_idx)

            bsb_vert_nn_pix_point = np.array([bsb_vert_nn_pix_x.item(), bsb_vert_nn_pix_y.item()])
            if not self.args.add_test_box:
                self.test_seg_click(self.args.img_path, bsb_vert_nn_pix_point, self.args.test_mask_idx, self.args.mask_color_nn_pix, self.args.show_nn_pix, self.args.save_dir, image, self.sam, external_name=external_name)
            else:
                self.test_seg_box(self.args.img_path, bsb_vert_nn_pix_point, self.args.test_box, self.args.test_mask_idx, self.args.mask_color_nn_pix, self.args.show_nn_pix, self.args.show_test_box, self.args.save_dir, image, self.sam, external_name=external_name)

            # print summary message
            bsb_str = 'Best Segmentation Buddy (BSB)' if is_bsb else 'Non Best Segmentation Buddy (non BSB)'
            nn_str = 'BSB' if is_bsb else 'non BSB'
            
            msg_str = '\n'.join([astrix_str,
                                 seg_str,
                                 '%s vertex index: %d' % (bsb_str, bsb_vert_idx),
                                 'Nearest Neighbor (NN) pixel for %s vertex: (%d, %d)' % (nn_str, bsb_vert_nn_pix_x, bsb_vert_nn_pix_y),
                                 'IoU between SAM mask for the NN pixel and the test mask: %.4f' % bsb_vert_iou,
                                 astrix_str
                                 ])
        else:
            msg_str = '\n'.join([astrix_str,
                                 seg_str,
                                 'No Best Segmentation Buddy (BSB) vertex was found!',
                                 astrix_str
                                 ])
            bsb_vert_idx = None
        
        print(msg_str)

        return bsb_vert_idx, compute_seg
