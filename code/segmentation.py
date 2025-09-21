# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Yunli Qi
"""
from tqdm import tqdm
import os,pdb
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import torch.nn as nn
# from func_3d.utils import get_network, set_log_dir,random_click, generate_bbox, create_logger,eval_seg
from utils import get_network,random_click, generate_bbox, set_log_dir,create_logger,eval_seg
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


    
class BTCV(Dataset):
    def __init__(self, args, image_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', seed=None, variation=0):

        # Set the data list for training
        self.name_list = os.listdir(os.path.join(image_path))
        
        # Set the basic information of the dataset
        self.image_path = image_path
        self.origin_mask = args.origin_mask
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_size, self.img_size)

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.image_path, name)     
        num_frame = len(os.listdir(img_path))               
        img_tensor = torch.zeros(num_frame, 3, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}
        if self.prompt=='autonomy':
            for frame_index in range(0, num_frame): 
                img = Image.open(os.path.join(img_path, f'{frame_index }.png')).convert('RGB')
                img = img.resize(newsize)
                img = torch.tensor(np.array(img)).permute(2, 0, 1)
                    
                obj_list=set()
                obj_list.add(1)

                img_tensor[frame_index , :, :, :] = img
                mask_dict[frame_index ] = {}
                pt_dict[frame_index ]={}                
                point_label_dict[frame_index ]={}
                for obj in obj_list:
                    pt_dict[frame_index ][obj]= torch.tensor([512, 512])
                    point_label_dict[frame_index ][obj] = 1

        else:
            origin_mask = os.path.join(self.origin_mask, name)
            data_seg_3d_shape = np.load(origin_mask + '/0.npy').shape
            data_seg_3d = np.zeros(data_seg_3d_shape + (num_frame,))
            for i in range(num_frame):
                data_seg_3d[..., i] = np.load(os.path.join(origin_mask, f'{i}.npy'))

            for frame_index in range(0, num_frame): 
                img = Image.open(os.path.join(img_path, f'{frame_index }.png')).convert('RGB')
                img = img.resize(newsize)
                img = torch.tensor(np.array(img)).permute(2, 0, 1)
                
                mask = data_seg_3d[..., frame_index]
                obj_list=set()
                obj_list.add(1)
                diff_obj_mask_dict = {}
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict = {}
                elif self.prompt == 'click' :
                    diff_obj_pt_dict = {}
                    diff_obj_point_label_dict = {}
                else:
                    raise ValueError('Prompt not recognized')
                for obj in obj_list:
                    obj_mask = mask == obj
                    # if self.transform_msk:
                    obj_mask = Image.fromarray(obj_mask)
                    obj_mask = obj_mask.resize(newsize)
                    obj_mask = torch.tensor(np.array(obj_mask)).unsqueeze(0).int()
                        # obj_mask = self.transform_msk(obj_mask).int()
                    diff_obj_mask_dict[obj] = obj_mask

                    if self.prompt == 'click':
                        diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)), point_label, seed=None)
                    if self.prompt == 'bbox':
                        diff_obj_bbox_dict[obj] = generate_bbox(obj_mask.squeeze(0).cpu().numpy(), variation=self.variation, seed=self.seed)
                        #获取当前frame中obj所在范围的矩阵的x最小值最大值和y最小值最大值
                img_tensor[frame_index , :, :, :] = img
                mask_dict[frame_index ] = diff_obj_mask_dict  #每一个diff_obj_mask_dict[obj] = obj_mask ，obj是对应frame的所有mask
                if self.prompt == 'bbox':
                    bbox_dict[frame_index ] = diff_obj_bbox_dict
                elif self.prompt == 'click':
                    pt_dict[frame_index ] = diff_obj_pt_dict
                    point_label_dict[frame_index ] = diff_obj_point_label_dict


        image_meta_dict = {'filename_or_obj':name}
        
        if self.prompt == 'bbox':
            return {
                'image':img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict':image_meta_dict,
            }
        elif self.prompt == 'click':
            return {
                'image':img_tensor,
                'label': mask_dict,
                'p_label':point_label_dict,
                'pt':pt_dict,
                'image_meta_dict':image_meta_dict,
            }
        elif self.prompt == 'autonomy':
            return {
                'image':img_tensor,
                'label': mask_dict,
                'p_label':point_label_dict,
                'pt':pt_dict,
                'image_meta_dict':image_meta_dict,
            }
                

class Segconfig:
    def __init__(self,args):
        self.args = args
        self.dataset=None
        self.net=None

    def get_dataloader_net(self):
        
        args = self.args
        btcv_test_dataset = BTCV(args, args.image_path, transform = None, transform_msk= None, mode = None, prompt=args.prompt)

        self.dataset = DataLoader(btcv_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)        
        GPUdevice = torch.device('cuda', args.gpu_device)

        self.net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
        self.net.to(dtype=torch.bfloat16)
        # breakpoint()
        if args.seg_model:
            # print(f"seg_model:{args.seg_model}")
            weights = torch.load(args.seg_model, map_location='cuda')
            self.net.load_state_dict(weights['model_state_dict'],strict=False)
        
    
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # args.path_helper = set_log_dir('logs', args.exp_name)

    def customized_validation_sam(self):
        self.get_dataloader_net()
        args = self.args
        val_loader = self.dataset
        net = self.net
        # eval mode
        net.eval()
        GPUdevice = torch.device('cuda', args.gpu_device)
        tot = 0
        threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
        mix_res = (0,)*1*2
        n_val = len(val_loader)  # the number of batch
        prompt_freq = args.prompt_freq
        pred_dir = args.pred_mask
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        # lossfunc = paper_loss

        prompt = args.prompt
        pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
        lossfunc=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
            for pack in val_loader:
                # breakpoint()
                imgs_tensor = pack['image']
                #name = pack['image_meta_dict']['filename_or_obj']
                if prompt == 'click' or prompt == 'autonomy':    
                    pt_dict = pack['pt']
                    point_labels_dict = pack['p_label']
                elif prompt == 'bbox':
                    bbox_dict = pack['bbox']
                if len(imgs_tensor.size()) == 5:
                    imgs_tensor = imgs_tensor.squeeze(0)
                frame_id = list(range(imgs_tensor.size(0)))
                
                train_state = net.val_init_state(imgs_tensor=imgs_tensor)
                
                prompt_frame_id = list(range(0, len(frame_id), prompt_freq))
                obj_list = set()
                # for id in frame_id:
                #     obj_list += list(mask_dict[id].keys())
                # obj_list = list(set(obj_list))
                obj_list.add(1)
                if len(obj_list) == 0:
                    continue

                name = pack['image_meta_dict']['filename_or_obj']
                # breakpoint()
                dir=os.path.join(pred_dir,name[0])
                # breakpoint()
                if not os.path.exists(dir):
                    os.mkdir(dir)
                
                loss = 0
                pred_iou = 0
                pred_dice = 0
                with torch.no_grad():
                    for id in prompt_frame_id:
                        for ann_obj_id in obj_list:
                            try:
                                if prompt == 'click' or prompt == 'autonomy':
                                    points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                                    labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                    _, _, _ = net.train_add_new_points(
                                        inference_state=train_state,
                                        frame_idx=id,
                                        obj_id=ann_obj_id,
                                        points=points,
                                        labels=labels,
                                        clear_old_points=False,
                                    )
                                elif prompt == 'bbox':
                                    bbox = bbox_dict[id][ann_obj_id]
                                    _, _, _ = net.train_add_new_bbox(
                                        inference_state=train_state,
                                        frame_idx=id,
                                        obj_id=ann_obj_id,
                                        bbox=bbox.to(device=GPUdevice),
                                        clear_old_points=False,
                                    )
                            except KeyError:
                                breakpoint()
                                _, _, _ = net.train_add_new_mask(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                                )
                    video_segments = {}  # video_segments contains the per-frame segmentation results
                
                    for out_frame_idx, out_obj_ids, out_mask_logits in net.propagate_in_video(train_state, start_frame_idx=0):
                        video_segments[out_frame_idx] = {
                            out_obj_id: out_mask_logits[i]
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
                    for id in frame_id:
                        obj_per_frame=torch.zeros(args.image_size,args.image_size, device=GPUdevice)
                        # obj_list_for_frame = list(set(list(mask_dict[id].keys())))
                        # os.mkdir(os.path.join(dir, str(id)))
                        for ann_obj_id in obj_list:
                            pred = video_segments[id][ann_obj_id]
                            pred = pred.unsqueeze(0)
                            # try:
                            #     mask = pack['label'][id][ann_obj_id].to(dtype = torch.float32, device = GPUdevice)
                            # except Exception as e:
                            #     print(f"Error: {e}")
                            # loss += lossfunc(pred, mask)
                            # temp = eval_seg(pred, mask, threshold)
                            # pred_iou += temp[0]
                            # pred_dice += temp[1]

                            pred_mask = (torch.sigmoid(pred) > 0.5).to(torch.uint8).squeeze()

                            obj_per_frame[pred_mask == 1] = ann_obj_id

                        

                        obj_np = obj_per_frame.cpu().numpy().astype(np.uint8)
                        save_path = os.path.join(dir, f"{id}.npy")
                        np.save(save_path, obj_np) 
                    # total_num = len(frame_id) * len(obj_list)
                    # loss = loss / total_num
                    # temp = (pred_iou / total_num, pred_dice / total_num)
                    # tot += loss

                    # mix_res = tuple([sum(a) for a in zip(mix_res, temp)])                 
                    # print(f"iou: {mix_res[0] / n_val}")
                    # print(f"dice: {mix_res[1] / n_val}")


                net.reset_state(train_state)
                pbar.update()
                                

def main(args=None):

     

    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    net.to(dtype=torch.bfloat16)
    # breakpoint()
    if args.seg_model:
        print(args.seg_model)
        weights = torch.load(args.seg_model, map_location='cuda')
        net.load_state_dict(weights['model_state_dict'],strict=False)
    
   
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # args.path_helper = set_log_dir('logs', args.exp_name)
    
    # customized_validation_sam(args, nice_test_loader, net)


if __name__ == '__main__':
    # breakpoint()
    main()


#python inference_3_prompts.py -prompt autonomy -image_path  /root/autodl-tmp/agent/image_test -pred_mask /root/autodl-tmp/agent/mask -seg_model /root/autodl-tmp/Medical_SAM2/logs/BTCV_MedSAM2_2025_04_21_19_32_12/Model/8_tol_10000.0.pth

#python inference_3_prompts.py -prompt click -origin_mask /root/autodl-tmp/agent/test_mask  -image_path  /root/autodl-tmp/agent/image_test -pred_mask /root/autodl-tmp/agent/mask -seg_model /root/autodl-tmp/Medical_SAM2/logs/BTCV_MedSAM2_2025_04_17_17_55_52/Model/latest_epoch_16_tol_0.013650267384946346.pth

