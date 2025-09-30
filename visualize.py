import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import argparse
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm 
from dataloader import get_loader
from network import get_model


def visualize_mask(image_tensor, mask_tensor):
    image = image_tensor.numpy().transpose(1, 2, 0) * 255 
    image = image.astype(np.uint8)  # [224, 224, 3]
    mask = mask_tensor.numpy()  # [224, 224], assuming the mask tensor is softmax output
    
    # Initialize the color mask
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_mapping = np.array([
        [0, 0, 0],        # bg
        [128, 64, 0],     # neck
        [200, 80, 80],    # skin
        [0, 192, 0],      # cloth
        [64, 0, 0],       # l_ear
        [192, 0, 0],      # r_ear
        [0, 128, 128],    # l_brow
        [128, 128, 128],  # r_brow
        [0, 0, 128],      # l_eye
        [128, 0, 128],    # r_eye
        [0, 128, 0],      # nose
        [64, 128, 0],     # mouth
        [64, 0, 128],     # l_lip
        [192, 128, 0],    # u_lip
        [192, 0, 128],    # hair
        [128, 128, 0],    # eye_g
        [64, 128, 128],   # hat
        [192, 128, 128],  # ear_r
        [0, 64, 0]        # neck_l
    ], dtype=np.uint8)

    for index, color in enumerate(color_mapping):
        color_mask[mask == index] = color

    overlayed_image = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)

    return overlayed_image, image, color_mask

def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1) 
    return tensor



def visualize(args, model, test_dataloaders, test_names):
    model.eval()
    for itr, test_dataloader in enumerate(test_dataloaders):
        with torch.no_grad():
            test_pbar = tqdm(test_dataloader, leave=True)
            i = 0
            for batch in test_pbar:
                images, labels, datasets = batch["image"], batch["label"], batch["dataset"]
                images = images.cuda()
                for k in labels.keys():
                    labels[k] = labels[k].cuda()
                datasets = datasets.cuda()
                seg_output = model(images, labels, datasets)
                preds = F.interpolate(seg_output, size=(args.input_resolution,args.input_resolution), mode='bilinear', align_corners=False)
                preds = preds.softmax(dim=1)
                mask = torch.argmax(preds, dim=1)

                mask, face, color_mask = visualize_mask(unnormalize(batch['image'][0]), mask[0].detach().cpu())
                cv2.imwrite(f"{args.visualization_path}/{args.expt_name}_{args.dataset}/face/{i}.jpeg", face[:, :, ::-1])
                cv2.imwrite(f"{args.visualization_path}/{args.expt_name}_{args.dataset}/pred/{i}.jpeg", color_mask[:, :, ::-1])
                cv2.imwrite(f"{args.visualization_path}/{args.expt_name}_{args.dataset}/pred_overlay/{i}.jpeg", mask[:, :, ::-1])
                mask, face, color_mask = visualize_mask(unnormalize(batch['image'][0]), labels["segmentation"][0].detach().cpu())
                cv2.imwrite(f"{args.visualization_path}/{args.expt_name}_{args.dataset}/gt/{i}.jpeg", color_mask[:, :, ::-1])
                cv2.imwrite(f"{args.visualization_path}/{args.expt_name}_{args.dataset}/gt_overlay/{i}.jpeg", mask[:, :, ::-1])
                i += 1

def test(args):
    #Config
    os.makedirs(os.path.join(args.visualization_path, f"{args.expt_name}_{args.dataset}"), exist_ok=True)
    os.makedirs(os.path.join(args.visualization_path, f"{args.expt_name}_{args.dataset}", "pred"), exist_ok=True)
    os.makedirs(os.path.join(args.visualization_path, f"{args.expt_name}_{args.dataset}", "face"), exist_ok=True)
    os.makedirs(os.path.join(args.visualization_path, f"{args.expt_name}_{args.dataset}", "pred_overlay"), exist_ok=True)
    os.makedirs(os.path.join(args.visualization_path, f"{args.expt_name}_{args.dataset}", "gt"), exist_ok=True)
    os.makedirs(os.path.join(args.visualization_path, f"{args.expt_name}_{args.dataset}", "gt_overlay"), exist_ok=True)

    test_dataloaders, test_names = get_loader(args, args.dataset, "test", args.train_bs, args.val_bs, args.test_bs, 0, args.seed, args.num_workers)
    print("Data Loaded")

    #Model Backbone
    model = get_model(args.backbone, args.input_resolution, 19, 19).cuda()
    model.eval()
    print("Model Loaded")

    weights_path = args.model_path
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['state_dict_backbone'])
    visualize(args, model, test_dataloaders, test_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expt_name", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--input_resolution", type=int)
    parser.add_argument("--train_bs", type=int, default=16)
    parser.add_argument("--val_bs", type=int, default=16)
    parser.add_argument("--test_bs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--visualization_path", type=str)
    args = parser.parse_args()  
    args.visualization_path = os.path.join(os.getenv('LOG_PATH'), "visualizations")
    test(args)