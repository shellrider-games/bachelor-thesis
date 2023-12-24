import argparse
import os.path
from matplotlib import pyplot as plt
from torchview import draw_graph
from torchinfo import summary
import graphviz
import torch.cuda
import torch.onnx
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import helper.transforms as T
from dataset import SketchDataset
from helper import utils
from helper.engine import train_one_epoch, evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--num_hidden_layers", help="Number of hidden layers for the predictor.", type=int, default=256)
parser.add_argument("--root", help="Path to the root directory containing the training data.", type=str,
                    default="../data/detection/")
parser.add_argument("--pth", help="Path which the model states are saved to.", type=str, default="./../pretrained/")
parser.add_argument("--load", help="Pth file which is loaded.", type=str, default=None)
parser.add_argument("--batch_size", help="Batch size for loading the data.", type=int, default=2)
parser.add_argument("--num_workers", help="Number of workers for loading.", type=int, default=4)
parser.add_argument("--lr", help="Learning rate.", type=float, default=0.005)
parser.add_argument("--momentum", help="Momentum (for learning).", type=float, default=0.9)
parser.add_argument("--weight_decay", help="Weight decay for stochastic gradient decent.", type=float, default=0.0005)
parser.add_argument("--gamma", help="Multiplicative factor of learning rate decay.", type=float, default=0.1)
parser.add_argument("--step_size", help="Period of learning rate decay.", type=int, default=3)
parser.add_argument("--epochs", help="Number of training epochs.", type=int, default=2000)


def get_model_instance_segmentation(num_classes=2, num_hidden_layer=256):
    """
    @brief Helper function for creating a RCNN based on resnet50 using a FastRCNNPredictor
    with the given number of hidden layers.
    @returns RCNN for detecting sketches
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    print(in_features)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    print(in_features_mask)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       num_hidden_layer,
                                                       num_classes)

    graphviz.set_jupyter_format('png')
    model_graph = draw_graph(model.roi_heads.mask_predictor, input_size=(256, 256, 2, 2), hide_inner_tensors=True, hide_module_functions=True, expand_nested=True, depth=6, save_graph=True, filename="mask_head")
    model_graph.visual_graph
    model_graph = draw_graph(model.roi_heads.box_predictor, input_size=(2, 1024, 1, 1), hide_inner_tensors=True, hide_module_functions=True, expand_nested=True, depth=6, save_graph=True, filename="box_predictor")
    model_graph
    model_graph = draw_graph(model.backbone, input_size=(1,3,224,224), hide_inner_tensors=True, hide_module_functions=True, expand_nested=False, depth=5, save_graph=True, filename="backbone")
    model_graph.visual_graph
    summary(model.backbone, depth=5)
    return model


def get_transforms(train):
    """
    @brief Helper function creating a transformation tensor if the data is being trained.
    Otherwise returns an empty tensor.
    @return Transformation Tensor
    """
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    args = parser.parse_args()

    model = get_model_instance_segmentation(num_hidden_layer=args.num_hidden_layers)

    device = torch.device("cuda")
    dataset = SketchDataset(args.root, get_transforms(train=True))
    dataset_test = SketchDataset(args.root, get_transforms(train=False))

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[:-50])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=utils.collate_fn)


    if args.load is not None:
        model.load_state_dict(torch.load(args.load))
        model.eval()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(2, args.epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)
        torch.save(model.state_dict(), os.path.join(args.pth, "detect_sketches_epoch_{}.pth".format(epoch)))


if __name__ == "__main__":
    main()
