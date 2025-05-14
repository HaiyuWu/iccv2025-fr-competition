import argparse
import numpy as np
import torch
from os import path, makedirs
from torch.nn import DataParallel
from model import iresnet, PartialFC_V2, get_vit
from data import TestDataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm


class Extractor(object):
    def __init__(self, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.create_model(args)

        if torch.cuda.device_count() > 0:
            print(f"Model will use {torch.cuda.device_count()} GPUs!")
            self.model = DataParallel(self.model)
            self.model = self.model.to(self.device)
        self.model.eval()

    def create_model(self, args):
        if args.model == "iresnet":
            model = iresnet(args.depth, fp16=True, mode=args.mode)
        elif args.model == "vit":
            model = get_vit(args.depth)
        model.load_state_dict(torch.load(args.model_path))
        return model

    def get_im_id(self, im_path):
        sep = im_path.split("/")
        return f"{sep[-2]}/{sep[-1][:-3]}"

    def l2_norm(self, input: np.array, axis=1):
        norm = np.linalg.norm(input, 2, axis, True)
        output = np.divide(input, norm)
        return output

    def extract(self, args):
        bs = args.batch_size
        name = args.image_paths.split("/")[-2]
        features = []
        images = np.load(args.image_paths)
        images = ((images / 255.) - 0.5) / 0.5
        for i in tqdm(range(0, len(images), bs)):
            im_or = torch.tensor(images[i:i+bs].astype(np.float32))
            im_flip = torch.tensor(images[i:i+bs][..., ::-1].astype(np.float32))
            f_or = self.model(im_or).cpu().detach().numpy()
            f_flip = self.model(im_flip).cpu().detach().numpy()
            features.append(f_or + f_flip)
        features = self.l2_norm(np.concatenate(features, axis=0))
        f1 = features[0::2]
        f2 = features[1::2]

        diff = np.subtract(f1, f2)
        res = np.sum(np.square(diff), 1)
        print(f"./{name}_{args.dataset_scale}_result.txt")
        np.savetxt(f"./{name}_{args.dataset_scale}_result.txt", res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image feature extraction."
    )
    parser.add_argument(
        "--model_path", "-model_path", help="model path.", type=str
    )
    parser.add_argument(
        "--model", "-model", help="iresnet/vit.", type=str, default="iresnet"
    )
    parser.add_argument(
        "--mode", "-mode", help="using SE attention [normal/se].", type=str, default="se"
    )
    parser.add_argument(
        "--depth", "-d",
        help="layers size: resnet [18, 34, 50, 100, 152, 200] / vit [s, b, l].",
        default="50",
        type=str
    )
    parser.add_argument("--batch_size", "-b", help="Batch size.", default=512, type=int)
    parser.add_argument("--image_paths", "-i", help="A file contains image paths.", type=str)
    parser.add_argument("--dataset_scale", "-scale", help="scale of the dataset.", choices=["10K", "20K", "100K"],
                        type=str, required=True)

    args = parser.parse_args()

    extractor = Extractor(args)
    extractor.extract(args)
