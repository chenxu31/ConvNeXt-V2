# -*- coding: utf-8 -*-

import argparse
import os
import sys
import pdb
import torch
import time
import numpy
import platform
import skimage.io
import glob
import models.fcmae as fcmae
import utils


if platform.system() == 'Windows':
    UTIL_DIR = r"E:\我的坚果云\sourcecode\python\util"
else:
    UTIL_DIR = r"/home/chenxu/Nutstore Files/我的坚果云/sourcecode/python/util"

sys.path.append(UTIL_DIR)
import common_metrics
import common_pelvic_pt as common_pelvic
import common_net_pt as common_net


def main(device, args):
    model = fcmae.__dict__["convnextv2_atto"](
        img_size=256,
        in_chans=1,
        mask_ratio=args.mask_ratio,
        decoder_depth=args.decoder_depth,
        decoder_embed_dim=args.decoder_embed_dim,
        clamp_out=1,
        norm_pix_loss=False
    )
    checkpoint = torch.load(args.checkpoint_file)

    """
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # remove decoder weights
    checkpoint_model_keys = list(checkpoint_model.keys())
    for k in checkpoint_model_keys:
        if 'decoder' in k or 'mask_token' in k or \
                'proj' in k or 'pred' in k:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    checkpoint_model = utils.remap_checkpoint_keys(checkpoint_model)
    utils.load_state_dict(model, checkpoint_model, prefix="")
    """
    model.load_state_dict(checkpoint["model"])

    model.to(device)
    model.eval()

    if args.modality == "ct":
        test_data, _, _, _ = common_pelvic.load_test_data(args.data_dir)
    elif args.modality == "cbct":
        _, test_data, _, _ = common_pelvic.load_test_data(args.data_dir)
    else:
        assert 0

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with torch.no_grad():
        in_patch = torch.from_numpy(test_data[0:1, 100:101, :, :]).to(device)

        loss, ret, mask = model(in_patch, mask_ratio=args.mask_ratio)
        print(loss)

        ret = model.unpatchify(ret.reshape(ret.shape[0], ret.shape[1], ret.shape[2]*ret.shape[3]).permute((0, 2, 1)))

        ret = ret.cpu().detach().numpy()[0, 0, :, :]
        ret = common_pelvic.data_restore(ret)

        skimage.io.imsave("ori.jpg", common_pelvic.data_restore(in_patch.cpu().detach().numpy()[0, 0, :, :]))#common_pelvic.data_restore(model.unpatchify(model.patchify(in_patch)).cpu().detach().numpy())[0, 0, :, :])
        skimage.io.imsave("syn.jpg", ret)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=int, default=0, help="gpu device id")
    parser.add_argument('--data_dir', type=str, default=r'/home/chenxu/datasets/pelvic/h5_data_nonrigid', help='path of the dataset')
    parser.add_argument('--checkpoint_file', type=str, default=r'/home/chenxu/training/checkpoints/convnextv2/atto_ct/checkpoint-799.pth', help='path of the dataset')
    parser.add_argument('--mask_ratio', default=0.6, type=float, help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--decoder_depth', type=int, default=1)
    parser.add_argument('--decoder_embed_dim', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='outputs', help="the output directory")
    parser.add_argument('--modality', type=str, default='ct', choices=["ct", "cbct"], help="the output directory")

    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")

    main(device, args)
