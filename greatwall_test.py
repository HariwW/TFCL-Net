import argparse
import multiprocessing as mp
import multiprocessing.pool as mpp
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import ttach as tta
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_supervision import *
from tools.tta_utils import DualInputSegmentationTTAWrapper


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [0, 0, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 255, 255]
    return mask_rgb


def img_writer(inp):
    mask, mask_id, rgb = inp
    if rgb:
        mask_name = mask_id + ".png"
        mask_rgb = label2rgb(mask)
        mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_name, mask_bgr)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name = mask_id + ".png"
        cv2.imwrite(mask_name, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to config")
    arg("-o", "--output_path", type=Path, required=True, help="Path where to save resulting masks")
    arg("--ckpt_path", type=Path, default=None, help="Checkpoint path; overrides the config value")
    arg("-t", "--tta", default=None, choices=[None, "d4", "lr"], help="Test time augmentation")
    arg("--rgb", action="store_true", help="Whether output rgb images")
    return parser.parse_args()


def main():
    seed_everything(42)
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    ckpt_path = str(args.ckpt_path) if args.ckpt_path is not None else os.path.join(
        config.weights_path, config.test_weights_name + ".ckpt"
    )
    model = Supervision_Train.load_from_checkpoint(ckpt_path, config=config)
    model = model.to("cuda")
    model.eval()

    test_dataset = getattr(config, "val_dataset", config.test_dataset)
    use_dem = getattr(config, "use_dem", None)
    use_ridge = getattr(config, "use_ridge", None)
    if use_dem is None:
        use_dem = bool(getattr(test_dataset, "use_dem", False))
    if use_ridge is None:
        use_ridge = bool(getattr(test_dataset, "use_ridge", False))

    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()

    if args.tta == "lr":
        transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        if use_dem or use_ridge:
            model = DualInputSegmentationTTAWrapper(model, transforms, merge_mode="mean")
        else:
            model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                # tta.Rotate90(angles=[90, 180, 270]),
                # # —— 光照/颜色增强（不改变尺寸）——
                tta.Multiply(factors=[0.9, 1.0, 1.1]),   # 明暗增强
                # tta.Gamma(gammas=[0.9, 1.0, 1.1]),       # Gamma 调整
                # tta.Scale(scales=[0.75, 1.0, 1.25], interpolation="bicubic", align_corners=True),
            ]
        )
        if use_dem or use_ridge:
            model = DualInputSegmentationTTAWrapper(model, transforms, merge_mode="mean")
        else:
            model = tta.SegmentationTTAWrapper(model, transforms)

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        results = []
        for inp in tqdm(test_loader):
            if (use_dem and "dem" in inp) or (use_ridge and "ridge" in inp):
                dem = inp.get("dem")
                ridge = inp.get("ridge")
                dem = dem.cuda() if dem is not None else None
                ridge = ridge.cuda() if ridge is not None else None
                raw_predictions = model(inp["img"].cuda(), dem, ridge)
            else:
                raw_predictions = model(inp["img"].cuda())
            if isinstance(raw_predictions, (tuple, list)):
                raw_predictions = raw_predictions[0]
            image_ids = inp["img_id"]
            masks_true = inp["gt_semantic_seg"]

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].cpu().numpy())
                mask_stem = Path(str(image_ids[i])).stem
                results.append((mask, str(args.output_path / mask_stem), args.rgb))

    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    OA = evaluator.OA()
    for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class, f1_per_class):
        print("F1_{}:{}, IOU_{}:{}".format(class_name, class_f1, class_name, class_iou))
    print("F1:{}, mIOU:{}, OA:{}".format(np.nanmean(f1_per_class), np.nanmean(iou_per_class), OA))

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    print("images writing spends: {} s".format(t1 - t0))


if __name__ == "__main__":
    main()
