import os
import os.path as osp
import random

import albumentations as albu
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CLASSES = ("background", "greatwall")
PALETTE = [[0, 0, 0], [255, 255, 255]]

ORIGIN_IMG_SIZE = (512, 512)
INPUT_IMG_SIZE = (512, 512)
TEST_IMG_SIZE = (512, 512)


def get_training_transform():
	train_transform = [
		albu.HorizontalFlip(p=0.5),
		albu.VerticalFlip(p=0.5),
		albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
		albu.Normalize(),
	]
	return albu.Compose(
		train_transform,
		additional_targets={
			"ridge": "mask",
			"dem": "mask",
			"connect0": "mask",
			"connect1": "mask",
			"connect2": "mask",
			"connect_d1_0": "mask",
			"connect_d1_1": "mask",
			"connect_d1_2": "mask",
		},
	)


def train_aug(
	img,
	mask,
	dem=None,
	ridge=None,
	connect0=None,
	connect1=None,
	connect2=None,
	connect_d1_0=None,
	connect_d1_1=None,
	connect_d1_2=None,
):
	img, mask = np.array(img), np.array(mask)
	extra = {}
	if dem is not None:
		extra["dem"] = np.array(dem)
	if ridge is not None:
		extra["ridge"] = np.array(ridge)
	if connect0 is not None:
		extra["connect0"] = np.array(connect0)
		extra["connect1"] = np.array(connect1)
		extra["connect2"] = np.array(connect2)
		extra["connect_d1_0"] = np.array(connect_d1_0)
		extra["connect_d1_1"] = np.array(connect_d1_1)
		extra["connect_d1_2"] = np.array(connect_d1_2)

	aug = get_training_transform()(image=img.copy(), mask=mask.copy(), **extra)
	img, mask = aug["image"], aug["mask"]
	if connect0 is not None:
		return (
			img,
			mask,
			aug.get("dem"),
			aug.get("ridge"),
			aug.get("connect0"),
			aug.get("connect1"),
			aug.get("connect2"),
			aug.get("connect_d1_0"),
			aug.get("connect_d1_1"),
			aug.get("connect_d1_2"),
		)
	if dem is not None or ridge is not None:
		return img, mask, aug.get("dem"), aug.get("ridge")
	return img, mask


def get_val_transform():
	val_transform = [
		albu.Normalize(),
	]
	return albu.Compose(
		val_transform,
		additional_targets={
			"ridge": "mask",
			"dem": "mask",
			"connect0": "mask",
			"connect1": "mask",
			"connect2": "mask",
			"connect_d1_0": "mask",
			"connect_d1_1": "mask",
			"connect_d1_2": "mask",
		},
	)


def val_aug(
	img,
	mask,
	dem=None,
	ridge=None,
	connect0=None,
	connect1=None,
	connect2=None,
	connect_d1_0=None,
	connect_d1_1=None,
	connect_d1_2=None,
):
	img, mask = np.array(img), np.array(mask)
	extra = {}
	if dem is not None:
		extra["dem"] = np.array(dem)
	if ridge is not None:
		extra["ridge"] = np.array(ridge)
	if connect0 is not None:
		extra["connect0"] = np.array(connect0)
		extra["connect1"] = np.array(connect1)
		extra["connect2"] = np.array(connect2)
		extra["connect_d1_0"] = np.array(connect_d1_0)
		extra["connect_d1_1"] = np.array(connect_d1_1)
		extra["connect_d1_2"] = np.array(connect_d1_2)

	aug = get_val_transform()(image=img.copy(), mask=mask.copy(), **extra)
	img, mask = aug["image"], aug["mask"]
	if connect0 is not None:
		return (
			img,
			mask,
			aug.get("dem"),
			aug.get("ridge"),
			aug.get("connect0"),
			aug.get("connect1"),
			aug.get("connect2"),
			aug.get("connect_d1_0"),
			aug.get("connect_d1_1"),
			aug.get("connect_d1_2"),
		)
	if dem is not None or ridge is not None:
		return img, mask, aug.get("dem"), aug.get("ridge")
	return img, mask


class GreatWallDataset(Dataset):
	def __init__(
		self,
		data_root="/home/dataica401/Dataset/slwang/output_hebei_complete_json_v5",
		split="train",
		transform=None,
		mosaic_ratio=0.0,
		img_dir="images",
		aug_images_dir="images_aug",
		aug_images_list="kept_images.txt",
		mask_dir="labels",
		dem_dir="dem",
		ridge_dir="ridge_tif_kernel_filtered_10m",
		connect_dir="connect_8_d1",
		connect_d1_dir="connect_8_d3",
		img_suffix=".png",
		mask_suffix=".png",
		mask_postfix="_mask",
		dem_suffix=".tif",
		ridge_postfix="_kde",
		ridge_suffix=".png",
		connect_suffix=".png",
		img_size=ORIGIN_IMG_SIZE,
		use_dem=False,
		use_ridge=False,
		use_connectivity=False,
		enable_aug_images=False,
	):
		self.data_root = data_root
		self.split = split
		self.transform = transform
		self.mosaic_ratio = mosaic_ratio
		self.img_dir = img_dir
		self.aug_images_dir = aug_images_dir
		self.aug_images_list = aug_images_list
		self.mask_dir = mask_dir
		self.dem_dir = dem_dir
		self.ridge_dir = ridge_dir
		self.connect_dir = connect_dir
		self.connect_d1_dir = connect_d1_dir
		self.img_suffix = img_suffix
		self.mask_suffix = mask_suffix
		self.mask_postfix = mask_postfix
		self.dem_suffix = dem_suffix
		self.ridge_postfix = ridge_postfix
		self.ridge_suffix = ridge_suffix
		self.connect_suffix = connect_suffix
		self.img_size = img_size
		self.use_dem = use_dem
		self.use_ridge = use_ridge
		self.use_connectivity = use_connectivity
		self.enable_aug_images = enable_aug_images
		self.img_ids = self.get_img_ids()
		self._custom_img_paths = {}
		self._aug_img_ids = set()
		self._append_aug_images()

	def __getitem__(self, index):
		p_ratio = random.random()
		if p_ratio < self.mosaic_ratio and self.split == "train":
			img, mask, dem, ridge, connect0, connect1, connect2, connect_d1_0, connect_d1_1, connect_d1_2 = self.load_mosaic_img_and_mask(index)
		else:
			img, mask, dem, ridge, connect0, connect1, connect2, connect_d1_0, connect_d1_1, connect_d1_2 = self.load_img_and_mask(index)

		if self.transform:
			if self.use_connectivity:
				img, mask, dem, ridge, connect0, connect1, connect2, connect_d1_0, connect_d1_1, connect_d1_2 = self.transform(
					img,
					mask,
					dem,
					ridge,
					connect0,
					connect1,
					connect2,
					connect_d1_0,
					connect_d1_1,
					connect_d1_2,
				)
			elif dem is not None or ridge is not None:
				img, mask, dem, ridge = self.transform(img, mask, dem, ridge)
			else:
				img, mask = self.transform(img, mask)
		else:
			img, mask = np.array(img), np.array(mask)
			if dem is not None:
				dem = np.array(dem)
			if ridge is not None:
				ridge = np.array(ridge)
			if self.use_connectivity:
				connect0 = np.array(connect0)
				connect1 = np.array(connect1)
				connect2 = np.array(connect2)
				connect_d1_0 = np.array(connect_d1_0)
				connect_d1_1 = np.array(connect_d1_1)
				connect_d1_2 = np.array(connect_d1_2)

		# Masks are saved as 0/255, convert them to class ids 0/1.
		mask = (mask > 0).astype(np.uint8)

		img = torch.from_numpy(img).permute(2, 0, 1).float()
		mask = torch.from_numpy(mask).long()
		if dem is not None:
			dem = dem.astype(np.float32)
			if dem.ndim == 3:
				dem = dem[:, :, 0]
			dem_min = float(dem.min())
			dem_max = float(dem.max())
			if dem_max > dem_min:
				dem = (dem - dem_min) / (dem_max - dem_min)
			dem = torch.from_numpy(dem).unsqueeze(0).float()
		if ridge is not None:
			ridge = ridge.astype(np.float32)
			if ridge.ndim == 3:
				ridge = ridge[:, :, 0]
			if ridge.max() > 1.0:
				ridge = ridge / 255.0
			ridge = torch.from_numpy(ridge).unsqueeze(0).float()
		if self.use_connectivity:
			connect0 = self._to_connect_tensor(connect0)
			connect1 = self._to_connect_tensor(connect1)
			connect2 = self._to_connect_tensor(connect2)
			connect_d1_0 = self._to_connect_tensor(connect_d1_0)
			connect_d1_1 = self._to_connect_tensor(connect_d1_1)
			connect_d1_2 = self._to_connect_tensor(connect_d1_2)
		img_id = self.img_ids[index]
		result = {"img": img, "gt_semantic_seg": mask, "img_id": img_id}
		if dem is not None:
			result["dem"] = dem
		if ridge is not None:
			result["ridge"] = ridge
		if self.use_connectivity:
			result.update(
				{
					"connect0": connect0,
					"connect1": connect1,
					"connect2": connect2,
					"connect_d1_0": connect_d1_0,
					"connect_d1_1": connect_d1_1,
					"connect_d1_2": connect_d1_2,
				}
			)
		return result

	def __len__(self):
		return len(self.img_ids)

	def get_img_ids(self):
		split_file = osp.join(self.data_root, f"{self.split}.txt")
		if not osp.exists(split_file):
			raise FileNotFoundError(f"Split file not found: {split_file}")

		with open(split_file, "r", encoding="utf-8") as f:
			img_ids = [line.strip() for line in f if line.strip()]
		return img_ids

	def _build_mask_name(self, img_name):
		stem, ext = osp.splitext(img_name)
		ext = ext if ext else self.mask_suffix
		return f"{stem}{self.mask_postfix}{ext}"

	def _build_mask_name_from_base(self, base_name):
		return f"{base_name}{self.mask_postfix}{self.mask_suffix}"

	def _build_dem_name(self, img_name):
		stem, _ = osp.splitext(img_name)
		return f"{stem}{self.dem_suffix}"

	def _build_connect_names(self, img_name):
		mask_name = self._build_mask_name(img_name)
		stem, _ = osp.splitext(mask_name)
		return [
			f"{stem}_0{self.connect_suffix}",
			f"{stem}_1{self.connect_suffix}",
			f"{stem}_2{self.connect_suffix}",
		]

	def _build_ridge_name(self, img_name):
		stem, _ = osp.splitext(img_name)
		return f"{stem}{self.ridge_postfix}{self.ridge_suffix}"

	def _parse_aug_base_name(self, img_name):
		stem, _ = osp.splitext(img_name)
		if "_mask_out_" in stem:
			return stem.split("_mask_out_")[0]
		return stem

	def _append_aug_images(self):
		if not self.enable_aug_images:
			return
		if self.split != "train":
			return
		list_path = osp.join(self.data_root, self.aug_images_list)
		if not osp.exists(list_path):
			return
		with open(list_path, "r", encoding="utf-8") as f:
			lines = [line.strip() for line in f if line.strip()]
		aug_dir = osp.join(self.data_root, self.aug_images_dir)
		for line in lines:
			img_path = osp.join(aug_dir, line)
			if not osp.isfile(img_path):
				continue
			self.img_ids.append(line)
			self._custom_img_paths[line] = img_path
			self._aug_img_ids.add(line)

	def _to_connect_tensor(self, connect):
		connect = connect.astype(np.float32)
		if connect.ndim == 2:
			connect = connect[:, :, None]
		if connect.max() > 1.0:
			connect = connect / 255.0
		return torch.from_numpy(connect).permute(2, 0, 1).float()

	def load_img_and_mask(self, index):
		img_id = self.img_ids[index]
		img_name = self._custom_img_paths.get(img_id, osp.join(self.data_root, self.img_dir, img_id))
		if img_id in self._aug_img_ids:
			base_name = self._parse_aug_base_name(img_id)
			mask_name = osp.join(self.data_root, self.mask_dir, self._build_mask_name_from_base(base_name))
			dem_name = osp.join(self.data_root, self.dem_dir, f"{base_name}{self.dem_suffix}")
			ridge_name = osp.join(self.data_root, self.ridge_dir, f"{base_name}{self.ridge_postfix}{self.ridge_suffix}")
			connect_names = self._build_connect_names(f"{base_name}{self.mask_suffix}")
		else:
			mask_name = osp.join(self.data_root, self.mask_dir, self._build_mask_name(img_id))
			dem_name = osp.join(self.data_root, self.dem_dir, self._build_dem_name(img_id))
			ridge_name = osp.join(self.data_root, self.ridge_dir, self._build_ridge_name(img_id))
			connect_names = self._build_connect_names(img_id)
		connect_paths = [osp.join(self.data_root, self.connect_dir, name) for name in connect_names]
		connect_d1_paths = [osp.join(self.data_root, self.connect_d1_dir, name) for name in connect_names]

		img = Image.open(img_name).convert("RGB")
		mask = Image.open(mask_name).convert("L")
		ridge = None
		connect0 = connect1 = connect2 = None
		connect_d1_0 = connect_d1_1 = connect_d1_2 = None
		if self.use_dem:
			dem = Image.open(dem_name)
		else:
			dem = None
		if self.use_ridge:
			ridge = Image.open(ridge_name).convert("L")
		if self.use_connectivity:
			connect0 = Image.open(connect_paths[0]).convert("RGB")
			connect1 = Image.open(connect_paths[1]).convert("RGB")
			connect2 = Image.open(connect_paths[2]).convert("RGB")
			connect_d1_0 = Image.open(connect_d1_paths[0]).convert("RGB")
			connect_d1_1 = Image.open(connect_d1_paths[1]).convert("RGB")
			connect_d1_2 = Image.open(connect_d1_paths[2]).convert("RGB")
		return img, mask, dem, ridge, connect0, connect1, connect2, connect_d1_0, connect_d1_1, connect_d1_2

	def load_mosaic_img_and_mask(self, index):
		indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
		img_a, mask_a, dem_a, ridge_a, connect0_a, connect1_a, connect2_a, connect_d1_0_a, connect_d1_1_a, connect_d1_2_a = self.load_img_and_mask(indexes[0])
		img_b, mask_b, dem_b, ridge_b, connect0_b, connect1_b, connect2_b, connect_d1_0_b, connect_d1_1_b, connect_d1_2_b = self.load_img_and_mask(indexes[1])
		img_c, mask_c, dem_c, ridge_c, connect0_c, connect1_c, connect2_c, connect_d1_0_c, connect_d1_1_c, connect_d1_2_c = self.load_img_and_mask(indexes[2])
		img_d, mask_d, dem_d, ridge_d, connect0_d, connect1_d, connect2_d, connect_d1_0_d, connect_d1_1_d, connect_d1_2_d = self.load_img_and_mask(indexes[3])

		img_a, mask_a = np.array(img_a), np.array(mask_a)
		img_b, mask_b = np.array(img_b), np.array(mask_b)
		img_c, mask_c = np.array(img_c), np.array(mask_c)
		img_d, mask_d = np.array(img_d), np.array(mask_d)
		if self.use_dem:
			dem_a = np.array(dem_a)
			dem_b = np.array(dem_b)
			dem_c = np.array(dem_c)
			dem_d = np.array(dem_d)
		if self.use_ridge:
			ridge_a = np.array(ridge_a)
			ridge_b = np.array(ridge_b)
			ridge_c = np.array(ridge_c)
			ridge_d = np.array(ridge_d)
		if self.use_connectivity:
			connect0_a = np.array(connect0_a)
			connect1_a = np.array(connect1_a)
			connect2_a = np.array(connect2_a)
			connect_d1_0_a = np.array(connect_d1_0_a)
			connect_d1_1_a = np.array(connect_d1_1_a)
			connect_d1_2_a = np.array(connect_d1_2_a)
			connect0_b = np.array(connect0_b)
			connect1_b = np.array(connect1_b)
			connect2_b = np.array(connect2_b)
			connect_d1_0_b = np.array(connect_d1_0_b)
			connect_d1_1_b = np.array(connect_d1_1_b)
			connect_d1_2_b = np.array(connect_d1_2_b)
			connect0_c = np.array(connect0_c)
			connect1_c = np.array(connect1_c)
			connect2_c = np.array(connect2_c)
			connect_d1_0_c = np.array(connect_d1_0_c)
			connect_d1_1_c = np.array(connect_d1_1_c)
			connect_d1_2_c = np.array(connect_d1_2_c)
			connect0_d = np.array(connect0_d)
			connect1_d = np.array(connect1_d)
			connect2_d = np.array(connect2_d)
			connect_d1_0_d = np.array(connect_d1_0_d)
			connect_d1_1_d = np.array(connect_d1_1_d)
			connect_d1_2_d = np.array(connect_d1_2_d)

		h, w = self.img_size
		start_x = w // 4
		start_y = h // 4
		offset_x = random.randint(start_x, w - start_x)
		offset_y = random.randint(start_y, h - start_y)

		crop_size_a = (offset_x, offset_y)
		crop_size_b = (w - offset_x, offset_y)
		crop_size_c = (offset_x, h - offset_y)
		crop_size_d = (w - offset_x, h - offset_y)

		extra_targets = {
			"ridge": "mask",
			"dem": "mask",
			"connect0": "mask",
			"connect1": "mask",
			"connect2": "mask",
			"connect_d1_0": "mask",
			"connect_d1_1": "mask",
			"connect_d1_2": "mask",
		}
		random_crop_a = albu.Compose([albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])], additional_targets=extra_targets)
		random_crop_b = albu.Compose([albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])], additional_targets=extra_targets)
		random_crop_c = albu.Compose([albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])], additional_targets=extra_targets)
		random_crop_d = albu.Compose([albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])], additional_targets=extra_targets)

		if self.use_connectivity:
			crop_kwargs_a = {
				"image": img_a.copy(),
				"mask": mask_a.copy(),
				"connect0": connect0_a.copy(),
				"connect1": connect1_a.copy(),
				"connect2": connect2_a.copy(),
				"connect_d1_0": connect_d1_0_a.copy(),
				"connect_d1_1": connect_d1_1_a.copy(),
				"connect_d1_2": connect_d1_2_a.copy(),
			}
			crop_kwargs_b = {
				"image": img_b.copy(),
				"mask": mask_b.copy(),
				"connect0": connect0_b.copy(),
				"connect1": connect1_b.copy(),
				"connect2": connect2_b.copy(),
				"connect_d1_0": connect_d1_0_b.copy(),
				"connect_d1_1": connect_d1_1_b.copy(),
				"connect_d1_2": connect_d1_2_b.copy(),
			}
			crop_kwargs_c = {
				"image": img_c.copy(),
				"mask": mask_c.copy(),
				"connect0": connect0_c.copy(),
				"connect1": connect1_c.copy(),
				"connect2": connect2_c.copy(),
				"connect_d1_0": connect_d1_0_c.copy(),
				"connect_d1_1": connect_d1_1_c.copy(),
				"connect_d1_2": connect_d1_2_c.copy(),
			}
			crop_kwargs_d = {
				"image": img_d.copy(),
				"mask": mask_d.copy(),
				"connect0": connect0_d.copy(),
				"connect1": connect1_d.copy(),
				"connect2": connect2_d.copy(),
				"connect_d1_0": connect_d1_0_d.copy(),
				"connect_d1_1": connect_d1_1_d.copy(),
				"connect_d1_2": connect_d1_2_d.copy(),
			}
			if self.use_dem:
				crop_kwargs_a["dem"] = dem_a.copy()
				crop_kwargs_b["dem"] = dem_b.copy()
				crop_kwargs_c["dem"] = dem_c.copy()
				crop_kwargs_d["dem"] = dem_d.copy()
			if self.use_ridge:
				crop_kwargs_a["ridge"] = ridge_a.copy()
				crop_kwargs_b["ridge"] = ridge_b.copy()
				crop_kwargs_c["ridge"] = ridge_c.copy()
				crop_kwargs_d["ridge"] = ridge_d.copy()
			crop_a = random_crop_a(**crop_kwargs_a)
			crop_b = random_crop_b(**crop_kwargs_b)
			crop_c = random_crop_c(**crop_kwargs_c)
			crop_d = random_crop_d(**crop_kwargs_d)
		elif self.use_dem or self.use_ridge:
			crop_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy(), dem=dem_a.copy() if self.use_dem else None, ridge=ridge_a.copy() if self.use_ridge else None)
			crop_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy(), dem=dem_b.copy() if self.use_dem else None, ridge=ridge_b.copy() if self.use_ridge else None)
			crop_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy(), dem=dem_c.copy() if self.use_dem else None, ridge=ridge_c.copy() if self.use_ridge else None)
			crop_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy(), dem=dem_d.copy() if self.use_dem else None, ridge=ridge_d.copy() if self.use_ridge else None)
		else:
			crop_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
			crop_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
			crop_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
			crop_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

		img_crop_a, mask_crop_a = crop_a["image"], crop_a["mask"]
		img_crop_b, mask_crop_b = crop_b["image"], crop_b["mask"]
		img_crop_c, mask_crop_c = crop_c["image"], crop_c["mask"]
		img_crop_d, mask_crop_d = crop_d["image"], crop_d["mask"]
		if self.use_dem:
			dem_crop_a = crop_a["dem"]
			dem_crop_b = crop_b["dem"]
			dem_crop_c = crop_c["dem"]
			dem_crop_d = crop_d["dem"]
		if self.use_ridge:
			ridge_crop_a = crop_a["ridge"]
			ridge_crop_b = crop_b["ridge"]
			ridge_crop_c = crop_c["ridge"]
			ridge_crop_d = crop_d["ridge"]
		if self.use_connectivity:
			connect0_crop_a = crop_a["connect0"]
			connect1_crop_a = crop_a["connect1"]
			connect2_crop_a = crop_a["connect2"]
			connect_d1_0_crop_a = crop_a["connect_d1_0"]
			connect_d1_1_crop_a = crop_a["connect_d1_1"]
			connect_d1_2_crop_a = crop_a["connect_d1_2"]
			connect0_crop_b = crop_b["connect0"]
			connect1_crop_b = crop_b["connect1"]
			connect2_crop_b = crop_b["connect2"]
			connect_d1_0_crop_b = crop_b["connect_d1_0"]
			connect_d1_1_crop_b = crop_b["connect_d1_1"]
			connect_d1_2_crop_b = crop_b["connect_d1_2"]
			connect0_crop_c = crop_c["connect0"]
			connect1_crop_c = crop_c["connect1"]
			connect2_crop_c = crop_c["connect2"]
			connect_d1_0_crop_c = crop_c["connect_d1_0"]
			connect_d1_1_crop_c = crop_c["connect_d1_1"]
			connect_d1_2_crop_c = crop_c["connect_d1_2"]
			connect0_crop_d = crop_d["connect0"]
			connect1_crop_d = crop_d["connect1"]
			connect2_crop_d = crop_d["connect2"]
			connect_d1_0_crop_d = crop_d["connect_d1_0"]
			connect_d1_1_crop_d = crop_d["connect_d1_1"]
			connect_d1_2_crop_d = crop_d["connect_d1_2"]

		top = np.concatenate((img_crop_a, img_crop_b), axis=1)
		bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
		img = np.concatenate((top, bottom), axis=0)

		top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
		bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
		mask = np.concatenate((top_mask, bottom_mask), axis=0)
		if self.use_dem:
			top_dem = np.concatenate((dem_crop_a, dem_crop_b), axis=1)
			bottom_dem = np.concatenate((dem_crop_c, dem_crop_d), axis=1)
			dem = np.concatenate((top_dem, bottom_dem), axis=0)
		if self.use_ridge:
			top_ridge = np.concatenate((ridge_crop_a, ridge_crop_b), axis=1)
			bottom_ridge = np.concatenate((ridge_crop_c, ridge_crop_d), axis=1)
			ridge = np.concatenate((top_ridge, bottom_ridge), axis=0)
		if self.use_connectivity:
			connect0_top = np.concatenate((connect0_crop_a, connect0_crop_b), axis=1)
			connect0_bottom = np.concatenate((connect0_crop_c, connect0_crop_d), axis=1)
			connect0 = np.concatenate((connect0_top, connect0_bottom), axis=0)
			connect1_top = np.concatenate((connect1_crop_a, connect1_crop_b), axis=1)
			connect1_bottom = np.concatenate((connect1_crop_c, connect1_crop_d), axis=1)
			connect1 = np.concatenate((connect1_top, connect1_bottom), axis=0)
			connect2_top = np.concatenate((connect2_crop_a, connect2_crop_b), axis=1)
			connect2_bottom = np.concatenate((connect2_crop_c, connect2_crop_d), axis=1)
			connect2 = np.concatenate((connect2_top, connect2_bottom), axis=0)
			connect_d1_0_top = np.concatenate((connect_d1_0_crop_a, connect_d1_0_crop_b), axis=1)
			connect_d1_0_bottom = np.concatenate((connect_d1_0_crop_c, connect_d1_0_crop_d), axis=1)
			connect_d1_0 = np.concatenate((connect_d1_0_top, connect_d1_0_bottom), axis=0)
			connect_d1_1_top = np.concatenate((connect_d1_1_crop_a, connect_d1_1_crop_b), axis=1)
			connect_d1_1_bottom = np.concatenate((connect_d1_1_crop_c, connect_d1_1_crop_d), axis=1)
			connect_d1_1 = np.concatenate((connect_d1_1_top, connect_d1_1_bottom), axis=0)
			connect_d1_2_top = np.concatenate((connect_d1_2_crop_a, connect_d1_2_crop_b), axis=1)
			connect_d1_2_bottom = np.concatenate((connect_d1_2_crop_c, connect_d1_2_crop_d), axis=1)
			connect_d1_2 = np.concatenate((connect_d1_2_top, connect_d1_2_bottom), axis=0)

		img = Image.fromarray(np.ascontiguousarray(img))
		mask = Image.fromarray(np.ascontiguousarray(mask))
		if self.use_dem:
			dem = Image.fromarray(np.ascontiguousarray(dem))
		else:
			dem = None
		if self.use_ridge:
			ridge = Image.fromarray(np.ascontiguousarray(ridge))
		else:
			ridge = None
		if self.use_connectivity:
			connect0 = Image.fromarray(np.ascontiguousarray(connect0))
			connect1 = Image.fromarray(np.ascontiguousarray(connect1))
			connect2 = Image.fromarray(np.ascontiguousarray(connect2))
			connect_d1_0 = Image.fromarray(np.ascontiguousarray(connect_d1_0))
			connect_d1_1 = Image.fromarray(np.ascontiguousarray(connect_d1_1))
			connect_d1_2 = Image.fromarray(np.ascontiguousarray(connect_d1_2))
		return img, mask, dem, ridge, connect0, connect1, connect2, connect_d1_0, connect_d1_1, connect_d1_2


greatwall_train_dataset = GreatWallDataset(split="train", transform=train_aug, mosaic_ratio=0.0)
greatwall_val_dataset = GreatWallDataset(split="val", transform=val_aug, mosaic_ratio=0.0)

# No dedicated test split: use validation split as test split.
greatwall_test_dataset = greatwall_val_dataset
