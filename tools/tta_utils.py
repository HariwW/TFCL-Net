import torch
from torch import nn


class DualInputSegmentationTTAWrapper(nn.Module):
    def __init__(self, model, transforms, merge_mode="mean"):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode

    def _is_intensity_transform(self, transformer):
        name = transformer.__class__.__name__.lower()
        return name in ("multiply", "add")

    def _augment_aux(self, transformer, tensor):
        if tensor is None:
            return None
        if self._is_intensity_transform(transformer):
            return tensor
        return transformer.augment_image(tensor)

    def forward(self, img, dem=None, ridge=None):
        merged_outputs = []
        for transformer in self.transforms:
            augmented_img = transformer.augment_image(img)
            augmented_dem = self._augment_aux(transformer, dem)
            augmented_ridge = self._augment_aux(transformer, ridge)

            if augmented_ridge is not None:
                augmented_output = self.model(augmented_img, augmented_dem, augmented_ridge)
            elif augmented_dem is not None:
                augmented_output = self.model(augmented_img, augmented_dem)
            else:
                augmented_output = self.model(augmented_img)

            if isinstance(augmented_output, (tuple, list)):
                augmented_output = augmented_output[0]

            merged_outputs.append(transformer.deaugment_mask(augmented_output))

        if self.merge_mode != "mean":
            raise ValueError(f"Unsupported merge mode `{self.merge_mode}` for dual-input TTA wrapper.")

        return torch.stack(merged_outputs, dim=0).mean(dim=0)