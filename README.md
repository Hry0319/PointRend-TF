# PointRend-TF
PointRend dnn layers implement in Tensorflow 1.15  
- This Head layers should put after the mask head of mask-rcnn (crop_bbox - >roi_Align -> mask_logits).
- The mask size 7x7 is default mask head predict, which were 14x14 downsapmling from 2x2 conv2d.
- This project is default predict 2 classes cls1/background0.
- Only take features from last fpn layer.

# example
```python 
from PointRendHead import PointHead_Branch
### put this layer after your mask-rcnn fpn_features -> rpn -> proposal -> roi_align layers
pointhead = PointHead_Branch(pred_classes, crop_rois, msk_logit, fpn_feats, is_training)

### training
#pred_cls (from mask-rcnn)
#gt_masks (from dataset)
msk_logit = pointhead.point_msk_logits
point_coords = pointhead.point_coords
### loss
loss = pointhead.roi_mask_point_loss(pred_cls, msk_logit, points_coord, gt_masks)

### evaluation
pointhead.point_rend_mask_logit
```

------
The whole algo are reference to:  
  [detectron2](https://github.com/facebookresearch/detectron2).  
  [PointRend: Image Segmentation as Rendering](https://arxiv.org/pdf/1912.08193.pdf).  


