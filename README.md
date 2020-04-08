# PointRend-TF
PointRend dnn layers implement in Tensorflow 1.xx.  
  
- This Head layers should put after the mask head of mask-rcnn (crop_bbox - >roi_Align -> mask_logits).
- The mask size 7x7 is default mask head predict, which were 14x14 downsapmling from 2x2 conv2d.
- This project is default predict 2 classes cls1/background0.
- Only take features from last fpn layer.
    
------
The whole algo are reference to:  
  [detectron2](https://github.com/facebookresearch/detectron2).  
  [PointRend: Image Segmentation as Rendering](https://arxiv.org/pdf/1912.08193.pdf).  


