import sys

import tensorflow as tf

NUM_CLASSES = 2
NUM_SUBDIVISION_STEPS = 5
RAW_IMAGE_SHAPE = (192, 256)
batch_size = 32

class PointHead_Branch():
    def __init__(self, pred_classes, crop_rois, msk_logit, fpn_feats, train):
        """ pred_classes: (b, top_k)
            crop_rois:    (b, ?, 4) normalized boxes, values range [0,1]
            msk_logit:    (b, p, 7, 7, 4)logit from MaskHead_Branch
            fpn_feats:    List of backbone feature_maps (b, feature_H, feature_W, c)
        """
        print("\n--PointHead-Branch--")
        self.num_classes = NUM_CLASSES
        shape = msk_logit.shape # b,p,7,7,nc
        self.batch_size = shape[0]
        self.N = int(shape[0]*shape[1]) # b * 50
        self.H = int(shape[2]) # 7 or 14 ?
        self.W = int(shape[3]) # 7 or 14 ?
        #--------------------------
        self.oversample_ratio = 3
        self.num_sampled = int(self.H*2 * self.W*2 * self.oversample_ratio) # sample h*2 w*2
        self.important_ratio = 0.75
        self.num_uncertain_points = int(self.important_ratio * self.num_sampled)
        #--------------------------
        self.num_subdivision_points = self.H*4 * self.W*4
        self.num_subdivision_steps = NUM_SUBDIVISION_STEPS

        self.dbg = [None] * 256
        with tf.device("gpu"):
            mask_logit = tf.reshape(msk_logit, (self.N, self.H, self.W, self.num_classes))
            with tf.variable_scope('PointHead_Branch'):
                if train:
                    """ training step:
                        1. gen uncertainty random coords, these coords are based on local proposal [0, 1].
                        2. extract {fine grained features}, and whole image scale uncertainty random coords.
                        3. sample coarse logits as {coarse features}, in local proposal level coords.
                        4. mlp_layer(2 + 3)
                    """
                    ### 1.
                    random_points = self.get_uncertain_pt_coords_randomness(mask_logit[..., 1][..., None])
                    random_points = random_points[..., ::-1]

                    ### 2.
                    fine_grained_features, point_coords_wrt_imgs = \
                        self.point_sample_fine_grained_features(fpn_feats, crop_rois, random_points, self.num_sampled)
                    print("fine_grained_features:\n  ", fine_grained_features) # (b*p, num_sampled, C)

                    ### 3.
                    coarse_coords = self.point_scale2img(random_points, self.H, self.W)
                    coarse_features = self.grid_nd_sample(mask_logit, coarse_coords, batch_dims=1)
                    print("coarse_features:\n  ", coarse_features) # (b*p, num_sampled, num_classes)

                    ### 4.
                    point_logits = self.mask_point_head(fine_grained_features, coarse_features, is_training=train, reuse=False)

                    ## output ##
                    self.point_msk_logits  = tf.reshape(point_logits,          (self.batch_size, -1, self.num_sampled, self.num_classes), name="point_msk_logits")
                    self.point_coords      = tf.reshape(point_coords_wrt_imgs, (self.batch_size, -1, self.num_sampled, 2))
                    self.coarse_coords     = tf.reshape(coarse_coords,         (self.batch_size, -1, self.num_sampled, 2))
                    print(self.point_msk_logits)  # (b, p, N, cls)
                    print(self.point_coords)  # (b, p, N, 2)
                    print(self.coarse_coords) # (b, p, N, 2)

                else:
                    """ inference step:
                        iterative
                    """
                    org_mask_logit = mask_logit
                    for subdivision_step in range(self.num_subdivision_steps):
                        print(subdivision_step)
                        ### upsampling
                        # mask_logit = backbones.UpSampling(mask_logit, upsampling_type="bilinear", sc_name="up_%d"%(subdivision_step))
                        ResizeShape = list(map(lambda a: int(a)*2, mask_logit.shape[1:3]))
                        mask_logit = tf.image.resize_bilinear(mask_logit, ResizeShape, align_corners=True, name='upsampleing')

                        R,sH,sW,C = map(int, mask_logit.shape)
                        if (self.num_subdivision_points >= 4*sH*sW and subdivision_step < self.num_subdivision_steps):
                            continue
                        ### get_uncertain_point_coords_on_grid
                        uncertainty_map = self.uncertainty(mask_logit, 1)
                        point_indices, point_coords = self.get_uncertain_point_coords_on_grid(uncertainty_map, self.num_subdivision_points)
                        point_coords = point_coords[..., ::-1]

                        ### fine_grained_features
                        ### extract from global fpn feat
                        ### return local feat and global coord
                        fine_grained_features, coord_wrt_img = self.point_sample_fine_grained_features(fpn_feats, crop_rois, point_coords, self.num_subdivision_points)

                        ### coarse_features
                        ### extract from local mask logits
                        coarse_coords = self.point_scale2img(point_coords, self.H, self.W) # local feat
                        coarse_features = self.grid_nd_sample(org_mask_logit, coarse_coords, batch_dims=1)

                        ### mask_point_head
                        point_logits = self.mask_point_head(fine_grained_features, coarse_features, is_training=train, reuse=tf.AUTO_REUSE)
                        # point_logits (R, 784, 2)

                        ### scatter update logits ###
                        # lin_mask_logit = tf.reshape(mask_logit, (R, sH*sW, C))
                        # point_indices = tf.tile(point_indices[..., None], (1, 1, self.num_classes))
                        # # lin_mask_logit (R, [784, 3136, 12544, ...], 2)
                        # # point_indices  (R, 784, 2)  value-> [0, sH*sW] >??????
                        #                                   # 10,3136,2              10,784              10,784,2
                        # mask_logit = tf.scatter_nd_update(lin_mask_logit, indices=point_indices, updates=point_logits)
                        # mask_logit = tf.reshape(mask_logit, (R, sH, sW, C))
                        inds = tf.cast(point_coords * tf.constant((sH,sW), tf.float32), tf.int32)
                        # [R, 28, 28, 2] mask_logit
                        # [R, 784, 2] inds
                        # [R, 784, 1] expdim
                        expdim = tf.tile(tf.range(0, R, dtype=tf.int32)[...,None], [1, inds.shape[1]])[...,None]
                        inds = tf.concat([expdim, inds], -1)
                        print(inds)
                        mask_logit = tf.tensor_scatter_nd_update(mask_logit, indices=inds, updates=point_logits)
                        print(mask_logit)
                        

                    self.point_msk_logits = mask_logit
                    self.point_coords = point_coords
                    self.point_rend_mask_logit = tf.reshape(mask_logit, (self.batch_size, -1, sH, sW, C))

                    ### debug zone -----------------------------------------------------------------------------------
                    gg = tf.zeros((R, sH, sW, 1))
                    ww = tf.ones((R, self.num_subdivision_points, 1))
                    gg = tf.tensor_scatter_nd_update(gg, indices=inds, updates=ww)
                    gg = tf.reshape(gg, (self.batch_size, -1, sH, sW))
                    self.uncertainty_map = uncertainty_map
                    self.point_indices = point_indices
                    self.dbg = [uncertainty_map, point_indices, point_coords, org_mask_logit, point_logits, gg]
                    ### debug zone -----------------------------------------------------------------------------------



        print("\n")

    def mask_point_head(self, fine_grained_features, coarse_features, is_training=True, reuse=False):
        with tf.variable_scope("mask_point_head"):
            net = tf.concat([fine_grained_features, coarse_features], axis=-1)    # (b*p, sample_points, c+cls)
            net = tf.layers.Conv1D(64, kernel_size=1, activation=tf.nn.relu, use_bias=True, kernel_initializer="glorot_normal",
                                  trainable=is_training, _reuse=reuse, name="lin0")(net) # (b*p, sample_points, C)
            net = tf.concat([fine_grained_features, coarse_features], axis=-1)    # (b*p, sample_points, c+cls)

            net = tf.layers.Conv1D(64, kernel_size=1, activation=tf.nn.relu, use_bias=True, kernel_initializer="glorot_normal",
                                  trainable=is_training, _reuse=reuse, name="lin1")(net) # (b*p, sample_points, C)
            net = tf.concat([fine_grained_features, coarse_features], axis=-1)    # (b*p, sample_points, c+cls)

            net = tf.layers.Conv1D(64, kernel_size=1, activation=tf.nn.relu, use_bias=True, kernel_initializer="glorot_normal",
                                  trainable=is_training, _reuse=reuse, name="lin2")(net) # (b*p, sample_points, C)
            net = tf.concat([fine_grained_features, coarse_features], axis=-1)    # (b*p, sample_points, c+cls)

            ### output
            net = tf.layers.Conv1D(self.num_classes, kernel_size=1, activation=tf.identity, use_bias=True,
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001),
                                   bias_initializer=tf.keras.initializers.Zeros(),
                                   trainable=is_training, _reuse=reuse, name="lin_out")(net) # (b*p, sample_points, cls)
            return net

    def point_sample_fine_grained_features(self, features_list, boxes, point_coords, num_sampled):
        """
        args:
            features_list: List of fpn feature_map (b, 24, 32, 96)
            boxes:         [0,1] yxyx (b*p, 4)
            point_coords:  [0,1]x[0,1] (b*p, 588, 2)
        return:
            fine_grained_features: tensor(b*p, 588, channel)
            point_coords_wrt_imgs: tensor(b*p, 588, 2)
        """
        with tf.variable_scope("point_sample_fine_grained_features", reuse=False):
            boxes = tf.reshape(boxes, (-1, 4))

            ### unnormalize points_coord to org src image scale
            ## 1. get unnormalize bbox of src image scale
            ## 2. get unnorm_points
            org_img_scale = tf.constant(RAW_IMAGE_SHAPE*2, "float32") ## list concat, not 2x
            unnorm_boxes = boxes * org_img_scale
            unnorm_points = self.get_point_coords_wrt_image(unnorm_boxes, point_coords)
            unnorm_points = tf.reshape(unnorm_points, (self.batch_size, -1, num_sampled, 2)) # (2,50,588,2)

            fine_grained_features = []
            for fidx, feature_map in enumerate(features_list):
                batch_size, fh, fw, chl = map(int, feature_map.shape)
                ### map image points to feat scale
                feat_coord = unnorm_points * (fh,fw) / org_img_scale[0:2]
                point_feat = self.grid_nd_sample(feature_map, feat_coord, batch_dims=1)
                fine_grained_features.append(point_feat)

            ### process List to tensor
            ### reshape to batch*proposal
            fine_grained_features = tf.concat(fine_grained_features, -1)
            fine_grained_features = tf.reshape(fine_grained_features, (self.N, num_sampled, -1))
            point_coords_wrt_imgs = tf.reshape(unnorm_points, (self.N, num_sampled, 2))

            return fine_grained_features, point_coords_wrt_imgs

    def get_point_coords_wrt_image(self, boxes, point_coords):
        """
        Convert box-normalized [0, 1] x [0, 1] point cooordinates to image-level coordinates.
        Args:
            boxes (Tensor): A tensor of shape (R, 4) that contains bounding boxes.
               normalized coordinates (y,x,y,x).
            point_coords (Tensor): A tensor of shape (R, P, 2) that contains
                [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.
        Returns:
            point_coords_wrt_image (Tensor): A tensor of shape (R, P, 2) that contains
                image-normalized coordinates of P sampled points.
        """
        with tf.variable_scope("get_point_coords_wrt_image", reuse=False):
            boxes = tf.stop_gradient(boxes)
            point_coords = tf.stop_gradient(point_coords)
            h = boxes[:, None, 2] - boxes[:, None, 0]
            w = boxes[:, None, 3] - boxes[:, None, 1]
            y1 = boxes[:, None, 0]
            x1 = boxes[:, None, 1]
            scale = tf.stack([h, w], axis=-1)
            trans = tf.stack([y1, x1], axis=-1)
            point_coords = point_coords * scale
            point_coords = point_coords + trans
            return point_coords

    def get_uncertain_point_coords_on_grid(self, uncertainty_map, num_points):
        """
        args:
            uncertainty_map: (B*P, H, W, 1)
                H&W_size: 28 -> 56 -> 112 -> 224
            num_points: 28*28
        return:
            points indices
            value[0, 1] points on local proposal
            shape (b*p, num_points, 2)
        """
        # print("get_uncertain_point_coords_on_grid")
        # print(uncertainty_map.shape)
        # print(num_points)
        R, H, W, C = map(int, uncertainty_map.shape)
        h_step = 1.0 / float(H)
        w_step = 1.0 / float(W)
        uncertainty_map = tf.reshape(uncertainty_map, (R, H*W))
        _, point_indices = tf.math.top_k(uncertainty_map, k=num_points)  # (R, 784)
        point_coords_y = w_step / 2.0 + tf.cast(point_indices % W, tf.float32) * w_step
        point_coords_x = h_step / 2.0 + tf.cast(point_indices // W, tf.float32) * h_step
        point_coords = tf.stack([point_coords_y, point_coords_x], 2)
        # point_coords = tf.concat([point_coords_y[..., None], point_coords_x[..., None]], -1)
        return point_indices, point_coords


    def get_uncertain_pt_coords_randomness(self, mask_coarse_logits):
        """
        algo steps:
            1. gen random_pts's coord as index for gather logits
            2. as torch func grid_sample => grid_nd_sample, to get the interp value of logits
            3. calculate step1&2's random interp points uncertainty from logits
            4. get the topK uncertainty points
            5. complete remain position
        args:
            mask_coarse_logits: (b*p, h, w, nb_classes)
        return:
            random_points: (b*p, num_sampled, 2)
                value [0, 1] on bbox proposal local coord
        """
        mask_coarse_logits = tf.stop_gradient(mask_coarse_logits)
        with tf.variable_scope("get_uncertain_pt_coords_randomness", reuse=False):
            ### 1.
            random_coords = tf.random.uniform(shape=(self.N, self.num_sampled, 2), minval=0, maxval=1)

            ### 2.
            unnorm_coords = self.point_scale2img(random_coords, self.H, self.W)
                                                  # 100,7,7,1         # 100,588,2
            point_logits = self.grid_nd_sample(mask_coarse_logits, unnorm_coords, batch_dims=1)

            ### 3.
            uncertainty_points = self.uncertainty(point_logits, 1)
            uncertainty_points = tf.reshape(uncertainty_points, (self.N, -1))

            ### 4.
            _, idx = tf.math.top_k(uncertainty_points, k=self.num_uncertain_points)  # (100,441)
            random_points = tf.gather_nd(random_coords, idx[..., None], batch_dims=1) # (100, 441, 2)

            ### 5.
            num_random_points = self.num_sampled - self.num_uncertain_points
            if num_random_points > 0:
                random_points = tf.concat(
                    [
                        random_points,
                        tf.random.uniform(shape=(self.N, num_random_points, 2), minval=0, maxval=1)
                    ], axis=1)

            random_points = tf.stop_gradient(random_points)
            return random_points

    def gen_regular_grid_coord(self,):
        """
        gen 8x8 regular grid
        """
        ### regular grid
        x = np.array(list(range(0,self.W))) / (self.W-1)
        y = np.array(list(range(0,self.H))) / (self.H-1)
        X,Y = tf.meshgrid(x,y)
        indices = tf.stack([X, Y])
        indices = tf.transpose(indices, (1, 2, 0))[None, ...]
        regular_coord_point = tf.tile(indices, (self.N,1,1,1))
        regular_coord_point = tf.cast(regular_coord_point, tf.float32)
        return regular_coord_point

    def uncertainty(self, logits, cls):
        """
        logits: (num_boxes, H, W, Class)
        """
        with tf.variable_scope("uncertainty", reuse=False):
            if logits.shape[-1] == 1:
                gt_class_logits = logits
            else:
                gt_class_logits = logits[..., cls][..., None]
            return -tf.abs(gt_class_logits)

    def grid_nd_sample(self, in_tensor, indices, batch_dims=0):
        """ gather_nd with interpolation as torch.grid_sample func
        Args:
            in_tensor: N-d tensor, NHWC
            indices: N-d tensor with last dim equals rank(in_tensor) - batch_dims
                assuming shape [..., [..., x, y]]
            batch_dims: number of batch dimensions
        """
        with tf.variable_scope("grid_nd_sample", reuse=False):
            interpolation_indices = indices[...,-2:]
            rounded_indices = indices[...,:-2]
            inter_floor = tf.floor(interpolation_indices)
            inter_ceil = tf.ceil(interpolation_indices)
            p1_indices = tf.concat([rounded_indices, inter_floor], axis=-1, name="p1_ind")
            p2_indices = tf.concat([rounded_indices, inter_ceil[...,:1], inter_floor[...,1:2]], axis=-1, name="p2_ind")
            p3_indices = tf.concat([rounded_indices, inter_floor[...,:1], inter_ceil[...,1:2]], axis=-1, name="p3_ind")
            p4_indices = tf.concat([rounded_indices, inter_ceil], axis=-1, name="p4_ind")
            mu = interpolation_indices - inter_floor
            with tf.name_scope("gather_corners"):
                p1v = tf.gather_nd(in_tensor, tf.cast(p1_indices, tf.int32), batch_dims=batch_dims, name="gather_p1")
                p2v = tf.gather_nd(in_tensor, tf.cast(p2_indices, tf.int32), batch_dims=batch_dims, name="gather_p2")
                p3v = tf.gather_nd(in_tensor, tf.cast(p3_indices, tf.int32), batch_dims=batch_dims, name="gather_p3")
                p4v = tf.gather_nd(in_tensor, tf.cast(p4_indices, tf.int32), batch_dims=batch_dims, name="gather_p4")
            mu_x, mu_y = tf.split(mu, 2, axis=-1)
            with tf.name_scope("interpolate_p12"):
                p12_interp = p1v * (1-mu_x) + p2v * mu_x
            with tf.name_scope("interpolate_p34"):
                p34_interp = p3v * (1-mu_x) + p4v * mu_x
            with tf.name_scope("interpolate_y"):
                vertical_interp = p12_interp * (1-mu_y) + p34_interp * mu_y
            return vertical_interp

    def point_scale2img(self, points, _H, _W):
        """ map normalized [0,1]x[0,1] points to image [0,H]x[0,W]
        args:
            points -> [..., 2]
        """
        with tf.variable_scope("point_scale2img", reuse=False):
            points = points * tf.constant([_H-1, _W-1], "float32")
            return points
