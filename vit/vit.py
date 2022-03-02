import tensorflow as tf
import numpy as np

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, emb_dim = 768, n_head = 12, dropout_rate = 0., kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0, stddev = 0.01), **kwargs):
        #ScaledDotProductAttention
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.emb_dim = emb_dim
        self.n_head = n_head
        if emb_dim % n_head != 0:
            raise ValueError("Shoud be embedding dimension % number of heads = 0.")
        self.projection_dim = emb_dim // n_head
        self.dropout_rate = dropout_rate
        self.query = tf.keras.layers.Dense(emb_dim, kernel_initializer = kernel_initializer)
        self.key = tf.keras.layers.Dense(emb_dim, kernel_initializer = kernel_initializer)
        self.value = tf.keras.layers.Dense(emb_dim, kernel_initializer = kernel_initializer)
        self.combine = tf.keras.layers.Dense(emb_dim, kernel_initializer = kernel_initializer)
    
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b = True)
        n_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(n_key)
        weight = tf.nn.softmax(scaled_score, axis = -1)
        if 0 < self.dropout_rate:
            weight = tf.nn.dropout(weight, self.dropout_rate)
        out = tf.matmul(weight, value)
        return out
    
    def separate_head(self, x):
        out = tf.keras.layers.Reshape([-1, self.n_head, self.projection_dim])(x)
        out = tf.keras.layers.Permute([2, 1, 3])(out)
        return out
    
    def call(self, inputs):
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)
        
        query = self.separate_head(query)
        key = self.separate_head(key)
        value = self.separate_head(value)

        attention = self.attention(query, key, value)
        attention = tf.keras.layers.Permute([2, 1, 3])(attention)
        attention = tf.keras.layers.Reshape([-1, self.emb_dim])(attention)
        
        out = self.combine(attention)
        return out
        
    def get_config(self):
        config = super(MultiHeadSelfAttention, self).get_config()
        config["emb_dim"] = self.emb_dim
        config["n_head"] = self.n_head
        config["projection_dim"] = self.projection_dim
        config["dropout_rate"] = self.dropout_rate
        return config
    
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_dim = 768, n_head = 12, n_feature = 3072, dropout_rate = 0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.emb_dim = emb_dim
        self.n_head = n_head
        self.n_feature = n_feature
        self.dropout_rate = dropout_rate
        
        self.attention = MultiHeadSelfAttention(emb_dim, n_head)
        self.feed_forward = [tf.keras.layers.Dense(n_feature, activation = tf.keras.activations.gelu), tf.keras.layers.Dense(emb_dim)]
        self.layer_norm = [tf.keras.layers.LayerNormalization(epsilon = 1e-6), tf.keras.layers.LayerNormalization(epsilon = 1e-6)]
        if 0 < dropout_rate:
            self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        out = self.layer_norm[0](inputs)
        out = self.attention(out)
        if 0 < self.dropout_rate:
            out = self.dropout(out)
        att_out = self.layer_norm[1](inputs + out)
        out = self.feed_forward[0](att_out)
        if 0 < self.dropout_rate:
            out = self.dropout(out)
        out = self.feed_forward[1](out)
        if 0 < self.dropout_rate:
            out = self.dropout(out)
        return att_out + out
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config["emb_dim"] = self.emb_dim
        config["n_head"] = self.n_head
        config["n_feature"] = self.n_feature
        config["dropout_rate"] = self.dropout_rate
        return config

class VisionTransformer(tf.keras.layers.Layer):
    def __init__(self, n_class = 1000, include_top = True, patch_size = 16, distillation = False, emb_dim = 768, n_head = 12, n_feature = 3072, n_layer = 12, dropout_rate = 0.1, ori_input_shape = None, method = "bicubic", **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.n_class = n_class
        self.include_top = include_top
        self.patch_size = patch_size if not isinstance(patch_size, int) else [patch_size, patch_size]
        self.distillation = distillation
        self.emb_dim = emb_dim
        self.n_head = n_head
        self.n_feature = n_feature
        self.n_layer = n_layer
        self.dropout_rate = dropout_rate
        self.ori_input_shape = ori_input_shape
        self.method = method
        
        self.patch_projection = tf.keras.layers.Dense(emb_dim)
        self.encoder = [TransformerBlock(emb_dim, n_head, n_feature, dropout_rate) for _ in range(n_layer)]
        if include_top:
            self.logits = tf.keras.layers.Dense(n_class, kernel_initializer = "zeros", name = "logits")
            if self.distillation:
                self.dist_logits = tf.keras.layers.Dense(n_class, kernel_initializer = "zeros", name = "kd_logits")
    
    def build(self, input_shape):
        n_patches = (input_shape[-3] // self.patch_size[0]) * (input_shape[-2] // self.patch_size[1])
        pos_dim = n_patches + 1
        self.class_emb = self.add_weight("class_embedding", shape = (1, 1,  self.emb_dim), trainable = self.trainable)
        if self.distillation:
            pos_dim += 1
            self.dist_emb = self.add_weight("kd_embedding", shape = (1, 1,  self.emb_dim), trainable = self.trainable)
        if self.ori_input_shape is not None:
            ori_n_patches = (self.ori_input_shape[0] // self.patch_size[0]) * (self.ori_input_shape[1] // self.patch_size[1])
            ori_pos_dim = ori_n_patches + 1
            if self.distillation:
                ori_pos_dim += 1
            self.pos_emb = self.add_weight("position_embedding", shape = (1, ori_pos_dim, self.emb_dim), trainable = self.trainable)
            self.pos_emb = self.resize_pos_embedding(self.pos_emb, pos_dim, self.distillation, self.method)
        else:
            self.pos_emb = self.add_weight("position_embedding", shape = (1, pos_dim, self.emb_dim), trainable = self.trainable)
        
    def extract_patches(self, images, patch_size):
        patches = tf.image.extract_patches(images = images, sizes = [1, patch_size[0], patch_size[1], 1], strides = [1, patch_size[0], patch_size[1], 1], rates = [1, 1, 1, 1], padding = "VALID")
        n_patch, patch_dim = tf.keras.backend.int_shape(patches)[-2:]
        patches = tf.keras.layers.Reshape([n_patch ** 2, patch_dim])(patches)
        return patches
    
    def resize_pos_embedding(self, pos_embedding, new_pos_dim, distillation = False, method = "bicubic"):
        pos_emb_token = pos_embedding[:, :1]
        pos_emb_grid = pos_embedding[:, 1:]
        new_pos_dim -= 1
        if distillation:
            pos_emb_dist_token = pos_embedding[:, -1:]
            pos_emb_grid = pos_embedding[:, 1:-1]
            new_pos_dim -= 1

        pos_dim, emb_dim = tf.keras.backend.int_shape(pos_emb_grid)[1:]
        n_patch = np.sqrt(pos_dim).astype(int)
        new_n_patch = np.sqrt(new_pos_dim).astype(int)

        pos_emb_grid = tf.reshape(pos_emb_grid, [1, n_patch, n_patch, emb_dim])
        pos_emb_grid = tf.image.resize(pos_emb_grid, [new_n_patch, new_n_patch], method = method)
        pos_emb_grid = tf.reshape(pos_emb_grid, [1, new_n_patch **2, emb_dim])

        pos_embedding = [pos_emb_token, pos_emb_grid]
        if distillation:
            pos_embedding.append(pos_emb_dist_token)
        pos_embedding = tf.concat(pos_embedding, axis = 1)
        return pos_embedding

    def call(self, inputs):
        out = self.extract_patches(inputs, self.patch_size)
        out = self.patch_projection(out)
        
        batch_size = tf.shape(inputs)[0]
        class_emb = tf.broadcast_to(self.class_emb, [batch_size, 1, self.emb_dim])
        out = [class_emb, out]
        if self.distillation:
            dist_emb = tf.broadcast_to(self.dist_emb, [batch_size, 1, self.emb_dim])
            out.append(dist_emb)
        out = tf.concat(out, axis = 1)
        out = out + self.pos_emb

        for encoder in self.encoder:
            out = encoder(out)

        if self.include_top:
            pre_logits = out[:, 0] #class token
            logits = self.logits(pre_logits)
            if self.distillation:
                pre_dist_logits = out[:, -1] #distillation token
                dist_logits = self.dist_logits(pre_dist_logits)
                out = [logits, dist_logits]
            else:
                out = logits
        return out
    
    def get_config(self):
        config = super(VisionTransformer, self).get_config()
        config["patch_size"] = self.patch_size
        config["distillation"] = self.distillation
        config["emb_dim"] = self.emb_dim
        config["n_head"] = self.n_head
        config["n_feature"] = self.n_feature
        config["n_layer"] = self.n_layer
        config["emb_dim"] = self.emb_dim
        config["dropout_rate"] = self.dropout_rate
        config["n_class"] = self.n_class
        config["include_top"] = self.include_top
        config["ori_input_shape"] = self.ori_input_shape
        config["method"] = self.method 
        return config
        
def train_model(input, logits, kd_logits = None, soft = True, alpha = 0.5, tau = 1.0):
    y_true = tf.keras.layers.Input(shape = (None,), name = "y_true", dtype = tf.float32)
    kd_true = None
    if kd_logits is not None:
        kd_true = tf.keras.layers.Input(shape = (None,), name = "kd_true", dtype = tf.float32)

    _y_true = tf.keras.layers.Lambda(lambda args: tf.cond(tf.equal(tf.shape(args[0])[-1], 1), true_fn = lambda: tf.one_hot(tf.cast(args[0], tf.int32), tf.shape(args[1])[-1])[:, 0], false_fn = lambda: args[0]))([y_true, logits])
    _y_true = tf.cast(_y_true, logits.dtype)
    if kd_logits is not None:
        _kd_true = tf.keras.layers.Lambda(lambda args: tf.cond(tf.equal(tf.shape(args[0])[-1], 1), true_fn = lambda: tf.one_hot(tf.cast(args[0], tf.int32), tf.shape(args[1])[-1])[:, 0], false_fn = lambda: args[0]))([kd_true, kd_logits])
        _kd_true = tf.cast(_kd_true, kd_logits.dtype)

    logits = tf.where(tf.equal(logits, 0), tf.keras.backend.epsilon(), logits)
    kd_logits = tf.where(tf.equal(kd_logits, 0), tf.keras.backend.epsilon(), kd_logits)

    accuracy = tf.keras.metrics.categorical_accuracy(_y_true, logits)
    loss = logits_loss = tf.keras.losses.categorical_crossentropy(_y_true, logits)
    if kd_logits is not None:
        if soft:
            kd_loss = tf.keras.losses.kl_divergence(tf.nn.softmax(_kd_true / tau), tf.nn.softmax(kd_logits / tau)) * (tau ** 2)
        else:
            kd_loss = tf.keras.losses.categorical_crossentropy(_kd_true, kd_logits)
        loss = (1 - alpha) * logits_loss + alpha * kd_loss
    
    model = tf.keras.Model([l for l in [input, y_true, kd_true] if l is not None], loss)
    
    model.add_metric(accuracy, name = "accuracy", aggregation = "mean")
    model.add_metric(loss, name = "loss", aggregation = "mean")
    model.add_loss(loss)
    return model
    
def vit_small(include_top = True, weights = None, input_tensor = None, input_shape = None, classes = 1000, distillation = False):
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape = input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor
    
    out = VisionTransformer(classes, include_top, patch_size = 16, distillation = distillation, emb_dim = 768, n_head = 8, n_feature = 2304, n_layer = 8, dropout_rate = 0.1, ori_input_shape = None, method = "bicubic")(img_input)
    model = tf.keras.Model(img_input, out)
    
    if weights is not None:
        model.load_weights(weights)
    return model
    
def vit_base(include_top = True, weights = None, input_tensor = None, input_shape = None, classes = 1000, distillation = False):
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape = input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor
    
    out = VisionTransformer(classes, include_top, patch_size = 16, distillation = distillation, emb_dim = 768, n_head = 12, n_feature = 3072, n_layer = 12, dropout_rate = 0.1, ori_input_shape = None, method = "bicubic")(img_input)
    model = tf.keras.Model(img_input, out)
    
    if weights is not None:
        model.load_weights(weights)
    return model
    
def vit_large(include_top = True, weights = None, input_tensor = None, input_shape = None, classes = 1000, distillation = False):
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape = input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor
    
    out = VisionTransformer(classes, include_top, patch_size = 16, distillation = distillation, emb_dim = 1024, n_head = 16, n_feature = 4096, n_layer = 24, dropout_rate = 0.1, ori_input_shape = None, method = "bicubic")(img_input)
    model = tf.keras.Model(img_input, out)
    
    if weights is not None:
        model.load_weights(weights)
    return model
    
def vit_huge(include_top = True, weights = None, input_tensor = None, input_shape = None, classes = 1000, distillation = False):
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape = input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor
    
    out = VisionTransformer(classes, include_top, patch_size = 16, distillation = distillation, emb_dim = 1280, n_head = 16, n_feature = 5120, n_layer = 32, dropout_rate = 0.1, ori_input_shape = None, method = "bicubic")(img_input)
    model = tf.keras.Model(img_input, out)
    
    if weights is not None:
        model.load_weights(weights)
    return model