import tensorflow as tf

def mobilenet_v1(tensor_in, num_classes, depth_multiplier, dropout_prob, is_training):
    """
    Constructs a Mobilenet V1 base convnet
    
    Args:
        tensor_in: a tensor of shape [NHWC]
        num_classes: number of channels of the final dense layer
        depth_multiplier: multiplier for number of channels, 
            should be 0.25, 0.5, 0.75 or 1.0
        dropout_prob: probability of the dropout layer before final dense
        is_training: the model is constructed for training or not
        
    Returns:
        logits: output tensor
    """
    
    # list of dicts specifying the base net architecture
    MOBILENET_V1_BASE_DEFS = [
        {'layer':'conv2d', 'name':'Conv_0',  'stride':2, 'depth':32  },
        {'layer':'convds', 'name':'Conv_1',  'stride':1, 'depth':64  },
        {'layer':'convds', 'name':'Conv_2',  'stride':2, 'depth':128 },
        {'layer':'convds', 'name':'Conv_3',  'stride':1, 'depth':128 },
        {'layer':'convds', 'name':'Conv_4',  'stride':2, 'depth':256 },
        {'layer':'convds', 'name':'Conv_5',  'stride':1, 'depth':256 },
        {'layer':'convds', 'name':'Conv_6',  'stride':2, 'depth':512 },
        {'layer':'convds', 'name':'Conv_7',  'stride':1, 'depth':512 },
        {'layer':'convds', 'name':'Conv_8',  'stride':1, 'depth':512 },
        {'layer':'convds', 'name':'Conv_9',  'stride':1, 'depth':512 },
        {'layer':'convds', 'name':'Conv_10', 'stride':1, 'depth':512 },
        {'layer':'convds', 'name':'Conv_11', 'stride':1, 'depth':512 },
        {'layer':'convds', 'name':'Conv_12', 'stride':2, 'depth':1024},
        {'layer':'convds', 'name':'Conv_13', 'stride':1, 'depth':1024}
    ]
    
    # hyperparams to use
    activation_fn = tf.nn.relu6
    normalizer_fn=tf.contrib.slim.batch_norm
    normalizer_params = {
        'is_training': is_training,
        'center': True, 
        'scale': True, 
        'decay': 0.9997, 
        'epsilon': 0.001, 
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }
    weights_initializer = tf.truncated_normal_initializer(stddev=0.09)
    weights_regularizer = tf.contrib.layers.l2_regularizer(0.00004)
    
    with tf.variable_scope('MobilenetV1', [tensor_in]):
        net = tensor_in
        # conv layers
        for layer_def in MOBILENET_V1_BASE_DEFS:
            if layer_def['layer']=='conv2d':
                net = tf.contrib.slim.conv2d( net,
                                              num_outputs=layer_def['depth']*depth_multiplier,
                                              kernel_size=[3,3],
                                              stride=layer_def['stride'],
                                              activation_fn=activation_fn,
                                              normalizer_fn=normalizer_fn,
                                              normalizer_params=normalizer_params,
                                              weights_initializer=weights_initializer,
                                              weights_regularizer=weights_regularizer,
                                              scope=layer_def['name'])
            elif layer_def['layer'] == 'convds':
                # depthwise conv
                net = tf.contrib.slim.separable_conv2d(net, 
                                                       num_outputs=None, # to skip pointwise stage
                                                       kernel_size=[3,3], 
                                                       stride=layer_def['stride'], 
                                                       activation_fn=activation_fn,
                                                       normalizer_fn=normalizer_fn,
                                                       normalizer_params=normalizer_params,
                                                       weights_initializer=weights_initializer,
                                                       scope=layer_def['name']+'_depthwise')
                # pointwise conv
                net = tf.contrib.slim.conv2d(net,
                                             num_outputs=layer_def['depth']*depth_multiplier,
                                             kernel_size=[1,1],
                                             activation_fn=activation_fn,
                                             normalizer_fn=normalizer_fn,
                                             normalizer_params=normalizer_params,
                                             weights_initializer=weights_initializer,
                                             weights_regularizer=weights_regularizer,
                                             scope=layer_def['name']+'_pointwise')
                
            else:
                raise ValueError('Unsupported layer type'+layer_def['layer'])
            
        # top layers
        convout_shape = net.get_shape().as_list()
        net = tf.contrib.slim.avg_pool2d(net, [convout_shape[1],convout_shape[2]], padding='VALID', scope='AvgPool')
        net = tf.contrib.slim.dropout(net, keep_prob=1-dropout_prob, is_training=is_training, scope='Dropout')
        logits = tf.contrib.slim.conv2d(net, 
                                num_outputs=num_classes, 
                                kernel_size=[1,1], 
                                activation_fn=None,
                                normalizer_fn=None, 
                                scope='Dense')
        logits = tf.squeeze(logits, axis=[1,2], name='Squeeze')
        return logits
        
