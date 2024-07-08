import tensorflow as tf
from keras.layers import *   
from keras.models import Model
import sys

class ConvModule(tf.keras.layers.Layer):
    def __init__(
        self, filters: int,
        kernel_size: tuple,
        strides: tuple = (1, 1),
        padding: str = "same",
        dilation_rate: tuple = (1, 1),
    ):
        super(ConvModule, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate

        self.conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate
        )

        self.bn = BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x

def Spatial_attention(inputs, ratio=8):
    init = inputs
    se_shape = (init.shape[1],init.shape[2],1)

    se = tf.reduce_mean(inputs, axis=3)
    se_max = tf.reduce_max(inputs, axis=3)
    
    se = Reshape(se_shape)(se)
    se_max = Reshape(se_shape)(se_max)
    
    se = Conv2D(1, (1,1), padding='same')(tf.concat([se,se_max],-1))
    se = Activation('sigmoid')(se)
    return se * inputs

def Channel_attention(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)
    
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, kernel_initializer='he_normal', use_bias=False)(se)
    
    se_max = GlobalMaxPooling2D()(init)
    se_max = Reshape(se_shape)(se_max)
    se_max = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se_max)
    se_max = Dense(filters, kernel_initializer='he_normal', use_bias=False)(se_max)
    
    x = Multiply()([init, Activation('sigmoid')(se+se_max)])
    return x

def Attention_block(g, x):
    filters = x.shape[-1]
   
    g_conv = Conv2D(filters, (1, 1), padding="same")(g)
    g_conv = BatchNormalization()(g_conv)
    g_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(g_conv)

    x_conv = Conv2D(filters, (1, 1), padding="same")(x)
    x_conv = BatchNormalization()(x_conv)
    
    gc_sum = Add()([g_pool, x_conv])

    gc_conv = Activation("relu")(gc_sum)
    gc_conv = Conv2D(filters, (1, 1), padding="same")(gc_conv)
    gc_conv = BatchNormalization()(gc_conv)
    gc_conv = Activation("sigmoid")(gc_conv)
    
    gc_mul = Multiply()([gc_conv, x])
    return gc_mul
  
def PASPP(x, filter):
  
  x1 = Conv2D(filter//4, (1,1), strides=1, kernel_initializer='he_normal', padding="same")(x)
  x1 = BatchNormalization()(x1)
  x1 = Activation("relu")(x1)
  
  x2 = Conv2D(filter//4, (1,1), strides=1, kernel_initializer='he_normal', padding="same")(x)
  x2 = BatchNormalization()(x2)
  x2 = Activation("relu")(x2)
  
  x3 = Conv2D(filter//4, (1,1), strides=1, kernel_initializer='he_normal', padding="same")(x)
  x3 = BatchNormalization()(x3)
  x3 = Activation("relu")(x3)
  
  x4 = Conv2D(filter//4, (1,1), strides=1, kernel_initializer='he_normal', padding="same")(x)
  x4 = BatchNormalization()(x4)
  x4 = Activation("relu")(x4)
  
  x1_2 = Add()([x1,x2])
  x3_4 = Add()([x3,x4])

  x1 = Conv2D(filter//4, (3,3), dilation_rate=1, kernel_initializer='he_normal',padding="same", use_bias=False)(x1)
  x1 = BatchNormalization()(x1)
  x1 = Activation("relu")(x1)
  
  x1 = Add()([x1,x1_2])
  
  x2 = Conv2D(filter//4, (3,3), dilation_rate=2, kernel_initializer='he_normal',padding="same", use_bias=False)(x2)
  x2 = BatchNormalization()(x2)
  x2 = Activation("relu")(x2)
  
  x2 = Add()([x2,x1_2])

  x3 = Conv2D(filter//4, (3,3), dilation_rate=4, kernel_initializer='he_normal',padding="same", use_bias=False)(x3)
  x3 = BatchNormalization()(x3)
  x3 = Activation("relu")(x3)
  
  x3 = Add()([x3,x3_4])

  x4 = Conv2D(filter//4, (3,3), dilation_rate=8, kernel_initializer='he_normal',padding="same", use_bias=False)(x4)
  x4 = BatchNormalization()(x4)
  x4 = Activation("relu")(x4)
  
  x4 = Add()([x4,x3_4])

  x1_2 = Concatenate()([x1,x2])
  x3_4 = Concatenate()([x3,x4])

  x1_2 = Conv2D(filter//2, (1,1), strides=1, kernel_initializer='he_normal', padding="same")(x1_2)
  x1_2 = BatchNormalization()(x1_2)
  x1_2 = Activation("relu")(x1_2)

  x3_4 = Conv2D(filter//2, (1,1), strides=1, kernel_initializer='he_normal', padding="same")(x3_4)
  x3_4 = BatchNormalization()(x3_4)
  x3_4 = Activation("relu")(x3_4)
  
  y = Concatenate()([x1_2,x3_4])
  y = Conv2D(filter, (1,1), strides=1, kernel_initializer='he_normal', padding="same")(y)
  y = BatchNormalization()(y)
  y = Activation("relu")(y)
  
  return y
  
class Decoder(tf.keras.layers.Layer):
    def __init__(self, filters: int, name:str):
        super(Decoder, self).__init__(name=name)
        self.filters = filters

        self.upsampling = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation='bilinear')
        self.conv_up1 = ConvModule(filters=filters, kernel_size=(3, 3))
        self.conv_up2 = ConvModule(filters=filters, kernel_size=(3, 3))
        self.conv_up3 = ConvModule(filters=filters, kernel_size=(3, 3))
        self.conv_up4 = ConvModule(filters=filters, kernel_size=(3, 3))
        self.conv_up5 = ConvModule(filters=2*filters, kernel_size=(3, 3))

        self.conv_concat_1 = ConvModule(filters=2*filters, kernel_size=(3, 3))
        self.conv_concat_2 = ConvModule(filters=3*filters, kernel_size=(3, 3))

        self.conv4 = ConvModule(filters=3*filters, kernel_size=(3, 3))
        self.conv5 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1))

    def call(self, rfb_feat1: tf.Tensor, rfb_feat2: tf.Tensor, rfb_feat3: tf.Tensor) -> tf.Tensor:
        rfb_feat1 = tf.nn.relu(rfb_feat1)
        rfb_feat2 = tf.nn.relu(rfb_feat2)
        rfb_feat3 = tf.nn.relu(rfb_feat3)

        x1_1 = rfb_feat1
        x2_1 = self.conv_up1(self.upsampling(rfb_feat1)) * rfb_feat2
        x3_1 = self.conv_up2(self.upsampling(self.upsampling(rfb_feat1)))  \
            * self.conv_up3(self.upsampling(rfb_feat2)) * rfb_feat3

        x2_2 = tf.concat([x2_1, self.conv_up4(
            self.upsampling(x1_1))], axis=-1)
        x2_2 = self.conv_concat_1(x2_2)

        x3_2 = tf.concat([x3_1, self.conv_up5(self.upsampling(x2_2))], axis=-1)
        x3_2 = self.conv_concat_2(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)

class Encoder_backbone():
    def __init__(self, model_architecture: str = 'efficientnet', inshape: tuple = (352, 352, 3), is_trainable: bool = True):
        self.inshape = inshape
        self._supported_arch = ['efficientnet']
        self.model_architecture = model_architecture
        self.is_trainable = is_trainable
        self.efficientnet_feature_extractor_layer_name = [                                     
            'block2d_add',  
            'block4a_expand_activation', 
            'block6a_expand_activation',  
            'top_activation',  
        ]
        self.backbone = tf.keras.applications.EfficientNetV2S(
                include_top=False, input_shape=self.inshape
            )
        self.backbone.trainable = self.is_trainable

    def get_fe_backbone(self) -> tf.keras.Model:
        layer_out = []
        for layer_name in self.efficientnet_feature_extractor_layer_name:
                layer_out.append(self.backbone.get_layer(layer_name).output)
            
        fe_backbone_model = tf.keras.models.Model(
            inputs=self.backbone.input, outputs=layer_out, name='efficientnet')
        
        return fe_backbone_model
      
def model(shape, args):
  inputs = Input(shape)
  features = Encoder_backbone(model_architecture=args.encoder_name,inshape=(args.image_size, args.image_size, 3), is_trainable=True ).get_fe_backbone()(inputs)
  p4 = features[3]
  p3 = features[2]
  p2 = features[1]
  p1 = features[0]

  a4 = PASPP(p4,32)
  a3 = PASPP(p3,32)
  a2 = PASPP(p2,32)
  
  out1 = Decoder(32,'out1')(a4, a3, a2)

  out1_s1 = tf.keras.layers.Resizing(args.image_size //4, args.image_size//4)(Activation('sigmoid')(out1))
  out1_s2 = tf.cast(out1_s1 > args.semantic_boundary, dtype = tf.float32)
  
  p1_s1 = Multiply()([Channel_attention(Spatial_attention)(p1) ,1-out1_s1])   
  a2_s1 = Attention_block(p1_s1,a2)
  a3_s1 = Attention_block(a2_s1,a3)
  a4_s1 = Attention_block(a3_s1,a4)

  out2 = Decoder(32,'out2')(a4_s1, a3_s1, a2_s1)

  p1_s2 = Multiply()([Channel_attention(Spatial_attention)(p1),1-out1_s2])   
  a2_s2 = Attention_block(p1_s2,a2)
  a3_s2 = Attention_block(a2_s2,a3)
  a4_s2 = Attention_block(a3_s2,a4)

  out3 = Decoder(32,'out3')(a4_s2, a3_s2, a2_s2)
  
  output1 = tf.keras.layers.Resizing(args.image_size, args.image_size, name='out_1')(out1)
  output2 = tf.keras.layers.Resizing(args.image_size, args.image_size, name='out_2')(out2)
  output3 = tf.keras.layers.Resizing(args.image_size, args.image_size, name='out_3')(out3)
 
  output = Concatenate()([output1,output2,output3])
  output = Conv2D(1,(1,1),activation='sigmoid')(output)

  return Model(inputs, output)

