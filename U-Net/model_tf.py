import tensorflow as tf 


class ConvBlock(tf.keras.Model):
    def __init__(self, filters):
        super(ConvBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3,padding = 'same', kernel_initializer = 'he_normal')
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3,padding = 'same', kernel_initializer = 'he_normal')

        self.activation = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.activation(x)

        return x

class EncoderBlock(tf.keras.Model):
    def __init__(self, filters):
        super(EncoderBlock, self).__init__()

        self.conv = ConvBlock(filters)
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        p = self.dropout(p)

        return x,p

class DecoderBlock(tf.keras.Model):
    def __init__(self, filters):
        super(DecoderBlock, self).__init__()

        self.up = tf.keras.layers.Conv2DTranspose(filters, 2, strides=(2,2), padding='same', kernel_initializer='he_normal')
        self.conv = ConvBlock(filters=filters)
    
    def call(self, inputs, conv_features):
        x = self.up(inputs)
        x = tf.keras.layers.concatenate([x, conv_features])
        x = self.conv(x)

        return x 

# UNET isbi-2012 dataset tensorflow 
class UNET_TF_ISBI_2012(tf.keras.Model):
    def __init__(self, num_classes):
        super(UNET_TF_ISBI_2012, self).__init__()

        # Contracting part
        self.encoder1 = EncoderBlock(filters=64)
        self.encoder2 = EncoderBlock(filters=128)
        self.encoder3 = EncoderBlock(filters=256)
        self.encoder4 = EncoderBlock(filters=512)

        self.b = ConvBlock(filters=1024)

        # Expansive part
        self.decoder1 = DecoderBlock(filters=512)
        self.decoder2 = DecoderBlock(filters=256)
        self.decoder3 = DecoderBlock(filters=128)
        self.decoder4 = DecoderBlock(filters=64)

        # output 
        self.outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='sigmoid')

    def call(self, inputs):
        s1, p1 = self.encoder1(inputs)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)

        b = self.b(p4)

        d1 = self.decoder1(b, s4)
        d2 = self.decoder2(d1, s3)
        d3 = self.decoder3(d2, s2)
        d4 = self.decoder4(d3, s1)

        output = self.outputs(d4)

        return output