import tensorflow as tf
from tensorflow.keras import layers

class VGG16(tf.keras.Model):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        
        # Block 1
        self.conv1_1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')
        self.conv1_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')
        self.pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')
        
        # Block 2
        self.conv2_1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')
        self.conv2_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')
        self.pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')
        
        # Block 3
        self.conv3_1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')
        self.conv3_2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')
        self.conv3_3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')
        self.pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')
        
        # Block 4
        self.conv4_1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')
        self.conv4_2 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')
        self.conv4_3 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')
        self.pool4 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')
        
        # Block 5
        self.conv5_1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')
        self.conv5_2 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')
        self.conv5_3 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')
        self.pool5 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')
        
        # Classification layers
        self.flatten = layers.Flatten(name='flatten')
        self.fc1 = layers.Dense(4096, activation='relu', name='fc1')
        self.fc2 = layers.Dense(4096, activation='relu', name='fc2')
        self.fc3 = layers.Dense(num_classes, activation='softmax', name='predictions')
        
    def call(self, inputs):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool1(x)
        
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)
        
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

# vgg16_model = VGG16()
