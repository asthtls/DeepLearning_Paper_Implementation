import tensorflow as tf
from tensorflow.keras import layers

class VGG11(tf.keras.Model):
    def __init__(self, num_classes):
        super(VGG11, self).__init__()
        # Block 1
        self.conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2))

        # Block 2
        self.conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2))

        # Block 3
        self.conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2))

        # Block 4
        self.conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        self.conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        self.pool4 = layers.MaxPooling2D((2, 2), strides=(2, 2))

        # Block 5
        self.conv7 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        self.conv8 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        self.pool5 = layers.MaxPooling2D((2, 2), strides=(2, 2))

        # Classification layers
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(4096, activation='relu')
        self.fc2 = layers.Dense(4096, activation='relu')
        self.fc3 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool3(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool4(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool5(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)

        return output