import tensorflow as tf 
from tf.kears import layers


class AlexNet(tf.keras.Model):
    def __init__(self, num_classes=1000):

        super(AlexNet, self).__init__()
        
        # AlexNet layers 선언
        self.conv1 = layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu')
        self.pool1 = layers.MaxPool2D(pool_size=3, strides=2)
        
        self.conv2 = layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu')
        self.pool2 = layers.MaxPool2D(pool_size=3, strides=2)
        
        self.conv3 = layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu')
        self.conv4 = layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu')
        self.conv5 = layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')
        self.pool3 = layers.MaxPool2D(pool_size=3, strides=2)
        
        # 완전 연결 계층 
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(units=4096, activation='relu')
        self.fc2 = layers.Dense(units=4096, activation='relu')
        self.fc3 = layers.Dense(units=1000, activation='softmax')
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x