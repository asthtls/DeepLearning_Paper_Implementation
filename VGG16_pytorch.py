import tensorflow as tf
from tensorflow.keras import layers

class VGG16(tf.keras.Model):
    def __init__(self, num_classes=2):
        super(VGG16, self).__init__()

        self.features = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=(224, 224, 3)),
            layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=2, strides=2),

            layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=2, strides=2),

            layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=2, strides=2),

            layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=2, strides=2),

            layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=2, strides=2)
        ])

        self.flatten = layers.Flatten()
        self.classifier = tf.keras.Sequential([
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs):
        x = self.features(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

# 모델 인스턴스 생성
model = VGG16()