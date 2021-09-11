import os
import tensorflow as tf
from make_data import DIV2K
from edsr import edsr

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(1)

div2k_train = DIV2K(scale=2, subset='train', downgrade='bicubic')
div2k_valid = DIV2K(scale=2, subset='valid', downgrade='bicubic')
weights_dir = '/home/ubuntu/bjh/Gan/SR-Webpage/Weights/upscale2.ckpt'
weights_file = lambda filename: os.path.join(weights_dir, filename)
# first
train_ds = div2k_train.dataset(batch_size=16, random_transform=True)
valid_ds = div2k_valid.dataset(batch_size=16, random_transform=True, repeat_count=1)

print(train_ds)


# 모델의 가중치를 저장하는 콜백 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weights_dir,
                                                 save_weights_only=True,
                                                 verbose=1)
model = edsr(scale=2)
model.summary()
def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-3, 5e-4])


model.compile(loss='mean_absolute_error',learning_rate=learning_rate,optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),metrics=[psnr])
model.fit(train_ds,epochs=200,steps_per_epoch=800,callbacks=[cp_callback],validation_data = valid_ds)