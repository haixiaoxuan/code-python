from tensorflow import keras

"""
    迁移学习模型参数可以参考
        https://keras.io/applications/
"""

height = 32
width = 32
channels = 3
batch_size = 32
num_classes = 10

# 自动读取数据并进行数据增强
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory='./',
    x_col='filepath',
    y_col='class',
    classes=class_names,
    target_size=(height, width),
    batch_size=batch_size,
    seed=7,
    shuffle=True,
    class_mode='sparse',    # categorical
)

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_num // batch_size,
                              epochs=epochs,
                              validation_data=valid_generator,
                              validation_steps=valid_num // batch_size)


# 迁移学习
resnet50_fine_tune = keras.models.Sequential()
resnet50_fine_tune.add(keras.applications.ResNet50(include_top=False,
                                                   pooling='avg',
                                                   weights='imagenet'))
resnet50_fine_tune.add(keras.layers.Dense(num_classes, activation='softmax'))
resnet50_fine_tune.layers[0].trainable = False

resnet50_fine_tune.compile(loss="categorical_crossentropy",
                           optimizer="sgd", metrics=['accuracy'])
resnet50_fine_tune.summary()


# 重新训练后五层
resnet50 = keras.applications.ResNet50(include_top=False,
                                       pooling='avg',
                                       weights='imagenet')
resnet50.summary()

for layer in resnet50.layers[0:-5]:
    layer.trainable = False

resnet50_new = keras.models.Sequential([
    resnet50,
    keras.layers.Dense(num_classes, activation='softmax'),
])
resnet50_new.compile(loss="categorical_crossentropy",
                     optimizer="sgd", metrics=['accuracy'])
resnet50_new.summary()


