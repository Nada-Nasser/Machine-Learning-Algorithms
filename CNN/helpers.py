from tensorflow.python.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils import to_categorical

# normalize data by divide each pixel in each image by 255(the max value for any pixel in the images)
def normalize_image_pixel(images_data):
    return images_data / 255.0


def prepare_training_and_testing_data():
    # All images have the same square size of 28×28 pixels.
    # the images are grayscale. ->
    (train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()

    print('Train: X=%s, y=%s' % (train_x.shape, train_x.shape))
    print('Test: X=%s, y=%s' % (test_x.shape, test_x.shape))

    # plot first few images
    for i in range(9):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(train_x[i], cmap=plt.get_cmap('gray'))
    # show the figure
    plt.show()

    # reshape dataset to set the number of color channels 
    # (color channel  = 1, because the images are grayscale)
    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
    test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))

    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    train_x, test_x = normalize_image_pixel(train_x), normalize_image_pixel(test_x)

    return (train_x, train_y), (test_x, test_y)


def create_sequential_model():
    model = models.Sequential() # define a Sequential model
    # adding the convelution layer
    # is a 2d layer with shape (3,3)
    # use the activation function rectified linear activation unit (ReLU)
    # this layer summarize the presence of features in an input image
    # results the down sampled feature maps to be the input for next layer
    # each feature map contain the precise position of features in the input image.
    model.add(layers.Conv2D(23, (3,3), activation="relu", input_shape=(28, 28, 1)))
    
    # add pooling layer after theconvelution layer
    # pooling layer create a new set of the same number of pooled feature maps.
    # the pooled feature map size is less than  the input maps.
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    
    # Flatten layer: doesnot change the number of pooled feature map
    # maps is converted to 1d lists
    model.add(layers.Flatten())
    
    # ???
    model.add(layers.Dense(100, activation='relu'))
	
    # adding the output layer with 10 nodes (0-9 classes)
    model.add(layers.Dense(10, activation='softmax'))
    
    # creat a gradient decent optimizer with learning rate equals 0.01
    opt = optimizers.SGD(lr=0.003)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def plt_history(history):
    print("Cost :")
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.show()
    
    print("Accuracy :")
    plt.plot(history.history['acc'], label='loss')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.show()
            
    
    