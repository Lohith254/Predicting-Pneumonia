# Import all the libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
# Initialising Sequential 
classifier = Sequential()

# The first Convolution Layer
classifier.add(Convolution2D(64, 3, 3, input_shape=(64, 64, 3), activation = 'relu'))

# The first Max-Pooling Layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# The second Convolution Layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))

# The Second Max-Pooling 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# The second Convolution Layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))

# The Second Max-Pooling 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flatten the layer
classifier.add(Flatten())

# Adding First ANN Layer
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dropout(p = 0.3))

# Adding Second Layer
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dropout(p = 0.2))

# Adding First ANN Layer
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(p = 0.2))

# Output Layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the Sequence 
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

"""
Data Pre-processing
"""

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                'chest_xray/train',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                            'chest_xray/test',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
classifier.fit(
                training_set,
                steps_per_epoch=5216,
                epochs=8,
                validation_data=test_set,
                validation_steps=624)

import numpy as np 
from keras.preprocessing import image

test_image = image.load_img('chest_xray/val/NORMAL/NORMAL2-IM-1438-0001.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Pneumonia: Positive'
else:
    prediction = 'Pneumonia: Negative'

classifier.predict(test_image)
print(prediction)

"""
Making the window for the application to work
"""

from tkinter import *
# loading Python Imaging Library
from PIL import ImageTk, Image
# To get the dialog box open when required
from tkinter import filedialog

def openfilename():
    """
    open file dialog box to select image
    The dialog box has a title "Open"
    """
    filename = filedialog.askopenfilename(title = 'open')
    return filename

def open_img():
    x = openfilename() # Select Imagename from a folder
    img = Image.open(x) # Opens the image
    
    test_image = image.load_img(x, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    training_set.class_indices
    
    if result[0][0] == 1:
        
        result = Label(root, text='Pneumonia: Positive', fg='red', font=("Helvetica", 18)).grid(row=4, columnspan=4)
        prediction = 'Pneumonia: Positive'
    else:
        result = Label(root, text='Pneumonia: Negative', fg='green', font=("Helvetica", 18)).grid(row=4, columnspan=4)
        prediction = 'Pneumonia: Negative'
    classifier.predict(test_image)
    print(prediction)
    img = img.resize((320,240), Image.ANTIALIAS) # Resize the image and apply high quality down sampling
    img = ImageTk.PhotoImage(img) # PhotoImage class is used to add image to widgets, icons etc
    panel = Label(root, image = img) # Create a Label
   
    # Set the image as img
    panel.image = img
    panel.grid(row=2, columnspan=2)
    
root = Tk() # Create a window
root.title('Pneumonia Predictor') # Set Title as Image Loader
root.geometry("430x600") # Set the resolution of window
root.resizable(width = True, height = True) # Allow window to be resizable
# Create a Button to add function
Heading = Label(root, text = 'Pneumonia Predictor', fg='black', font=("Helvetica", 20), justify="center").grid(row=0, columnspan=4)
instructions = Label(root, text="Instructions\n Click on 'uploaad' button to" +
                     "upload the image.\n Immediately when the image is "+
                     "uploaded to the program \n and will run to find out if"+
                     " theres is chance of pneumonia.", fg='grey', justify="center").grid(row=1, columnspan=2)
btn = Button(root, text='upload', command = open_img, justify="center", padx=190, font=("Helvetica", 14)).grid(row=3,column=0, columnspan=1) 
root.mainloop()