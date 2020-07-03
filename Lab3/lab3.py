import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import MeanShift

from PIL import Image

size = 100, 100

img_names = ["shapes.png", "strawberry.png"]
ext_names = ["coins.png", "kiwi.png"]

images = [i for i in img_names]
ext_images = [i for i in ext_names]


def plot_three_images(figure_title, image1, label1,
                      image2, label2, image3, label3):
    fig = plt.figure(figsize=[14, 5], dpi=100)
    fig.suptitle(figure_title)

    # Display the first image
    fig.add_subplot(1, 3, 1)
    plt.imshow(image1)
    plt.axis('off')
    plt.title(label1)

    # Display the second image
    fig.add_subplot(1, 3, 2)
    plt.imshow(image2)
    plt.axis('off')
    plt.title(label2)

    # Display the third image
    fig.add_subplot(1, 3, 3)
    plt.imshow(image3)
    plt.axis('off')
    plt.title(label3)

    plt.show()


for img_path in images:
    img = Image.open(img_path)
    img.thumbnail(size)  # Convert the image to 100 x 100
    # Convert the image to a numpy matrix
    img_mat = np.array(img)[:, :, :3]

    #
    # +--------------------+
    # |     Task 1         |
    # +--------------------+
    #
    # TODO: perform MeanShift on image
    # Follow the hints in the lab spec.  

    # Step 1 - Extract the three RGB colour channels
    # Hint: It will be useful to store the shape of one of the colour
    # channels so we can reshape the flattened matrix back to this shape.
    shape = img_mat.shape
    img_flattend = img_mat.reshape(-1, 3)

    # Step 2 - Combine the three colour channels by flatten each channel 
    # then stacking the flattened channels together.
    # This gives the "colour_samples"
    colour_samples = img_flattend

    # Step 3 - Perform Meanshift clustering
    # For larger images, this may take a few minutes to compute.
    ms_clf = MeanShift(bin_seeding=True)
    ms_labels = ms_clf.fit_predict(colour_samples)

    # Step 4 - reshape ms_labels back to the original image shape 
    # for displaying the segmentation output 
    ms_labels = ms_labels.reshape(shape[0], shape[1])
    #%%
    #
    # +--------------------+
    # |     Task 2         |
    # +--------------------+
    #

    # TODO: perform Watershed on image
    # Follow the hints in the lab spec.

    # Step 1 - Convert the image to gray scale
    # and convert the image to a numpy matrix
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    img_array = rgb2gray(img_mat)
    
    # Step 2 - Calculate the distance transform
    # Hint: use     ndi.distance_transform_edt(img_array)
    distance = ndi.distance_transform_edt(img_array)
    
    # Step 3 - Generate the watershed markers
    # Hint: use the peak_local_max() function from the skimage.feature library
    # to get the local maximum values and then convert them to markers
    # using ndi.label() -- note the markers are the 0th output to this function
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                            labels=img_array)
    markers = ndi.label(local_maxi)[0]
    
    # Step 4 - Perform watershed and store the labels
    # Hint: use the watershed() function from the skimage.morphology library
    # with three inputs: -distance, markers and your image array as a mask
    ws_labels = watershed(-distance, markers, mask=img_array)
    

    # Display the results
    plot_three_images(img_path, img, "Original Image", ms_labels, "MeanShift Labels",
                      ws_labels, "Watershed Labels")

    # If you want to visualise the watershed distance markers then try
    # plotting the code below.
#     plot_three_images(img_path, img, "Original Image", -distance, "Watershed Distance",
#                        ws_labels, "Watershed Labels")

    
    
#%%
#
# +-------------------+
# |     Task 3        |
# +-------------------+
#
# Loop for the extension component
for img_path in ext_images:
    img = Image.open(img_path)
    img.thumbnail(size)

    img_mat = np.array(img)[:, :, :3]
    #%%

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    img_array = rgb2gray(img_mat)
    
    # fore
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(img_array,cv2.MORPH_OPEN,kernel,iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations = 3)
    #dist_transform = cv2.distanceTransform(opening,1,5)
    dist_transform = ndi.distance_transform_edt(opening)
    ret,sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    # plt.imshow(sure_fg)
    
    
    #distance = ndi.distance_transform_edt(img_array)
    local_maxi = peak_local_max(dist_transform, indices=False, footprint=np.ones((3, 3)),
                            labels=sure_fg)
    markers = ndi.label(local_maxi)[0]
    ws_labels = watershed(-dist_transform, markers, mask=sure_fg)

    # Display the results
    plot_three_images(img_path, img, "Original Image", -dist_transform, "Watershed Distance",
                       ws_labels, "Watershed Labels")
    