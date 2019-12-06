# Cat Breed Classifier
**![](https://lh4.googleusercontent.com/Rxd9OiUPJLx5YJ3xJU-Zqbl-eixqvfBGtaxCEiHOKD8Z7yb3-oa4pdbX2_zHI_rzfbU9wvdTOzJZUNjxEBualfk8-ZlCacfk0nwk87OGWPSqaPgwvQoVmTpLiTuOuKl6MM9L6zUFv5c)**
## 1. Problem Statement
Build a model that can successfully classify any cats into the closest breeds and show the images of other cats that are also in that breed.

## 2. Image Collection and Pre-processing
The initial dataset I had to train the model is downloaded from Google Image and Pinterest, the tools and packages used to download the images are:
- **[Google Image Downloader](https://pypi.org/project/google_images_download/)**
- **[PinDown](https://chrome.google.com/webstore/detail/pindown/flieckppkcgagklbnnhnkkeladdghogp)**

The cat breeds is from **[List_of_cat_breeds](https://en.wikipedia.org/wiki/List_of_cat_breeds)** on wikipedia. I picked  relatively unique 55 breeds among the 96 breeds to make the model easier to train and to produce a better result.

The final image file contains 31,207 labeled images with 55 cat breeds. The data size 3.78GB and it won't fit in GitHub, so the data is not uploaded.

### Resize

Since InceptionV3 model default input size is 224x224 with RGB mode, all the image data is resized to 224x224 RGB mode.



### Augmentation
Data collected are biased. Since some breed is more popular and got more input online. The breed with most images has 980 images, and the breed with least images has only 157.
**![](https://lh6.googleusercontent.com/LNPc5QTHdy7H_cr8rnlBR5NrnYTVG9tfHGw3bYsXLog2IVf23dbb1dW17FY0_edpwef52UAGWt-e7jayis8_kvYCkReEIveireB_uLj0QXTiyYze4ukX0eV9J1o19pAdki2Ep3ly1Eg)**

The package used in this project for augmentation is **[Image_aug](https://imgaug.readthedocs.io/en/latest/source/examples_basics.html)**
Images are randomly:
-   Scale/Zoom
-   Translate/Move
-   Rotate
-   Shear

For x amount of time on randomly selected pictures in a breed so the total number of pictures in each breed are 1000.

(Eg. Breed ragdoll has 651 images in the folder, we will randomly selected 1000 - 651 = 349 images from it and do random augmentation on them)

**
![](https://lh4.googleusercontent.com/sgCIe5wC5GjOBSSYTQZ-kbzKBFCZtgOcTfSTH2anPhDrw9N7sIfNGwXJtuVBccObAylJH3G5Ly1TDHiSZQ718EvI09gthYvkirkhF8fpViUL1zs8GRESTo5hZ_F5hKAUFpsg7kATL5s)**

After resizing and Augmentation and eventually tain_test_split, our final data set looks like:
-  The final shape of training data np.array(38,456, 244, 244, 3), The size of training data is 23.17GB
 - The final shape of testing data np.array(16,482, 244, 244, 3)The size of training data is 9.93GB
## 3. Model Building
### Transfer Learning

Transfer learning is to use existing pre-trained model as top layer of our neural network. Adding data and train the model addition on the pre-trained weight to get the output we need.

The pre-trained models I used in this project are:
-   [InceptionV3](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3)
-   [InceptionResnetV2](https://keras.io/applications/#inceptionresnetv2)
### InceptionV3 and ImageNet
**[InceptionV3](https://cloud.google.com/tpu/docs/inception-v3-advanced)** is a widely-used image recognition model that has been shown to attain greater than 78.1% accuracy on the ImageNet dataset. **[ImageNet](https://devopedia.org/imagenet)** is a large database or dataset of over 14 million images. It was designed by academics intended for computer vision research. It was the first of its kind in terms of scale. Images are organized and labelled in a hierarchy.

### Cloud Computing
With more than 30GB training and testing data, the requirement for computational power from our computer is massive. One good way of solving this issue is to use cloud computing. The website used in this project is **[FloydHub](https://www.floydhub.com/jobs)**. It's a very powerful, easy to use and cheap resource to run neural network model on GPU, The flowing is the best model.

Epoch 17/20
38456/38456 [==============================] - 685s 18ms/step - loss: 1.9859 - acc: 0.4356 - val_loss: 2.8279 - val_acc: 0.2505

## 4. Model Evaluation
### Accuracy Chart
The accuracy of the model is about 25%. Which is a big jump from the baseline model 1.82%.
|  | Training | Testing|
|--|--|--|
| Baseline | 1.82% |1.82%|
| InceptionV3 | 40.8% |25.05%|
| InceptionResnetV2 | 46.2% |24.4%|

The model is still predicting the breed pretty inaccurately. However if we increase the tolerance for the model and let it predict the top X breeds. The accuracy of the true value fall into one of the top X breeds predictions is as follow:

**![](https://lh6.googleusercontent.com/x6heqHonpEV4fBRTZQ0eYP0ActlVrbhSHWUihawj-9fPZyKM67DtBRj0s4yHtTfmveArgifeYs9OwkdHojtQXUVUjJAFO8Mj6vf4rZ-TsNYmzRVBSgHczdblipl0b1BXNvSuIUveqFg)**
### AUC_score Graph
**![](https://lh4.googleusercontent.com/NEWtWf6PlFpkCyqSOtwFc-1b8inlKBlp8J0KxOOia0_srQZWvV-3YIC_22YJyF3TenHZOxUrFcCOIHvw7F2jfnVtOXq9PCOdjxKhgFabj6wB9de_rWUujqRvGlr5XcPMh8Gk9KW2wDE)**
We can see the top 3 performance breeds are Bombay, Siamese, and Lykoi. They all have obvious body features that can be easily catched my model. The bottom 3 performance breeds are Munchkin, American Bobtail, and American Shorthair. They are lacking of unique features that’s ‘visible’ by the model.

## 5 .Prediction
**![](https://lh5.googleusercontent.com/YOTsKSkj4nVez6YI5ji4qTBybfWmKJ1qBY9jzeN44wA03UkpCn_tAcTv6YnqcZ9bw5D7TlNdg8RVoB_SNjIYrhJ5Tb6nHtQHtz-Ql0rhFc0cFu1fXD4egbsIKcyaf_cCl_41kvdGvro)**

**![](https://lh4.googleusercontent.com/tKL64wh9sZKeLbQCsI3SNOD_yg0QNJqzg9ePlZdUsGySPEitmy70VGSfFS6JETKZiZrc-iToYTJtobOn6RZZt3dtEYqp3iq4J_PsEVflTguWv3WKzVQk5wH7l-2nFChS8anYsgE3rvQ)**

## 6.Files

- **[Image Collection](./code/google_image_collect.py)** This is the code using googleing image downloader to get images, also using web scrubing to get breed list from wiki
- **[Preprocessing](./code/preprocessing.py)** Preprocessing include, resizing and augmentation. Saveing everything into npy files.
- **[Building Model](./code/build_model.py)** Building neural network
- **[Presentation](https://docs.google.com/presentation/d/1iA0DLP5R2RQxeNgS-ONyNMZZDWPAtXi5AAHUICKJS8A/edit?usp=sharing)**
