
# coding: utf-8

# <a href="https://www.nvidia.com/en-us/deep-learning-ai/education/"> <img src="files/DLI Header.png" alt="Header" style="width: 400px;"/> </a>

# # Improving your Model

# Now that you've learned to successfully train a model, let's work to train a state of the art model. In this lab, we'll learn the levers that you, as a deep learning practitioner, will use to navigate towards desired results. In the process, we'll start to peel back the layers around the technology that makes this possible.
# 
# Let's bring back our handwritten digit classifier. Go to DIGITS' home screen by clicking the DIGITS logo on the top left of the screen. Here, you'll see (at least) two models. Choose the first model you created, in our case, "My first model."
# 
# Among other things, DIGITS will display the graph that was generated as the model was being trained. 
# 
# ![](graphfromfirsttraining.png)
# 
# Three quantities are reported: training loss, validation loss, and accuracy.  The values of training and validation loss should have decreased from epoch to epoch, although they may jump around some.  The accuracy is the measure of the ability of the model to correctly classify the validation data.  If you hover your mouse over any of the data points, you will see its exact value.  In this case, the accuracy at the last epoch is about 87%.  Your results might be slightly different than what is shown here, since the initial networks are generated randomly.
# 
# Analyzing this graph, one thing that jumps out is that accuracy is increasing over time and that loss is decreasing. A natural question may be, "will the model keep improving if we let it train longer?" This is the first intervention that we'll experiment with and discuss. 
# 
# ## Study more
# 
# Following the advice of parents and teachers everywhere, let's work to improve the accuracy of our model by asking it to study more. 
# 
# An **epoch** is one complete presentation of the data to be learned to a learning machine. Let's make sense of what is happening during an **epoch.**
# 
# 1. Neural networks take the first image (or small group of images) and make a prediction about what it is (or they are).
# 2. Their prediction is compared to the actual label of the image(s).
# 3. The network uses information about the difference between their prediction and the actual label to adjust itself.
# 4. The network then takes the next image (or group of images) and make another prediction.
# 5. This new (hopefully closer) prediction is compared to the actual label of the image(s).
# 6. The network uses information about the difference between this new prediction and the actual label to adjust again. 
# 7. This happens repeatedly until the network has looked at each image. 
# 
# Compare this to a human study session using flash cards:
# 
# 1. A student looks at a first flashcard and makes a guess about what is on the other side.
# 2. They check the other side to see how close they were.
# 3. They adjust their understanding based on this new information.
# 4. The student then looks at the next card and makes another prediction.
# 5. This new (hopefully closer) prediction is compared to the answer on the back of the card.
# 6. The student uses information about the difference between this new prediction and the right answer to adjust again.
# 7. This happens repeatedly until the student has tried each flashcard once. 
# 
# You can see that one epoch can be compared to one trip through a deck of flashcards.
# 
# In the model that we trained, we asked our network for 5 epochs. The blue curve on the graph above shows how far off each prediction was from the actual label.
# 
# Like a human student, the point of studying isn't just to be able to replicate someone else's knowledge. The green curve shows the difference between the model's predictions and actual labels for NEW data that it hasn't learned from. The orange curve is simple the inverse of that loss. 
# 
# **Loss** can be measured in many ways, but conceptually, it's simply the difference between predicted and actual labels.
# 
# If after taking a test and earning 87%, a human student wanted to improve performance, they might be advised to study more. At this point, they'd go back to their flashcards. 
# 
# A deep learning practitioner could request more epochs. It *is* possible to start from scratch and request more epochs when creating your model for the first time. However, our model has already spent some time learning! Let's use what it has learned to improve our model instead of start from scratch.
# 
# Head back to DIGITS and scroll to the bottom of your model page and click the big green button labeled: "Make Pretrained Model."
# 
# ![](pretrainedbutton.PNG)
# 
# This will save two things:
# 
# 1. The "network architecture" that you chose when selecting "AlexNet."
# 2. What the model has "learned" in the form of the parameters that have been adjusted as your network worked through the data in the first 5 epochs.
# 
# We can now create a new model from this starting point. Go back to DIGITS' home screen and create a new Image Classfication model like before. New Model (Images) -> Classification 
# 
# - Select the same dataset - (Default Options Small Dataset)
# - Choose some number of epochs between 3 and 8. (Note that in creating a model from scratch, this is where you could have requested more epochs originally.)
# - This time, instead of choosing a "Standard Network," select "Pretrained Networks." 
# - Select the pretrained model that you just created, likely "My first model."
# - Name your model - We chose "Study more"
# - Click Create
# 
# Your settings should look like:
# 
# ![](pretrainedmodelsetup.png)
# 
# When you create the model, you'll get the following graph. 
# 
# ![](pretrainedgraph.PNG)
# 
# Note the following:
# 
# 1. As expected, the accuracy starts close to where our first model left off, 86%.
# 2. Accuracy DOES continue to increase, showing that increasing the number of epochs often does increase performance.
# 3. The *rate* of increase in accuracy slows down, showing that more trips through the same data can't be the only way to increase performance. 
# 
# Let's test our new and improved model using the same image from before. At the bottom of our model page, "Test a single image." We'll test the same "2" to compare performance. As a reminder, the image path is:  
# <code> /data/test_small/2/img_4415.png </code>
# 
# ![](classifyoneunlabeled.png)
# 
# Recall that our original model correctly identified this as a 2 as well, but with a confidence of 85%. This is clearly a better model.
# 
# Feel free to try a few more images by changing the number after "img_ "

# Let's try testing the model with a set of images.  They are shown below.
# 
# <img src="test_images/image-1-1.jpg" width="64px" /> <div style="text-align:center;">image-1-1.jpg</div>
# <img src="test_images/image-2-1.jpg" width="64px" /> <div style="text-align:center;">image-2-1.jpg</div>
# <img src="test_images/image-3-1.jpg" width="64px" /> <div style="text-align:center;">image-3-1.jpg</div>
# <img src="test_images/image-4-1.jpg" width="64px" /> <div style="text-align:center;">image-4-1.jpg</div>
# <img src="test_images/image-7-1.jpg" width="64px" /> <div style="text-align:center;">image-7-1.jpg</div>
# <img src="test_images/image-8-1.jpg" width="64px" /> <div style="text-align:center;">image-8-1.jpg</div>
# <img src="test_images/image-8-2.jpg" width="64px" /> <div style="text-align:center;">image-8-2.jpg</div>
# 
# We can classify multiple files if we put them in the list. In the link below, execute the code block and a link to the file an_image.list will appear.  Right click on an_image.list and save that to a file on your local computer(right click and "Save As"). Remember the directory in which it is saved.

# In[2]:


from IPython.display import FileLink, FileLinks
FileLinks('test_images_list')


# On the right side of the DIGITS model page, there is an option to "test a list of images".  Press the button **Browse** and select the `an_image.list` file you just downloaded. Then press the **Classify Many** button.  After several seconds, you will see the results from Caffe trying to classify these images with the generated model.  In the image name, the first number is the digit in the image (ex. image-3-1.jpg is a 3). Your results should be similar to this:
# 
# ![](images/classify-many-images-small.png)
# 
# What is shown here is the probability that the model predicts the class of the image.  The results are sorted from highest probability to lowest.  Our model didn't do so well.
# 
# While the accuracy of the model was 87%, it could not correctly classify any of the images that we tested.  What can we do to improve the classification of these images?

# At this point it's time to be a bit more intentional. After we can successfully train a model, what comes next comes from understanding and experimentation. To build a better understanding of THIS project, we should start with our data. To build a better understanding of anything, we should start with primary sources. 
# 
# The dataset that we are learning from is a subset of the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. A close read of the documentation would likely yield a lot of insight.
# 
# One key observation we'll use is that the images start as 28x28 grayscale images. When we loaded our dataset, we stuck with defaults, which were 256x256 and color. You may have noticed that the images were a bit blurry. In the next section we'll explore the benefits that can come from matching your data to the right model. 
# 
# ## The right model
# 
# Let's start from scratch using what we've learned. This time, we'll load our data as 28x28 grayscale images and pick a model that is built to accept that type of data, LeNet. To compare to our previous model, use the total number of epochs that you trained with so far, eg. the 5 in "my first model" and the additional epochs trained from your pretrained model. In my case I'll use 8. 
# 
# Here's an image of the settings that would create the dataset.
# 
# ![](grayscaledata.png)
# 
# Feel free to "explore the db" again. I notice immediately that the images are no longer blurry.
# 
# Next, create a model using the settings in the image below. Note that the LeNet model is designed to take 28x28 grayscale images. You'll see shortly that this network was actually purpose-built for digit classification. 
# 
# ![](LeNetmodelcreate.png)
# 
# Woah. You should have noticed two things. 
# 
# 1. Your model improved performance. Mine was accurate to more than 96%.
# 2. Your model trained faster. In far less than two minutes, my model ran through 8 epochs.
# 
# ![](LeNetmodeltrain.png)
# 
# We haven't done anything to address the problem of more diverse data, but if we can train faster, we can experiment a lot more. 

# ## Train with more data
# 
# In our last attempt at training, we only used 10% of the full MNIST dataset.  Let's try training with the complete dataset and see how it improves our training.  We can use the clone option in DIGITS to simplify the creation of a new job with similar properties to an older job.  Let's return to the home page by clicking on **DIGITS** in the upper left corner.  Then select **Dataset** from the left side of the page to see all of the datasets that you have created.  You will see your **Mnist small** dataset.  When you select that dataset, you will be returned to the results window of that job.  In the right corner you will see a button: **Clone Job**. 
# 
# ![](images/digits_clone_dataset.png)
# 
# Press the **Clone Job** button.  
# 
# From here you will see the create dataset template populated with all the options you used when you created the Default Options Small Digits dataset.  To create a database with the full MNIST data, change the following settings:
# 
# - Training Images - /data/train_full
# - Dataset Name - MNIST full
# 
# Then press the **Create** button.  This dataset is ten times larger than the other dataset, so it will take a few minutes to process.
# 
# After you have created your new database, follow the same procedure to clone your training model.  In the template, change the following values:
# 
# - Select the MNIST full dataset
# - Change the name to MNIST full
# 
# Then create the model.
# 
# With much more data, the model will take longer to run. It still should complete in less than a minute. What do you notice that is different about the results? Both the training and validation loss function values are much smaller. In addition, the accuracy of the model is around 99%, possibly greater. That is saying the model is correctly identifying most every image in its validation set. This is a significant improvement. However, how well does this new model do on the challenging test images we used previously?
# 
# Using the same procedure from above to classify our set of test images, here are the new results:
# 
# ![](images/digits_classify_many_full.png)
# 
# The model was still only able to classify one of the seven images.  While some of the classifications came in a close second, our model's predictive capabilities were not much greater.  Are we asking too much of this model to try and classify non-handwritten, often colored, digits with our model? 
# 

# ## Improving Model Results - Data Augmentation
# 
# You can see with our seven test images that the backgrounds are not uniform.  In addition, most of the backgrounds are light in color whereas our training data all have black backgrounds.   We saw that increasing the amount of data did help for classifying the handwritten characters, so what if we include more data that tries to address the contrast differences?
# 
# Let's try augmenting our data by inverting the original images.  Let's turn the white pixels to black and vice-versa.  Then we will train our network using the original and inverted images and see if classification is improved.
# 
# To do this, follow the steps above to clone and create a new dataset and model.  The directories for the augmented data are:
# 
# - Training Images - /data/train_invert
# 
# Remember to change the name of your dataset and model.  When the new dataset is ready, explore the database.  Now you should see images with black backgrounds and white numbers and also white backgrounds and black numbers.
# 
# Now train a new model.  Clone your previous model results, and change the dataset to the one you just created with the inverted images.  Change the name of the model and create a new model.  When the training is complete, the accuracy hasn't really increased over the non-augmented image set.  In fact, the accuracy may have gone down slightly.  We were already at 99% so it is unlikely we were going to improve our accuracy.  Did using an augmented dataset help us to better classify our images?  Here is the result:
# 
# ![](images/digits_classify_many_invert.png)
# 
# By augmenting our dataset with the inverted images, we could identify six of the seven images.  While our result is not perfect, our slight change to the images to increase our dataset size made a significant difference.

# ## Advanced - Improving Model Results by Modifying the Network (Optional)
# 
# Augmenting the dataset improved our results, but we are not identifying all our test images. Let's try modifying the LeNet network directly. You can create custom networks to modify the existing ones, use different networks from external sources, or create your own. To modify a network, select the Customize link on the right side of the Network dialog box.
# 
# ![](images/network-dialog.png)
# 
# This will open an editor with the LeNet model configuration.  Scroll through the window and look at the code.  The network is defined as a series of layers.  Each layer has a name that is a descriptor of its function.  Each layer has a top and a bottom, or possibly multiples of each, indicating how the layers are connected.  The *type* variable defined what type the layer is.  Possibilities include **Convolution**, **Pool**, and **ReLU**.  All the options available in the Caffe model language are found in the [Caffe Tutorial](http://caffe.berkeleyvision.org/tutorial/).
# 
# At the top of the editor, there is a **Visualize** button.  Pressing this button will visualize all of the layers of the model and how they are connected.  In this window, you can see that the data are initially scaled, there are two sets of Convolution and Pooling layers, and two Inner Products with a Rectilinear Unit (ReLU) connected to the first Inner Product.  At the bottom of the network, there are output functions that return the accuracy and loss computed through the network.
# 
# We are going to make two changes to the network.  First, we are going to connect a ReLU to the first pool.  Second, we are going to change the values of num_output to 75 for the first Convolution (conv1) and 100 for the second Convolution (conv2).  The ReLU layer definition should go below the pool1 definition and look like:
# 
# <code>
# layer {
#   name: "reluP1"
#   type: "ReLU"
#   bottom: "pool1"
#   top: "pool1"
# }
# </code>
# 
# The Convolution layers should be changed to look like:
# 
# <code>
# layer {
#   name: "conv1"
#   type: "Convolution"
#   bottom: "scaled"
#   top: "conv1"
# ...
#   convolution_param {
#     num_output: **75**
# ...
# </code>
# <code>
# layer {
#   name: "conv2"
#   type: "Convolution"
#   bottom: "pool1"
#   top: "conv2"
# ...
#   convolution_param {
#     num_output: **100**
# ...
# </code>
# Note, the ellipis (...) just indicates we removed some of the lines from the layer for brevity.  The only change you have to make is to the value of num_output.
# 
# After making these changes, visualize your new model.  You should see the ReLU unit appear similar to:
# 
# ![](images/add-relu.png)
# 
# Now change the name of the model and press the **Create** button.  When it is complete test the data again.  The results should be similar to:
# 
# ![](images/classify-invert-relu.png)
# 
# Were you able to correctly identify them all? If not, why do you think the results were different?

# ## Next Steps
# 
# In our example here, we were able to identify most of our test images successfully.  However, that is generally not the case.   How would go about improving our model further?  Typically, hyper-parameter searches are done to try different values of model parameters such as learning-rate or different solvers to find settings that improve model accuracy.  We could change the model to add layers or change some of the parameters within the model associated with the performance of the convolution and pooling layers.  In addition, we could try other networks.

# ## Summary
# 
# In this tutorial you were introduced to Deep Learning and all of the steps necessary to classify images including data processing, training, testing, and improving your network through data augmentation and network modifications.  In the training phase, you learned about the parameters that can determine the performance of training a network. By training a subset of the MNIST data, a full set, different models, augemented data, etc. you saw the types of options you have to control performance. In testing our model, we found that although the test images were quite different than the training data, we could still correctly classify them.
# 
# Now that you have a basic understanding of Deep Learning and how to train using DIGITS, what you do next is limited only by your own imagination.  Test what you have learned, there are several good datasets with which to practice.  We have included the [CalTech 101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) dataset at <pre>/data/101_ObjectCategories</pre> Feel free to run any part of this lab again with that (more complex) dataset.
# 
# There is still much more to learn! Start a Deep Learning project or take another course. The Nvidia Deep Learning Institute has a comprehensive library of hands-on labs for both fundamental and advanced topics.

# <a href="https://www.nvidia.com/en-us/deep-learning-ai/education/"> <img src="files/DLI Header.png" alt="Header" style="width: 400px;"/> </a>

# # Appendix
# 
# For a high performing and generalizable model:
# 
# 1. Load the dataset stored at <pre>/data/train_invert</pre> as 28x28 grayscale images. 
# 
# 2. Train a model using the LeNet network for 5 epochs.

# In[ ]:




