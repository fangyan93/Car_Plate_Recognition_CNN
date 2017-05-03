# Car_Plate_Recognition_CNN

This program is used to do simple car plate recognition. It runs in both OpenCV and Tensorflow environment. To run it, first install Tensorflow and OpenCV in the same environment.

For Tensorflow, the [official installation](https://www.tensorflow.org/install/install_mac) is recommanded for Ubuntu and Mac OS X, for Rasberry Pi Raspbian, I directly use the pre-compiled [wheel](https://github.com/samjabrahams/tensorflow-on-raspberry-pi). 

For OpenCV, installation on Ubuntu and Mac OS X is easy, one line command should work, I use sudo apt-get install on Ubuntu and pip install on Mac, however, install OpenCV on Raspberry Pi is a little timecosuming, I followed this [instruction](http://www.pyimagesearch.com/2015/10/26/how-to-install-opencv-3-on-raspbian-jessie/). 

To test its performance, first activate the environment in which Tensorflow and OpenCV are installed, for a single image, you can modify the file name at the top of Main.py file, and run python Main.py, for multiple images, you may put all of them into a folder, then specify the folder path at the top of test.py file, and run python test.py. The convolutional neural network model is saved in 3 file whose name starts with my_model and the checkpoint file, it will be used in recognition. 

Also, a Pi camera was used to do some practical test in real time, I used motion to capture images, and test my code on them. To install motion in Raspbian on Raspberry Pi, this [link](http://sjj.azurewebsites.net/?p=701) is helpful.
