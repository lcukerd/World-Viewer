## ABSTRACT
This project helps in overcoming the problems associated with visual impairment by effectively informing blind people or visually impaired people about their surroundings. Not only does the project help in perception of the surrounding but also instils a sense of independence amongst the user.
 The objects are detected and represented on a screen with their names and then converted to speech. After careful, extensive and scrupulous study over the previous detection algorithms namely Convolution Neural Network (CNN), Region based CNN(RCNN), Fast RCNN, Faster RCNN and YOLO, our project makes the use of YOLO (You Only Look Once) detection models for object detection. This choice was made after considering the limitations of all the above detection techniques on the basis of computational and storage cost.
 The 3D location of the objects is estimated from the location and the size of the bounding boxes from the detection algorithm. This detection is implemented actively on the handheld device of user or mobile which utilizes its camera and microphone for real time input feed and audio respectively. The sound is transmitted to the user via a speakerphone or earphones depending upon on the ease on user. It is played at intervals of few seconds or when the recognized object is different from previous one, whichever earliest.
 
## INTRODUCTION

Millions of people live in this world with incapacities of understanding the environment due to visual impairment. Although they can develop alternative approaches to deal with daily routines, they also suffer from certain navigation difﬁculties as well as social awkwardness. For example, it is very difﬁcult for them to ﬁnd a room in an unfamiliar environment. And blind and visually impaired people ﬁnd it difﬁcult to know whether a person is talking to them or someone else during a conversation. Computer vision technologies, especially the deep convolutional neural network, have been rapidly developed in recent years. It is promising to use the state-of-art computer vision techniques to help people with vision loss. In this project, we want to explore the possibility of using the hearing sense to understand visual objects. The sense of sight and hearing sense share a striking similarity: both visual object and audio sound can be spatially localized. It is not often realized by many people that we are capable at identifying the spatial location of a sound source just by hearing it with two ears. In our project, we build a real-time object detection and position estimation pipeline, with the goal of informing the user about surrounding object and their spatial position using binaural sound.

### OBJECTIVES
•	Detection of an object in real time with high accuracy
•	Identification of an object in real time
•	Deploying object identification model on mobile device
•	Proximity estimation of the object with respect to the user in real time
•	Generating an audio output of the object’s name and its relative proximity

### SCOPE
The project aims to build an aiding application for the visually challenged which helps them identify their surroundings without the need of any special hardware or even an internet connection. The machine learning model trained on the COCO dataset helps in real time identification of surrounding objects with high accuracy. All the user needs is an android smartphone with a working camera.
Presently, objects belonging to the 80 classes of the COCO dataset can be identified in real time along with proximity estimation and the output is audio transmitted to the user. 

### SUMMARY
This report contains, enlists and summarises all possible aspects and dimensions related to the project under discussion. From the initial research motivation to obtaining meaningful results, it has been attempted to encapsulate it all.
The abstract summarizes the project technology, applications and motivations.
The 1st chapter titled “Introduction” acquaints the reader with the objectives and subsequent motive of the project along with its scope.
The 2nd chapter, according to its title, gives a detailed description of the literature survey carried out prior to initiating the research related to the project. It contains references to the research papers studied and referred to, during the course of research.
The 3rd chapter titled “Implementation” has to sub-sections. While the first one gives an overview of the technology and fields of study that have been dug into and used so as to manifest the project ideation and corresponding research. The second one deals with the step by step implementation of the previously discussed technology in order to put it to relevant use to build the actual project. It also contains code snippets corresponding to the implementation steps.
The 4th chapter titled “Results and Discussions” presents a record of the observed results in the form of screenshots of the running application and the results logged on the console. Then it explains those results, their applications, and talks about their possible improvements.
The 5th and last chapter titled “Conclusion” sums up the working and application of the project backed by the results discussed previously. It basically tries to conclude the research by establishing a fulfilment to its motive.
The last chapter is followed by the list references (in IEEE format) used in the literature survey.
The report ends with the “Appendices” section containing the research paper written by the authors of this report. 
 
## Implementation

### Technology Overview and Use
#### Computer vision
Computer vision is an interdisciplinary scientific field that deals with how computers can be made to gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to automate tasks that the human visual system can do.
Computer vision tasks include methods for acquiring, processing, analysing and understanding digital images, and extraction of high-dimensional data from the real world in order to produce numerical or symbolic information, e.g., in the forms of decisions. Understanding in this context means the transformation of visual images (the input of the retina) into descriptions of the world that can interface with other thought processes and elicit appropriate action. This image understanding can be seen as the disentangling of symbolic information from image data using models constructed with the aid of geometry, physics, statistics, and learning theory.
As a scientific discipline, computer vision is concerned with the theory behind artificial systems that extract information from images. The image data can take many forms, such as video sequences, views from multiple cameras, or multi-dimensional data from a medical scanner. As a technological discipline, computer vision seeks to apply its theories and models for the construction of computer vision systems.
Sub-domains of computer vision include scene reconstruction, event detection, video tracking, object recognition, 3D pose estimation, learning, indexing, motion estimation, and image restoration.
The classical problem in computer vision, image processing, and machine vision is that of determining whether or not the image data contains some specific object, feature, or activity. Different varieties of the recognition problem are described as:
•	Object recognition (also called object classification) – one or several pre-specified or learned objects or object classes can be recognized, usually together with their 2D positions in the image or 3D poses in the scene. Blippar, Google Goggles and LikeThat provide stand-alone programs that illustrate this functionality.
•	Identification – an individual instance of an object is recognized. Examples include identification of a specific person's face or fingerprint, identification of handwritten digits, or identification of a specific vehicle.
•	Detection – the image data are scanned for a specific condition. Examples include detection of possible abnormal cells or tissues in medical images or detection of a vehicle in an automatic road toll system. Detection based on relatively simple and fast computations is sometimes used for finding smaller regions of interesting image data which can be further analysed by more computationally demanding techniques to produce a correct interpretation.
Currently, the best algorithms for such tasks are based on convolutional neural networks. An illustration of their capabilities is given by the ImageNet Large Scale Visual Recognition Challenge; this is a benchmark in object classification and detection, with millions of images and hundreds of object classes. Performance of convolutional neural networks, on the ImageNet tests, is now close to that of humans. The best algorithms still struggle with objects that are small or thin, such as a small ant on a stem of a flower or a person holding a quill in their hand. They also have trouble with images that have been distorted with filters (an increasingly common phenomenon with modern digital cameras). By contrast, those kinds of images rarely trouble humans. Humans, however, tend to have trouble with other issues. For example, they are not good at classifying objects into fine-grained classes, such as the particular breed of dog or species of bird, whereas convolutional neural networks handle this with ease.

#### Neural networks
A neural network is a network or circuit of neurons, or in a modern sense, an artificial neural network, composed of artificial neurons or nodes. Thus a neural network is either a biological neural network, made up of real biological neurons, or an artificial neural network, for solving artificial intelligence (AI) problems. The connections of the biological neuron are modelled as weights. A positive weight reflects an excitatory connection, while negative values mean inhibitory connections. All inputs are modified by a weight and summed. This activity is referred as a linear combination. Finally, an activation function controls the amplitude of the output. For example, an acceptable range of output is usually between 0 and 1, or it could be −1 and 1.
Unlike von Neumann model computations, artificial neural networks do not separate memory and processing and operate via the flow of signals through the net connections, somewhat akin to biological networks.
These artificial networks may be used for predictive modelling, adaptive control and applications where they can be trained via a dataset. Self-learning resulting from experience can occur within networks, which can derive conclusions from a complex and seemingly unrelated set of information.

#### Deep learning
Deep learning (also known as deep structured learning or hierarchical learning) is part of a broader family of machine learning methods based on artificial neural networks. Learning can be supervised, semi-supervised or unsupervised.
Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, audio recognition, social network filtering, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases superior to human experts.
Neural networks were inspired by information processing and distributed communication nodes in biological systems. ANNs have various differences from biological brains. Specifically, neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic (plastic) and analog.

#### YOLO
YOLO (You Only Look Once), is an algorithm for object detection. The object detection task consists in determining the location on the image where certain objects are present, as well as classifying those objects. Previous methods for this, like R-CNN and its variations, used a pipeline to perform this task in multiple steps. This can be slow to run and also hard to optimize, because each individual component must be trained separately. YOLO, does it all with a single neural network.
The input image is divided into an S x S grid of cells. For each object that is present on the image, one grid cell is said to be “responsible” for predicting it. That is the cell where the centre of the object falls into.
Each grid cell predicts B bounding boxes as well as C class probabilities. The bounding box prediction has 5 components: (x, y, w, h, confidence). The (x, y) coordinates represent the centre of the box, relative to the grid cell location (remember that, if the centre of the box does not fall inside the grid cell, then this cell is not responsible for it). These coordinates are normalized to fall between 0 and 1. The (w, h) box dimensions are also normalized to [0, 1], relative to the image size.
## Technology implementation
The following steps were followed in order to implement the project:
*	Planning and designing the application based on the needs of potential users.
*	Comparing different techniques available for object detection and classification.
*	Exploring resources available for YOLO by original authors.
*	Converting TensorFlow model and weights for YOLO, available publically by YOLO authors, to TensorFlow lite (tflite) format. This was done to allow YOLO to run on Android, that uses TensorFlow lite.
*	Loading the YOLO model in memory in the device and feeding realtime video frames to the model.
*	The output of model is unformatted to get confidence score, class and bounding box for each detected object.
*	Objects with confidence score below a threshold are ignored.
*	Perimeter of bounding box for each object is calculated for every frame and is compared with its perimeter for previous frame.
*	If the perimeter is increasing then the object is moving closer, if decreasing then it is moving away.
*	If the difference between both perimeters is less than a threshold then object is stationary.
*	Proximity Estimation, along with object class, is fed to Google Text-to-speech engine for audio feedback.


## CONCLUSION

It was concluded that out of all the detection algorithms studied, the project performed more robust detection under the usage of YOLO to provide acceptable results and help the project culminate into being an aid for affected users.
The tfLite model ran smoothly and successfully on the mobile device, hence providing am affordable and efficient platform to harness and inculcate the benefits of artificial intelligence in the daily lives of people.

 
## REFERENCES

[1] J. Redmon and A. Angelova, “Real-time grasp detection using convolutional neural networks,” 2015 IEEE International Conference on Robotics and Automation (ICRA), 2015.
[2] R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation,” 2014 IEEE Conference on Computer Vision and Pattern Recognition, 2014.
[3] R. Girshick, “Fast R-CNN,” 2015 IEEE International Conference on Computer Vision (ICCV), 2015.
[4] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no. 6, pp. 1137–1149, 2017.
[5] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
[6] H. Caesar, J. Uijlings, and V. Ferrari, “COCO-Stuff: Thing and Stuff Classes in Context,” 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018.
[7] S. Gupta, R. Girshick, P. Arbeláez, and J. Malik, “Learning Rich Features from RGB-D Images for Object Detection and Segmentation,” Computer Vision – ECCV 2014 Lecture Notes in Computer Science, pp. 345–360, 2014.
[8] A. Karpathy and L. Fei-Fei, “Deep visual-semantic alignments for generating image descriptions,” 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.
[9] J. Xiao, K. Ramdath, M. Iosilevish, D. Sigh, and A. Tsakas, “A low cost outdoor assistive navigation system for blind people,” 2013 IEEE 8th Conference on Industrial Electronics and Applications (ICIEA), 2013.

[10] “TensorFlow Lite | TensorFlow,” TensorFlow. [Online]. Available: https://www.tensorflow.org/lite. [Accessed: 24-Mar-2019].
[11] “An introduction to Text-To-Speech in Android,” Android Developers Blog, 23-Sep-2009. [Online]. Available: https://android-developers.googleblog.com/2009/09/introduction-to-text-to-speech-in.html. [Accessed: 24-Mar-2019].
