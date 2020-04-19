# Social-Distancing-Analyser-wrt.-COVID-19
An AI tool to prevent spreading of coronavirus (COVID-19) by using computer vision on video surveillance.

## For education purpose only, meant as my contribution towards society

Social Distancing Analyser automatically detects the extent to which social distancing protocols are followed in the area.
Deploying it on current surveillance systems and drones used by police to monitor large areas can help to prevent coronavirus by allowing automated and better tracking of activities happening in the area. It shows analytics of the area in real time. It can also be used to alert police in case of considerable violation of social distancing protocols in a particular area. 

  ### Please fork the repository and give credits if you use any part of it.
  ## It took me time and effort to code it. I would really appreciate if you give it a star.
  ## YOU ARE NOT ALLOWED TO MONETIZE THIS CODE IN ANY FORM.(Not even youtube)
 
#### Result:

![](output.gif)

## Features:
* Get the areal time nalytics such as:
   - Number of people in a particular area
   - Number of people in high risk
   - The extent of risk to a particular person.
* Doesn't collect any data of a particular person
* Stores a video output for review

## Things needed to be improved :
* Auto-calibration [For the given sample video, I've calibrated the model by simulating a 3D depth factor based on the camera position and orientation.
* Faster processing

## Installation:
* Fork the repository and download the code.
* Download the following files and place it in the same directory
   - https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
   - https://pjreddie.com/media/files/yolov3.weights
* For slower CPUs, use yolov3-tiny (link in the code comments)
* Install all the dependenices
* Run social_distancing_analyser.py

### Credits:
##### A big thanks to Ardian Rosebrock (www.pyimagesearch.com) for CV tutorial on detection.
