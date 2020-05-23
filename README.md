# Social-Distancing-Analyser-COVID-19
An AI tool to prevent spreading of coronavirus (COVID-19) by using computer vision on video surveillance.
A social distancing analyzer AI tool to regulate social distancing protocol using video surveillance of CCTV cameras and drones. Social Distancing Analyser to prevent COVID19


## For education purpose only, meant as my contribution towards society

Social Distancing Analyser automatically detects the extent to which social distancing protocols are followed in the area.
Deploying it on current surveillance systems and drones used by police to monitor large areas can help to prevent coronavirus by allowing automated and better tracking of activities happening in the area. It shows analytics of the area in real time. It can also be used to alert police in case of considerable violation of social distancing protocols in a particular area. 

  ### Please fork the repository and give credits if you use any part of it. :slightly_smiling_face:
  ## It took me time and effort to code it.
  ## I would really appreciate if you give it a star. :star:
  ## YOU ARE NOT ALLOWED TO COMMERCIALIZE OR MONETIZE THIS CODE IN ANY FORM.(Not even youtube)
  ## IF ANYONE FOUND TO BE VIOLATING THESE TERMS AND LICENCE, LEGAL ACTION WOULD BE TAKEN AGAINST THE PERSON.
  ## ASK FOR MY PERMISSION via email (mr.skilled.coder@gmail.com) before publicizing any part of code or output generated from it. Read Licence to avoid problems
 
#### v1.0 output:

![](output.gif)

#### v2.0 output:
![](op2.gif)

### V3.0 output (code coming soon)
## Features:
* Get the areal time analytics such as:
   - Number of people in a particular area
   - Number of people in high risk
   - The extent of risk to a particular person.
* Doesn't collect any data of a particular person
* Stores a video output for review

## Things needed to be improved :
* ~~Auto-calibration [For the given sample video, I've calibrated the model by simulating a 3D depth factor based on the camera position and orientation.]~~ (Check out v2.0)
* Faster processing
#### Please Note: angle factor is needed to be set between 0 to 1 for v2.0 according to the angle of camera (move towards one as angle becomes verticle)
## Installation:
* Fork the repository and download the code.
* Download the following files and place it in the same directory
   - https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
   - https://pjreddie.com/media/files/yolov3.weights
* For slower CPUs, use yolov3-tiny (link in the code comments)
* Install all the dependenices
* Run social_distancing_analyser.py or social_distancing_analyser 2.0.py

### Credits:

@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}

copyright © 2020 Ankush Chaudhari | All rights reserved
