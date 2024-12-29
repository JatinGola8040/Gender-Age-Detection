import cv2
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
from PIL import Image


def gender_age_detection(fr_cv):
    # fr_cv
    #fr_cv = cv2.imread('harsha.jpg')
    #fr_cv = cv2.resize(fr_cv, (720, 640))
    
    fr_cv = np.array(fr_cv)
    plt.imshow(fr_cv)

    """**Importing Models**"""

    face1 = "opencv_face_detector.pbtxt"
    face2 = "opencv_face_detector_uint8.pb"
    age1 = "age_deploy.prototxt"
    age2 = "age_net.caffemodel"
    gen1 = "gender_deploy.prototxt"
    gen2 = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    # Face
    face = cv2.dnn.readNet(face2, face1)

    # age
    age = cv2.dnn.readNet(age2, age1)

    # gender
    gen = cv2.dnn.readNet(gen2, gen1)

    """Defining Categories for age & gender"""

    # Categories of distribution
    la = ['(0-2)', '(3-7)', '(8-12)', '(13-18)', '(19-20)', '(20-22)', '(22-24)', '(25-28)', '(28-30)', '(31-45)', '(46-60)', '(61-100)']
    lg = ['Male', 'Female']

    # Copy fr_cv
    #fr_cv = fr_cv.copy()

    """Identifying Face blob"""

    # Face detection
    fr_h = fr_cv.shape[0]
    fr_w = fr_cv.shape[1]
    blob = cv2.dnn.blobFromfr_cv(fr_cv, 1.0, (300, 300),
                                [104, 117, 123], True, False)

    face.setInput(blob)
    detections = face.forward()

    """Creating Bounding box"""

    # Face bounding box creation
    faceBoxes = []
    for i in range(detections.shape[2]):

        #Bounding box creation if confidence > 0.8
        confidence = detections[0, 0, i, 2]
        if confidence > 0.8:

            x1 = int(detections[0, 0, i, 3]*fr_w)
            y1 = int(detections[0, 0, i, 4]*fr_h)
            x2 = int(detections[0, 0, i, 5]*fr_w)
            y2 = int(detections[0, 0, i, 6]*fr_h)

            faceBoxes.append([x1, y1, x2, y2])

            cv2.rectangle(fr_cv, (x1, y1), (x2, y2),
                        (0, 255, 0), int(round(fr_h/150)), 8)

    

    """Implementing gender & age detection on face"""

    # Checking if face detected or not
    '''
    if not faceBoxes:
        print("No face detected")
    else:
        # Loop if faces are detected
        for faceBox in faceBoxes:

            # Extracting face as per the faceBox
            face = fr_cv[max(0, faceBox[1] - 15):
                        min(faceBox[3] + 15, fr_cv.shape[0] - 1),
                        max(0, faceBox[0] - 15):min(faceBox[2] + 15,
                                                    fr_cv.shape[1] - 1)]

            # Extracting the main blob part
            blob = cv2.dnn.blobFromfr_cv(
                face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Prediction of gender
            gen.setInput(blob)
            genderPreds = gen.forward()
            gender = lg[genderPreds[0].argmax()]

            # Prediction of age
            age.setInput(blob)
            agePreds = age.forward()

            age_conf_threshold = 0.6

            if np.max(agePreds[0]) > age_conf_threshold:
                age_index = np.argmax(agePreds)
                predictedAge = la[age_index]

                # Putting text of age & gender
                # At the top of box
                cv2.putText(fr_cv,
                            f'{gender}, {predictedAge}',
                            (faceBox[0] - 150, faceBox[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.3,
                            (217, 0, 0),
                            4,
                            cv2.LINE_AA)
    '''
    
    for faceBox in faceBoxes:

            # Extracting face as per the faceBox
            face = fr_cv[max(0, faceBox[1] - 15):
                        min(faceBox[3] + 15, fr_cv.shape[0] - 1),
                        max(0, faceBox[0] - 15):min(faceBox[2] + 15,
                                                    fr_cv.shape[1] - 1)]

            # Extracting the main blob part
            blob = cv2.dnn.blobFromfr_cv(
                face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Prediction of gender
            gen.setInput(blob)
            genderPreds = gen.forward()
            gender = lg[genderPreds[0].argmax()]

            # Prediction of age
            age.setInput(blob)
            agePreds = age.forward()

            age_conf_threshold = 0.6

            if np.max(agePreds[0]) > age_conf_threshold:
                age_index = np.argmax(agePreds)
                predictedAge = la[age_index]

                # Putting text of age & gender
                # At the top of box
                cv2.putText(fr_cv,
                            f'{gender}, {predictedAge}',
                            (faceBox[0] - 150, faceBox[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.3,
                            (217, 0, 0),
                            4,
                            cv2.LINE_AA)
    
        
    '''plt.imshow(fr_cv)
        plt.show()
        plt.savefig(fr_cv.png)'''

        #return fr_cv
        
    '''
        buf= io.bytesIO()
        plt.savefig(buf, format='png')

        buf.seek(0)
        pil_image=Image.open(buf)
        
        image_data = Image.toarray(pil_image)
        
        return image_data '''



iface= gr.Interface(
    fn=gender_age_detection,
    inputs=gr.Image(),
    outputs=gr.Image()
)

iface.launch()