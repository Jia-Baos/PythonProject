# import OpenCV module
import cv2 as cv

# setup webcam reader
videoHandler = cv.VideoCapture(0)
succeed, inputImage = videoHandler.read()

# allocate a retina instance with input size equal to the one of the loaded image
retina = cv.bioinspired_Retina.create((inputImage.shape[1], inputImage.shape[0]))

# retina parameters management methods use sample
# -> save current (here default) retina parameters to a xml file (you may use it only one time to get the file and modify it)
retina.write("retinaParams.xml")
# -> load retina parameters from a xml file : here we load the default parameters that we just wrote to file
retina.setup("retinaParams.xml")

# main processing loop
stillProcess = True
while stillProcess is True:

    # grab a new frame and display it
    stillProcess, inputImage = videoHandler.read()
    cv.imshow("input frame", inputImage)

    # run retina on the input image
    retina.run(inputImage)

    # grab retina outputs
    retinaOut_parvo = retina.getParvo()
    retinaOut_magno = retina.getMagno()

    # draw retina outputs
    cv.imshow("retina parvo out", retinaOut_parvo)
    cv.imshow("retina magno out", retinaOut_magno)

    # wait a little to let the time for figures to be drawn
    cv.waitKey(2)
