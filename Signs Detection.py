import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from numpy.core.fromnumeric import argmax
from tensorflow.keras.models import load_model



def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_hist_equalized)
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized

def LaplacianOfGaussian(image):
    LoG_image = cv2.GaussianBlur(image, (3,3), 0)           # paramter 
    gray = cv2.cvtColor( LoG_image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian( gray, cv2.CV_8U,3,3,2)       # parameter
    LoG_image = cv2.convertScaleAbs(LoG_image)
    return LoG_image


def binarization(image):
    thresh = cv2.threshold(image,32,255,cv2.THRESH_BINARY)[1]
    #thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return thresh

def removeSmallComponents(image, threshold=200):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img2 = np.zeros((output.shape),dtype = np.uint8)
    #for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = constrastLimit(image)
    image = LaplacianOfGaussian(image)
    image = binarization(image)
    image = removeSmallComponents(image,200)
    return image

def largest_cont(edg):
    cnts, hierarchy  = cv2.findContours(edg.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #select contour with the biggest area
    cnt = max(cnts , key = cv2.contourArea)
    #perimeter approximation (True --> closed contour)
    return cnt

def draw_cont(img,edg):
    #draw contours
    cnts, hierarchy  = cv2.findContours(edg.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts , key = cv2.contourArea)
    return cv2.drawContours(img.copy(), [cnt], -1, (0,255,0), 3)

def correct(cont,img):
    lig=cont.shape[0]
    cont=cont.reshape(lig,2)
    rect = np.zeros((4,2), dtype="float32")

    #quadrilateral estimation
    s = np.sum(cont, axis=1)
    rect[0] = cont[np.argmin(s)]
    rect[2] = cont[np.argmax(s)]

    diff = np.diff(cont, axis=1)
    rect[1] = cont[np.argmin(diff)]
    rect[3] = cont[np.argmax(diff)]

    (A, B, C, D) = rect

    #quadrilateral max(hauteur,largeur)
    widthA = np.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2 )
    widthB = np.sqrt((D[0] - C[0])**2 + (D[1] - C[1])**2 )
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt((A[0] - D[0])**2 + (A[1] - D[1])**2 )
    heightB = np.sqrt((B[0] - C[0])**2 + (B[1] - C[1])**2 )
    maxHeight = max(int(heightA), int(heightB))

    #reference quadrilateral
    dst = np.array([
    [0,0],
    [maxWidth-1, 0],
    [maxWidth-1, maxHeight-1],
    [0, maxHeight-1]], dtype="float32")

    #transformation matrix (original quad --> reference quad)
    BansformMaBix = cv2.getPerspectiveTransform(rect, dst)
    #affine transformation
    scan = cv2.warpPerspective(img.copy(), BansformMaBix, (maxWidth, maxHeight),borderMode=cv2.BORDER_REPLICATE)

    return scan

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def getCalssName(classNo):
    if classNo == 0:
        return 'Stop'
    elif classNo == 1:
        return 'Turn right ahead'
    elif classNo == 2:
        return 'Turn left ahead'
    elif classNo == 3:
        return 'Danger'

def detect_circles(image,gray,dp=1.2,minDist=100):

    '''image: 8-bit, single channel image. If working with a color image, convert to grayscale first.
    method: Defines the method to detect circles in images. Currently, the only implemented method is cv2.HOUGH_GRADIENT, which corresponds to the Yuen et al. paper.
    dp: This parameter is the inverse ratio of the accumulator resolution to the image resolution (see Yuen et al. for more details). Essentially, the larger the dp gets, the smaller the accumulator array gets.
    minDist: Minimum distance between the center (x, y) coordinates of detected circles. If the minDist is too small, multiple circles in the same neighborhood as the original may be (falsely) detected. If the minDist is too large, then some circles may not be detected at all.
    param1: Gradient value used to handle edge detection in the Yuen et al. method.
    param2: Accumulator threshold value for the cv2.HOUGH_GRADIENT method. The smaller the threshold is, the more circles will be detected (including false circles). The larger the threshold is, the more circles will potentially be returned.
    minRadius: Minimum size of the radius (in pixels).
    maxRadius: Maximum size of the radius (in pixels).'''

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, minDist)
    circles = np.uint16(np.around(circles))
    height = image.shape[0]
    width = image.shape[1]
    ROIs = []
    for i in circles[0, :]:

        # Prepare a black canvas:
        canvas = np.zeros((height, width))

        # Draw the outer circle:
        color = (255, 255, 255)
        thickness = -1
        centerX = i[0]
        centerY = i[1]
        radius = i[2]
        cv2.circle(canvas, (centerX, centerY), radius, color, thickness)

        # Create a copy of the input and mask input:
        imageCopy = image.copy()
        imageCopy[canvas == 0] = (0, 0, 0)

        # Crop the roi:
        x = centerX - radius
        y = centerY - radius
        h = 2 * radius
        w = 2 * radius

        croppedImg = imageCopy[y:y + h, x:x + w]
        croppedImg[croppedImg==0]=255
        # Store the ROI:
        ROIs.append(croppedImg) 
    return ROIs



def predict(img):
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = grayscale(img)
    img = img / 255
    img = img.reshape(1, 32, 32, 1)
    model = load_model('model.h5')
    pred = model.predict(img)
    classIndex = argmax(pred)
    return classIndex

if __name__=='__main__':
    path = 'test.png'
    orig_image = cv2.imread(path)
    image = preprocess_image(orig_image)
    #roi = correct(largest_cont(image),orig_image)
    roi = detect_circles(orig_image,image)
    print(getCalssName(predict(roi[1])))
    '''cv2.imshow('roi',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    



