import cv2 as cv

image = cv.imread("Car_plate_en3.jpg", 1)

# cv.IMREAD_GRAYSCALE    0
# cv.IMREAD_COLOR        1
# cv.IMREAD_UNCHANGED    -1


cv.imshow('IMAGE display', image)
cv.waitKey(0)
cv.destroyAllWindows()
