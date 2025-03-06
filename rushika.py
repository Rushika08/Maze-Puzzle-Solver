import cv2
import numpy as np

# Load image
image = cv2.imread("1.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Noise Removal
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge Detection
edges = cv2.Canny(blur, 50, 150)

# Find Contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Show Result
cv2.imshow("Maze Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
