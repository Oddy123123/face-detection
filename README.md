# My Face Detector

A simple class for face detection using ResNet SSD detector model from OpenCV.

![download](https://user-images.githubusercontent.com/95940642/228975346-11d311a3-9160-4089-aca6-2d99deae6443.png)

*Image source: https://regalgentleman.com/blogs/blog/how-to-find-your-face-shape-for-men*

## Example

```python
from my_face_detector import FaceDetector

# create a face detector object
detector = FaceDetector('face_detector/')

# detect faces from RGB img
faces = detector.detect(img)
```

The `faces` variable contains a list of dictionaries, where each dictionary represents a detection and contains 3 keys:
1. confidence: Confidence of detection as a float between 0 and 1
2. box: bounding box of a face as a tuple with 4 integer values (x, y, width, height)
3. img: Image of the face

See test.ipynb for more details on usage
