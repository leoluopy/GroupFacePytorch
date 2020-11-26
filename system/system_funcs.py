import sys, os, cv2
import numpy as np


def visualise(path1, path2, predicted_id, GT_id, score, timeElapse=0):
    im1 = cv2.imread(path1)
    im2 = cv2.imread(path2)

    im1 = cv2.resize(im1, (224, 224))
    im2 = cv2.resize(im2, (224, 224))

    im = np.concatenate((im1, im2), 1)

    cv2.putText(im, "similarity:{:.2f}".format(score), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(im, "predicted:{}".format(predicted_id), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(im, "ground truth:{}".format(GT_id), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("result", im)
    cv2.waitKey(timeElapse)
