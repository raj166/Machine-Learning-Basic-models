import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        #store the preprocessors
        self.preprocessors = preprocessors

        #if the preprocessors are not specified then initialize them as empty list
        if self.preprocessors is None:
            self.preprocessors = []
    
    def load(self, imagePaths, verbose = -1):
        #initialize the list of features and labels
        data = []
        labels = []

        #loop over the image directories
        for (i, imagePath) in enumerate(imagePaths):
            #load the image and extract the class labels assuming that the dir is a directory
            #/path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            #check if the preprocessors are nonempty
            if self.preprocessors is not None:
                #loop over each preprocessor and apply each preprocessor to the image
                for p in self.preprocessors:
                    image = p.preprocess(image)


            #treat our pre processed image as a feature vector
            data.append(image)
            labels.append(label)

            #show an update every 'verbose' image
            if verbose > 0 and i > 0 and (i+1)%verbose==0:
                print("[INFO] Preprocessed {}/{}. ".format(i+1,len(imagePaths)))

        #return a tuple of the data and labels
        return (np.array(data), np.array(labels))
