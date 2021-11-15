from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import glob
from pyimageseacrch.preprocessing.SimplePreprocessor import SimplePreprocessor
from pyimageseacrch.dataset.SimpleDatasetLoader import SimpleDatasetLoader
from imutils import paths
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the dataset")
ap.add_argument("-k" , "--neighbors" , type=int, default=1, help="Number of neighbors")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs for kNN distance(-1 uses all available)")
args = vars(ap.parse_args())

# grab the list of images that well be describing 
print("[INFO] loading Images...")
imagePath = list(paths.list_images(args['dataset']))

#initialize the image preprocessor, load the dataset from disk

sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, lables) = sdl.load(imagePath , verbose=500)
data = data.reshape((data.shape[0],3072))


#show some information on memmory consumption of the imagePath
print("[INFO] feature matrix: {:.1f}MB".format(data.nbytes / (1024*1000.0)))

le = LabelEncoder()
lables = le.fit_transform(lables)

(trainX , testX, trainY, testY) = train_test_split(data , lables , test_size= 0.35 , random_state=42)


print("[INFO] evaluating k-nn classifier...")
model = KNeighborsClassifier(n_neighbors=args['neighbors'], n_jobs= args['jobs'])
model.fit(trainX, trainY)

print(classification_report(testY , model.predict(testX), target_names=le.classes_))