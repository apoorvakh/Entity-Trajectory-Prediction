# Entity-Trajectory-Prediction

### **Preprocessing :**

>Download the Stanford Drone Dataset

>Preprocess the data: 
>>Run the files in following order - preprocess.py, trainTestSplit.py, frameSplit.py, classesSplit.py and entitiesClassesSplit.py .

>Navigate to newannotation/train/classes/entities - copy all files belonging to a particular video into a separate folder, for example, copy all files with prefix ‘bookstore0’ into a folder called bookstore0entities.

### **Running Source Code :**

>Create an ‘output’ folder with subfolders ‘entity’ and ‘frame’.

>Run the files for each method, namely, pooling, hierarchical and attention with correctly configured local paths for input and output.
