CropScan — AI Plant Disease Detection

CropScan is a web app that helps farmers detect plant diseases by simply uploading a photo of a crop leaf. It analyses the image using a deep learning model and instantly returns the disease name, confidence score, symptoms, causes, treatment steps, and prevention tips. The app covers 38 disease conditions across 14 crop types including tomatoes, potatoes, apples, grapes, and corn. It was built using Python, Flask and TensorFlow and includes features like drag and drop upload, a searchable disease library, dark mode, and full mobile support. Built as part of a community service initiative.

How to Run:
Install dependencies: "py -3.12 -m pip install flask werkzeug tensorflow pillow numpy"
Download model: "py -3.12 -c "from tensorflow.keras.applications import MobileNetV2; m = MobileNetV2(weights='imagenet'); m.save('mobilenet_base.h5')"
Run:"py -3.12 app.py" then open "http://localhost:5000"
