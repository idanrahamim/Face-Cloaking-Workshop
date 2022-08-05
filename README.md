# Face-Cloaking-Workshop

Workshop on usable security and privacy, under the supervision of Dr. Mahmood Sharif at Tel-Aviv University

### How to run the project

The final product of this project is a website that the user can upload an image and then get side-by-side view of the
image with adversarial examples that would with certain presented probability fools real-world face recognition systems
as described in details in our report.

#### To run this website you should

0. clone the repository and ```cd``` to the project folder.
1. install the requirements.txt file: ```pip install -r requirements.txt```
2. create a database by running ```python ./website_project/dbs_create.py```
3. execute the flask app with: ```python ./website_project/flask_app.py```

The rest of the code is the code we used for research and to create the report. In order to reproduce the presented
results follow the documentation of the file ```facenet_adversarial_generate.py```.
