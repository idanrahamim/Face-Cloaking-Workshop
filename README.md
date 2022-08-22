# Face-Cloaking-Workshop

Workshop on usable security and privacy, under the supervision of Dr. Mahmood Sharif at Tel-Aviv University.

### How to run the project

The final product of this project is a website where the user can upload an image. After that, he gets a side-by-side view of the
image with adversarial examples, that would with certainly presented probability fools real-world face recognition systems,
as described in detail in our report.

#### To run the website server you should:

0. Download and install Python 3.10.6
1. Clone the repository and ```cd``` to the project folder.
2. Install the requirements.txt file: ```pip install -r requirements.txt```
3. Create a database by running ```python ./website_project/dbs_create.py```
4. Execute the flask app with: ```python ./website_project/flask_app.py```
5. To open the website, run a browser and type localhost in the address bar. 

The rest of the code is the code we used for research and creating the report. To reproduce the presented
results follow the documentation of the file ```facenet_adversarial_generate.py```.
