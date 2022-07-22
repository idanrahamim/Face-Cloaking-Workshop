import facenet_adversarial_generate
import sqlite3
from flask import Flask, flash, request, redirect, render_template, abort, send_from_directory
import os
from werkzeug.utils import secure_filename
from waitress import serve
from apscheduler.schedulers.background import BackgroundScheduler
from hashlib import md5
from datetime import datetime

app = Flask(__name__)
SECRET_KEY = os.urandom(12).hex()
app.secret_key = SECRET_KEY
app.config['RES_FOLDER'] = './static/users_adversarial_examples/'
app.config['UPLOAD_FOLDER'] = './users_uploaded_images/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

statistic_dic = {'original': 0, 'fifty': 0, 'eighty': 0}


def apply_adversarial_example(fp):
    from PIL import Image
    new_path_orig, new_path_ulixes_50, new_path_ulixes_20 = facenet_adversarial_generate.execute_attack(fp, app.config['RES_FOLDER'])
    # a = Image.open(fp)
    # a = a.convert("1")
    # new_path = fp.replace(UPLOAD_FOLDER, RES_FOLDER)
    # a.save(new_path)
    return new_path_orig.replace('./static/', ''), new_path_ulixes_50.replace('./static/', ''), \
           new_path_ulixes_20.replace('./static/', '')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    print(f'[{datetime.now().replace(microsecond=0)}]', "Client IP:", request.remote_addr)
    return render_template('/index.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = md5(str(request.remote_addr).encode('utf-8')).hexdigest() \
                   + secure_filename(file.filename)
        fp = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(fp)
        original_fp, faceoff_fp, ulixes_fp = apply_adversarial_example(fp)
        return render_template("/submission.html", original_image=original_fp, a45_image=faceoff_fp, a20_image=ulixes_fp)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/download', methods=['POST'])
def download():  # sends to the user the pic he wish to download
    for key in request.form:
        try:
            privacy_lvl, dot, path = key.partition('.')
            path = path[1:].replace('static/users_adversarial_examples/', '')
            privacy_lvl = privacy_lvl.split("_")[1]
            statistic_dic[privacy_lvl] += 1
            user_ip = str(request.remote_addr)
            if md5(user_ip.encode('utf-8')).hexdigest() in path:
                print(f'[{datetime.now().replace(microsecond=0)}]', "Client IP:", user_ip, "Request download:", privacy_lvl)
                return send_from_directory(app.config['RES_FOLDER'], path, as_attachment=True)
            else:
                abort(404)
        except:
            abort(404)


def db_update_func():  # updates db for download button statistics
    for val in statistic_dic.values():
        if val != 0:
            connection = sqlite3.connect("./statistic.db")
            cursor = connection.cursor()
            cursor.execute(f"update statistic set original = original+{statistic_dic['original']},"
                           f" fifty = fifty +{statistic_dic['fifty']}, eighty = eighty + {statistic_dic['eighty']}")
            connection.commit()
            for key in statistic_dic:
                statistic_dic[key] = 0

            #for row in cursor.execute("select * from statistic"):
            #   print(row)


def txt_update_func():  # updates txt file for download button statistics
    for val in statistic_dic.values():
        if val != 0:
            with open("./statistics.txt", "r+") as f:
                lines = f.readlines()
                new = []
                for line in lines:
                    splatted_lst = line.split(",")
                    splatted_lst[1] = str(int(splatted_lst[1]) + statistic_dic[splatted_lst[0]])
                    new.append(splatted_lst[0] + "," + splatted_lst[1] + "\n")
                f.seek(0)
                f.writelines(new)
            for key in statistic_dic:
                statistic_dic[key] = 0


if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=db_update_func, trigger="interval", seconds=2)  # can chose managing db or txt file for
    # download button statistics
    scheduler.start()
    serve(app, host='0.0.0.0', port=80, threads=6)
