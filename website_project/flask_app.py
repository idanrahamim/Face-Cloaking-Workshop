import facenet_adversarial_generate
import sqlite3
from flask import Flask, flash, request, redirect, render_template
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

statistic_dic = {'Original': 0, 'Low': 0, 'Medium': 0, 'High': 0}
privacy_lvl_to_success_percent = {'Low': '20%', 'Medium': '50%', 'High': '80%'}
privacy_lvl_to_amplification = {'Low': 3.2, 'Medium': 4.2, 'High': 6.}


def apply_adversarial_example(fp, first_amp, second_amp):
    new_path_orig, new_path_ulixes_first, new_path_ulixes_second =\
        facenet_adversarial_generate.execute_attack(fp, app.config['RES_FOLDER'], first_amp, second_amp)
    return new_path_orig.replace('./static/', ''), new_path_ulixes_first.replace('./static/', ''), new_path_ulixes_second.replace('./static/', '')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    print(f'[{datetime.now().replace(microsecond=0)}]', "Client IP:", request.remote_addr)
    return render_template('/index.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'first_privacy' not in request.form or 'second_privacy' not in request.form:
        flash('Privacy level missed')
        return redirect(request.url)
    first_privacy_lvl = request.form['first_privacy'].split(" ", 1)[0]
    second_privacy_lvl = request.form['second_privacy'].split(" ", 1)[0]
    if first_privacy_lvl not in ['Low', 'Medium', 'High'] or second_privacy_lvl not in ['Low', 'Medium', 'High']:
        flash('Wrong privacy level')
        return redirect(request.url)
    if first_privacy_lvl == second_privacy_lvl:
        flash('Select different privacy levels')
        return redirect(request.url)
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
        original_fp, first_ulixes_fp, second_ulixes_fp = apply_adversarial_example(fp,
            privacy_lvl_to_amplification[first_privacy_lvl], privacy_lvl_to_amplification[second_privacy_lvl])
        return render_template("/submission.html", original_image=original_fp, a45_image=first_ulixes_fp,
                               a20_image=second_ulixes_fp,
                               first_privacy_lvl=first_privacy_lvl, second_privacy_lvl=second_privacy_lvl,
                               first_success_percent=privacy_lvl_to_success_percent[first_privacy_lvl],
                               second_success_percent=privacy_lvl_to_success_percent[second_privacy_lvl])
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/download', methods=['POST'])
def download():
    for key in request.form:
        if key in statistic_dic:
            statistic_dic[key] += 1
    return '', 204


def db_update_func():  # updates db for download button statistics
    for val in statistic_dic.values():
        if val != 0:
            connection = sqlite3.connect("./statistic.db")
            cursor = connection.cursor()
            cursor.execute(f"update statistic set original = original+{statistic_dic['Original']},"
                           f" Low = Low+{statistic_dic['Low']}, Medium = Medium+{statistic_dic['Medium']},"
                           f" High = High+ {statistic_dic['High']}")
            connection.commit()
            for key in statistic_dic:
                statistic_dic[key] = 0

            # for row in cursor.execute("select * from statistic"):
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
