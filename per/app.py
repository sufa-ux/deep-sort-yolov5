import sys
sys.path.append(r'C:\Users\waka\Desktop\DeepSORT_YOLOv5_Pytorch-master')
from flask import Flask,render_template,request,flash,redirect,url_for,session,Response,send_from_directory
# 1、引入SQLAlchemy
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash,check_password_hash
import os
from datetime import datetime,date

from PIL import Image, ImageDraw, ImageFont

from utils_ds.draw import draw_boxes

import numpy as np
from yolov5.utils.general import check_img_size
from utils_ds.draw import draw_boxes

import cv2
import argparse 
from main import VideoTracker


class apli(VideoTracker):
    def __init__(self,args,path=None):
        super().__init__(args=args)
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY']=os.urandom(24)
        self.basedir=os.path.dirname(__file__)
        self.app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///'+os.path.join(self.basedir,'../data.db')
        self.app.config['UPLOAD_FOLDER']=r'C:/Users/waka/Desktop/DeepSORT_YOLOv5_Pytorch-master/per/static/videos/'
        self.db=SQLAlchemy(self.app)
        self.camera=cv2.VideoCapture(path)

        class Username(self.db.Model):
            id = self.db.Column(self.db.Integer, primary_key=True, autoincrement=True, nullable=False)
            username = self.db.Column(self.db.String(100),nullable=False)
            password = self.db.Column(self.db.String(60),nullable=False)


        @self.app.route("/")
        def index():
            if "username" in session:

                return render_template("index.html")
            else:
                return redirect(url_for("login"))
        @self.app.route('/',methods=['POST'])
        def upload_video():
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('No image selected for uploading')
                return redirect(request.url)
            else:
                filename = file.filename
                file.save(os.path.join(self.app.config['UPLOAD_FOLDER'], filename))
                # print('upload_video filename: ' + filename)
                flash('Video successfully uploaded and displayed below')
                return render_template('index2.html', filename=filename)
        @self.app.route("/display/<path:filename>")
        def display_video(filename):
            path=os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
            print(path)
            self.camera=cv2.VideoCapture(path)
            return Response(self.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')                          

        @self.app.route('/video')
        def video():
            self.camera=cv2.VideoCapture(0)
            return Response(self.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/show_video')
        def show_video():
            return render_template("videos.html")

        #登录页面
        @self.app.route("/login", methods=['GET', 'POST'])
        def login():
            if request.method == 'POST':
                username = request.form.get('username')
                pwd = request.form.get('pwd')

                user = Username.query.filter_by(username=username).filter_by(password=pwd).first()  # 从数据库中查找匹配的用户

                if user:
                    session['logged_in'] = True
                    session['username'] = username
                    session['pwd'] = pwd
                    flash("登录成功")
                    return redirect(url_for("index"))
                else:
                    flash("您的用户名或者密码不对")

            return render_template("login.html")

        @self.app.route("/register",methods=['GET','POST'])
        def register():
            if request.method=='POST':
                username=request.form['username']
                pwd=request.form['pwd']
                pwd1=request.form['pwd1']
                # yzm=request.form['yzm']
                if not all([username,pwd,pwd1]):
                    flash('请填入完整信息')
                elif len(username) < 3 and len(username) > 0 or len(username) >15:
                    flash('用户名长度应大于 3 位小于 15 位',category='info')
                elif pwd==pwd1:
                    flash('注册成功！请登录')
                    # 将注册成功的信息存储到会话中
                    session['registered_username'] = username
                    session['registered_pwd'] = pwd
                    # 在这里添加将用户信息保存到数据库的操作
                    new_user = Username(username=username, password=pwd)
                    self.db.session.add(new_user)
                    self.db.session.commit()
                    return redirect(url_for("login"))
                else:
                    flash('密码不相符，请重新注册')
                    return redirect(url_for("register"))
            return render_template("register.html")


    def run(self,port=8888,host='127.0.0.1'):
        self.app.run(port=port,host=host)
