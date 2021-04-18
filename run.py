from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from flask_cors import CORS
from datetime import datetime
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import cv2
import numpy as np
import json
import datetime
import time
import os
import io
import sys
import random
import base64
from random import randint
from tasks_3_4 import save_hist_image, save_hist_hist_image, save_dft_image, save_dft_hist_image, save_dct_image, save_dct_hist_image, save_scale_image, save_scale_hist_image, save_gradient_image, save_gradient_hist_image, save_hist_3d_image, save_dft_3d_image, save_dct_3d_image, save_scale_3d_image, save_gradient_3d_image, save_parallel_system

app = Flask(__name__)
CORS(app)
api = Api(app)

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def get_sym_lines():
    imagesEncoded = []
    while len(imagesEncoded) < 7:
        imgName = str(random.randint(1,228)) + '.jpg'
        img = cv2.imread("./Dataset/" + imgName, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        image_copy = img.copy()

        for (x,y,width,height) in faces:
            face_cropped = gray.copy()[y:y+height, x:x+width]

            size = face_cropped.shape[::-1]
            w = size[0]//3

            max_value = float('inf')
            central, min_dist = w, max_value
            for line in range(w,size[0] - w):
                cur_dist = 0           
                for i in range(1,w+1):
                    for j in range(size[1]):
                        cur_dist += np.abs(int(face_cropped[j][line-i]) - int(face_cropped[j][line+i]))

    #             cur_dist = np.sum(np.abs(face_cropped[:,line-w:line][::-1]-face_cropped[:,line+1:line+w+1]))
                if cur_dist < min_dist:
                    central, min_dist = line, cur_dist
            cv2.line(image_copy, (x+central, y+height//4), (x+central, y+3*height//4), (255,0,0), 2) 

            w1 = w//2
            left, left_dist = w1, max_value
            right, right_dist = central + w1, max_value
            left_line = w1
            right_line = central + w1
            while left_line < central - w1 and right_line < size[0] - w1:
                cur_left_dist, cur_right_dist = 0, 0
                for i in range(1, w1+1):
                    for j in range(0, int(3*size[1]/4)):
                        cur_left_dist += np.abs(int(face_cropped[j][left_line-i]) - int(face_cropped[j][left_line+i]))
                        cur_right_dist += np.abs(int(face_cropped[j][right_line-i]) - int(face_cropped[j][right_line+i]))
                if cur_left_dist < left_dist:
                    left, left_dist = left_line, cur_left_dist
                if cur_right_dist < right_dist:
                    right, right_dist = right_line, cur_right_dist
                left_line += 1
                right_line += 1

            cv2.line(image_copy, (x+left, y+height//3), (x+left, y+2*height//4), (255,0,0), 2) 
            cv2.line(image_copy, (x+right, y+height//3), (x+right, y+2*height//4), (255,0,0), 2)

            img = cv2.cvtColor(image_copy,cv2.COLOR_BGR2GRAY)

            imgNameOut = "out_" + imgName
            cv2.imwrite("./SymmetryLinesOut/"+ imgNameOut, img)
            with open("./SymmetryLinesOut/" + imgNameOut, "rb") as image_file:
                encodedString = base64.b64encode(image_file.read())
            imagesEncoded.append(str(encodedString))

    return imagesEncoded

def getViolaJonesImages():
    imagesEncoded = []
    for x in range(6):
        imgName = str(random.randint(1,228)) + '.jpg'
        img = cv2.imread("./Dataset/" + imgName, 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imgNameOut = "out_" + imgName
        cv2.imwrite("./ViolaJonesOut/"+ imgNameOut, img)
        with open("./ViolaJonesOut/" + imgNameOut, "rb") as image_file:
            encodedString = base64.b64encode(image_file.read())
        imagesEncoded.append(str(encodedString))

    return imagesEncoded

def getImages(forFacePart):
    imagesEncoded = []
    template = cv2.imread('0.jpg',0)[80:190,75:175]

    if forFacePart == "face":
        template = cv2.imread('0.jpg',0)[80:190,75:175]
    elif forFacePart == "upperFace":
        template = cv2.imread('0.jpg',0)[80:150,75:175]
    elif forFacePart == "eyes":
        template = cv2.imread('0.jpg',0)[100:130,75:175]
    elif forFacePart == "nose":
        template = cv2.imread('0.jpg',0)[100:155,110:140]
    elif forFacePart == "mouth":
        template = cv2.imread('0.jpg',0)[145:175,100:155]

    w, h = template.shape[::-1]

    imgName = str(random.randint(1,228)) + '.jpg'

    for meth in methods:
        img = cv2.imread("./Dataset/" + imgName, 0)
        method = eval(meth)

        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img,top_left, bottom_right, 255, 2)
        
        imgNameOut = "out_" + imgName
        cv2.imwrite("./TemplateMatchingOut/"+ imgNameOut, img)
        with open("./TemplateMatchingOut/" + imgNameOut, "rb") as image_file:
            encodedString = base64.b64encode(image_file.read())
            imagesEncoded.append(str(encodedString))

    return imagesEncoded

def getTemplate():
    with open("0.jpg", "rb") as image_file:
        encodedString = base64.b64encode(image_file.read())
    return str(encodedString)

class Generate(Resource):
    def get(self):            
        response = app.response_class(
            response=json.dumps({"message": "Everything went fine!", "data": getTemplate()}),
            status=201,
            mimetype="application/json",
        )
        return response

class GenerateImages(Resource):
    def get(self):
        facePart = request.args.get('facePart')

        response = app.response_class(
            response=json.dumps({"message": "Everything went fine!", "data": getImages(facePart)}),
            status=201,
            mimetype="application/json",
        )
        return response

class GenerateViolaJonesImages(Resource):
    def get(self):
        response = app.response_class(
            response=json.dumps({"message": "Everything went fine!", "data": getViolaJonesImages()}),
            status=201,
            mimetype="application/json",
        )
        return response

class GenerateSymmetryLinesImages(Resource):
    def get(self):
        response = app.response_class(
            response=json.dumps({"message": "Everything went fine!", "data": get_sym_lines()}),
            status=201,
            mimetype="application/json",
        )
        return response

class GenerateHistExampleImages(Resource):
    def get(self):
        photos = [randint(1, 75), randint(76, 150), randint(151, 250)]
        imagesEncoded = []
        imagesEncoded.append(save_hist_image(photos))
        imagesEncoded.append(save_hist_hist_image(photos))
        response = app.response_class(
            response=json.dumps({"message": "Everything went fine!", "data": imagesEncoded}),
            status=201,
            mimetype="application/json",
        )
        return response

class GenerateDftExampleImages(Resource):
    def get(self):
        photos = [randint(1, 75), randint(76, 150), randint(151, 250)]
        imagesEncoded = []
        imagesEncoded.append(save_dft_image(photos))
        imagesEncoded.append(save_dft_hist_image(photos))
        response = app.response_class(
            response=json.dumps({"message": "Everything went fine!", "data": imagesEncoded}),
            status=201,
            mimetype="application/json",
        )
        return response

class GenerateDctExampleImages(Resource):
    def get(self):
        photos = [randint(1, 75), randint(76, 150), randint(151, 250)]
        imagesEncoded = []
        imagesEncoded.append(save_dct_image(photos))
        imagesEncoded.append(save_dct_hist_image(photos))
        response = app.response_class(
            response=json.dumps({"message": "Everything went fine!", "data": imagesEncoded}),
            status=201,
            mimetype="application/json",
        )
        return response

class GenerateScaleExampleImages(Resource):
    def get(self):
        photos = [randint(1, 75), randint(76, 150), randint(151, 250)]
        imagesEncoded = []
        imagesEncoded.append(save_scale_image(photos))
        imagesEncoded.append(save_scale_hist_image(photos))
        response = app.response_class(
            response=json.dumps({"message": "Everything went fine!", "data": imagesEncoded}),
            status=201,
            mimetype="application/json",
        )
        return response

class GenerateGradientExampleImages(Resource):
    def get(self):
        photos = [randint(1, 75), randint(76, 150), randint(151, 250)]
        imagesEncoded = []
        imagesEncoded.append(save_gradient_image(photos))
        imagesEncoded.append(save_gradient_hist_image(photos))
        response = app.response_class(
            response=json.dumps({"message": "Everything went fine!", "data": imagesEncoded}),
            status=201,
            mimetype="application/json",
        )
        return response

class GenerateHistParameterSelectionImage(Resource):
    def get(self):
        best_params, best_score, array_acc, imageEncoded = save_hist_3d_image()
        response = app.response_class(
            response=json.dumps({"message": "Everything went fine!", "best_parameter": str(best_params[0]), "number_of_etalons": str(best_params[1]*10), "number_of_test_images": str(10 - best_params[1]*10), "best_score": str(best_score), "data": imageEncoded}),
            status=201,
            mimetype="application/json",
        )
        return response

class GenerateDftParameterSelectionImage(Resource):
    def get(self):
        best_params, best_score, array_acc, imageEncoded = save_dft_3d_image()
        response = app.response_class(
            response=json.dumps({"message": "Everything went fine!", "best_parameter": str(best_params[0]), "number_of_etalons": str(best_params[1]*10), "number_of_test_images": str(10 - best_params[1]*10), "best_score": str(best_score), "data": imageEncoded}),
            status=201,
            mimetype="application/json",
        )
        return response

class GenerateDctParameterSelectionImage(Resource):
    def get(self):
        best_params, best_score, array_acc, imageEncoded = save_dct_3d_image()
        response = app.response_class(
            response=json.dumps({"message": "Everything went fine!", "best_parameter": str(best_params[0]), "number_of_etalons": str(best_params[1]*10), "number_of_test_images": str(10 - best_params[1]*10), "best_score": str(best_score), "data": imageEncoded}),
            status=201,
            mimetype="application/json",
        )
        return response

class GenerateScaleParameterSelectionImage(Resource):
    def get(self):
        best_params, best_score, array_acc, imageEncoded = save_scale_3d_image()
        response = app.response_class(
            response=json.dumps({"message": "Everything went fine!", "best_parameter": str(best_params[0]), "number_of_etalons": str(best_params[1]*10), "number_of_test_images": str(10 - best_params[1]*10), "best_score": str(best_score), "data": imageEncoded}),
            status=201,
            mimetype="application/json",
        )
        return response

class GenerateGradientParameterSelectionImage(Resource):
    def get(self):
        best_params, best_score, array_acc, imageEncoded = save_gradient_3d_image()
        response = app.response_class(
            response=json.dumps({"message": "Everything went fine!", "best_parameter": str(best_params[0]), "number_of_etalons": str(best_params[1]*10), "number_of_test_images": str(10 - best_params[1]*10), "best_score": str(best_score), "data": imageEncoded}),
            status=201,
            mimetype="application/json",
        )
        return response

class GenerateParallelSystemImages(Resource):
    def get(self):
        imagesEncoded = save_parallel_system()
        response = app.response_class(
            response=json.dumps({"message": "Everything went fine!", "data": imagesEncoded}),
            status=201,
            mimetype="application/json",
        )
        return response

api.add_resource(Generate, "/generate")
api.add_resource(GenerateImages, "/generate-images")
api.add_resource(GenerateViolaJonesImages, "/generate-viola-jones-images")
api.add_resource(GenerateSymmetryLinesImages, "/generate-symmetry-lines-images")
api.add_resource(GenerateHistExampleImages, "/generate-hist-example-images")
api.add_resource(GenerateDftExampleImages, "/generate-dft-example-images")
api.add_resource(GenerateDctExampleImages, "/generate-dct-example-images")
api.add_resource(GenerateScaleExampleImages, "/generate-scale-example-images")
api.add_resource(GenerateGradientExampleImages, "/generate-gradient-example-images")
api.add_resource(GenerateHistParameterSelectionImage, "/generate-hist-parameter-selection-image")
api.add_resource(GenerateDftParameterSelectionImage, "/generate-dft-parameter-selection-image")
api.add_resource(GenerateDctParameterSelectionImage, "/generate-dct-parameter-selection-image")
api.add_resource(GenerateScaleParameterSelectionImage, "/generate-scale-parameter-selection-image")
api.add_resource(GenerateGradientParameterSelectionImage, "/generate-gradient-parameter-selection-image")
api.add_resource(GenerateParallelSystemImages, "/generate-parallel-system-images")

if __name__ == "__main__":
    app.run(port=5000, debug=True)


