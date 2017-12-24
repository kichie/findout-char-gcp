#!/usr/bin/python
#coding:utf-8
import cv2
import numpy as np
import base64
import json
from requests import Request, Session
from bs4 import BeautifulSoup


def recognize_captcha(str_image_path):
        bin_captcha = open(str_image_path, 'rb').read()

        str_encode_file = base64.b64encode(bin_captcha).decode("utf-8")

        str_url = "https://vision.googleapis.com/v1/images:annotate?key="

        key_f = open("keys.json","r")
        api_json = json.load(key_f)
        str_api_key = api_json["api"]
        print(str_api_key)
        str_headers = {'Content-Type': 'application/json'}

        str_json_data = {
            'requests': [
                {
                    'image': {
                        'content': str_encode_file
                    },
                    'features': [
                        {
                            'type': "TEXT_DETECTION",
                            'maxResults': 10
                        }
                    ]
                }
            ]
        }

        print("begin request")
        obj_session = Session()
        obj_request = Request("POST",
                              str_url + str_api_key,
                              data=json.dumps(str_json_data),
                              headers=str_headers
                              )
        obj_prepped = obj_session.prepare_request(obj_request)
        obj_response = obj_session.send(obj_prepped,
                                        verify=True,
                                        timeout=60
                                        )
        print("end request")

        if obj_response.status_code == 200:
            #print (obj_response.text)
            with open('data.json', 'w') as outfile:
                json.dump(obj_response.text, outfile)
            return obj_response.text
        else:
            return "error"

def binary(input_data_path,output_data_path):
    img_src = cv2.imread(input_data_path)
    img_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)

    thresh  = 150
    max_pixel = 255
    ret, img_dst = cv2.threshold(img_gray,
                                 thresh,
                                 max_pixel,
                                 cv2.THRESH_BINARY)
    cv2.imshow("SHOW",img_dst)
    cv2.imwrite(output_data_path,img_dst)

if __name__ == '__main__':
    binary("./testdata/orange_name.jpg","./outdata/binary.jpg")
    data = json.loads(recognize_captcha("./outdata/binary.jpg"))
    data = data["responses"]
    print(data)
    for i in data:
        print(i["fullTextAnnotation"]["text"])
