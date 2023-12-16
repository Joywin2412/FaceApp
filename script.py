import face_recognition
import cv2
import numpy as np

from flask import *
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
import tensorflow as tf
import keras
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sklearn.decomposition import PCA

# use it later
from imgaug import augmenters as iaaz

# ica
from sklearn.decomposition import FastICA

import subprocess

# import rs decoder
from reedsolo import RSCodec, ReedSolomonError
# goign to vary this
rsc = RSCodec(3)
joywin_uri = "mongodb+srv://joy:ryHVuNxW2ATaMtyy@cluster0.vffdptk.mongodb.net/Cluster()?retryWrites=true&w=majority"
uri = joywin_uri

client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client["Security"]
users_collection = db["User"]

# FingerPrint cnn model
import pickle 
with open('filename.pickle', 'rb') as handle:
    cnn_model = pickle.load(handle)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/registerFace', methods=['GET'])
def register_face():
    # Initialize the camera
    # cap = cv2.VideoCapture(0)
    # frame = ""
    # while True:
    #     ret, frame = cap.read()
    #     cv2.imshow("Capture Photo", frame)
        
        
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # Release the camera
    # cap.release()
    # cv2.destroyAllWindows()

    frame = cv2.imread("Image_db.png")
    face_encodings = face_recognition.face_encodings(frame)
    # Extract face encodings from the photo
    
    if not face_encodings:
        return "No face found in the captured image."

    # Store user data in the database
    return jsonify({"face_encodings": face_encodings[0].tolist()})


@app.route('/inputFinger', methods=['GET'])
def input_Finger():
    print("Finger")
    exe_path = 'C:\\Program Files\\Mantra\\MFS100\\Driver\\MFS100Test\\MANTRA.MFS100.Test.exe'

    try:
        subprocess.run(f'"{exe_path}"', shell=True)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    return "yo"


@app.route('/registerFinger', methods=['GET'])
def register_Finger():
    # try:
        # minitae points
        fingerprint_database_image = cv2.imread("download.jpeg")
        fingerprint_database_image = cv2.resize(fingerprint_database_image,(90,90))
        fingerprint_database_image = cv2.cvtColor(fingerprint_database_image, cv2.COLOR_BGR2GRAY)
        fingerprint_database_image = tf.expand_dims(fingerprint_database_image,axis = -1)
        fingerprint_database_image = np.expand_dims(fingerprint_database_image,axis=0)

        fingerprint_database_image2 = cv2.imread("now.png")
        fingerprint_database_image2 = cv2.resize(fingerprint_database_image2,(90,90))
        fingerprint_database_image2 = cv2.cvtColor(fingerprint_database_image2, cv2.COLOR_BGR2GRAY)
        fingerprint_database_image2 = tf.expand_dims(fingerprint_database_image2,axis = -1)
        fingerprint_database_image2 = np.expand_dims(fingerprint_database_image2,axis=0)
        
        fc2 = cnn_model.predict([fingerprint_database_image,fingerprint_database_image2])
        print(fc2[0])
        return jsonify({"fingerprint_encodings":fc2[0].tolist()})
        sift = cv2.SIFT_create()
        
        keypoints_1, descriptors_1 = sift.detectAndCompute(fingerprint_database_image, None)
        
        descriptors_list = descriptors_1.tolist()

        # Create a document with keypoints and descriptors
        fingerprint_data = {
            "keypoints": [{"x": kp.pt[0], "y": kp.pt[1], "size": kp.size, "angle": kp.angle, "response": kp.response, "octave": kp.octave, "class_id": kp.class_id} for kp in keypoints_1],
            "descriptors": descriptors_list
        }

    # Return the fingerprint data
        return jsonify(fingerprint_data)

    # except Exception as e:
    #     print("Hello")
    #     return jsonify({"error": str(e)})
    
    
@app.route('/registerUser', methods=['POST'])
def register_user():
    # try:
        # print('hello')
        import json
        user_data = request.get_json()
        # print(user_data1)
        # user_data = json.loads(user_data)
        # print(user_data['faceEncodings'])
        # print(user_data['fingerData'])
        faceEncodings = user_data['faceEncodings']
        fingerEncodings = user_data['fingerData']
        # fingerEncodings = fingerEncodings['fingerprint_encodings']

        BLA_intermediate = np.outer(np.transpose(np.array(faceEncodings)),np.array(fingerEncodings))
        # print(faceEncodings)
        # print(fingerEncodings)

        
        list = []
        
        threshold = np.mean(BLA_intermediate)
        count1 = 0
        count2 = 0
        variance = 0
        for row in BLA_intermediate:
            threshold_row = np.mean(row)
            now_bit = 0
            count2 = 0
            if(threshold_row > threshold):
                now_bit = 1
            import math
            for ele in row:
                variance = variance + np.square(ele-threshold_row)

            variance = variance / (len(BLA_intermediate[0])) 

            for ele in row:
                now_val = np.abs(threshold_row - threshold)
                reliability = 1 + math.erf(float(now_val)/float(np.sqrt(2*variance*variance)))
                list.append([count1,count2,reliability])
                BLA_intermediate[count1][count2] = now_bit
                count2 = count2 + 1
            
            count1 = count1 + 1
            # print(count1)
                
        list.sort(key = lambda x : x[2])
                       
        count_keys = 100
        lc_count = 0
        private_message = []
        for val in list:
            private_message.append(int(BLA_intermediate[val[0]][val[1]]))
            lc_count = lc_count + 1
            if(lc_count == count_keys):
                break
        # print("encode")
        # print(private_message)
        import rsa
        import random_string
        import random
        import hashlib
        special_string = random_string.get_random_string(150).encode('utf-8')
        hashed = hashlib.sha256(special_string).hexdigest()
        encoded_private_message = rsc.encode(special_string)
        encoded_private_arr = random_string.string2array(encoded_private_message)
        print("Encoded below which we will get later")
        print(encoded_private_message)
        print("The encoded array is")
        print(encoded_private_arr)
        for i in range(len(private_message),len(encoded_private_message)):
            private_message.append(0)
        # padding

        xr = []
        for i in range(len(encoded_private_message)):
            xr.append((private_message[i] ^ encoded_private_arr[i]))
        import rsa
        
        
        obj = {}
        obj['name'] = user_data["name"]
        obj['hash'] = hashed
        obj['xor'] = xr
        result = users_collection.insert_one(obj)
        if result.acknowledged:
            return jsonify({"message": "User data stored in MongoDB."})
        else:
            return jsonify({"error": "Failed to store user data."})
    # except Exception as e:
    #     print("Error")
        return jsonify({"error": "hi"})

# Function to capture a user's photo and perform login
@app.route('/login', methods=['POST'])
def login_user():
    name = request.get_json()['name']
    # Initialize the camera
    
    faceEncodings = request.get_json()['faceEncodings']
    fingerEncodings = request.get_json()['fingerData']
    # fingerEncodings = fingerEncodings['fingerprint_encodings']
    print(name)
    print("Hi")
    print(faceEncodings)
    print("Bye")
    print(fingerEncodings)
    if not faceEncodings:
        return "No face found in the captured image."
    
    user_data = users_collection.find_one({"name": name})

    if not user_data:
        return "User not found."

    hashed_enroll = user_data.get("hash")
    xor_data = user_data.get("xor")



    BLA_intermediate = np.outer(np.transpose(np.array(faceEncodings)),np.array(fingerEncodings))
        
    list = []
    
    threshold = np.mean(BLA_intermediate)
    count1 = 0
    count2 = 0
    variance = 0
    for row in BLA_intermediate:
        threshold_row = np.mean(row)
        now_bit = 0
        count2 = 0
        if(threshold_row > threshold):
            now_bit = 1
        import math

        for ele in row:
            variance = variance + np.square(ele-threshold_row)

        variance = variance / (len(BLA_intermediate[0])) 

        for ele in row:
            now_val = np.abs(threshold_row - threshold)
            reliability = 1 + math.erf(float(now_val)/float(np.sqrt(2*variance*variance)))
            list.append([count1,count2,reliability])
            BLA_intermediate[count1][count2] = now_bit
            count2 = count2 + 1
        
        count1 = count1 + 1
        # print(count1)
            
    list.sort(key = lambda x : x[2])
                    
    count_keys = 100
    lc_count = 0
    private_message = []
    for val in list:
        private_message.append(int(BLA_intermediate[val[0]][val[1]]))
        lc_count = lc_count + 1
        if(lc_count == count_keys):
            break
    
    # get the xored data to be sent to decoder

    encoded_message = []

    for i in range(len(private_message),len(xor_data)):
        private_message.append(0)

    for i in range(0,len(xor_data)):
        encoded_message.append(xor_data[i] ^ (private_message[i]))
    print(encoded_message)
    import rsa
    import random_string
    encoded_message = random_string.array2string(encoded_message)
    encoded_message = bytes(encoded_message, encoding='utf8')
    print("The encoded message is here:")
    print(encoded_message)
    decoded_message = rsc.decode(encoded_message)[0]
    print("The actual decoded message which was the random string is : ")
    print(decoded_message)
    import hashlib
    hashed_auth = hashlib.sha256(decoded_message).hexdigest()
    print("hashed someothing")
    print(hashed_auth)
    print(hashed_enroll)
    
    if hashed_auth == hashed_enroll:
        return jsonify({"message": "Login Successful."})
    else:
        return "Face recognition failed. Login unsuccessful."


# Example usage
if __name__ == "__main__":
    app.run(debug=True)
