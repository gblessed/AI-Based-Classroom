from flask import Flask, request
from flask_restful import Resource, Api
import flask
 
app = Flask(__name__)
api = Api(app)

class ParameterServer(Resource):
    def get(self):
        return {'parameters': 'received'}
    def post(self):
        somejson = request.get_json()
        print(somejson['features'])
        return flask.jsonify(somejson)
    
api.add_resource(ParameterServer, '/')
 
app.run(host='0.0.0.0', port= 8090)
