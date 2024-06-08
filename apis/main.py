from flask import Flask
from flask_cors import CORS
from video_controller import video_bp
from firebase_utils import save_server_ip

app = Flask(__name__)
CORS(app)
app.register_blueprint(video_bp)

if __name__ == "__main__":
    save_server_ip()
    app.run(debug=True,host='0.0.0.0', port=9999)
