from flask import Flask
from video_controller import video_bp

app = Flask(__name__)
app.register_blueprint(video_bp)

if __name__ == '__main__':
    app.run(debug=True)