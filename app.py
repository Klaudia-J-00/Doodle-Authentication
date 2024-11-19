from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
from werkzeug.utils import secure_filename
import base64
from detect import Detect


app = Flask(__name__)
app.secret_key = os.urandom(24)

# In-memory db
users_db = {}

# Folder for uploaded doodles during registration
UPLOAD_FOLDER = 'doodles'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Label file and model path
labels_file = "labels/labels.txt"
model_path = "model/mobilenet_doodle_model.pth"
# Initialize the detector
detector = Detect(labels_file=labels_file, model_path=model_path)

# Function to check if the file is allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Main route, checks if user is logged in (session)
@app.route('/')
def index():
    if 'user' in session:
        return render_template('index.html', logged_in=True, username=session['user'])
    return render_template('index.html', logged_in=False)

# Login route, checks if user is already logged in and if not, checks if the login and password are correct
# If they are, checks if the doodle is correct and logs the user in
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['login']
        password = request.form['password']
        doodle_data = request.form['doodle']

        if username in users_db and users_db[username]['password'] == password:
            if doodle_data:
                # Convert doodle to image
                img_data = base64.b64decode(doodle_data.split(',')[1])
                img_filename = f'{username}_doodle.png'
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
                # Save the doodle in doodles folder
                with open(img_path, 'wb') as f:
                    f.write(img_data)

                # Use the detector to detect the class of the doodle
                detected_class = detector.detect_class(img_path)

                # Check if the detected class is the same as the one saved in the db
                if detected_class != users_db[username].get('doodle_class'):
                    flash("Doodle się nie zgadza. Spróbuj ponownie.", "error")
                    return render_template('login.html')

            # If the doodle is correct, log the user in
            session['user'] = username
            flash("Zalogowano pomyślnie!", "success")
            return redirect(url_for('index'))
        else:
            # If the login or password is incorrect, flash an error message
            flash("Niepoprawny login lub hasło", "error")

    return render_template('login.html')

# Logout route, pops the user from the session and flashes a message
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Zostałeś wylogowany", "success")
    return redirect(url_for('index'))

# Register route, checks if the passwords match, saves the password in the db
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['login']
        password = request.form['password']
        password_conf = request.form['password-conf']
        file = request.files['file']
        doodle_data = request.form['doodle']

        # Check if the passwords match
        if password != password_conf:
            return "Hasła się nie zgadzają!"

        # Save the password in the db
        users_db[username] = {'password': password}

        # Priority to file - if file is uploaded, save it and detect the class
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Detect the class of the file
            detected_class = detector.detect_class(file_path)
            users_db[username]['file'] = file_path
            users_db[username]['doodle_class'] = detected_class
        elif doodle_data:
            # If there was no file uploaded, check if the doodle is provided
            # Convert doodle to image
            img_data = base64.b64decode(doodle_data.split(',')[1])
            # Save the doodle in doodles folder
            img_filename = f'{username}_doodle.png'
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
            with open(img_path, 'wb') as f:
                f.write(img_data)

            # Detect the class of the doodle
            detected_class = detector.detect_class(img_path)
            users_db[username]['doodle'] = img_path
            users_db[username]['doodle_class'] = detected_class

        # Print the db for debugging purposes
        print(users_db)

        return redirect(url_for('index'))

    return render_template('register.html')

# Route for "Zaakceptuj obrazek" button, checks if the doodle was provided,
# saves it temporarily and detects the class of the doodle (used in register and login.html)
@app.route('/detect_doodle', methods=['POST'])
def detect_doodle():
    data = request.get_json()
    doodle_data = data.get('doodle')

    if not doodle_data:
        return {"error": "No doodle data provided"}, 400

    # Convert doodle to image
    img_data = base64.b64decode(doodle_data.split(',')[1])
    # Save the doodle temporarily
    temp_path = "temp_doodle.png"
    with open(temp_path, 'wb') as f:
        f.write(img_data)

    # Detect the class of the doodle
    detected_class = detector.detect_class(temp_path)
    os.remove(temp_path)  # Remove temporary file

    return {"class": detected_class}

# Route for "Zaakceptuj plik" button, checks if the file was provided,
# saves it temporarily and detects the class of the file (used in register.html)
@app.route('/detect_file', methods=['POST'])
def detect_file():
    if 'file' not in request.files:
        return {"error": "No file provided"}, 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(temp_path)

        detected_class = detector.detect_class(temp_path)
        os.remove(temp_path)

        return {"class": detected_class}
    return {"error": "Invalid file"}, 400


if __name__ == '__main__':
    app.run(debug=True)
