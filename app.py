from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
from werkzeug.utils import secure_filename
from io import BytesIO
import base64
from detect import Detect


app = Flask(__name__)
app.secret_key = os.urandom(24)

# In-memory "baza danych"
users_db = {}

# Konfiguracja folderu do przechowywania plików
UPLOAD_FOLDER = 'doodles'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ścieżki do modelu i pliku z etykietami
labels_file = "labels/labels.txt"
model_path = "model/mobilenet_doodle_model.pth"
detector = Detect(labels_file=labels_file, model_path=model_path)

# Sprawdzanie dozwolonych rozszerzeń plików
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Sprawdzamy, czy użytkownik jest zalogowany
    if 'user' in session:
        return render_template('index.html', logged_in=True, username=session['user'])
    return render_template('index.html', logged_in=False)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['login']
        password = request.form['password']
        doodle_data = request.form['doodle']  # Odczytujemy doodle narysowane przez użytkownika

        if username in users_db and users_db[username]['password'] == password:
            # Sprawdzamy, czy doodle się zgadza
            if doodle_data:
                # Konwersja doodle na obraz
                img_data = base64.b64decode(doodle_data.split(',')[1])
                img_filename = f'{username}_doodle.png'
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
                with open(img_path, 'wb') as f:
                    f.write(img_data)

                # Wykrycie klasy doodle'a
                detected_class = detector.detect_class(img_path)

                # Porównanie wykrytej klasy z zapisaną w bazie danych
                if detected_class != users_db[username].get('doodle_class'):
                    flash("Doodle się nie zgadza. Spróbuj ponownie.", "error")
                    os.remove(img_path)  # Usuwamy zapisany plik doodle
                    return render_template('login.html')

                # Usuwamy plik doodle po porównaniu
                os.remove(img_path)

            # Jeśli hasło i doodle są poprawne, logujemy użytkownika
            session['user'] = username
            flash("Zalogowano pomyślnie!", "success")
            return redirect(url_for('index'))
        else:
            flash("Niepoprawny login lub hasło", "error")

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)  # Usuwamy użytkownika z sesji
    flash("Zostałeś wylogowany", "success")
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['login']  # Zmieniamy 'login' na 'username'
        password = request.form['password']
        password_conf = request.form['password-conf']
        file = request.files['file']
        doodle_data = request.form['doodle']

        # Sprawdzamy, czy hasła się zgadzają
        if password != password_conf:
            return "Hasła się nie zgadzają!"

        # Zapisywanie hasła (w prawdziwej aplikacji użyj bcrypt do haszowania!)
        users_db[username] = {'password': password}

        # Priorytet: jeśli plik jest obecny, zapisujemy plik, ignorując doodle
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Detekcja klasy
            detected_class = detector.detect_class(file_path)
            users_db[username]['file'] = file_path
            users_db[username]['doodle_class'] = detected_class
        elif doodle_data:
            # Jeśli plik nie został dodany, zapisujemy doodle
            img_data = base64.b64decode(doodle_data.split(',')[1])
            img_filename = f'{username}_doodle.png'  # Zmieniamy 'login' na 'username'
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
            with open(img_path, 'wb') as f:
                f.write(img_data)

            detected_class = detector.detect_class(img_path)
            users_db[username]['doodle'] = img_path
            users_db[username]['doodle_class'] = detected_class

        print(users_db)

        return redirect(url_for('index'))

    return render_template('register.html')

@app.route('/detect_doodle', methods=['POST'])
def detect_doodle():
    data = request.get_json()
    doodle_data = data.get('doodle')

    if not doodle_data:
        return {"error": "No doodle data provided"}, 400

    # Konwersja doodle na obraz
    img_data = base64.b64decode(doodle_data.split(',')[1])
    temp_path = "temp_doodle.png"
    with open(temp_path, 'wb') as f:
        f.write(img_data)

    # Detekcja klasy
    detected_class = detector.detect_class(temp_path)
    os.remove(temp_path)  # Usuwamy tymczasowy plik

    return {"class": detected_class}

@app.route('/detect_file', methods=['POST'])
def detect_file():
    if 'file' not in request.files:
        return {"error": "No file provided"}, 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        # Zapisz tymczasowy plik
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(temp_path)

        # Wykryj klasę
        detected_class = detector.detect_class(temp_path)
        os.remove(temp_path)  # Usuń tymczasowy plik

        return {"class": detected_class}
    return {"error": "Invalid file"}, 400


if __name__ == '__main__':
    app.run(debug=True)
