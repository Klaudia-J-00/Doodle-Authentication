from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
from werkzeug.utils import secure_filename
from io import BytesIO
import base64

app = Flask(__name__)
app.secret_key = os.urandom(24)

# In-memory "baza danych"
users_db = {}

# Konfiguracja folderu do przechowywania plików
UPLOAD_FOLDER = 'doodles'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
        username = request.form['login']  # Zmieniamy 'login' na 'username'
        password = request.form['password']

        if username in users_db and users_db[username]['password'] == password:
            session['user'] = username  # Zapisujemy login użytkownika w sesji
            flash("Zalogowano pomyślnie!", "success")
            print("Redirecting to index page...")
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
            users_db[username]['file'] = file_path
        elif doodle_data:
            # Jeśli plik nie został dodany, zapisujemy doodle
            img_data = base64.b64decode(doodle_data.split(',')[1])
            img_filename = f'{username}_doodle.png'  # Zmieniamy 'login' na 'username'
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
            with open(img_path, 'wb') as f:
                f.write(img_data)
            users_db[username]['doodle'] = img_path

        print(users_db)

        return redirect(url_for('index'))

    return render_template('register.html')

if __name__ == '__main__':
    app.run(debug=True)
