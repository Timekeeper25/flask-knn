from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import re

app = Flask(__name__)

app.config['SECRET_KEY'] = 'secret123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(db.Model, UserMixin):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), index=True, unique=True)
    phone_number = db.Column(db.String(15), index=True, unique=True)
    password = db.Column(db.String(255), index=True, unique=True)
    role = db.Column(db.String(10), default='user')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        phone_number = request.form.get('phone_number')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if not username or not phone_number or not password or not confirm_password:
            flash('All fields are required!', 'danger')
            return redirect(url_for('register'))
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))
        if not re.match(r'^[0-9]{10,15}$', phone_number):
            flash('Invalid phone number format!', 'danger')
            return redirect(url_for('register'))
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'danger')
            return redirect(url_for('register'))
        existing_phone = User.query.filter_by(phone_number=phone_number).first()
        if existing_phone:
            flash('Phone number already registered. Please use a different one.', 'danger')
            return redirect(url_for('register'))
        hashed_password = generate_password_hash(password, method='sha256')
        new_user = User(username=username, phone_number=phone_number, password=hashed_password)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash('An error occurred while registering. Please try again.', 'danger')
            print(e)
    return render_template('register.html')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    # Paths to the CSV files
    jadwal_path = os.path.join('dataset', 'jadwal.csv')
    kelas_path = os.path.join('dataset', 'kelas.csv')

    # Read the CSV files into DataFrames
    df_jadwal = pd.read_csv(jadwal_path)
    df_kelas = pd.read_csv(kelas_path)

    df_jadwal.fillna('', inplace=True)

    # Prepare data to pass to the template
    jadwal_data = {
        'column_names': df_jadwal.columns.values,
        'row_data': df_jadwal.values.tolist()
    }

    kelas_data = {
        'column_names': df_kelas.columns.values,
        'row_data': df_kelas.values.tolist()
    }

    return render_template('dashboard.html',
                       jadwal_data=jadwal_data,
                       kelas_data=kelas_data,
                       user_role=current_user.role)

@app.route('/perhitungan', methods=['GET', 'POST'])
@login_required
def perhitungan():
    data_file = os.path.join('training_data', 'data_train_sample.csv')
    data = pd.read_csv(data_file)

    # Preprocess the data
    # Convert categorical columns to numerical values
    label_encoder = LabelEncoder()
    data['Kelas'] = label_encoder.fit_transform(data['Kelas'])

    # Define features and target
    X = data[['Usia', 'Potensi', 'Pengalaman Sebelumnya', 'Keinginan Orangtua']]
    y = data['Kelas']

    # Initialize and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    # Function to predict the class for new data
    def predict_class(new_data):
        new_data_df = pd.DataFrame([new_data], columns=['Usia', 'Potensi', 'Pengalaman Sebelumnya', 'Keinginan Orangtua'])
        predicted_class = knn.predict(new_data_df)
        return label_encoder.inverse_transform(predicted_class)[0]

    if request.method == 'POST':
        save_data = os.path.join('dataset', 'data_siswa.csv')

        Nama = request.form.get('nama')
        Usia = request.form.get('usia')
        Potensi = request.form.get('potensi')
        Pengalaman = request.form.get('pengalaman')
        Keinginan = request.form.get('keinginan')

        new_siswa = {'Usia': Usia, 'Potensi': Potensi, 'Pengalaman Sebelumnya': Pengalaman, 'Keinginan Orangtua': Keinginan}
        predicted_class = predict_class(new_siswa)

        df = pd.read_csv(save_data)

        # Generate new ID
        if df.empty:
            new_id = 1
        else:
            new_id = df['ID'].max() + 1

        new_row = pd.DataFrame([{'ID': new_id, 'Nama': Nama, 'Usia': Usia, 'Potensi': Potensi, 'Pengalaman Sebelumnya': Pengalaman, 'Keinginan Orangtua': Keinginan, 'Kelas': predicted_class}])
        df = pd.concat([df, new_row], ignore_index=True)
        save_data = os.path.join('dataset', 'data_siswa.csv')
        df.to_csv(save_data, index=False)

        return render_template('perhitungan.html', predicted_class=predicted_class)

    return render_template('perhitungan.html')

@app.route('/edit/<int:id>', methods=['GET', 'POST'])
@login_required
def edit(id):
    data_file = os.path.join('training_data', 'data_train_sample.csv')
    data = pd.read_csv(data_file)

    label_encoder = LabelEncoder()
    data['Kelas'] = label_encoder.fit_transform(data['Kelas'])

    X = data[['Usia', 'Potensi', 'Pengalaman Sebelumnya', 'Keinginan Orangtua']]
    y = data['Kelas']

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    def predict_class(new_data):
        new_data_df = pd.DataFrame([new_data], columns=['Usia', 'Potensi', 'Pengalaman Sebelumnya', 'Keinginan Orangtua'])
        predicted_class = knn.predict(new_data_df)
        return label_encoder.inverse_transform(predicted_class)[0]

    df = pd.read_csv(os.path.join('dataset', 'data_siswa.csv'))
    row_to_edit = df[df['ID'] == id].iloc[0]
    if request.method == 'POST':
        new_name = request.form['name']
        new_age = request.form['age']
        new_potensi = request.form['potensi']
        new_pengalaman = request.form['pengalaman']
        new_keinginan = request.form['keinginan']

        new_siswa = {'Usia': new_age, 'Potensi': new_potensi, 'Pengalaman Sebelumnya': new_pengalaman, 'Keinginan Orangtua': new_keinginan}
        predicted_class = predict_class(new_siswa)

        df.loc[df['ID'] == id, ['Nama', 'Usia', 'Potensi', 'Pengalaman Sebelumnya', 'Keinginan Orangtua', 'Kelas']] = [new_name, new_age, new_potensi, new_pengalaman, new_keinginan, predicted_class]
        df.to_csv(os.path.join('dataset', 'data_siswa.csv'), index=False)
        return redirect(url_for('siswa'))

    return render_template('edit.html', row=row_to_edit)

@app.route('/delete/<int:id>', methods=['GET', 'POST'])
@login_required
def delete(id):
    df = pd.read_csv(os.path.join('dataset', 'data_siswa.csv'))
    df.drop(df[df['ID'] == id].index, inplace=True)
    df.to_csv(os.path.join('dataset', 'data_siswa.csv'), index=False)
    return redirect(url_for('siswa'))

@app.route('/siswa', methods=['GET', 'POST'])
@login_required
def siswa():
    if current_user.role != 'admin':
        flash('Access denied. You do not have permission to view this page.', 'danger')
        return redirect(url_for('dashboard'))

    search_query = request.args.get('search', '')
    df = pd.read_csv(os.path.join('dataset', 'data_siswa.csv'))
    if search_query:
        filtered_data = df[df['ID'].astype(str).str.contains(search_query, case=False)]
    else:
        filtered_data = df

    return render_template('siswa.html', search_query=search_query, column_names=filtered_data.columns.values, row_data=list(filtered_data.values.tolist()), dataset=len(list(filtered_data.values.tolist())), zip=zip)

@app.route('/kelas', methods=['GET', 'POST'])
@login_required
def kelas():
    try:
        data_file = os.path.join('dataset', 'kelas.csv')
        df = pd.read_csv(data_file)
        print("Loaded kelas.csv successfully.")
    except Exception as e:
        print(f"Error loading kelas.csv: {e}")
        return "Error loading kelas.csv", 500

    try:
        data_file = os.path.join('dataset', 'data_siswa.csv')
        data = pd.read_csv(data_file)
        print("Loaded data_siswa.csv successfully.")
    except Exception as e:
        print(f"Error loading data_siswa.csv: {e}")
        return "Error loading data_siswa.csv", 500

    try:
        class_counts = data['Kelas'].value_counts().reset_index()
        class_counts.columns = ['Kelas', 'Jumlah Siswa']
        print("Calculated class counts successfully.")

        if 'Jumlah Siswa' in df.columns:
            df = df.drop(columns=['Jumlah Siswa'])
            print("Dropped 'Jumlah Siswa' from df.")

        df = df.merge(class_counts, on='Kelas', how='left')
        df['Jumlah Siswa'] = df['Jumlah Siswa'].fillna(0).astype(int)
        print("Merged and updated df with class counts.")

        if current_user.role != 'admin':
            df = df.drop(columns=['Jumlah Siswa'], errors='ignore')
            print("Dropped 'Jumlah Siswa' for non-admin user.")
    except Exception as e:
        print(f"Error processing data: {e}")
        return "Error processing data", 500

    return render_template('kelas.html', column_names=df.columns.values, row_data=list(df.values.tolist()), dataset=len(list(df.values.tolist())), zip=zip, user_role=current_user.role)


@app.route('/training', methods=['GET', 'POST'])
@login_required
def training():
    df = pd.read_csv(os.path.join('training_data', 'data_train_sample.csv'))
    return render_template('training.html', column_names=df.columns.values, row_data=list(df.values.tolist()), dataset=len(list(df.values.tolist())), zip=zip)

@app.route('/testing', methods=['GET', 'POST'])
@login_required
def testing():
    data_file = os.path.join('training_data', 'data_train_sample.csv')
    data = pd.read_csv(data_file)

    # Preprocess the data
    label_encoder = LabelEncoder()
    data['Kelas'] = label_encoder.fit_transform(data['Kelas'])

    # Define features and target
    X = data[['Usia', 'Potensi', 'Pengalaman Sebelumnya', 'Keinginan Orangtua']]
    y = data['Kelas']
    names = data['Nama']  # Keep track of the names

    # Split the data
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(X, y, names, test_size=0.1, random_state=42)

    # Initialize and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Predict on the test set
    y_pred = knn.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Function to predict the class for new data
    def predict_class(new_data):
        new_data_df = pd.DataFrame([new_data], columns=['Usia', 'Potensi', 'Pengalaman Sebelumnya', 'Keinginan Orangtua'])
        predicted_class = knn.predict(new_data_df)
        return label_encoder.inverse_transform(predicted_class)[0]

    # Inverse transform the predicted classes
    predicted_classes = label_encoder.inverse_transform(y_pred)

    # Convert DataFrame to a list of lists
    X_test_list = X_test.values.tolist()  # Convert to list of lists
    y_test_list = label_encoder.inverse_transform(y_test).tolist()  # Inverse transform y_test
    names_test_list = names_test.tolist()  # Convert names to list

    return render_template('testing.html', accuracy=accuracy, X_test=X_test_list, y_test=y_test_list, y_pred=predicted_classes, names=names_test_list)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
