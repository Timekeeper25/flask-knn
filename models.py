from flask_sqlalchemy import SQLAlchemy
          
db = SQLAlchemy()
          
class Users(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), index=True, unique=True)
    phone_number = db.Column(db.String(15), index=True, unique=True)
    password = db.Column(db.String(255), index=True, unique=True)
    role = db.Column(db.String(10), default='user')
