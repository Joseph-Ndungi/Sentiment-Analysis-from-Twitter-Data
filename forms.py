from wtforms import DateField,SelectField, StringField
from flask_wtf import FlaskForm
from datetime import datetime
from wtforms.validators import InputRequired



class DateForm(FlaskForm):
    startDate = DateField(default=datetime.strptime('2023-01-01','%Y-%m-%d'),validators=[InputRequired()])
    endDate = DateField(default=datetime.strptime('2023-12-31','%Y-%m-%d'),validators=[InputRequired()])
    query = StringField('Text', validators=[InputRequired()])
