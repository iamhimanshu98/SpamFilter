from flask import Flask, url_for, render_template, request
import joblib, re

converter = joblib.load("./models/countvectorizer.lb")
model = joblib.load("./models/multinomialnaivebayes.lb")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction',methods = ['GET','POST'])
def prediction():
    if request.method == "POST":
        mail = request.form.get('mail','')
        temp = mail
        mail = mail.lower()
        mail = re.sub(r'[^a-zA-z ]', '', mail)

        mail_transformed = converter.transform([mail])
        prediction = model.predict(mail_transformed)[0]
        label ={"0":"Ham","1":"Spam"}
        output = label.get(str(prediction))

    return render_template("output.html",
                           output=output,
                           mail=temp)

if __name__ == "__main__":
    app.run(debug=True)