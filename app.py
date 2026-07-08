from flask import Flask, render_template, request,redirect
import pandas as pd, numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
import os

app=Flask(__name__)

r=pd.read_csv("dataset.csv")

#TRAINING THE DATA
model=LogisticRegression()
ss=StandardScaler()
le=LabelEncoder()

features=["review_text","rating","account_age_days","verified_purchase","review_length_words","all_caps_ratio","exclamation_count","repeated_phrases","profile_photo"]

r["verified_purchase"]=r["verified_purchase"].map({"Yes":1, "No":0})
r["profile_photo"]=r["profile_photo"].map({"Yes":1, "No":0})

x=r[features]
y=r["classification"].map({"Fake":0, "Suspicious":1, "Genuine":2})  #target

preprocessor = make_column_transformer(
    (TfidfVectorizer(), "review_text"),
    (
        StandardScaler(),
        [
            "rating",
            "account_age_days",
            "verified_purchase",
            "review_length_words",
            "all_caps_ratio",
            "exclamation_count",
            "repeated_phrases",
            "profile_photo"
        ]
    )
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier())
])

pipeline.fit(x,y)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict",methods=["POST"])
def submit():



    review=request.form.get("Review Text: ")      #TAKING REVIEW AS INPUT
    rate=int(request.form.get("Rating: "))         #RATING INPUT

    mapping={"yes":1, "no":0}

    ver_pur=mapping.get(request.form.get("Verified Purchase: ").lower(), -1)   #IS THE USER VERIFIED OR NOT

    pfp=mapping.get(request.form.get("Profile Photo: ").lower(), -1)           #DOES USER HAVE PFP OR NOT

    account_age = r["account_age_days"].mean()                                  #ACCOUNT AGE

    length = len(review.split())                                                # REVIEW TEXT LENGTH

    letters=[c for c in review if c.isalpha()]
    ratio=(sum(1 for c in letters if c.isupper())/len(letters)                  #CAPS RATIO
           if len(letters)>0 else 0)

    exclamation_count = review.count("!")                                       # EXCLAMATION MARK COUNT

    a = review.lower().split()
    cntr = Counter(a)
    repeated = 1 if any(c >= 3 for c in cntr.values()) else 0                   # TO COUNT REPEATED PHRASE

    # PREDICTING OUTPUT-

    new_review = pd.DataFrame({
        "review_text": [review],
        "rating": [rate],
        "account_age_days": [account_age],
        "verified_purchase": [ver_pur],
        "review_length_words": [length],
        "all_caps_ratio": [ratio],
        "exclamation_count": [exclamation_count],
        "repeated_phrases": [repeated],
        "profile_photo": [pfp]
    })

    prediction = pipeline.predict(new_review)

    label_map = {
        0: "Fake",
        1: "Suspicious",
        2: "Genuine"
    }

    if not review:
        return render_template("index.html",error="Enter review too")
    return render_template(
        "base.html",
        prediction=label_map[prediction[0]],
        review=review)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


