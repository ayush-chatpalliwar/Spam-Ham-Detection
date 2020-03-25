import pandas as pd 
import numpy as np
from tkinter import *
from tkinter import filedialog, scrolledtext 
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter  import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


text1=''


df_review=pd.read_csv('cactus_new.csv')
df_rev=df_review.copy()

N=df_rev.shape[0]

corpus=[]
#nltk.download('stopwords')

ps=PorterStemmer()

for i in range(0,N):
    review=re.sub('[^a-zA-Z]', ' ', df_rev['Text'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review
            if not word in set(stopwords.words('english'))]
    review=" ".join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(use_idf=True, strip_accents='ascii')

y=df_rev['spam']
x=vectorizer.fit_transform(corpus)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=0, test_size=0.25)

from sklearn.tree import DecisionTreeClassifier
dt_clf=DecisionTreeClassifier(class_weight='balanced')

dt_clf.fit(x_train,y_train)
y_pred=dt_clf.predict(x_test)

#Split the test and train
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#Confussion Matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)

#Classification Report
cr=classification_report(y_test, y_pred)
print(cr)

#Accuracy Score
acc=accuracy_score(y_test, y_pred)
acc=round(acc*100, 2)
print(acc)


def browse(stemail):
    #name= fd.askopenfilename() 
    #print(name)

    window.fileName=filedialog.askopenfilename(filetypes=(("text files",".txt"),("All files","*.*")))
    print(window.fileName)
    window.title(window.fileName)
    global text1
    text1=open(window.fileName).read()
    print(text1)
    stemail.insert(END, text1)
    
def filter(stfil,text1):
    
    ps=PorterStemmer()
    rev=re.sub('[^a-zA-Z]', ' ', text1)
    rev=rev.lower()
    rev=rev.split()
    rev=[ps.stem(word) for word in rev
            if not word in set(stopwords.words('english'))]
    #text1=''
    rev=" ".join(rev)
    text1=rev
    stfil.insert(END,text1)
    
def pred_dt(enpre, text1):
    ps=PorterStemmer()
    rev=re.sub('[^a-zA-Z]', ' ', text1)
    rev=rev.lower()
    rev=rev.split()
    rev=[ps.stem(word) for word in rev
            if not word in set(stopwords.words('english'))]
    #text1=''
    rev=" ".join(rev)
    text1=rev
    
    corpus.append(text1)
    
     
    y_new=df_rev['spam']
    x_new=vectorizer.fit_transform(corpus)
        
    print(corpus[999])
        
        #print(x_new[:3141])
        
    x_train_new=x_new[:999]
    y_train_new=y_new
        
    x_test_new=x_new[999]
        
        
        
    
        
    dt_clf.fit(x_train_new,y_train_new)
    y_pred_new=dt_clf.predict(x_test_new)
    
    corpus.pop()
    
    if(y_pred_new[0]==1):
        enpre.insert(END, 'HAM')
    else:
        enpre.insert(END, 'SPAM')
        
    #enaccdt.insert(END, acc)
    
def accu(acc, enacc):
    enacc.insert(END, acc)
    


def clear(stemail, stfil, enpre, enacc):
    stemail.delete(1.0, END)
    stfil.delete(1.0, END)
    enpre.delete(0, END)
    enacc.delete(0, END)
    


window=Tk()
window.geometry('1400x820+0+0')

window.configure(background='Salmon')

btnsel=Button(window, text='SELECT FILE',font=('Arial',18),bd='5',command = lambda:browse(stemail),fg='black')
btnsel.place(x=470, y=120)

stemail=scrolledtext.ScrolledText(window, width='60', height='5',)
stemail.place(x=470, y=180)

lblhead=Label(window, text="Spam and Ham Detection", fg='Black',bg='Salmon', font=("Arial Bold", 40))
lblhead.place(x=430, y=10)

btnfil=Button(window, text="Filter", fg='Black',bd='5', font=("Arial", 20), command = lambda:filter(stfil, text1))
btnfil.place(x=470, y=280)

stfil=scrolledtext.ScrolledText(window, width='60', height='5')
stfil.place(x=470, y=350)


lblpred=Label(window, text="Prediction by Decision Tree", fg='black', font=("Arial", 25),bg='Salmon')
lblpred.place(x=470, y=450)



btnpre=Button(window, text="Predicted Answer",fg='Black',bd='5',font=("Arial", 18), command = lambda:pred_dt(enpre, text1))
btnpre.place(x=470, y=520)
enpre=Entry(window,bd=5,width=30)
enpre.place(x=700, y=530)


btnacc=Button(window, text="Accuracy",fg='Black',bd='5',font=("Arial ", 18), command = lambda:accu(acc, enacc))
btnacc.place(x=470, y=600)
enacc=Entry(window,bd=5,width=30)
enacc.place(x=700, y=610)


btn2=Button(window, text="Clear", fg='Black',font=('Arial',14),bd=5,anchor=CENTER, command = lambda:clear(stemail, stfil, enpre, enacc))
btn2.place(x=650, y=700)



window.mainloop()
