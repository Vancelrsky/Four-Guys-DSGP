import gzip
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import Functions
with gzip.open('cleaned_data.zip','rb') as file:
    feature_data = pd.read_csv(file,index_col=[0,1])

with gzip.open('new_label_data.zip','rb') as file:
    new_label_data = pd.read_csv(file,index_col=[0,1])

data = pd.concat([feature_data,new_label_data],join='inner',ignore_index=False,axis=1)


X_train, X_test, Y_train, Y_test = Functions.splitdata(data,0.1)
print(feature_data.shape)
print(new_label_data.value_counts())
X_train = tf.reshape(X_train,(-1,5,10))
X_test = tf.reshape(X_test,(-1,5,10))
input_dim = 10
batch_size = int(len(X_train)//10000)
units = 128
output_size = 7  # labels are from 0 to 6

# Build the RNN model
def build_model():
    lstm_layer = keras.layers.LSTM(units, input_shape=(None,input_dim))
    model = keras.models.Sequential(
        [
            lstm_layer,
            keras.layers.Dropout(0.5),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(output_size, activation='softmax'),
        ]
    )
    return model
model = build_model()
#model.summary()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer="Adam",
    metrics=["accuracy"]
)
LSTM_model = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size, epochs=50, verbose=1)
import matplotlib.pyplot as plt
import seaborn as sns  
sns.set_style("whitegrid") 


acc = LSTM_model.history['accuracy']     #获取训练集准确性数据
val_acc = LSTM_model.history['val_accuracy']    #获取验证集准确性数据
loss = LSTM_model.history['loss']          #获取训练集错误值数据
val_loss = LSTM_model.history['val_loss']  #获取验证集错误值数据

epochs = range(1,len(acc)+1)
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(121)
ax1.plot(epochs,acc,'g',label='Trainning acc')     #以epochs为横坐标，以训练集准确性为纵坐标
ax1.plot(epochs,val_acc,'b--',label='Vaildation acc') #以epochs为横坐标，以验证集准确性为纵坐标
ax1.legend()   #绘制图例，即标明图中的线段代表何种含义
plt.ylim(0,1)
ax2 = fig.add_subplot(122)
ax2.plot(epochs,loss,'g',label='Trainning loss')
ax2.plot(epochs,val_loss,'b--',label='Vaildation loss')
ax2.legend()  ##绘制图例，即标明图中的线段代表何种含义
plt.ylim(0,3)
plt.show()
from sklearn import metrics

predictions = model.predict(X_test)

y_test_pred = np.argmax(predictions, axis=1)
confusion = metrics.ConfusionMatrixDisplay.from_predictions(Y_test,y_test_pred,cmap='Blues',normalize='true',values_format='.2f')
confusion.figure_.suptitle("Confusion Matrix")
confusion.figure_.set_size_inches(8,6)
plt.grid(visible=None)
plt.show()
