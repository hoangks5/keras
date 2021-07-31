from sklearn.model_selection import train_test_split
from numpy import loadtxt
from keras import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy

data = loadtxt('data-pima.csv',delimiter=',') # Đọc giá trị sau mỗi dấu ','
X = data[:,0:8]
y = data[:,8]
X_train_val, X_test, y_train_val, y_test = train_test_split(X,y,test_size=0.2)
# Lấy 20% dữ liệu để test còn 80% để train và val
X_train, X_val, y_train, y_val = train_test_split(X_train_val,y_train_val,test_size=0.2)
# Lấy 20% dữ liệu val 80% để train

# Xây dựng 1 model với 2 lớp Hiden layer
model = Sequential() # Xây dựng mô hình NN
model.add(Dense(16,input_dim=8, activation='relu')) # Thêm một lớp với 16 layer với input vào là 8
model.add(Dense(8,activation='relu')) # Thêm một lớp với 8 layer
model.add(Dense(1,activation='sigmoid')) # Thêm một lớp với 1 layer đây cũng là output

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# Cấu hình Compile model 

model.fit(X_train,y_train,epochs=100,batch_size=8,validation_data=(X_val,y_val))
# Train 100 lần với input vào mỗi lần là 8, kiểm tra với hàm validation_data

model.save('mymodel.h5') # Save model

model = load_model('mymodel.h5')
loss, acc = model.evaluate(X_test,y_test)
print("Loss =",loss)
print("Acc =",acc)
# Test dữ liệu kiểm tra độ chính xác model

X_test_1 = X_test[10]
y_test_1 = y_test[10]
X_test_1 = numpy.expand_dims(X_test_1,axis=0) # Convert thành số
y_predict = model.predict(X_test_1)
print("Giá trị dự đoán =",y_predict)
print("Giá trị thật =",y_test_1)
# Dự đoán kiểm tra một giá trị

