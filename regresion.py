import sqlite3
import tensorflow as tf 
import numpy as np
from tkinter import ttk
from tkinter import *

import matplotlib.pyplot as plt

class Ecuacion:
    
    dbname = 'datos.db'
    
    def __init__(self,window):
        self.wind = window
        self.wind.title('Aplicacion de regresion lineal')
        self.x = np.array([1,2,3,4,5,6,7,8],dtype=float)
        self.y = np.array([58,67,79,93,107,122,138,155],dtype=float)
        self.formula=''
        
        
        #crear frame
        frame = LabelFrame(self.wind, text="Ingresa los valores de X")
        frame.pack()
        #inputx
        Label(frame,text="ValorX: ").pack()
        self.xval = Entry(frame)
        self.xval.pack()
        #
        
        #inputy
        Label(frame,text="ValorY: ").pack()
        self.yval = Entry(frame)
        self.yval.pack()
        #button
        ttk.Button(frame,text="Agregar Valores",command=self.addDatos).pack()
        ttk.Button(frame,text="Entrenar Red",command=self.trainNetwork).pack()
        ttk.Button(frame,text="Borrar Datos",command=self.deleteRecords).pack()
        #tabla
        self.tree = ttk.Treeview(height=10,columns =('X','Y','XY','X^2','Y^2','Yr','YYr2','YYp2'))
        
        self.tree['show'] = 'headings'
        self.tree.heading('#0',text="", anchor=CENTER)
        self.tree.heading('X',text="X", anchor=CENTER)
        self.tree.heading('Y',text="Y", anchor=CENTER)
        self.tree.heading('XY',text="XY", anchor=CENTER)
        self.tree.heading('X^2',text="X^2", anchor=CENTER)
        self.tree.heading('Y^2',text="Y^2", anchor=CENTER)
        self.tree.heading('Yr',text="Yr", anchor=CENTER)
        self.tree.heading('YYr2',text="(Y-Yr)^2", anchor=CENTER)
        self.tree.heading('YYp2',text="(Y-Yp)^2", anchor=CENTER)
        self.i = 0
        self.tree.pack()
        self.getDatos()
    def AnadirValores(self):
        
        np.append(self.x,float(self.xval.get()))
        np.append(self.y,float(self.yval.get()))
        
        for j in self.tree.get_children():
            self.tree.delete(j)
        for xvalue in self.x:
            self.tree.insert('', index='end', values=(xvalue,self.y[self.i]))
            self.i = self.i+1
        
        self.i = 0

    def mQuery(self,query,parameters={}):
        with sqlite3.connect(self.dbname) as conn:
            cursor = conn.cursor()
            result = cursor.execute(query,parameters)  
            conn.commit()  
        return result
    
    def getDatos(self):
        records = self.tree.get_children()
        for record in records:
            self.tree.delete(record)
        query = 'SELECT * FROM datos ORDER BY X ASC'
        dbrows = self.mQuery(query)
        xsum = 0.
        ysum = 0.
        xysum = 0.
        xsqr = 0.
        ysqr = 0.
        yrsum = 0.
        yyrsum = 0.
        yypsum=0.
        
        for row in dbrows:
            self.tree.insert('',index='end',values=(row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7]))
            xsum = xsum + row[0]
            ysum = ysum + row[1]
            xysum = xysum + row[2]
            xsqr = xsqr + row[3]
            ysqr = ysqr + row[4]
            if row[5] and row[6] and row[7]:
                yrsum = yrsum + row[5]
                yyrsum = yyrsum + row[6]
                yypsum = yypsum + row[7]
            
        if np.array(self.tree.get_children()).size:
            if row[5]:
                self.tree.insert('', index='end', values=(xsum,ysum,xysum,xsqr,ysqr,yrsum,yyrsum,yypsum))
                print(np.array(self.tree.get_children()).size)
            else:
                self.tree.insert('', index='end', values=(xsum,ysum,xysum,xsqr,ysqr))
                print(np.array(self.tree.get_children()).size)

            
            
    def validation(self):   
        return len(self.xval.get())!=0 and len(self.yval.get()) !=0
    
    def addDatos(self):
        if self.validation():
            query = "INSERT INTO datos (X,Y,XY,X2,Y2) VALUES(?, ?, ?, ?, ?)"
            parameters = (self.xval.get(),self.yval.get(),float(self.xval.get()) * float(self.yval.get()),float(self.xval.get())**2,float(self.yval.get())**2)
            self.mQuery(query,parameters)
            print('Datos guardados')
            
            
            self.getDatos()
        else:
            print('Ingresa datos validos')
    
    def trainNetwork(self):
        query = 'SELECT * FROM datos ORDER BY X ASC'
        dbrows = self.mQuery(query)
        xarr = []
        yarr = []
        for row in dbrows:
            xarr.append(row[0])
            yarr.append(row[1])
        npx = np.array(xarr,dtype=float)
        npy = np.array(yarr,dtype=float)
        
        self.capa = tf.keras.layers.Dense(units=1,input_shape=[1])
        self.modelo = tf.keras.Sequential([self.capa])
        self.modelo.compile(
           optimizer=tf.keras.optimizers.Adam(0.1),
           loss='mean_squared_error'
        )
        print("Empieza entrenamiento de modelo")
        historial = self.modelo.fit(npx,npy,epochs=10000,verbose=False)
        predict = self.modelo.predict([9])
        print("La prediccion es: " + str(predict[0][0]))
        print(npx,npy)
        resultado = self.capa.get_weights()
        print("La formula ajustada es Y = " + str(resultado[1][0]) + " + " + str(resultado[0][0][0]) + "x")
        yrsum = 0.
        sum1 = 0
        for xvalue in npx:
            Yr = resultado[1][0] + resultado[0][0][0] * xvalue
            queryyr2 = "UPDATE datos SET YYr2=? WHERE X=?"
            parametersyyr2 = (float((npy[sum1]-Yr)**2),xvalue)
            
            self.mQuery(queryyr2,parametersyyr2)
            
            queryyp2 = "UPDATE datos SET YYp2=? WHERE X=?"
            parametersyyp2 = (float((npy[sum1]-(npy.sum()/npy.size))**2),xvalue)
            
            self.mQuery(queryyp2,parametersyyp2)
            
            
            
            query = "UPDATE datos SET Yr=? WHERE X=?"
            parameters = (float(Yr),xvalue)
            self.mQuery(query,parameters)
            sum1+=1
        
            
        
        T = Label(self.wind, height = 5, width = 52,text="La formula ajustada es Y = " + str(resultado[1][0]) + " + " + str(resultado[0][0][0]) + "x")
        T.pack()
        self.getDatos()
        
        
        
        
    def deleteRecords(self):
        query = 'DELETE FROM datos'
        self.mQuery(query)
        self.getDatos()
        
            
            
            
        
        
    
if __name__ == '__main__':
    window = Tk()
    applicacion = Ecuacion(window)
    window.mainloop()

