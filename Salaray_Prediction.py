import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import Linear_Regression as LR
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

salary_data = pd.read_csv('salary_data.csv')

print(salary_data.head())

print(salary_data.shape)


print(salary_data.isnull().sum())

X = salary_data.iloc[:,:-1].values
Y = salary_data.iloc[:,1].values

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.33,random_state=2)

model = LR.Linear_Regression(learning_rate=0.02, no_of_iterations=1000)
model.fit(X_train, Y_train)

test_data_preditction = model.predict(X_test)

def predict_salary():
    try:
        years_exp = float(entry.get())
        if(years_exp >= 0):
            prediction = model.predict(np.array([[years_exp]]))
            result_label.config(text=f"Estimated Salary: ₹{prediction[0]:,.2f}")
        elif(years_exp < 0):
            result_label.config(text=f"Invalid years of experience!!")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid number for years of experience.")


# Set up the main window
root = tk.Tk()
root.title("Salary Prediction App")
root.geometry("450x400")
root.configure(bg="#eaf4f4")
root.resizable(False, False)

# Styles
style = ttk.Style()
style.theme_use("clam")
style.configure("TButton",
                foreground="white",
                background="#009688",
                font=("Segoe UI", 11),
                padding=10)
style.map("TButton",
          background=[("active", "#00796B")])

# Title
title = tk.Label(root, text="👨‍💼 Salary Predictor", font=("Segoe UI", 22, "bold"), fg="#004D40", bg="#eaf4f4")
title.pack(pady=25)

# Entry Label
entry_label = tk.Label(root, text="Enter Years of Experience:", font=("Segoe UI", 13), bg="#eaf4f4", fg="#00695C")
entry_label.pack()

# Entry Box
entry = tk.Entry(root, font=("Segoe UI", 14), justify="center", relief="solid", bd=1, width=20)
entry.pack(pady=10)

# Predict Button
predict_btn = ttk.Button(root, text="🎯 Predict Salary", command=predict_salary)
predict_btn.pack(pady=15)

# Result Label
result_label = tk.Label(root, text="", font=("Segoe UI", 16, "bold"), fg="#1B5E20", bg="#eaf4f4")
result_label.pack(pady=30)

# Footer
footer = tk.Label(root, text="Made by Neev", font=("Segoe UI", 10), fg="#757575", bg="#eaf4f4")
footer.pack(side="bottom", pady=10)

# Start GUI loop
root.mainloop()