💼 Salary Predictor using Custom Linear Regression
This is a simple and beautiful machine learning app that predicts the salary of an individual based on their years of experience using a custom implementation of Linear Regression (no scikit-learn models used for training — just pure Python magic).

🚀 Features
🧠 Implements Linear Regression from scratch
📊 Trained on a real-world salary dataset (salary_data.csv)
🎨 Beautiful and interactive Tkinter GUI
📝 User-friendly design with emojis, smooth colors, and clear instructions
💬 Real-time predictions based on user input

🛠️ How It Works
1) The model is trained using the classic Years of Experience vs Salary dataset.
2) A custom Linear_Regression.py file performs gradient descent to learn the best-fit line.
3) The app takes user input for years of experience via a simple GUI.
4) On clicking “Predict Salary,” it shows the estimated salary instantly.
5) Everything is wrapped in a clean and modern Tkinter interface.

🧪 Tech Stack
Python 🐍
NumPy
Pandas
Tkinter (for GUI)
Custom-built Linear Regression (No pre-built ML libraries for model)

📦 File Structure
.
├── salary_data.csv             # Dataset (YearsExperience, Salary)
├── Linear_Regression.py        # Custom Linear Regression model
├── salary_predictor_gui.py     # Beautiful GUI app (main file)
