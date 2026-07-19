HANDWRITTEN TEXT RECOGNITION SYSTEM

FILES
1. train.py - Trains the CNN model and creates model.h5.
2. gui.py - Opens the Tkinter digit drawing application.
3. app.py - Your second application file, corrected to use model.h5.
4. requirements.txt - Required Python libraries.

RUNNING ON A COMPUTER
1. Install packages:
   pip install -r requirements.txt

2. Train the model:
   python train.py

3. Run the application:
   python gui.py

IMPORTANT
- model.h5 will be created only after train.py finishes.
- Keep model.h5 in the same folder as gui.py and app.py.
- TensorFlow and Tkinter projects generally cannot run directly on most Android phones.
