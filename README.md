# Doodle-Authentication
This application is a doodle-based authentication system where users can register and log in by p
roviding a username, password, and a doodle (either drawn or uploaded). The application uses a trained MobileNet model to classify doodles for user authentication.

--------
#### Author: Klaudia Jędryszczak

--------
## Features 
- **Registration**: Users can register by providing a username, password, and a doodle. 
The doodle can be drawn on the canvas or uploaded from the device.
- **Login**: Users can log in by providing their username, password, and doodle. 
The doodle can be drawn on the canvas or uploaded from the device.
- **Doodle Detection**: The application uses a trained MobileNet model to classify doodles.
---------

## Pre-requisites
Ensure you have the following installed:
- Python 3.6 or higher
- Virtual environment tool (e.g. venv)
---------

## Setting Up the Project 
1. Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate # For Linux/MacOS 
venv\Scripts\activate # For Windows
```
2. Install the Required Libraries
```bash
pip install -r requirements.txt
```
----------
## Directory Structure
```plaintext
├── doodles/                # Directory for storing user doodles (empty by default)
├── labels/
│   ├── labels.txt          # Classes for doodle detection
├── model/
│   ├── mobilenet_doodle_model.pth  # Trained MobileNet model
│   ├── model.py                    # Training script for the model
├── static/
│   ├── css/                # Stylesheets
│   ├── images/             # Images used in the app
├── templates/
│   ├── index.html          # Homepage template
│   ├── login.html          # Login page template
│   ├── register.html       # Registration page template
├── app.py                  # Main application logic
├── detect.py               # Doodle class detection logic
├── requirements.txt        # List of required Python packages
├── README.md               # Documentation
```
----------
## Running the Application
1. Start the Server
```bash
python app.py
```

2. Access the Application
Open a web browser and go to `http://127.0.0.1:5000/`

--------
## Using the App 
1. **Registration** 
- Go to the registration page by clicking the "Register" button on the homepage or by going to `http://127.0.0.1:5000/register`
- Enter a username, password, and confirm your password.
- Either upload a doodle image or draw a doodle on the canvas.
- You can click on "Wyczyść" button to clear the canvas and draw a new doodle.
- You can click on "Zaakceptuj obrazek" button to see the detected class of the doodle. 
- Submit the form to register.

2. **Login**
- Go to the login page by clicking the "Login" button on the homepage or by going to `http://127.0.0.1:5000/login`
- Enter your username, password, and draw a doodle on the canvas.
- You can click on "Wyczyść" button to clear the canvas and draw a new doodle.
- You can click on "Zaakceptuj obrazek" button to see the detected class of the doodle.
- Submit the form to log in.
- If the doodle class is the correct class you will be logged in, taken to the homepage, and greeted with your username.

--------
## Homepage
The homepage displays a welcome message describing what doodle authentication is 
and two buttons to navigate to the registration and login pages.

![Homepage](/static/images/homepage.png)

--------
## Registration Page
The registration page allows users to register by providing a username, password, and doodle.
![Registration Page](/static/images/register.png)

After drawing or providing a doodle image, users can click the "Zaakceptuj obrazek" button to see the detected class of the doodle.
![Doodle Detection](/static/images/detect.png)

## Login Page
After registering, users can log in by providing their username, password, and doodle. The doodle 
doesn't have to be the same doodle used during registration, but it must be the same class (e.g., "flower").
![Login Page](/static/images/login.png)

Successful login will take the user to the homepage, where they will be greeted with their username.
![Homepage Greeting](/static/images/greet.png)

--------
## Training the MobileNet Model 
The MobileNet model used for doodle classification was trained using the 
[Doodle Dataset](https://www.kaggle.com/datasets/ashishjangra27/doodle-dataset) from Kaggle.
The dataset was limited to 71 classes, and each class contained 3000 samples. 

1. Dataset and DataLoader
   - A `DoodleDataset` class was created to load the doodle images and labels from the dataset directory 
   - Images were transformed using `torchvision.transforms` to resize and normalize the images
   - The dataset was split into traing (70%), validation (15%), and test (15%) sets using `torch.utils.data.random_split`
   - DataLoaders were created for each set using `torch.utils.data.DataLoader`
   - The DataLoader used batch size of 128 and shuffled the training data

2. Model Preparation 
   - MobileNetV2 (a lightweight convolutional neural network) was loaded with pretrained weights from `torchvision.models`
   - The classifier layer was replaced with a new linear layer with 71 output features

3. Training Loop
   - The model was trained for 10 epochs using the Adam optimizer and CrossEntropyLoss function. 
   - The training loop included: 
     - Zeroing the gradients
     - Forward propagation
     - Backward propagation
     - Updating weights
     - Using the optimizer 
     - Calculating the loss and accuracy

4. Saving the Model
   - The trained model was saved to a file using `torch.save` 
   - The model was saved as `mobilenet_doodle_model.pth` 

5. Model Evaluation
   - The model was evaluated on the test set to calculate the accuracy of the model
   - Predictions and ground truth labels were collected to compute the confusion matrix and classification report

Final accuracy of the model was 91.2% on the test set. 
You can find the training script in `model/model.py`. 