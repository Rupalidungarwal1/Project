import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta
import geocoder
import csv
import sys
import json # New Import for persistence

# --- Configuration ---
DETAILS_FILE = 'criminal_details.json' # File to store persistent details
path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
print(f"Detected files in Training_images: {myList}")

# Load training images and class names
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    # The criminal name is derived from the filename (e.g., 'john_doe.jpg' -> 'john_doe')
    classNames.append(os.path.splitext(cl)[0])
print(f"Known Criminals (from images): {classNames}")

# --- Persistence Functions ---

def load_criminal_details():
    """Loads criminal details from the JSON file."""
    if os.path.exists(DETAILS_FILE):
        try:
            with open(DETAILS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode {DETAILS_FILE}. Starting with empty details.")
            return {}
    return {}

def save_criminal_details(details):
    """Saves criminal details to the JSON file."""
    try:
        with open(DETAILS_FILE, 'w') as f:
            json.dump(details, f, indent=4)
        print(f"\n--- Criminal details saved to {DETAILS_FILE} ---")
    except Exception as e:
        print(f"Error saving criminal details: {e}")

# --- Dynamic Input/Loading for Criminal Details ---
criminal_details = load_criminal_details()
new_details_added = False # Flag to track if we need to save

print("\n--- Criminal Details Check ---")
# Get the set of names from images and the set of names in the loaded details
image_names_upper = {name.upper() for name in classNames}
loaded_names = set(criminal_details.keys())

# Find criminals from images that are NOT in the loaded details
names_to_prompt = image_names_upper - loaded_names

if names_to_prompt:
    print(f"New criminals detected! Input required for: {list(names_to_prompt)}")
    
    # Iterate through only the new names and prompt the user for details
    for name in names_to_prompt:
        print(f"\n[ Details for: {name} ]")
        
        # 1. Get Crime
        crime = input("  Enter Crime Committed: ")
        
        # 2. Get Cases Registered (Ensure it's a number)
        while True:
            try:
                cases = int(input("  Enter Cases Registered: "))
                break
            except ValueError:
                print("  Invalid input. Please enter a whole number for Cases Registered.")
        
        # 3. Get Danger Rating (Ensure it's a number and within a reasonable range)
        while True:
            try:
                rating = int(input("  Enter Danger Rating (1-10): "))
                if 1 <= rating <= 10:
                    break
                else:
                    print("  Rating must be between 1 and 10.")
            except ValueError:
                print("  Invalid input. Please enter a whole number for the Danger Rating.")

        # Store the gathered details in the dictionary
        criminal_details[name] = {
            "Crime": crime,
            "Cases Registered": cases,
            "Danger Rating": rating
        }
        new_details_added = True # Set flag to save later
else:
    print("All known criminals have registered details. Skipping input.")

# Save the updated details if new entries were added
if new_details_added:
    save_criminal_details(criminal_details)

print(f"\n--- Criminal details loaded successfully ---")
print(f"Loaded details for: {list(criminal_details.keys())}")


# --- Functions (Unchanged) ---

def findEncodings(images):
    encodeList = []
    for img in images:
        if img is None:
            print("Image not loaded properly.")
            continue

        print("Before conversion:", img.dtype, img.shape)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print("After conversion:", img.dtype, img.shape)

        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 0:
            encodeList.append(encodings[0])
        else:
            print("No face found in image.")

    return encodeList

def get_location():
    # ... (function body remains the same)
    try:
        g = geocoder.ip('me')
        if g.ok:
            return f"{g.city}, {g.country}"
        else:
            return "Unknown Location"
    except Exception:
        return "Location Service Failed"

def markcriminal_record(name, place, last_detected, details):
    # ... (function body remains the same, but it uses the correct 'details' dictionary structure)
    current_time = datetime.now()
    dtString = current_time.strftime('%I:%M:%S %p')
    dateString = current_time.strftime('%Y-%m-%d')
    
    crime = details.get("Crime", "N/A")
    cases = details.get("Cases Registered", "N/A")
    rating = details.get("Danger Rating", "N/A")
    
    new_row = [name, dtString, dateString, place, crime, cases, rating]

    try:
        file_exists = os.path.exists('criminal_record.csv')
        
        with open('criminal_record.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            
            if not file_exists or os.stat('criminal_record.csv').st_size == 0:
                writer.writerow(['Name', 'Time', 'Date', 'Place', 'Crime', 'Cases Registered', 'Danger Rating'])
            
            writer.writerow(new_row)
            
        last_detected[name] = current_time
    
    except PermissionError:
        print("\n!!! ERROR: [Errno 13] Permission denied: 'criminal_record.csv'")
        print("!!! ACTION: Please close the CSV file if it is open in Excel or another program.")
    except Exception as e:
        print(f"Error writing to CSV: {e}")
        
# --- Main Execution (Unchanged from original logic flow) ---

encodeListKnown = findEncodings(images)
print('Encoding Complete')

if len(encodeListKnown) != len(classNames):
    print("FATAL ERROR: Encoding failed for one or more images. Check if faces are clear.")
    sys.exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()

place = get_location() 
print(f"System location: {place}\nStarting Webcam...")

last_detected = {}

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame. Exiting...")
        break
        
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        if len(faceDis) > 0:
            matchIndex = np.argmin(faceDis)
        else:
            continue

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            
            # --- GET AND DISPLAY CRIMINAL DETAILS ---
            details = criminal_details.get(name, {})
            
            rating_text = f"Rating: {details.get('Danger Rating', 'N/A')}/10"
            crime_text = f"CRIME: {details.get('Crime', 'UNKNOWN')}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 40), (x2, y2), (0, 0, 255), cv2.FILLED)
            
            cv2.putText(img, f"{name} | {rating_text}", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(img, crime_text, (x1 + 6, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

            # --- LOGGING WITH COOLDOWN ---
            if name not in last_detected or datetime.now() - last_detected[name] >= timedelta(minutes=5):
                markcriminal_record(name, place, last_detected, details)

        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 165, 0), cv2.FILLED)
            cv2.putText(img, "UNKNOWN", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


    cv2.imshow('Criminal Detection System', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()