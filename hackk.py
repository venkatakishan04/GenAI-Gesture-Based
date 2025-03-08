import cv2
import mediapipe as mp
import requests
import threading
import numpy as np
import time
from math import radians, sin, cos, sqrt, atan2

# 🔑 Replace with your Google Maps API key and Gemini API key
GOOGLE_MAPS_API_KEY = "AlzaSy9K80VZUUa_5jVNVv14LLLGv4RCZa-mJqk"
GEMINI_API_KEY = "AIzaSyA2h44PnhvnR9qrgxL7le1Kn6Yc4BxLDgI"  # Replace with your Gemini API key

# 📍 Airport location
AIRPORT_LAT, AIRPORT_LON = 17.2403, 78.4294  

# Cache API responses
cached_places = {"restaurant": [], "police": [], "airport": [], "shopping_mall": []}
lock = threading.Lock()  # Prevents multiple API calls at the same time

# Track hover state for location selection
hover_start_time = None
selected_location = None
pointer_position = (300, 200)  # Initial pointer position on the map

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth using the Haversine formula."""
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def get_google_places(place_type):
    """Fetch places from Google Maps API and cache results."""
    with lock:
        if cached_places[place_type]:  # Use cache if available
            return cached_places[place_type]

        url = "https://maps.gomaps.pro/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{AIRPORT_LAT},{AIRPORT_LON}",
            "radius": 5000,  # Search within 5 km
            "type": place_type,
            "key": GOOGLE_MAPS_API_KEY
        }

        try:
            response = requests.get(url, params=params, timeout=5)
            data = response.json()

            places = [
                {
                    "name": place["name"],
                    "place_id": place.get("place_id"),  # Store place_id for detailed info
                    "lat": place["geometry"]["location"]["lat"],
                    "lng": place["geometry"]["location"]["lng"],
                    "rating": place.get("rating", "N/A"),
                    "user_ratings_total": place.get("user_ratings_total", "N/A"),
                    "vicinity": place.get("vicinity", "N/A"),
                    "distance": haversine(AIRPORT_LAT, AIRPORT_LON, place["geometry"]["location"]["lat"], place["geometry"]["location"]["lng"])
                }
                for place in data.get("results", []) if "name" in place
            ]

            cached_places[place_type] = places if places else []
            return cached_places[place_type]
        except Exception as e:
            print(f"API Error: {e}")
            return []

def generate_map_image(places):
    """Generate a map image with markers."""
    if not places:
        return None

    markers = [f"label:{place['name'][0]}|{place['lat']},{place['lng']}" for place in places]
    airport_marker = f"label:A|{AIRPORT_LAT},{AIRPORT_LON}"
    markers_str = f"&markers={airport_marker}&markers=" + "&markers=".join(markers)

    map_url = f"https://maps.gomaps.pro/maps/api/staticmap?center={AIRPORT_LAT},{AIRPORT_LON}&zoom=14&size=600x400&maptype=roadmap{markers_str}&key={GOOGLE_MAPS_API_KEY}"
    
    response = requests.get(map_url)
    if response.status_code == 200:
        image = np.frombuffer(response.content, dtype=np.uint8)
        return cv2.imdecode(image, cv2.IMREAD_COLOR)
    return None

def get_place_details(place_id):
    """Fetch detailed information about a place using its place_id."""
    url = "https://maps.gomaps.pro/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "name,rating,user_ratings_total,formatted_address,reviews",
        "key": GOOGLE_MAPS_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        return data.get("result", {})
    except Exception as e:
        print(f"API Error: {e}")
        return {}

def get_gemini_info(lat, lng):
    """Fetch additional information about a location using the Gemini API."""
    # Simulated Gemini API call (replace with actual API endpoint)
    print(f"Fetching Gemini info for location: {lat}, {lng}")  # Debugging
    time.sleep(2)  # Simulate API delay
    return f"Additional information from Gemini about the location at ({lat}, {lng})."

def smooth_move_pointer(target_x, target_y):
    """Smoothly move the pointer to the target position."""
    global pointer_position
    current_x, current_y = pointer_position

    # Linear interpolation for smooth movement
    steps = 50
    dx = (target_x - current_x) / steps
    dy = (target_y - current_y) / steps

    for _ in range(steps):
        current_x += dx
        current_y += dy
        pointer_position = (int(current_x), int(current_y))
        time.sleep(0.01)  # Adjust for smoothness

def display_places_list(places):
    """Display the list of places with names and distances in a blank window."""
    # Create a blank white image
    blank_image = np.ones((400, 600, 3), dtype=np.uint8) * 255  # White background

    # Add text to the image
    y_offset = 30
    for place in places:
        text = f"{place['name']} - {place['distance']:.2f} km"
        cv2.putText(blank_image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_offset += 30

    # Show the blank window
    cv2.imshow("Places List", blank_image)

# 🖐 Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open Webcam
cap = cv2.VideoCapture(0)

def is_finger_extended(landmarks, tip, pip):
    """Check if a finger is extended."""
    return landmarks[tip].y < landmarks[pip].y

last_gesture = None  # Track last detected gesture

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    fingers_extended = 0  
    places = []

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = hand_landmarks.landmark

        # Count extended fingers
        fingers_extended = sum([
            is_finger_extended(landmarks, 8, 6),  # Index finger
            is_finger_extended(landmarks, 12, 10),  # Middle finger
            is_finger_extended(landmarks, 16, 14),  # Ring finger
            is_finger_extended(landmarks, 20, 18)  # Pinky
        ])

        # Gesture Mapping
        place_type = None
        if fingers_extended == 1:
            place_type = "restaurant"  
        elif fingers_extended == 2:
            place_type = "police"  
        elif fingers_extended == 3:
            place_type = "airport"  
        elif fingers_extended == 4:
            place_type = "shopping_mall"  

        # Fetch places if gesture changes
        if place_type and place_type != last_gesture:
            last_gesture = place_type
            thread = threading.Thread(target=lambda: cached_places.update({place_type: get_google_places(place_type)}))
            thread.start()

        places = cached_places.get(place_type, [])

    # Display fingers count
    cv2.putText(frame, f"Fingers: {fingers_extended}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display Map in another window
    if places:
        map_image = generate_map_image(places)
        if map_image is not None:
            # Draw pointer on the map
            cv2.circle(map_image, pointer_position, 3, (0, 0, 255), -1)
            cv2.imshow("Places Map", map_image)

            # Check if pointer is hovering over a location
            for place in places:
                # Convert lat/lng to map coordinates (simplified for demonstration)
                map_x = int((place["lng"] - AIRPORT_LON) * 1000 + 300)
                map_y = int((AIRPORT_LAT - place["lat"]) * 1000 + 200)

                if abs(pointer_position[0] - map_x) < 20 and abs(pointer_position[1] - map_y) < 20:
                    if hover_start_time is None:
                        hover_start_time = time.time()
                    elif time.time() - hover_start_time > 10:  # Hover for 10 seconds
                        selected_location = place
                        hover_start_time = None
                        # Smoothly move the pointer to the location
                        threading.Thread(target=smooth_move_pointer, args=(map_x, map_y)).start()
                        # Fetch and display detailed location info
                        details = get_place_details(place["place_id"])
                        gemini_info = get_gemini_info(place["lat"], place["lng"])
                        # Display detailed info in a new window
                        info_window = np.zeros((400, 600, 3), dtype=np.uint8)
                        cv2.putText(info_window, f"Name: {details.get('name', 'N/A')}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(info_window, f"Rating: {details.get('rating', 'N/A')}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(info_window, f"Total Ratings: {details.get('user_ratings_total', 'N/A')}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(info_window, f"Address: {details.get('formatted_address', 'N/A')}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(info_window, f"Reviews: {len(details.get('reviews', []))}", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(info_window, f"Gemini Info: {gemini_info}", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.imshow("Location Info", info_window)
                        # Display distance in a fourth window
                        distance_window = np.zeros((200, 400, 3), dtype=np.uint8)
                        cv2.putText(distance_window, f"Distance: {place['distance']:.2f} km", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.imshow("Distance Info", distance_window)
                    break
            else:
                hover_start_time = None

        # Display the list of places in the third window
        display_places_list(places)

    # Show Video Feed
    cv2.imshow("Gesture Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()