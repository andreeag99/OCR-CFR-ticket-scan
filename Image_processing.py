import json
import re
from datetime import datetime
import mysql.connector
from imutils.perspective import four_point_transform
import cv2
from fuzzywuzzy import process, fuzz
from matplotlib import pyplot as plt
import numpy as np
import easyocr



def insert_data(data_list):
    index_mapping = {
        "TicketID": 10,
        "Tren": 0,
        "Data": 9,
        "Plecare": 4,
        "Sosire": 6,
        "OraPlecare": 5,
        "OraSosire": 7,
        "Clasa": 1,
        "Loc": 3,
        "Pret": 8,
        "Vagon": 2
    }
    try:
        # Connect to the database
        db = mysql.connector.connect(
            host="localhost",
            database="ocrdb",
            user="root",
            password="ama1568"
        )

        if db.is_connected():
            cursor = db.cursor()

            # Insert data into the database
            for row in data_list:
                # Map the data based on the provided index_mapping
                data_tuple = (
                    row[index_mapping["TicketID"]],
                    row[index_mapping["Tren"]],
                    datetime.strptime(row[index_mapping["Data"]], "%Y-%m-%d").date(),  # Convert to date
                    row[index_mapping["Plecare"]],
                    row[index_mapping["Sosire"]],
                    datetime.strptime(row[index_mapping["OraPlecare"]], "%H:%M").time(),  # Convert to time
                    datetime.strptime(row[index_mapping["OraSosire"]], "%H:%M").time(),  # Convert to time
                    row[index_mapping["Clasa"]],
                    int(row[index_mapping["Loc"]]),  # Convert to integer
                    float(row[index_mapping["Pret"]]),  # Convert to float
                    row[index_mapping["Vagon"]]
                )
                cursor.execute("""
                    INSERT INTO tickets ( TicketID, Tren, Data, Plecare, Sosire, OraPlecare, OraSosire, Clasa, Loc, Pret, Vagon)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, data_tuple)

            # Commit the transaction
            db.commit()

            cursor.close()
            db.close()

            return "Data imported successfully"

    except mysql.connector.Error as err:
        return f"Error: {err}"


def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)


def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width = im_data.shape[:2]

    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.axis('off')

    ax.imshow(im_data, cmap='gray')

    plt.show()


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def thin_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)


def thick_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)


def remove_borders(img):

    #  Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of ticket color in HSV
    lower_blue = np.array([10, 0, 50])
    upper_blue = np.array([100, 100, 250])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask)

    # Find contours of BW mask
    contours, hiearchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = cntsSorted[-1]


    # find the perimeter of the first closed contour
    perim = cv2.arcLength(cnt, True)

    # setting the precision
    epsilon = 0.05 * perim

    # approximating the contour with a polygon
    approxCorners = cv2.approxPolyDP(cnt, epsilon, True)

    # check how many vertices has the approximate polygon
    approxCornersNumber = len(approxCorners)
    p1 = approxCorners[0]
    p2 = approxCorners[1]
    p3 = approxCorners[2]
    p4 = approxCorners[3]
    points = np.array([[p1[0][0], p1[0][1]], [p2[0][0], p2[0][1]], [p3[0][0], p3[0][1]], [p4[0][0], p4[0][1]]])
    borderless = four_point_transform(img, points)

    cv2.drawContours(img, [cnt], 0, (0, 255, 0), 5)
    cv2.drawContours(img, [approxCorners], 0, (0, 0, 255), 3)

    return borderless


def image_rotate(image):
    height, width = image.shape[:2]

    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 0.8)
    new_image = cv2.warpAffine(image, matrix, (width, height))

    return new_image



def extract_ticket_id(image) -> list:


    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    # Adjusted to include pink
    lower_pink = np.array([140, 50, 50])
    upper_pink = np.array([180, 255, 255])


    # mask based on purple color range
    mask = cv2.inRange(hsv_image, lower_pink, upper_pink)

    # Save the HSV image and mask for debugging
    cv2.imwrite('hsv_image.jpg', hsv_image)
    cv2.imwrite('mask.jpg', mask)

    # Apply the mask to the original BGR image
    purple_image = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite('purple.jpg', purple_image)

    reader = easyocr.Reader(['ro'])

    try:
        result = reader.readtext(purple_image)
        return result
    except ValueError:
        print('ID not detected')

def convert_to_hh_mm(time_str):

    # input string already has the format "HH:MM"
    time_str = re.sub(r'\D', '0', time_str)

    if re.match(r'^\d{2}:\d{2}$', time_str):
        hour, minute = time_str.split(':')
        print(hour, minute)
        hour = int(hour)
        minute = int(minute)

        # Ensure the values are within the valid range
        hour = min(max(hour, 0), 23)
        minute = min(max(minute, 0), 59)
        return f"{hour:02}:{minute:02}"


    if time_str:

        hour = time_str[:2]
        minute = time_str[2:]


        hour = int(hour)

        minute = int(minute)

        if hour > 23:
            hour = 23
        if minute > 59:
            minute = 59


        return f"{hour:02}:{minute:02}"


def extract_target(image) -> list:

    coord_list = [
        [176, 372], [452, 466], # train nr
        [140, 457], [258, 556], # class
        [822, 455], [986, 555], # wagon
        [285, 500], [430, 605], # seat
        [160, 600], [820, 690], # leaving
        [1090, 605], [1353, 691], # leaving hour
        [160, 680], [780, 760], # arriving
        [1090, 680], [1353, 764], # arriving hour
        [1080, 843], [1361, 931], # price
        [827, 370], [1340, 464] # date
    ]

    result_list = []



    for i in range(0, len(coord_list) - 1, 2):
        x1, y1 = coord_list[i][0], coord_list[i][1]   # Top-left corner coordinates
        x2, y2 = coord_list[i + 1][0], coord_list[i + 1][1]   # Bottom-right corner coordinates
        roi = image[y1:y2, x1:x2]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        reader = easyocr.Reader(['ro'])

        try:
            result = reader.readtext(roi)
            result_texts = []
            if result:
                for detection in result:
                    if detection[1]:
                        result_texts.append(detection[1])
                    else:
                        result_texts.append("null")
                result_list.append(' '.join(result_texts))
            else:
                result_list.append("null")


            for detection in result:
                box = detection[0]
                text = detection[1]
                plt.plot([box[i][0] for i in range(4)] + [box[0][0]], [box[i][1] for i in range(4)] + [box[0][1]],
                         linewidth=2)
                plt.text(box[0][0], box[0][1], text, color='black')

        except ValueError as e:
            print(e)

    cv2.imwrite('rectangles.jpg', image)
    plt.show()
    return result_list



# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage


# pattern finding
def findTemplate(image):
    template = cv2.imread('SNTFC_Pattern.jpg', cv2.IMREAD_GRAYSCALE)
    assert template is not None, "file could not be read, check with os.path.exists()"
    method = cv2.TM_CCOEFF
    res = cv2.matchTemplate(image, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    return top_left



def resemble(cityOCR):
    with open("cities_ro", 'r', encoding='utf-8') as file:
        cities = json.load(file)

    match = process.extractOne(cityOCR, cities, scorer=fuzz.ratio)
    if match and match[1] >= 80:
        return match[0]
    return "None"

def convert_to_price(price):


    price = re.sub(r'[a-zA-Z]', '0', price)


    if '.' in price:
        lei, bani = price.split('.')

        return f"{lei}.{bani}"

    if price:
        # Replace non-digit characters with '0'
        if ' ' in price:
            price = ' '.join(price)
        price = re.sub(r'\D', '0', price)

        if len(price) >=3:
            lei = price[:-2]
            lei = lei[-3:]
            bani = price[-2:]
        else:
            lei = price
            bani = "00"

        return f"{lei}.{bani}"


def convert_to_date(date):
    # input string already has the format "DD.MM.YYYY"

    date = re.sub(r'[a-zA-Z]', '1', date)

    if re.match(r'^\d{2}\.\d{2}\.\d{4}$', date):
        day, month, year = date.split('.')
        day = int(day)
        month = int(month)
        year = int(year)


        # Ensure the values are within the valid range
        day = min(max(day, 1), 31)
        month = min(max(month, 1), 12)

        year = min(max(year, 2010), 2024)
        return f"{day:02}.{month:02}.{year:04}"


    if date:

        # Replace non-digit characters with '0'
        date = re.sub(r'\D', '0', date)

        # input string is at least 8 characters long (DDMMYYYY)
        date = date.ljust(8, '0')

        day = int(date[:2])
        month = int(date[3:5])
        year = int(date[6:9])  # Ensure year has 4 digits


        # values are within the valid range
        day = min(max(day, 1), 31)
        month = min(max(month, 1), 12)
        year = min(max(year, 2010), 2024)


        return f"{day:02}.{month:02}.{year:04}"
