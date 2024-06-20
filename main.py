import cv2
import easyocr
import Image_processing
import re
import ImportButton
FIXED_WIDTH = 1500
REF_COORD = (35, 58)

# initialize the reader with Romanian language


reader = easyocr.Reader(['ro'], gpu=False)

# load an image
im_read = cv2.imread("test1.jpg")
no_borders = Image_processing.remove_borders(im_read)



# Image processing
# binarization, grayscale and straighten
gray_image = Image_processing.grayscale(no_borders)
thresh, img_bin = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

# rescale image
w, h = img_bin.shape[::-1]
scale = FIXED_WIDTH / w
new_h = int(h * scale)
new_w = int(w * scale)
image = cv2.resize(img_bin, (new_w, new_h),
                   interpolation=cv2.INTER_NEAREST)
cv2.imwrite('testResize.jpg', image)

# thicken the font
thick_image = Image_processing.thick_font(image)

# noise removal
no_noise = Image_processing.noise_removal(thick_image)



# save final image
cv2.imwrite('final.jpg', no_noise)


# find locating pattern
#locator = Image_processing.findTemplate(no_noise)
#offset = np.subtract(locator, REF_COORD)


# extract id from ticket
rotate_image = Image_processing.image_rotate(im_read)
id = Image_processing.extract_ticket_id(rotate_image)


# create list with targeted data
database_data = Image_processing.extract_target(no_noise)
for detection in id:
    database_data.append(detection[1])


#verify input in the final list
database_data[4] = Image_processing.resemble(database_data[4])
database_data[6] = Image_processing.resemble(database_data[6])
database_data[1] = ''.join(re.findall(r'\d+', database_data[1]))
database_data[2] = ''.join(re.findall(r'\d+', database_data[2]))
database_data[3] = ''.join(re.findall(r'\d+', database_data[3]))
database_data[5] = Image_processing.convert_to_hh_mm(database_data[5])
database_data[7] = Image_processing.convert_to_hh_mm(database_data[7])
database_data[8] = Image_processing.convert_to_price(database_data[8])
database_data[9] = Image_processing.convert_to_date(database_data[9])



# [0]train, [1]class, [2]wagon, [3]place, [4]leave, [5]leave time, [6]arrival, [7]arrival time, [8]price, [9]date, [10]id
print(database_data)

Image_processing.insert_data(database_data)

