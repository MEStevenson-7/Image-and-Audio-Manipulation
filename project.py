"""
DSC 20 Project
Name(s): Arah Sanders, Morgan Stevenson
PID(s):  A18583710, A16798052
Sources: Lecture Notes
"""

import numpy as np
import os
from PIL import Image
from pydub import AudioSegment

NUM_CHANNELS = 3


# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Saves the given RGBImage instance to the given path
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    Represents an image in RGB format
    """

    def __init__(self, pixels):
        """
        Initializes a new RGBImage object

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        >>> pixels = [[(20, 20, 20), (30, 30, 30)], [(40, 50, 70), (5, 2, 3)]]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError
        >>> pixels = [[[-1, 2, 10], [4, 2, 5]]]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        ValueError
        >>> pixels = [[[20, 30, 40, 50], [60, 21, 31]]]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError
        """
        if type(pixels) != list:
            raise TypeError()
        elif len(pixels) == 0:
            raise TypeError()
        elif all([isinstance(row, list) for row in pixels]) is False:
            raise TypeError()
        elif all([len(row) > 0 for row in pixels]) is False:
            raise TypeError()
        elif len(set([len(row) for row in pixels])) != 1:
                raise TypeError()
        elif all([isinstance(column, list) for row in pixels \
                for column in row]) is False:
            raise TypeError()
        elif all([len(column) == 3 for row in pixels for column in row]) \
            is False:
            raise TypeError()
        elif all([isinstance(intensity, int) for row in pixels \
                for column in row for intensity in column]) is False:
            raise TypeError()
        elif all([True if intensity >= 0 and intensity <= 255 else False \
            for row in pixels for column in row for intensity in column]) is False:
            raise ValueError()

        self.pixels = pixels
        self.num_rows = len(self.pixels)
        self.num_cols = len(self.pixels[0])
        

    def size(self):
        """
        Returns the size of the image in (rows, cols) format

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        Returns a copy of the image pixel array

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        return [[list(column) for column in row] \
                for row in self.pixels]

    def copy(self):
        """
        Returns a copy of this RGBImage object

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        img_copy = RGBImage(self.get_pixels())
        return img_copy
        

    def get_pixel(self, row, col):
        """
        Returns the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)

        #added doc tests
        #testing if row is not an int
        >>> pixels_test1 = [
        ...              [['', 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels_test1)
        Traceback (most recent call last):
        ...
        TypeError
        """
        if isinstance(row, int) is False or isinstance(col, int) is False:
            raise TypeError()
        elif row >= len(self.pixels) or row < 0:
            raise ValueError()
        elif col < 0 or col >= len(self.pixels[0]):
            raise ValueError()

        return tuple(self.pixels[row][col])

    def set_pixel(self, row, col, new_color):
        """
        Sets the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        if isinstance(row, int) is False or isinstance(col, int) is False:
            raise TypeError()
        elif row >= len(self.pixels) or row < 0:
            raise ValueError()
        elif col < 0 or col >= len(self.pixels[0]):
            raise ValueError()
        elif isinstance(new_color, tuple) is False or len(new_color) != 3 \
            or all([isinstance(intensity, int) for intensity in new_color]) \
            is False:
            raise TypeError()
        elif all([intensity <= 255 for intensity in new_color]) is False:
            raise ValueError()

        current_color = self.pixels[row][col]
        updated_color = [current_color[i] if new_color[i] < 0 \
                        else new_color[i] \
                        for i in range(len(new_color))]
        self.pixels[row][col] = updated_color


# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    Contains assorted image processing methods
    Intended to be used as a parent class
    """

    def __init__(self):
        """
        Creates a new ImageProcessingTemplate object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0

    def get_cost(self):
        """
        Returns the current total incurred cost

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        return self.cost

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img)
        >>> id(img) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img = img_read_helper('img/test_image_32x32.png')                 # 2
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')  # 3
        >>> img_negate = img_proc.negate(img)                               # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/test_image_32x32_negate.png', img_negate)# 6
        """
        if isinstance(image, RGBImage) is False:
            raise TypeError()

        return RGBImage([[[255 - intensity for intensity in column] for column in row] \
                for row in image.copy().get_pixels()])

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_gray.png')
        >>> img_gray = img_proc.grayscale(img)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/test_image_32x32_gray.png', img_gray)
        """
        if isinstance(image, RGBImage) is False:
            raise TypeError()

        return RGBImage([[[sum(column) // 3, sum(column) // 3, sum(column) // 3] \
                for column in row] for row in image.copy().get_pixels()])

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/test_image_32x32_rotate.png', img_rotate)
        """
        if isinstance(image, RGBImage) is False:
            raise TypeError()

        return RGBImage([[list(column) for column in row[::-1]] \
                for row in image.copy().get_pixels()[::-1]])

    def get_average_brightness(self, image):
        """
        Returns the average brightness for the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.get_average_brightness(img)
        86
        """
        if isinstance(image, RGBImage) is False:
            raise TypeError()

        avg_each_pix = [sum(col) // 3 for row in image.copy().get_pixels() \
                        for col in row]
        return sum(avg_each_pix) // len(avg_each_pix)

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_adjusted.png')
        >>> img_adjust = img_proc.adjust_brightness(img, 1.2)
        >>> img_adjust.pixels == img_exp.pixels # Check adjust_brightness
        True
        >>> img_save_helper('img/out/test_image_32x32_adjusted.png', img_adjust)
        """
        if isinstance(image, RGBImage) is False:
            raise TypeError()
        elif isinstance(intensity, float) is False:
            raise TypeError()
        elif intensity < 0:
            raise ValueError()

        new_img = image.copy()
        new_img_pixels = [[[255 if i * intensity > 255 else int(i * intensity) \
                for i in column] \
                for column in row] \
                for row in new_img.get_pixels()]
        return RGBImage(new_img_pixels)




# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    Represents a standard tier of an image processor
    """

    def __init__(self):
        """
        Creates a new StandardImageProcessing object

        # Check that the cost was assigned
        >>> img_proc = StandardImageProcessing()
        >>> img_proc.cost
        0
        """
        self.cost = 0
        self.coupons = 0

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')
        >>> img_negate = img_proc.negate(img)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        if self.coupons > 0:
            self.coupons -= 1
        else:
            self.cost += 5

        return super().negate(image)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        """
        if self.coupons > 0:
            self.coupons -= 1
        else:
            self.cost += 6

        return super().grayscale(image)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image
        """
        if self.coupons > 0:
            self.coupons -= 1
        else:
            self.cost += 10

        return super().rotate_180(image)

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level
        """
        if self.coupons > 0:
            self.coupons -= 1
        else:
            self.cost += 1

        return super().adjust_brightness(image, intensity)

    def redeem_coupon(self, amount):
        """
        Makes the given number of methods calls free

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        if isinstance(amount, int) is False:
            raise TypeError()
        elif amount <= 0:
            raise ValueError()

        self.coupons += amount


# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Represents a paid tier of an image processor
    """

    def __init__(self):
        """
        Creates a new PremiumImageProcessing object

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        self.cost = 50

    def pixelate(self, image, block_dim):
        """
        Returns a pixelated version of the image, where block_dim is the size of 
        the square blocks.

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_pixelate = img_proc.pixelate(img, 4)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_pixelate.png')
        >>> img_exp.pixels == img_pixelate.pixels # Check pixelate output
        True
        >>> img_save_helper('img/out/test_image_32x32_pixelate.png', img_pixelate)
        """
        if not isinstance(image, RGBImage):
            raise TypeError()
        if not isinstance(block_dim, int) or block_dim <= 0:
            raise ValueError("block_dim must be a positive integer")

        pixels = image.get_pixels()
        num_rows, num_cols = image.size()

        pixelated_pixels = [[None] * num_cols for _ in range(num_rows)]

        for row in range(0, num_rows, block_dim):
            for col in range(0, num_cols, block_dim):
                r_total, g_total, b_total, count = 0, 0, 0, 0
                for i in range(row, min(row + block_dim, num_rows)):
                    for j in range(col, min(col + block_dim, num_cols)):
                        r, g, b = pixels[i][j]
                        r_total += r
                        g_total += g
                        b_total += b
                        count += 1
                r_avg = r_total // count
                g_avg = g_total // count
                b_avg = b_total // count

                for i in range(row, min(row + block_dim, num_rows)):
                    for j in range(col, min(col + block_dim, num_cols)):
                        pixelated_pixels[i][j] = [r_avg, g_avg, b_avg]

        return RGBImage(pixelated_pixels)

    def edge_highlight(self, image):
        """
        Returns a new image with the edges highlighted

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_edge = img_proc.edge_highlight(img)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_edge.png')
        >>> img_exp.pixels == img_edge.pixels # Check edge_highlight output
        True
        >>> img_save_helper('img/out/test_image_32x32_edge.png', img_edge)
        """
        # YOUR CODE GOES HERE #


# --------------------------------------------------------------------------- #

def audio_read_helper(path):
    """
    Creates an AudioWave object from the given audio file
    """
    # Load the audio file
    audio = AudioSegment.from_file(file = path,  
                                  format = path.split('.')[-1])    
    # Convert to mono if it's stereo
    audio = audio.set_channels(1)
    # Get the raw audio data
    raw_data = np.array(audio.get_array_of_samples()).tolist()
    return AudioWave(raw_data)


def audio_save_helper(path, audio, sample_rate = 44100):
    """
    Saves the given AudioWave instance to the given path
    """
    # Convert list to numpy array
    audio_array = np.array(audio.wave).astype(np.int16)
    # Convert the NumPy array back to an AudioSegment
    audio_segment = AudioSegment(
        audio_array.tobytes(), 
        frame_rate=sample_rate,
        sample_width=audio_array.dtype.itemsize,
        channels=1
    )
    
    # Export the audio segment to a wav file
    audio_segment.export(path, format="wav")


# --------------------------------------------------------------------------- #
# Part 5: Multimedia Processing
class AudioWave():
    """
        Represents audio through a 1-dimensional array of amplitudes
    """
    def __init__(self, amplitudes):
        self.wave = amplitudes

## TODO: PremiumPlusMultimediaProcessing Class Implementation



# Part 6: Image KNN Classifier #
class ImageKNNClassifier:
    """
    Represents a simple KNNClassifier
    """

    def __init__(self, k_neighbors):
        """
        Creates a new KNN classifier object
        """
        # YOUR CODE GOES HERE #

    def fit(self, data):
        """
        Stores the given set of data and labels for later
        """
        # YOUR CODE GOES HERE #

    def distance(self, image1, image2):
        """
        Returns the distance between the given images

        >>> img1 = img_read_helper('img/steve.png')
        >>> img2 = img_read_helper('img/knn_test_img.png')
        >>> knn = ImageKNNClassifier(3)
        >>> knn.distance(img1, img2)
        15946.312896716909
        """
        # YOUR CODE GOES HERE #

    def vote(self, candidates):
        """
        Returns the most frequent label in the given list

        >>> knn = ImageKNNClassifier(3)
        >>> knn.vote(['label1', 'label2', 'label2', 'label2', 'label1'])
        'label2'
        """
        # YOUR CODE GOES HERE #

    def predict(self, image):
        """
        Predicts the label of the given image using the labels of
        the K closest neighbors to this image

        The test for this method is located in the knn_tests method below
        """
        # YOUR CODE GOES HERE #


def knn_tests(test_img_path):
    """
    Function to run knn tests

    >>> knn_tests('img/knn_test_img.png')
    'nighttime'
    """
    # Read all of the sub-folder names in the knn_data folder
    # These will be treated as labels
    path = 'knn_data'
    data = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        # Ignore non-folder items
        if not os.path.isdir(label_path):
            continue
        # Read in each image in the sub-folder
        for img_file in os.listdir(label_path):
            train_img_path = os.path.join(label_path, img_file)
            img = img_read_helper(train_img_path)
            # Add the image object and the label to the dataset
            data.append((img, label))

    # Create a KNN-classifier using the dataset
    knn = ImageKNNClassifier(5)

    # Train the classifier by providing the dataset
    knn.fit(data)

    # Create an RGBImage object of the tested image
    test_img = img_read_helper(test_img_path)

    # Return the KNN's prediction
    predicted_label = knn.predict(test_img)
    return predicted_label
