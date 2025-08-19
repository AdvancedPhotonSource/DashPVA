import pvaccess as pva
from epics import camonitor, caget
import numpy as np
import matplotlib.pyplot as plt

CROP_PADDING = 0
ROI = 'ROI2'
PVA = 'Pva1'
TEST_ROW = 69

class ROICropping:
    """
    Crop an image based on the ROI
    """
    def __init__(self):
        self.channel : pva.Channel = None
        self.pva_obj : pva.PVObject = None

        # size
        self.image : np.ndarray = None
        self.shape : tuple = (0,0)
        self.shaped_img : np.ndarray = None
        
        # cropping
        self.cropped_image : np.ndarray = None

        # minx,maxx,miny,maxy
        self.crop_size : tuple = (0,0,0,0)
        self.cropped_col_avg : float = 0.0
        self.cropped_row_avg : float = 0.0
        

    def crop_img(self) -> None:
        """
        This function crops the Pva1 to the size of the ROI
        and displays it on matplotlib

        """
        
        self.get_image()
        self.get_roi()
        self.shape_image()
        self.crop_shaped_image(ROI_NUM=3)
        self.calc_average()

        # DEBUG
        # print(f'\
        #       {PVA} Image: {self.image}\
        #       {ROI}: {self.roi_data}\
        #       Image Size: {self.shape}\
        #       Crop Size: {self.crop_size}')
        # print(f'\
        #       Before Crop: {self.image}\
        #       After Crop: {self.cropped_image}\n\n\
        #       Average Column: {self.cropped_col_avg}\n\n\
        #       Average Row: {self.cropped_row_avg}\
        #       ')
        print(type(self.shaped_img))
        self.display_image()
        
        

    def get_image(self) -> None:
        # Gets a single image
        self.channel = pva.Channel(f'dp-ADSim:{PVA}:Image', pva.PVA)

        # The fields I want to be visible
        self.pva_obj = self.channel.get('field(value,dimension,timeStamp,uniqueId)')

        # Get and set the image from the dictionary
        self.image = self.pva_obj['value'][0]['ubyteValue']


        
    def get_roi(self) -> None:
        # Get the ROI of that single image
        self.roi_data = caget(f'dp-ADSim:{ROI}:MinX')



    def shape_image(self) -> None:
        """
        Turns the PVAObject into an image 

        Return: None
        """
        # Check if dimensions are in image
        if 'dimension' in self.pva_obj:
            # grab the shape and store them in a tuple 
            self.shape = tuple([dim['size'] for dim in self.pva_obj['dimension']])

            # Reshape into a 2d image
            self.shaped_img = np.array(self.image).reshape(self.shape, order='F')
            
        else:
            print(f'Dimension not in {PVA} object')




    def crop_shaped_image(self, ROI_NUM:int = None):
        """
        Crops the shaped_img to the specific ROI's size
        
        Args: ROI_NUM(int) - For the specific ROI you want to crop to
        """

        # There is an ROI provided
        if ROI_NUM: 
                # Get the ROI's dimension
                min_x = caget(f'dp-ADSim:ROI{ROI_NUM}:MinX')
                min_y = caget(f'dp-ADSim:ROI{ROI_NUM}:MinY')
                max_x = caget(f'dp-ADSim:ROI{ROI_NUM}:SizeX')
                max_y = caget(f'dp-ADSim:ROI{ROI_NUM}:SizeY')

                # Slice the needed ROI dimensions from the image
                self.cropped_image = self.shaped_img[min_x:min_x+max_x, min_y:min_y+max_y]
        
        # If not provided use the images dimensions
        else:
            self.crop_size = reversed(self.shaped_img.shape)
            self.cropped_image = self.shaped_img



    def calc_average(self) -> None:
        """
        Calculate the averages of the column and rows
        """
        # Average the cropped images column
        self.cropped_col_avg = np.mean(self.cropped_image,0)
        # Average the cropped images row
        self.cropped_row_avg = np.mean(self.cropped_image,1)



    def display_image(self):
        """
        Display data using matplotlib
        """

        # 
        X = np.arange(0, self.cropped_image.shape[1], 1.0)
        Y = np.arange(0, self.cropped_image.shape[0], 1.0)

        # Used to plot multiple graphs
        figure, axis = plt.subplots(2,2)
    
        # Plot the graphs on the sublot
        # Top Right
        axis[0,0].title.set_text('Cropped ROI Image')
        axis[0,0].imshow(self.cropped_image)

        # Top Left
        axis[0,1].title.set_text('Row Average')
        axis[0,1].plot(self.cropped_row_avg, Y)
        axis[0,1].invert_yaxis()

        # Bottom Right
        axis[1,0].title.set_text('Column Average')
        axis[1,0].plot(X, self.cropped_col_avg)

        # Bottom Left
        axis[1,1].title.set_text('Full Image')
        axis[1,1].imshow(self.shaped_img)

        plt.show()



# Call and start function
roi_cropping = ROICropping()
roi_cropping.crop_img()
