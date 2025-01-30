#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import all necessary packages
import pandas as pd           
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 
import geemap    
import ee        
import numpy as np
import datetime
from scipy.stats import ttest_ind
from scipy import stats
import urllib.request
from io import BytesIO
from PIL import Image
from datetime import datetime


# In[ ]:


# Authenticate with Earth Engine (Kindly note that this activities is only needed once on a laptop, uncomment if you have not done so before)
# ee.Authenticate() 

# Initialize Earth Engine
ee.Initialize()


# In[ ]:


# create interactive map for visual representations
Map = geemap.Map()
Map.add_basemap('HYBRID')
Map.addLayerControl()
Map


# In[ ]:


# This will ensure that the map is displayed in the note without error 
# !jupyter nbextension enable --py widgetsnbextension


# In[ ]:


# After loading the shap file into the project assets on the googleearth engine code editor
# Access the asset using its ID
roi = ee.FeatureCollection('projects/ee-ayotundenew/assets/omo_forest')

# Optionally, convert it to a geometry
roi = roi.geometry()

# Print the geometry to check it
# print(roi.getInfo())

# Add the ROI to the map
Map.addLayer(roi, {}, 'ROI Polygon')
# Center the map on the ROI
Map.centerObject(roi, 10)  # You can adjust the zoom level
# Show the map
# Map


# ## Obtain the data from google earth engine using the landsat 8 data clipping to the ROI

# In[ ]:


# Define Landsat 8 Collection 2, Tier 1 data filtered by date, ROI, path/row, and cloud cover
L8_collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                 .filterBounds(roi)  # Filter by region of interest (ROI)
                 .filterDate('2014-01-01', '2024-12-10')  # Filter by date range
                 .filter(ee.Filter.eq('WRS_PATH', 190))  # Filter by Path (Landsat 8 specific)
                 .filter(ee.Filter.eq('WRS_ROW', 55))    # Filter by Row
                 .filter(ee.Filter.lt('CLOUD_COVER', 5)))  # Filter by cloud cover (limit to <5%)

# Clip each image in the collection to the ROI
clipped_L8_collection = L8_collection.map(lambda image: image.clip(roi))

# Sort the image collection by acquisition date
clipped_L8_collection = clipped_L8_collection.sort('DATE_ACQUIRED')

# Get the number of images in the clipped collection
image_count = clipped_L8_collection.size()
print("Number of images retrieved:", image_count.getInfo())

# Optionally, retrieve and display the first image to check the clipping
first_image = clipped_L8_collection.first()
Map.addLayer(first_image, {}, 'First_L8_Clipped Image')

# Show the Map (if working in the Earth Engine environment)
# Map


# In[ ]:


# Convert the image colleccloud_free_collectiontion to a list
image_list = clipped_L8_collection.toList(clipped_L8_collection.size())

# Get the second image (index 1)
first_image = ee.Image(image_list.get(0))

# Define visualization parameters
vis_params = {
    'min': 8000,
    'max': 15000,
    'bands': ['SR_B4', 'SR_B3', 'SR_B2']  # RGB bands for Landsat Surface Reflectance
}

# Add the second image to the map
Map.addLayer(first_image, vis_params, 'first Image')

# Show the map
Map


# In[ ]:


# Convert the image colleccloud_free_collectiontion to a list
image_list = clipped_L8_collection.toList(clipped_L8_collection.size())

# Get the second image (index 1)
Last_image = ee.Image(image_list.get(10))

# Define visualization parameters
vis_params = {
    'min': 9500,
    'max': 20000,
    'bands': ['SR_B4', 'SR_B3', 'SR_B2']  # RGB bands for Landsat Surface Reflectance
}

# Add the second image to the map
Map.addLayer(Last_image, vis_params, 'Last Image')

# Show the map
Map


# In[ ]:


# Get metadata for each image in the clipped collection
image_list = clipped_L8_collection.toList(image_count)

# Iterate over each image in the list
for i in range(image_count.getInfo()):
    image = ee.Image(image_list.get(i))
    
    # Retrieve specific metadata properties
    acquisition_date = image.get('DATE_ACQUIRED').getInfo()
    cloud_cover = image.get('CLOUD_COVER').getInfo()
    image_id = image.id().getInfo()
    
    # Print out metadata for the image
    print(f"Image {i + 1} - ID: {image_id}, Date: {acquisition_date}, Cloud Cover: {cloud_cover}%")


# ### As a result of the existence of cloud in some of the images as shown in the first and last image through the meta data, I will be conducting a cloud masking

# In[ ]:


# Function to mask clouds and cloud shadows based on the QA_PIXEL band
def mask_clouds_and_shadows(image):
    # Get the 'QA_PIXEL' band
    qa_pixel = image.select(['QA_PIXEL'])
    
    # Cloud and cloud shadow masking:
    # Cloud detection: bit 3 (cloud flag) should be 0 (no cloud)
    cloud_mask = qa_pixel.bitwiseAnd(1 << 3).eq(0)
    
    # Cloud shadow detection: bit 5 (cloud shadow flag) should be 0 (no shadow)
    cloud_shadow_mask = qa_pixel.bitwiseAnd(1 << 5).eq(0)
    
    # Combine cloud and cloud shadow masks
    total_mask = cloud_mask.And(cloud_shadow_mask)
    
    # Apply the mask to the image
    return image.updateMask(total_mask)

# Apply the cloud masking function to the image collection
masked_L8_collection = clipped_L8_collection.map(mask_clouds_and_shadows)

# Optionally, visualize the first image after cloud masking
first_masked_image = masked_L8_collection.first()
if masked_L8_collection.size().getInfo() == 0:
    print("The image collection is empty after masking!") #Checking if the collection is empty after the masking of cloud
else:
    Map.addLayer(first_masked_image, {}, 'First Masked L8 Image')


# ### Print the masked collection images adjusted cloud percentage

# In[ ]:


# Function to calculate adjusted cloud cover percentage
def calculate_cloud_cover_percentage(image):
    # Total pixel count
    total_pixels = image.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=image.geometry(),
        scale=30,
        maxPixels=1e13
    ).values().get(0)
    
    # Clear pixel count
    clear_pixels = image.mask().reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=image.geometry(),
        scale=30,
        maxPixels=1e13
    ).values().get(0)
    
    # Calculate the cloud cover percentage
    cloud_cover_percentage = ee.Number(1).subtract(
        ee.Number(clear_pixels).divide(total_pixels)
    ).multiply(100)
    
    # Return the image with the adjusted cloud cover property
    return image.set('Adjusted_Cloud_Cover', cloud_cover_percentage)

# Apply the function to all images in the collection
masked_L8_collection_with_cloud_cover = masked_L8_collection.map(calculate_cloud_cover_percentage)

# Function to print metadata for all images in the collection
def print_metadata_with_adjusted_cloud_cover(image_list):
    for i in range(image_list.size().getInfo()):
        # Get the image
        image = ee.Image(image_list.get(i))
        
        # Extract properties
        image_id = image.get('system:index').getInfo()
        acquisition_date = image.date().format('YYYY-MM-dd').getInfo()
        original_cloud_cover = image.get('CLOUD_COVER').getInfo()
        adjusted_cloud_cover = image.get('Adjusted_Cloud_Cover').getInfo()
        
        # Print formatted metadata
        print(
            f"Image {i + 1} - ID: {image_id}, Date: {acquisition_date}, "
            f"Original Cloud Cover: {original_cloud_cover}%, "
            f"Adjusted Cloud Cover: {adjusted_cloud_cover:.2f}%"
        )

# Convert the collection to a list
image_list = masked_L8_collection_with_cloud_cover.toList(masked_L8_collection_with_cloud_cover.size())

# Call the function to print metadata
print_metadata_with_adjusted_cloud_cover(image_list)


# In[ ]:


# get information on projection and spatial resolution on a selected specific band 
image_list = masked_L8_collection.toList(image_count)
for i in range(image_count.getInfo()):
    image = ee.Image(image_list.get(i))
    
    # Using the Near Infrared Band band
    band = image.select('SR_B5')
    
    # Get the projection of the selected band
    projection = band.projection()
    
    # Get the spatial resolution (pixel size) in meters
    spatial_resolution = projection.nominalScale().getInfo()
    
    # Get the CRS information including the EPSG code
    crs_info = projection.getInfo()
    
    # print(f"Image {i + 1} - Spatial Resolution (meters): {spatial_resolution}")
    print(f"Image {i + 1} - CRS Info: {crs_info}")


# In[ ]:


# Get the first image in the collection (most recent date)
first_image = ee.Image(masked_L8_collection.sort('system:time_start', True).first())

# Compute the minimum and maximum values within the ROI for the selected image
stats = first_image.reduceRegion(
    reducer=ee.Reducer.minMax(),
    geometry=roi,
    scale=30,  # Scale set to Landsat's resolution
    bestEffort=True  # This helps avoid computation timeouts for large ROIs
)

# Retrieve min and max values for the blue (SR_B2), green (SR_B3), and red (SR_B4) bands
min_val_blue = stats.get('SR_B2_min').getInfo()  # Min value for blue
max_val_blue = stats.get('SR_B2_max').getInfo()   # Max value for blue
min_val_green = stats.get('SR_B3_min').getInfo()  # Min value for green
max_val_green = stats.get('SR_B3_max').getInfo()   # Max value for green
min_val_red = stats.get('SR_B4_min').getInfo() # Min value for red
max_val_red = stats.get('SR_B4_max').getInfo() # Max value for red

# Print the min and max values for each band
print(f"Min Value (Blue Band): {min_val_blue}")
print(f"Max Value (Blue Band): {max_val_blue}")
print(f"Min Value (Green Band): {min_val_green}")
print(f"Max Value (Green Band): {max_val_green}")
print(f"Min Value (Red Band): {min_val_red}")
print(f"Max Value (Red Band): {max_val_red}")


# In[ ]:


# Define visualization parameters using computed min and max values
vis_params = {
    'bands': ['SR_B4', 'SR_B3', 'SR_B2'],  # Red, Green, Blue bands for true color
    'min': min(min_val_blue, min_val_green, min_val_red),  # Use the minimum value across bands
    'max': max(max_val_blue, max_val_green, max_val_red),  # Use the maximum value across bands
    'gamma': 0.8  # Gamma correction for better visualization
}

# Add the true color composite to the map
Map.addLayer(first_image, vis_params, 'First Cloud Masked Landsat 8 RGB')

# Display the map
Map


# In[ ]:


# Get the last image in the collection (most latter date)
last_image = ee.Image(masked_L8_collection.sort('system:time_start', False).first())

# Compute the minimum and maximum values within the ROI for the selected image
stats = last_image.reduceRegion(
    reducer=ee.Reducer.minMax(),
    geometry=roi,
    scale=30,  # Scale set to Landsat's resolution
    bestEffort=True  # This helps avoid computation timeouts for large ROIs
)

# Retrieve min and max values for the blue (SR_B2), green (SR_B3), and red (SR_B4) bands
min_val_blue = stats.get('SR_B2_min').getInfo()  # Min value for blue
max_val_blue = stats.get('SR_B2_max').getInfo()   # Max value for blue
min_val_green = stats.get('SR_B3_min').getInfo()  # Min value for green
max_val_green = stats.get('SR_B3_max').getInfo()   # Max value for green
min_val_red = stats.get('SR_B4_min').getInfo() # Min value for red
max_val_red = stats.get('SR_B4_max').getInfo() # Max value for red

# Print the min and max values for each band
print(f"Min Value (Blue Band): {min_val_blue}")
print(f"Max Value (Blue Band): {max_val_blue}")
print(f"Min Value (Green Band): {min_val_green}")
print(f"Max Value (Green Band): {max_val_green}")
print(f"Min Value (Red Band): {min_val_red}")
print(f"Max Value (Red Band): {max_val_red}")


# In[ ]:


# Define visualization parameters using computed min and max values
vis_params = {
    'bands': ['SR_B4', 'SR_B3', 'SR_B2'],  # Red, Green, Blue bands for true color
    'min': min(min_val_blue, min_val_green, min_val_red),  # Use the minimum value across bands
    'max': max(max_val_blue, max_val_green, max_val_red),  # Use the maximum value across bands
    'gamma': 0.8  # Gamma correction for better visualization
}

# Add the true color composite to the map
Map.addLayer(last_image, vis_params, 'Last Cloud Masked Landsat 8 RGB')

# Display the map
Map


# In[ ]:


# Select the NIR (SR_B5), Red (SR_B4), and Green (SR_B3) bands of the first Image
first_nir_red_green = first_image.select(['SR_B5', 'SR_B4', 'SR_B3'])

# Define min and max values for the composite visualization
# Ensure the values correspond to the selected bands
nir_min_val = stats.get('SR_B5_min').getInfo()
nir_max_val = stats.get('SR_B5_max').getInfo()

# Add the NIR color composite to the map
Map.addLayer(first_nir_red_green, {
    'bands': ['SR_B5', 'SR_B4', 'SR_B3'],  # NIR, Red, Green
    'min': nir_min_val,
    'max': nir_max_val,
    'gamma': 0.8  # Adjust gamma for better contrast
}, 'First Image NIR Color Composite')


# In[ ]:


# Select the NIR (SR_B5), Red (SR_B4), and Green (SR_B3) bands of the last Image
last_nir_red_green = last_image.select(['SR_B5', 'SR_B4', 'SR_B3'])

# Define min and max values for the composite visualization
# Ensure the values correspond to the selected bands
nir_min_val = stats.get('SR_B5_min').getInfo()
nir_max_val = stats.get('SR_B5_max').getInfo()

# Add the NIR color composite to the map
Map.addLayer(last_nir_red_green, {
    'bands': ['SR_B5', 'SR_B4', 'SR_B3'],  # NIR, Red, Green
    'min': nir_min_val,
    'max': nir_max_val,
    'gamma': 0.8  # Adjust gamma for better contrast
}, 'Last Image NIR Color Composite')


# In[ ]:


# Select the SWIR (SR_B7), NIR (SR_B5), and Red (SR_B4) bands of the first
first_false_color_composite_754 = first_image.select(['SR_B7', 'SR_B5', 'SR_B4'])

# Compute min and max values for the SWIR, NIR, and Red bands
swir_min_val = stats.get('SR_B7_min').getInfo()
swir_max_val = stats.get('SR_B7_max').getInfo()

# Define visualization parameters
viz_params_754 = {
    'bands': ['SR_B7', 'SR_B5', 'SR_B4'],  # SWIR, NIR, Red
    'min': swir_min_val,
    'max': swir_max_val,
    'gamma': 0.8  # Adjust gamma for better contrast
}

# Add the false-color composite to the map
Map.addLayer(first_false_color_composite_754, viz_params_754, 'First image False Color Composite 754')


# In[ ]:


# Select the SWIR (SR_B7), NIR (SR_B5), and Red (SR_B4) bands
last_false_color_composite_754 = last_image.select(['SR_B7', 'SR_B5', 'SR_B4'])

# Compute min and max values for the SWIR, NIR, and Red bands
swir_min_val = stats.get('SR_B7_min').getInfo()
swir_max_val = stats.get('SR_B7_max').getInfo()

# Define visualization parameters
viz_params_754 = {
    'bands': ['SR_B7', 'SR_B5', 'SR_B4'],  # SWIR, NIR, Red
    'min': swir_min_val,
    'max': swir_max_val,
    'gamma': 0.8  # Adjust gamma for better contrast
}

# Add the false-color composite to the map
Map.addLayer(last_false_color_composite_754, viz_params_754, 'Last image False Color Composite 754')


# ## Calculating the NDVI of the images
# ### NDVI= NIR+Red/NIR−Red
# ​

# In[ ]:


# Function to calculate NDVI for an image
def calculate_ndvi(image):
    # Calculate NDVI using the formula
    ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    
    # Add NDVI as a new band to the image
    return image.addBands(ndvi)

# Apply the NDVI calculation function to each image in the collection
ndvi_collection = masked_L8_collection.map(calculate_ndvi)

# Print confirmation
print("NDVI calculated and added to the image collection.")


# In[ ]:


# Function to calculate mean NDVI for an image within the ROI
def calculate_mean_ndvi(image):
    # Reduce the NDVI band over the ROI to get the mean value
    mean_ndvi = image.select('NDVI').reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=30,  # Scale set to Landsat's resolution
        bestEffort=True  # Avoid computation timeouts for large ROIs
    ).get('NDVI')
    
    # Add the mean NDVI as a property to the image
    return image.set('Mean_NDVI', mean_ndvi)

# Apply the function to all images in the NDVI collection
ndvi_with_mean = ndvi_collection.map(calculate_mean_ndvi)

# Function to print NDVI values for all images
def print_ndvi_values(image_list):
    for i in range(image_list.size().getInfo()):
        # Get the image
        image = ee.Image(image_list.get(i))
        
        # Extract metadata
        image_id = image.get('system:index').getInfo()
        acquisition_date = image.date().format('YYYY-MM-dd').getInfo()
        mean_ndvi = image.get('Mean_NDVI').getInfo()
        
        # Print formatted information
        print(f"Image {i + 1} - ID: {image_id}, Date: {acquisition_date}, Mean NDVI: {mean_ndvi:.4f}")

# Convert the collection to a list and print NDVI values
image_list = ndvi_with_mean.toList(ndvi_with_mean.size())
print_ndvi_values(image_list)


# In[ ]:


# Add NDVI layers to the map
for i in range(image_count.getInfo()):
    image = ee.Image(image_list.get(i))
    
    # Get the image year from metadata
    image_year = image.date().get('year').getInfo()
    
    # Extract the NDVI band
    ndvi = image.select('NDVI')
       
    # Define visualization parameters for the NDVI
    vis_params = {
        'min': 0.05, 'max': 0.4, 'palette': ['red', 'yellow', 'green']
    }
    
# Add the NDVI layer to the map
Map.addLayer(ndvi, vis_params, name=f'NDVI - {image_year}')
    
# Show the Map
Map


# ### Plot the the NDVI Images

# In[ ]:


# Set up the number of rows and columns for the plot
num_columns = 2

# Set figure size
plt.figure(figsize=(10, 30))

# Loop through the images and calculate NDVI
for i in range(image_count.getInfo()):
    image = ee.Image(image_list.get(i))
    
    # Get the image year from metadata
    image_year = image.date().get('year').getInfo()
    
    # Calculate NDVI using NIR (SR_B5) and Red (SR_B4)
    ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')

    # Set the visualization parameters on the NDVI image
    ndvi_vis = ndvi.visualize(min=0.05, max=0.38, palette=['red', 'yellow', 'green'])

    # Get the URL for downloading the NDVI image data with visualization parameters
    url = ndvi_vis.getDownloadURL({
        'scale': 30,  # Specify the scale of the image in meters (e.g., 30m for Landsat)
        'region': roi,  # Set region to the area of interest
        'format': 'png',  # Image format for downloading
    })

    # Download the image data using urllib
    img_data = urllib.request.urlopen(url).read()
    img = Image.open(BytesIO(img_data))
    
    # Convert the image to a numpy array for visualization
    img_array = np.array(img)
    
    # Saving the LST plot
   # plt.savefig(f'NDVI_{image_year}.png')
    
    # Plot the NDVI image with a color bar
    plt.subplot((image_count.getInfo() + num_columns - 1) // num_columns, num_columns, i + 1)
    plt.imshow(img_array)
    plt.title(f'NDVI - {image_year}')
    plt.axis('off')  # Turn off axis labels
    plt.colorbar(plt.imshow(img_array, cmap='RdYlGn', vmin=-0.5, vmax=0.5))  # Add color bar

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# ### Plotting the time series of the NDVI

# In[ ]:


# Example NDVI time series data
dates = ['2014-12-30', '2015-01-15', '2015-12-17', '2016-01-02', '2017-01-04', 
         '2018-12-25', '2019-12-28', '2020-02-14', '2022-12-20', '2023-12-23', 
         '2024-02-09']
ndvi_values = [0.2673, 0.3479, 0.3537, 0.3074, 0.3312, 0.2939, 0.2931, 0.1779, 
               0.3490, 0.2755, 0.1822]


    
plt.figure(figsize=(10, 5))
plt.plot(dates, ndvi_values, marker='o', label='Mean NDVI')
plt.xlabel('Date')
plt.ylabel('Mean NDVI')
plt.xticks(rotation=45)
plt.title('NDVI Time Series')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('NDVI Time Series.png') # Save the time series
plt.show()


# ### Calculating the Land Surface Temperature (LST)
# ### LST=(TIR×0.00341802+149.0)−273.15
# #### Where:
#         - TIR = 
#         - 0.00341802 = multiplicative factor to convert the radiance to brightness temperature (in Kelvin).
#         - 149.0 = additive factor to adjust the result (this is specific to Landsat 8 and Band 10).
#         - 273.15 = used to convert to Celsius from Kelvin.

# In[ ]:


# Define the LST calculation function
def compute_lst(image):
    # Select the thermal infrared band
    tir = image.select('ST_B10')  # Thermal Infrared band (B10)
    
    # Convert the temperature from Kelvin to Celsius (if needed)
    lst = tir.multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')  # Conversion factors depend on the specific sensor
    return image.addBands(lst)

def print_lst_values(image, index):
    # Compute LST
    image_with_lst = compute_lst(image)
    
    # Select LST band
    lst = image_with_lst.select('LST')

    # Get the mean LST value over the whole image
    mean_lst = lst.reduceRegion(
        reducer=ee.Reducer.mean(),
        scale=30,
        maxPixels=1e8
    ).get('LST').getInfo()

    print(f"Image {index + 1} - Mean LST: {mean_lst} °C")

# Loop through the collection and print LST values
for i, feature in enumerate(masked_L8_collection.getInfo()['features']):
    image = ee.Image(feature['id'])  # Access image by its ID
    print_lst_values(image, i)


# In[ ]:


# Set up the number of rows and columns for the plot
num_columns = 2

# Set figure size
plt.figure(figsize=(10, 30))

# Define the LST calculation function
def compute_lst(image):
    # Select the thermal infrared band
    tir = image.select('ST_B10')  # Thermal Infrared band (B10)
    
    # Convert the temperature from Kelvin to Celsius (if needed)
    lst = tir.multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')  # Conversion factors depend on the specific sensor
    return image.addBands(lst)

# Loop through the images and plot LST values
for i in range(image_count.getInfo()):
    image = ee.Image(image_list.get(i))
    
    # Get the image year from metadata
    image_year = image.date().get('year').getInfo()
    
    # Compute LST
    image_with_lst = compute_lst(image)
    
    # Select LST band
    lst = image_with_lst.select('LST')

    # Get the mean LST value over the whole image
    mean_lst = lst.reduceRegion(
        reducer=ee.Reducer.mean(),
        scale=30,
        maxPixels=1e8
    ).get('LST').getInfo()

    # Set the visualization parameters on the LST image
    lst_vis = lst.visualize(min=25, max=40, palette=['blue', 'cyan', 'green', 'yellow', 'red'])

    # Get the URL for downloading the LST image data with visualization parameters
    url = lst_vis.getDownloadURL({
        'scale': 30,  # Specifying the scale of the image in meters
        'region': roi,  # Setting region to the area of interest
        'format': 'png',  # Image format for downloading
    })

    # Download the image data using urllib
    img_data = urllib.request.urlopen(url).read()
    img = Image.open(BytesIO(img_data))
    
    # Convert the image to a numpy array for visualization
    img_array = np.array(img)
    
    # Plot the LST image with a color bar
    plt.subplot((image_count.getInfo() + num_columns - 1) // num_columns, num_columns, i + 1)
    plt.imshow(img_array, cmap='inferno')  # Use 'inferno' colormap for temperature
    plt.title(f'LST - {image_year}')
    plt.axis('off')  # Turn off axis labels
    plt.colorbar(plt.imshow(img_array, cmap='inferno', vmin=-10.0, vmax=50.0))  # Add color bar

    # Saving the LST plot
    # plt.savefig(f'LST_{image_year}.png')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# ### Plotting the LST time series

# In[ ]:


### Get LST dates and index values

def get_lst_values(image, index):
    # Compute LST
    image_with_lst = compute_lst(image)
    
    # Select LST band
    lst = image_with_lst.select('LST')

    # Get the mean LST value over the whole image
    mean_lst = lst.reduceRegion(
        reducer=ee.Reducer.mean(),
        scale=30,
        maxPixels=1e8
    ).get('LST').getInfo()

    # Get the image date
    date_str = image.date().format('YYYY-MM-dd').getInfo()
    date = datetime.strptime(date_str, '%Y-%m-%d')
    
    return date, mean_lst

# Loop through the collection and collect LST values and dates without printing
dates = []
mean_lst_values = []

for i, feature in enumerate(masked_L8_collection.getInfo()['features']):
    image = ee.Image(feature['id'])  # Access image by its ID
    date, mean_lst = get_lst_values(image, i)
    dates.append(date)
    mean_lst_values.append(mean_lst)

# Plot the time series of mean LST values
plt.figure(figsize=(10, 6))
plt.plot(dates, mean_lst_values, marker='o', linestyle='-', color='b')
plt.xlabel('Date')
plt.ylabel('Mean LST (°C)')
plt.title('Time Series of Mean LST Values')
plt.grid(True)
plt.show()


# ### Using sentinel 1 images with 10m resolution

# In[ ]:


collectionS1 =ee.ImageCollection('COPERNICUS/S1_GRD')\
    .filter(ee.Filter.eq('instrumentMode','IW'))\
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH'))\
    .filterMetadata('resolution_meters','equals' ,10)\
    .filter(ee.Filter.eq('orbitProperties_pass','ASCENDING'))\
    .filterDate('2014-01-01', '2024-12-31')\
    .filterBounds(roi)

# Define a function to clip each image in the collection to the ROI
def clip_to_roi(image):
    return image.clip(roi)

# Map the clipping function over the collection
collectionS1 = collectionS1.map(clip_to_roi)


# In[ ]:


# get number of images for time series
print(f"Number of images: {collectionS1.size().getInfo()}.")

# get band names
print(f"Bands available: {collectionS1.first().bandNames().getInfo()}.")


# In[ ]:


# Filter Sentinel-1 images for the to get the 2014 and 2023 seperately
# Filter Sentinel-1 images for the year 2015
collection_2015 = ee.ImageCollection('COPERNICUS/S1_GRD')\
                 .filterDate('2015-01-01', '2015-12-31')\
                 .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
                 .filterMetadata('resolution_meters', 'equals', 10)\
                 .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))\
                 .filterBounds(roi)\

# Filter Sentinel-1 images for the year 2023
collection_2023 = ee.ImageCollection('COPERNICUS/S1_GRD')\
            .filterDate('2023-01-01', '2023-12-31')\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
            .filterMetadata('resolution_meters', 'equals', 10)\
            .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))\
            .filterBounds(roi)\

# Define a function to clip each image in the collection to the ROI
def clip_to_roi(image):
    return image.clip(roi)

# Map the clipping function over the collections
collection_2015 = collection_2015.map(clip_to_roi)
collection_2023 = collection_2023.map(clip_to_roi)

# Get the number of images for each year
num_images_2015 = collection_2015.size().getInfo()
num_images_2023 = collection_2023.size().getInfo()

print(f"Number of images for 2015: {num_images_2015}.")
print(f"Number of images for 2023: {num_images_2023}.")

# Get band names for each collection
bands_2015 = collection_2015.first().bandNames().getInfo()
bands_2023 = collection_2023.first().bandNames().getInfo()

print(f"Bands available for collection_2015: {bands_2015}.")
print(f"Bands available for collection_2023: {bands_2023}.")


# In[ ]:


# Compute the mean VH backscatter ratio for 2015 and 2024
mean_2015 = collection_2015.select('VH').mean()
mean_2023 = collection_2023.select('VH').mean()

# Calculate the difference between the two years
difference = mean_2023.subtract(mean_2015)

# Define visualization parameters
vis_params = {
    'min': -20, 
    'max': 0
}

diff_vis_params = {
    'min': -5,
    'max': 5,
    'palette': ['green', 'white', 'red']
}


# Create a map using geemap
Map = geemap.Map(center=[roi.centroid().coordinates().get(1).getInfo(), roi.centroid().coordinates().get(0).getInfo()], zoom=10)

# Add layers to the map
Map.addLayer(mean_2015, vis_params, 'Mean VH 2015')
Map.addLayer(mean_2023, vis_params, 'Mean VH 2023')
Map.addLayer(difference, diff_vis_params, 'Difference VH 2015-2023')

# Display the map
Map.addLayerControl()
Map


# In[ ]:


# Compute the mean VH backscatter ratio for 2015 and 2023
# Define a reducer to compute the mean value over the region
reducer = ee.Reducer.mean()

# Calculate the mean VH backscatter for 2015 over the ROI
mean_2015_value = mean_2015.reduceRegion(
    reducer=reducer,
    geometry=roi,
    scale=30,
    maxPixels=1e9
).get('VH').getInfo()

# Calculate the mean VH backscatter for 2023 over the ROI
mean_2023_value = mean_2023.reduceRegion(
    reducer=reducer,
    geometry=roi,
    scale=30,
    maxPixels=1e9
).get('VH').getInfo()

# Difference in mean VH backscatter between the two years
difference_value = mean_2023_value - mean_2015_value

# Print the results
print(f"Mean VH backscatter for 2015: {mean_2015_value}")
print(f"Mean VH backscatter for 2023: {mean_2023_value}")
print(f"Difference in VH backscatter: {difference_value}")

# Define visualization parameters
vis_params = {
    'min': -15, 
    'max': 15,
    'palette': ['green', 'white', 'red']
}

diff_vis_params = {
    'min': -2,
    'max': 2,
    'palette': ['green', 'white', 'red']
}


# Add layers to the map
Map.addLayer(mean_2015, vis_params, 'Mean VH 2015')
Map.addLayer(mean_2023, vis_params, 'Mean VH 2023')
Map.addLayer(difference, diff_vis_params, 'Difference VH 2015-2023')

# Display the map
Map


# ### Plotting the time series map for 2015 and 2024

# In[ ]:


# Sort the filtered collections by date in ascending order
sorted_collection_2015 = collection_2015.sort('system:time_start')
sorted_collection_2023 = collection_2023.sort('system:time_start')

# Define a function to extract and format data from the image collection
def extract_data(image_collection):
    dates = []
    backscatter_values = []
    for image in image_collection.toList(image_collection.size()).getInfo():
        ee_image = ee.Image(image['id'])
        date_str = ee_image.get('system:time_start').getInfo()
        date = datetime.utcfromtimestamp(int(date_str) // 1000)
        dates.append(date)
        vh_band = ee_image.select('VH')
        stats = vh_band.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, scale=10, maxPixels=1e9)
        backscatter_value = stats.get('VH').getInfo()
        backscatter_values.append(backscatter_value)
    return dates, backscatter_values

# Extract data for 2015 and 2023 periods
dates_collection_2015, backscatter_collection_2015 = extract_data(sorted_collection_2015)
dates_collection_2023, backscatter_collection_2023 = extract_data(sorted_collection_2023)

# Create DataFrames for 2015 and 2023 periods
df_collection_2015 = pd.DataFrame({'Date': dates_collection_2015, 'VH Backscatter': backscatter_collection_2015})
df_collection_2023 = pd.DataFrame({'Date': dates_collection_2023, 'VH Backscatter': backscatter_collection_2023})

# Plot the time series for both 2015 and 2023 periods
plt.figure(figsize=(12, 6))
plt.plot(df_collection_2015['Date'], df_collection_2015['VH Backscatter'], marker='o', linestyle='-', color='g', label='2015')
plt.plot(df_collection_2023['Date'], df_collection_2023['VH Backscatter'], marker='o', linestyle='-', color='r', label='2023')
plt.xlabel('Year')
plt.ylabel('VH Backscatter')
plt.title('Sentinel-1 VH Backscatter Time Series for 2015 and 2023')
plt.legend()
plt.grid(True)
plt.gca().set_xticks([datetime(2015, 1, 1), datetime(2023, 1, 1)]) # showing only the two comparing years
plt.gca().set_xticklabels(['2015', '2023'])
plt.savefig('Time series for 2015 and 2023.png')
plt.tight_layout()
plt.show()


# ### Showing the time series from 2015 to 2024

# In[ ]:


# Sort the filtered collection by date in ascending order
sorted_collectionS1 = collectionS1.sort('system:time_start')

# Initialize empty lists to store dates and backscatter values
dates = []
backscatter_values = []

# Iterate through the sorted collection
for image in sorted_collectionS1.toList(sorted_collectionS1.size()).getInfo():
    # Convert the image object to an Earth Engine image
    ee_image = ee.Image(image['id'])
    
    # Get the incidence date as a string
    date_str = ee_image.get('system:time_start').getInfo()
    
    # Parse the string date into a Python datetime object
    date = datetime.utcfromtimestamp(int(date_str) // 1000)
    dates.append(date)
    
    # Get the VH band from the image
    vh_band = ee_image.select('VH')
    
    # Calculate the mean backscatter value within the ROI
    stats = vh_band.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, scale=10, maxPixels=1e9)
    backscatter_value = stats.get('VH').getInfo()
    backscatter_values.append(backscatter_value)

# Create a DataFrame to store the data
data = {'Date': dates, 'VH Backscatter': backscatter_values}
prosna = pd.DataFrame(data)

# Plot the VH backscatter values over time
plt.figure(figsize=(12, 6))
plt.plot(prosna['Date'], prosna['VH Backscatter'], marker='o', linestyle='-', color='b', label='VH Backscatter')
plt.title('Change in VH Backscatter Over Time from 2015 till 2024')
plt.xlabel('Date')
plt.ylabel('VH Backscatter (dB)')
plt.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)

# Set x-axis ticks and format
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)

# Add a legend
plt.legend()

# Save the plot
# plt.savefig('Time series from 2015 to 2024.png')

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()


# ### End of code
