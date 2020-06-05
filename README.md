
# RB-FCL - Region Based - Fully Connected Layer

The Region-based feature layer output (RB-FCL) segmentation of several deep learning models to be able to predict bone age.The Region-Based Feature Connected Layer (RB-FCL) extract essential segmented region of hand x-ray. We treat the deep learning models as the feature extraction for each region of the hand x-ray bone. The Fully Connected Layers are the output from the trained important region such as 1-radius-ulna, 2-carpal, 3-metacarpal, 4-phalanges, and 5-ephypisis. DenseNet121, InceptionV3, and InceptionResNetV2 are the deep learning models that we used to train the critical region

## Please download all dataset from this sources

Please download all dataset provided from 
- Digital Hand Atlas Database System, https://ipilab.u sc.edu/research /baaweb/, Access December, 15, 2019.
- RSNA Dataset, https://www.rsna.org/en/education/ai-resources-and-training/ai-image-challenge , Access November, 20, 2019.

### Prerequisites

What things you need to install the software and how to install them

```
pip install -r requirements
```

### 1. Standard deep learning simulation

This ipynb file use for doing standard deep learning simulation

```
1-bone-age-standard.ipynb
```

Please edit

```
img_dir = 'hand-atlas/JPEGimages/combine/'
csv_path = 'hand-atlas/export_dicom.csv'
```

### 2. Region extraction by using Faster R-CNN

We utilize Faster R-CNN to extract the region of hand x-ray 1-radius-ulna, 2-carpal, 3-metacarpal, 4-phalanges, and 5-ephypisis. Please use this [tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) for details


### 3. Training for each of regions

This ipynb file use for training each of the regions of hand x-ray image. 1-radius-ulna, 2-carpal, 3-metacarpal, 4-phalanges, and 5-ephypisis

You can download the models from this [model repository](https://drive.google.com/open?id=1MPzi-ayeBmvaQb9DMZ55NwUwTw-W2jIR). Place this model together with the root folder of this source code.

```
3-train-bone-age-3-dicom-tl-segmented-class-1-5.ipynb
```

Please edit this file, The df['class'] == int(5) represent the region 1-radius-ulna, 2-carpal, 3-metacarpal, 4-phalanges, and 5-ephypisis. dicom_index_segmented.csv is the index for each region files.

```
#Reading data
print("Reading data...")
img_dir = 'hand-atlas/JPEGimages/output/'
csv_path = 'hand-atlas/dicom_index_segmented.csv'
all_df = pd.read_csv(csv_path)
age_df = all_df.loc[all_df['class'] == int(5)]
print (age_df.shape)
```

### 4. Produce FCL output for each region

This ipynb file is use to produce FLC output from each of the regions. 

```
4-predict-3-bone-age-tl-dicom-class-1-5.ipynb
```

Please edit this file The df['class'] == int(5) represent the region 1-radius-ulna, 2-carpal, 3-metacarpal, 4-phalanges, and 5-ephypisis

```
#Reading data
print("Reading data...")
#pdb.set_trace()
img_dir = 'hand-atlas/JPEGimages/output/'
csv_path = 'hand-atlas/dicom_index_segmented.csv'
all_df = pd.read_csv(csv_path)
age_df = all_df.loc[all_df['class'] == int(5)]
print (age_df.shape)
```
output file
```
result-3-dicom-inception-resnet-6-tl-class-5.csv
result-3-dicom-inception-resnet-6-tl-class-5-gt.csv
```

You can download the output file in this [repository](https://drive.google.com/open?id=1oE9lveSbU_q28uWfX83gLBO5dIWpczZP)


### 5. Combine FCL output

This ipynb file is to combine segmented region for each of hand x-ray. 

```
5-traditional-bone-dicom-segmented-to-weighted
```
```
input file
df_fmaps_1 = pd.read_csv("result-3-dicom-inception-resnet-6-tl-class-5.csv", header=None)
df_class = pd.read_csv("result-3-dicom-inception-resnet-6-tl-class-5-gt.csv")
```
output file
```
result-3-dicom-inception-resnet-6-tl-class-5-weighted.csv
```
You can download the example of input and output file in this [repository]https://drive.google.com/open?id=1oE9lveSbU_q28uWfX83gLBO5dIWpczZP)




### 6. Regression experiment to predict bone age

This ipynb file use for doing several experiments of regressios simulation by using FCL output from a deep learning models.
```
6-traditional-bone-combine-dicom-weighted-calculate
```

Input File

```
df_fmaps_31 = pd.read_csv("result-3-dicom-inception-resnet-6-tl-class-1-weighted.csv", header=None)
df_fmaps_32 = pd.read_csv("result-3-dicom-inception-resnet-6-tl-class-2-weighted.csv", header=None)
df_fmaps_33 = pd.read_csv("result-3-dicom-inception-resnet-6-tl-class-3-weighted.csv", header=None)
df_fmaps_34 = pd.read_csv("result-3-dicom-inception-resnet-6-tl-class-4-weighted.csv", header=None)
df_fmaps_35 = pd.read_csv("result-3-dicom-inception-resnet-6-tl-class-5-weighted.csv", header=None)
```
You can download the example of input file in this [repository](https://drive.google.com/open?id=1oE9lveSbU_q28uWfX83gLBO5dIWpczZP)



## Authors

* **Ari Wibisono** - 
* **Petrus Mursanto** - 


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
