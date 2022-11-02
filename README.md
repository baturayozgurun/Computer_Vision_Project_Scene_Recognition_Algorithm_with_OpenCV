# Computer Vision Project -- Scene Recognition Algorithm with OpenCV
[![Software: C++](https://img.shields.io/badge/Software-C%2B%2B-blue)](http://www.cplusplus.org/)  [![Library: OpenCV](https://img.shields.io/badge/Library-OpenCV-green)](https://www.opencv.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](https://mit-license.org/) 

Developer: Baturay Özgürün

Copyright © 2018 Baturay Özgürün

If you have any requests or questions please do not hesitate to contact me via baturay.ozgurun@gmail.com

---

## Project Description

This project aims to perform a scene recognition technique capitalizing on the Bag of Visiual Words method. A small training dataset (street images) is utilized to create vocabulary and calculate histogram distributions. Scene recognition is performed by finding the correlation between the histogram distributions of training (with the referance image) and test images. This operation allows us to evaluate scene similarities. This code also provides a similarity matrix (evaluation of the scene recognition task) and vocabulary (generated using training dataset) in text files.

---

## Software Requirements

The project was developed on macOS 10.13.6 environment. 

In order to run the project, it is required to install the following programs:
- Xcode 10.1
- OpenCV 3.4.0

---

## Installation

### Step 1: Install Xcode
First, it is required to download and install **Xcode** to run OpenCV projects for C++ on macOS environment. To install Xcode go to **App Store** -> **Search bar** -> **Type Xcode** -> **Click Install App** (see the following figure). Make sure that Xcode is stored in the Applications folder. The OpenCV project presented here was developed on **Xcode 10.1** and **macOS 10.13.6**.

![fig1](https://user-images.githubusercontent.com/76174639/197363657-32774932-cf69-407a-b30e-f058c19f1d61.png)

### Step 2: Download OpenCV Package
To download and install **OpenCV 3.4.0** package go to www.opencv.org and then go to **Library** -> **Releases** -> **Download OpenCV 3.4.0 Sources**. Open the zip file and place it on Desktop. In the opencv-3.4.0 folder, create a new folder and name it *build* (see the following figure).

![fig2](https://user-images.githubusercontent.com/76174639/197363690-788fce3a-d926-4c46-87b5-00680dc9ab35.png)

### Step 3: Install OpenCV
Go to www.cmake.org then click Download button. The binary distributions for MacOS are listed on this page (see the following figure). Since the platform is macOS 10.13, download and install **make-3.25.0-rc2-macos-universal.dmg**. Make sure that CMake is stored in the Applications folder.

![fig3](https://user-images.githubusercontent.com/76174639/197363696-e8b0fe57-9071-481e-abee-883fa1ca9bfe.png)

After installing the CMake software, open it and select the source and build folders as shown in the following figure. Then press **Configure** (please see the following figure).

![fig4](https://user-images.githubusercontent.com/76174639/197363701-1b20712b-bb4a-4c84-b821-d9477b289b8d.png)

Select **Unix Makefiles** and **use default native compilers**. Then press **Done**, and wait until the configuration is done (please see the following figure).

![fig5](https://user-images.githubusercontent.com/76174639/197363740-df8a5e2c-2088-4e64-87c8-b6facad92787.png)

After the configuration is done, check the build files out, and then press **Generate** in the CMake software (please see the following figure). 

![fig6](https://user-images.githubusercontent.com/76174639/197363705-6d8c75e5-53fc-4fc1-a0dd-26f731515349.png)

When the generation is done, close the CMake software (please see the following figure). 

![fig7](https://user-images.githubusercontent.com/76174639/197363706-f4f97304-9c75-4844-93d4-9a3aaeb92dbc.png)

Open the terminal, and type the following lines of codes.

``` 
$ cd Desktop/opencv-3.4.0/build
$ make
$ sudo make install
```

### Step 4: Link the OpenCV Libraries to Xcode

After installing OpenCV, open the Xcode project (the file you downloaded; main.xcodeproj). Go to **Build Settings**, and type **paths** in the search bar. Check if **Header Search Paths** and **Library Search Paths** are indicated as shown in the following figure. 

![fig8](https://user-images.githubusercontent.com/76174639/197363708-677a6925-52eb-44e3-99b1-441271f40751.png)

Make sure that the OpenCV libraries are linked to Xcode, and do not forget that the full path name of your libraries might be different than those in the downloaded Xcode project. Therefore, type **linking** in the search bar, and then clear contents in **Other Linker Flags**. Hold and drag the libraries from **Desktop/opencv-3.4.0/build/lib**, and then drop them to **Other Linker Flags** (please see the following figure).

![fig9](https://user-images.githubusercontent.com/76174639/197363714-ab300d14-0022-4c13-9fc9-ffeca5c789a1.png)

### Step 5: Change the Working Directory in Xcode

It is required to define a custom working directory to read and write images in Xcode. To do this, in Xcode, go to **Product** -> **Scheme** -> **Edit Scheme**. Then go to **Options**, and define 'your' **Working Directory**. It should contain the path name of the *main* folder you downloaded. In my case, it is /Users/BATURAY_OZGURUN/Desktop/main (please see the following figure).

![fig10](https://user-images.githubusercontent.com/76174639/197363718-feaa4af1-9e7c-4512-919a-1422d431dd3a.png)

Now the OpenCV project is ready to run.

#### Remarks: Setting Up OpenCV Extra Modules

This project may require an extra module, e.g., xfeatures2d. To build the extra module, go to https://github.com/opencv/opencv_contrib and download **opencv_contrib-3.4** folder. In this folder, find the module (e.g., xfeatures2d), copy and paste it into the **modules** folder of opencv-3.4.0. Open the CMake software, and then repeat Step 3 and Step 4.

---

## Citation

If you use this project in your research, please cite it as follows:

B. Ozgurun, Computer Vision Project -- Scene Recognition Algorithm with OpenCV, (2022), GitHub repository, https://github.com/baturayozgurun/Computer_Vision_Project_Scene_Recognition_Algorithm_with_OpenCV

---

## License

This project is licensed under [MIT License](https://mit-license.org/).
