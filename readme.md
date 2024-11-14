# License Plate Recognition (LPR) BENCHMARK
## How to add models to the benchmark
1. Add model/code and requirements into ``Models/YOUR_FOLDER_NAME/``.
   Result should look like this: 
   ``Models/example/example.py`` 
   ``Models/example/requirements.txt``
2. In your code, create a function called ``predict()``:
   - It must take one argument - path of the input photo (as ``string``)
   - It must return license plate characters (as ``string``)
3. The function should contain or run only code needed to recognize license plate, meaning any unnecessary plots or ``print()`` functions shouldn't be present.
4. If model needs specific environment (specific version of tensorflow, opencv, python etc.) it must be specified in a comment above ``import`` section.
## data
photos should be downloaded manually from [kaggle](https://www.kaggle.com/datasets/piotrstefaskiue/poland-vehicle-license-plate-dataset) and pasted into data folder
