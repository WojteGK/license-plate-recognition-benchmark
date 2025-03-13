# License Plate Recognition (LPR) BENCHMARK
## Adding models and/or projects to the benchmark
1. In your project files, create an entry-point script wrapping all necessary scripts and functions called ``bench_entry.py``.

2. Entry-point script must contain:
   - ``predict()``: 
      - must take one argument being path of the input photo (as ``string``)
      - must return predicted license plate characters (as ``string``)


3. Add all files, entry-point python script and ``requirements.txt`` into ``Models\YOUR_FOLDER_NAME\`` like this: 
   `Models\`
   ├── `example\`
   │   ├── `bench_entry.py`
   │   ├── `requirements.txt`
   │   ├── `other files...`

## Adding photos to benchmark
1. Go to [kaggle](https://www.kaggle.com/datasets/piotrstefaskiue/poland-vehicle-license-plate-dataset) and download our dataset.
2. Unzip file and paste photos into `data\photos` folder.

## Automated run of all modules/subprojects
1. Make sure you have properly followed previous points
2. Open the config file `Benchmark\config.py`
3. Add your module/project following the instructions in comment
4. Make sure you have downloaded all necessary python versions
5. Check python executable paths and correct them if necessary
6. Execute `run.py`
7. Results will be stored inside `Benchmark\results\` folder


## Manual run chosen module/subproject
1. Make sure you have properly followed previous points
2. Change directory to chosen model `cd .\Modules\MODULE_NAME`
3. Create venv using proper python version `virtualenv -p path\to\proper\python\version`
4. Activate venv `.\Scripts\activate`
5. Install requirements `pip install -r requirements.txt`
6. Adjust values like paths and number of iterations in `run_bench.py`
7. Run `python run_bench.py`
8. Results will be displayed in `Benchmark\data\` folder.
