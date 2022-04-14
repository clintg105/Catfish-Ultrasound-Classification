# Contents

## Data
- Novelo (781 MB) Ultrasound screenshots provided by Dr. Novelo and collected at the Aquaculture Research Station.
	- The file NoveloData.csv attempts to organize the metadata read from these screenshots.
- raw (117 MB) The isolated ultrasounds without ultrasound GUI elements.
- CroppedAndCorrected (58 MB) Images preprocessed as described in [[1]](#1).

## Transfer Learning
1. The transfer learning pipline is contained in the notebook **Final_Results_Collection**.
	- This file relies on the nets in the **PretrainedModels** directory, which becomes populated after running the cells in the **Downloading Networks** section.
2. The code in **Final_Results_Collection** populates the **FinalResults** folder, and figures are generated in the **Figure_Creation** notebook.

## Dimension Reduction
1. Running the notebook **Feature_Extractor** will populate the **DimensionReduction** folder with CSV files of vectorized images.
2. Traditional models are trained and scored by the script **Python\ultrasound_DR**.
	- Corss validation scores and confusion matricies are stored in the **Results** directory.
3. The notebook **Figure_Creation_DR** plots the results.

# References
<a id="1">[1]</a> Graham, C. A., Shamkhalichenar, H., Browning, V. E., Byrd, V. J., Liu, Y., Gutierrez-Wing, M. T., … Tiersch, T. R. (2022). A practical evaluation of machine learning for classification of ultrasound images of ovarian development in channel catfish (Ictalurus punctatus). Aquaculture, 552, 738039. doi:10.1016/j.aquaculture.2022.738039