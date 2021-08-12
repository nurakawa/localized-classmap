# Localized Class Map

This repository contains the implementation of the localized class map, a tool
for visualizing classification results. It also includes some examples using benchmark datasets, found in the folder "examples."

The localized class map is an extension of the class map of [Raymaekers, Rousseeuw and Hubert (2021)](doi:10.1080/00401706.2021.1927849), which is available on CRAN as R 
package `classmap`. By modifying the definition of _farness_ to use local neighborhood distances, the localized classmap can be used with any classifier.  

The implementation of the localized class map is made to be compatible with `caret`. A user first trains a model in `caret`, then inputs it directly into `plotClassMap()` to visualize the results. See `examples/example-iris.R` for a simple example.  

