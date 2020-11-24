# MiraBest

The MiraBest Batched Dataset is a labelled dataset of Fanaroff-Riley (FR) galaxies drawn from [Miraghaei and Best 2017](https://academic.oup.com/mnras/article/466/4/4346/2843096). At present, the dataset consists of FRI, FRII and hybrid FR sources, labelled as either confidently or uncertainly classified, and with any nonstandard morphology noted.

## The MiraBest Batched Dataset

The MiraBest Batched Dataset is comprised of a total of 1256 images, with their class denoted by a three-digit number.

First digit: FRI (1), FRII (2), hybrid source (3).
Second digit: confidently classified (0), uncertaintly classified (1).
Third digit: standard morphology (0), double-double (1), wide-angle tail (2), head-tail (4).

Images were obtained from the VLA FIRST sky survey via SkyView virtual telescope.  All images have dimensions of 150 x 150 pixels, with one pixel equivalent to an angular size of 1.8". The dataset is split into 7 training batches and a test batch, each composed of 157 images. The number of objects of each class in each batch is roughly consistent between batches, and the test batch contains at least one example of every class.
