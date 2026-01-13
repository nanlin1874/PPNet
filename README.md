Underwater Image Enhancement<br> 
===
This is the repo for "Underwater Image Enhancement via Prior-guided Prior-embedding Network".<br> 

Requirements<br> 
----
* install the requirements.txt<br>

Testing<br>
----
Our pretrained models are provided in folder `checkpoint`<br> 
* First, You can use `RedChannelPrior` to produce red-channel transmission and put in folder `rcp`<br>
* Second, use `ContrastPrior` to produce contrast-prior images and put in folder `cp`<br>
* run `main_test.py`<br>

Test_datasets<br>
----
**UIEB dataset** [[UIEB]](https://li-chongyi.github.io/proj_benchmark.html)<br>
paper : An Underwater Image Enhancement Benchmark Dataset and Beyond<br>

**SQUID dataset** [[SQUID]](https://csms.haifa.ac.il/profiles/tTreibitz/datasets/ambient_forwardlooking/index.html)<br>
paper : Underwater Single Image Color Restoration Using Haze-Lines and a New Quantitative Dataset<br>

**RUIE dataset** [[RUIE]](https://github.com/dlut-dimt/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark)<br>
paper :  Real-world Underwater Enhancement: Challenges, Benchmarks, and Solutions(RUIE-Net)<br>

**EUVP、UFO-120 datasets** [[EUVP]](http://irvlab.cs.umn.edu/resources/euvp-dataset)<br>
paper :  Fast Underwater Image Enhancement for Improved Visual Perception<br>

**Color-Check7 dataset** [[Color-Check7]](https://github.com/fergaletto/Color-Balance-and-fusion-for-underwater-image-enhancement.-.)<br>
paper :  Color Balance and fusion for underwater image enhancement<br>
