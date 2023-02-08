# MeNSP: Matching Exemplar as Next Sentence Prediction
Official implementation to our paper _Matching Exemplar as Next Sentence Prediction (MeNSP): Zero-shot Prompt Learning for Automatic Scoring in Science Education_, which is available at [2301.08771v1.pdf (arxiv.org)](https://arxiv.org/pdf/2301.08771v1.pdf).

#### Setup

* Assuming that you are managing your environments with conda.

  ```sh
  >>> conda create -n MeNSP python=3.9
  >>> conda activate MeNSP
  ```

* Installing your dependencies by:

  ```shell
  >>> cd src
  >>> pip install -r requirements.txt
  ```

#### Reproduction

* Our experiment results are recorded at logs/final_test.log

* You may reproduce this result by:

  ```shell
  >>> python -u experiments.py 2023 0 > logs/my_experiment.log
  ```

#### TODO

* We will upload a service version of our model to Huggingface library.
* We will develop a simple website to  let you further explore this method.
