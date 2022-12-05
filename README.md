<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Módulo 3 NLP Module Project">Módulo 3 NLP Module Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## Módulo 3 NLP Module Project
The `run.py` generates results for 3 NLP activities.

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites
This is an example of how to list things you need to use the software and how to install them.
* nltk
  ```sh
  pip install nltk
  ```
* flair
  ```sh
  pip install flair
  ```
* torch
  ```sh
  pip install torch
  ```
* transformers
  ```sh
  pip install transformers
  ```
* wrapt
  ```sh
  pip install wrapt
  ```
* scikit-learn
  ```sh
  pip install scikit-learn
  ```
* googletrans
  ```sh
  pip install googletrans
  ```
`requirements.txt` also has all packages required.

### Executing program

* Make sure you have all the files required in a same folder 
* Make sure you have all required packages installed
* Run `run.py` with the files in the Resources folder

## Results
* NER Model
The following graph show the loss and F1 Score for the train, validation and test set, using a 40% of the full dataset due to RAM issues.
![NER_error](https://user-images.githubusercontent.com/63175363/205551823-1c54464d-7fd0-4a9a-bfab-d7938008044c.png)

## Author
Erandi del Angel

<p align="right">(<a href="#readme-top">back to top</a>)</p>
