---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [ ] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [x] Remember to comply with good coding practices (`pep8`) while doing the project
* [x] Do a bit of code typing and remember to document essential parts of your code
* [ ] Setup version control for your data or part of your data
* [x] Construct one or multiple docker files for your code
* [x] Build the docker files locally and make sure they work as intended
* [x] Write one or multiple configurations files for your experiments
* [x] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [ ] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [x] Write unit tests related to the data part of your code
* [x] Write unit tests related to model construction and or model training
* [x] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [ ] Create a trigger workflow for automatically building your docker images
* [ ] Get your model training in GCP using either the Engine or Vertex AI
* [x] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [ ] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [x] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

99

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s190464, s214622, s214649

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used "Optuna" as our third-party framework in this project. As we've in the past semester had the course "active machine learning" we thought it would be nice to revisit Hyperparameter setting optimization with another framework. We used this GitHub as inspiration for our settings: "https://github.com/elena-ecn/optuna-optimization-for-PyTorch-CNN/blob/main/optuna_optimization.py" and optimized for learning rate, optimizer dropout and experimented with a flexible depth in our FFNN. We especially found the pruning of obvious bad runs quite smart and efficient alongside the general ease of used compared to GPYOpt which we've used earlier and has very poor documentation. Optuna helped us achieve better performance and was much faster, computationally speaking, than we though it would be.

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

As a prerequisite, we recommend setting up a virtual environment for this project using Anaconda.

'''bash
$ conda create --name [your_environment_name] python=3.9
$ conda activate [your_environment_name] 
'''

!!!!!!!!!!!!!!!!!!!!!!!!!!!
Der skal skrives noget her
!!!!!!!!!!!!!!!!!!!!!!!!!!
We have made a requirements.txt file in which we have stated all the libraries and packages. This file is used for a new member of the time to set up the right environment.
The way the member can do this is by:
1. $cd [path]
2. $pip install -r requirements.txt
   
Afterwards, you should be good to go.
Furthermore, we've made a docker file, which builds an exact image of the project and can run both the training- and predict files.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

We used the cookiecutter template given in the course. It was important for us to keep the structure of the template to force us to work in a more structured manner - We then removed the following folders at the end, due to the fact we did not implement these parts in the project [...].
The main part of the folder are [...].

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

Add that we used pep8 and linting [...].

Throughout this project work and other projects, we've noticed the importance of using git to share code that makes for small parts in a greater whole in the project. This means that no one person has the full picture of over details in the code without reading and understanding other's code. Rules like pep8 and just good coding practices in general enhance readability, thereby hastening and improving the quality overall in the project.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

The main idea was to make tests for train, model architecture and data, as these parts was changed the most throughout the project, making it important to test often. We made a total of [...] that tested [...]. We then used the "Coverage" library to check over coverage and reiterated our testing, both in general but also as our project grew in size.

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

Our code coverage is [...] ....

We think that unit-testing and thereby test-coverage in important to check for problems that might arise when growing/changing the project. But as most ML, especially DL is largely dynamic, the errors can also be very abstract. It's easy to make tests for tensor-sizes and data cleaning functions, as the answer to these tests are obvious. The real issue arises when ones loss is acting out or the accuracy suddenly is much lower after a change that even went through the tests. In this project we even had issues with data and models even at a simple FFNN - The solution was much more abstract than a simple unittest is able to handle. Therefore in ML we cannot trust test to makes us error-free, but it helps in certain areas.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We first started our workflow on the "main" branch. We quickly figured out why branching was necessary, even in small to medium projects like this. The main issue we had only using the main branch was that we all worked on this code at the same time, making the push/pull experience bad, as we had to continuously fix the merging issues.
We started adding more branches for certain situations - One for clean implementation and one for more experimental testing. For our project, this was more than fine in a 3-person group, but in bigger projects giving a branch to every person is probably the way to go.
The pull request feature was quite brilliant, as we suddenly saw zero merging issues on our main branch, meaning that the code on the main branch was always working for the actual ML testing.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

Initially we didn't use it due to [...], but [...].

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:


Unittesting: In our project, we use pytest as unittester. The way we do this is by creating a new folder in which
three files: test_data.py, test_model.py, test_model_structure.py are placed. These files contain functions that test  different aspects of our project
like do we remove all grenades from the dataset. The first one tests all the data-related code and the test_model.py test if the model is properly running.
test_model_structure.py test the structure of our code.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We mainly used Hydra to configure our experiments, which has all its parameter settings in a config file and saves all important experiment data in an output folder with timestamps.
As an addition, we've also made a "click"-library-based file for command line training and prediction, where one can input settings directly into the command line.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

For experimental reproducibility, we used both Hydra and Wandb to store important information from our experiments. Hydra took care of the config files and saved a local file every time we experimented with hyperparameters, data information, loss, and the trained model. This was enough to let us reproduce the experiments if needed. However, we lacked the online functionality of Wandb, so everyone on the team could follow training in real-time as it was being made. Wandb also added the ability to share loss, accuracy etc.-plots on the go, making the inference analysis much easier as a team. The mixture of these two tools went a long way in making reproducibility easier and safer and we were in general very happy about working with increased reproducibility. 

We added different workflows in the code, such that we in a code/ML-debugging scenario could stop and start the tracking if need-be.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

Docker has also been a focus of interest in our project, both on its own, but also to use alongside our API and Google Cloud implementation. Separately we used two docker images: one for training and one for inference. We mostly ran the images out of the box, but it is possible to change hyperparameter settings by changing the Hydra config file. Link to training file "https://github.com/GorenzelgProjects/MLOpsProject/blob/main/train_model.dockerfile".
Furthermore, we used docker images for our API deployment, as this would make sure we could run our solution on any PC, just by running the docker file.
Lastly, we also used a docker image for our Google Cloud setup.

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

In our project, we used Visual Studio Codes debugger, which helped a lot in getting our project to work. We found this technique to be very time-efficient for debugging plus efficient code checking (run sub-parts of the code). Using the breakpoint function and debugger console allows us to move beyond print-statements and we found this very efficient. However, the debugger tool still works best for "old-school" code problems, whereas other tools like plots and shape prints are still necessary for ML issues/bugs debugging.
Already in the early project days, we focused on implementing profiling to see which parts of the code were slow. This helped us speed up the code, as we now had a better idea if where we could improve the most.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- question 17 fill here ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- question 22 fill here ---

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 23 fill here ---

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

--- question 24 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

Lav figure først [...]

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

Throughout the process, we had different challenges. Firstly, we had program issues with our model to work on our data. This needed a mixture of debugging, profiling, and wandb to solve it. Debugging helped with issues regarding folder layout giving data-loading issues, profiling to speed up very slow parts of the code, and wandb+shape-print-statements to fix model issues. Hydra also gave us issues with config files in the training loop, but in the end, the tools we've gotten both before this course and throughout it helped us overcome most of the local deployment issues and made for a good ground for reproducible experiments 

Secondly, we had the Google Cloud issue (surprise, surprise) which took a huge amount of time working. To solve the first problem we had to look through several videos on YouTube and experiment with code
before it worked. For the second problem, we searched through the internet/YouTube to solve it, and it seemed to be a huge struggle in the beginning. The first couple of steps was fine in which we mean creating
a project and activating API and bucket. But from there it was time-consuming.

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

--- question 27 fill here ---
