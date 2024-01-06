# MLOpsProject
Exam project for MLOps course DTU

### Participants:
* Martin Christoffersen, s190464@student.dtu.dk
* Mikkel Grøngård, s214649@student.dtu.dk
* Rasmus Sørensen, s214622@student.dtu.dk

<h1 align="center">
  <br>
  Kaggle: CS:GO Round Winner Classification and strategy recommendation MLOps project
  <br>
</h1>

## Overall goal of the project
Our basic goal of this project is to predict whether a given CSGO round is a win or not given information from the game. If possible in the time-frame, we want to make a recommendation in game - For example if one should save or spend earned money during that round to improve the winning chances throughout a whole match. This means that we have to implement some kind of next-sequence prediction. We hope that a Transformer based model's positional embedding can take care of this. The motivation of the project is to make a recommendation system of eco, buy, or force buy in-game at real-time. This already exists for many other games as a layover, but not for CSGO as of yet.
The Transformer-based model is the experimental part of our project, meaning we'll see how far we get. Our main focus is to use MLops strategies and methods from the MLOps course.

## What framework are you going to use and you do you intend to include the framework into your project?
The basic setup going to use Pytorch. We're planning to evolve our model into an AutoEncoder with a Transformer. We're expecting to use Huggingface to look for a model of this sort. Furthermore we're looking into the suggested pytorch frameworks given from the course for some new interesting implementation throughout our project.

## What data are you going to run on (initially, may change)
The Kaggle dataset: CS:GO Round Winner Classification (link: https://www.kaggle.com/datasets/christianlillelund/csgo-round-winner-classification). This dataset includes around 97 parameters, where most of them are the different weapon types. Furthermore, it includes, the most important general parameters, like map, eco, health, and so on. As the "y-label" we have round win or loss, which we wish to predict from the other 96 parameters of information from the game.

## What models do you expect to use
On day one, we want to create a simple linear neural network as a beginning to make a simple loop starting from data processing to train-/testloop. The output we expect one day:1 is the probability of winning a CSGO match given specific parameters. Throughout the rest of the project days, we want to look into an autoencoder with a transformer implementation in the sparse latent space. - We hope to not only be able to predict if a team is winning or losing but also give recommendations of what might help to win given specific circumstances. - This could for example be used for economic strategy in CSGO.
