# MLOpsProject
Exam project for MLOps course DTU

Template by: @amitmerchant1990 | https://github.com/amitmerchant1990/electron-markdownify#readme

<h1 align="center">
  <br>
  Kaggle: CS:GO Round Winner Classification
  <br>
</h1>

## Overall goal of the project
Our overall goal is to predict whether you win or not. And if time possible you should save or spend earned money in a round to win a round and at last end win the whole game. In in-game language eco, buy, or force buy. The model is the experimental part of our project, but our focus is to use MLops strategies and methods to make this experimentation easier, and for us to better share work.

## What framework are you going to useand you do you intend to include the framework into your project?
We are going to use pytorch with a hugging face framework.

## What data are you going to run on (initially, may change)
We are going to use Kaggle CS:GO Round Winner Classification (link: https://www.kaggle.com/datasets/christianlillelund/csgo-round-winner-classification). This dataset includes around 97 parameters, where most of them are the different weapon types. And of course, the most important is the more general parameters, like map, eco, health, and so on.

## What models do you expect to use
As a starting point, we want to create a linear neural network to quickly get started. This should give us the probability of winning given specific parameters.  Afterward, we want to look into an autoencoder with a transformer, including an encoder and a decoder. If time we want to use this as a form of experiment. Which maybe could be used for economic strategy.
