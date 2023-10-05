# Chrome searchBar Suggestions auto-encoder-NLP Project

<p>
whenever users will search anything in the chrome search bar at that it gives their    suggestions
for the users. So how it is gives the suggestions to the users, this is the problem statement of this project.
and i will solve this problem statement in this project by building own NLP
model and Training Pipeline to give the suggestions to the users.
</p>

### Build a Robust Training pipeline for this project.
i Build a Robust Training Pipeline by the python Oop concepts, because in future, if i will may have good amount of chrome user histories. then i can easily again execute my Training Pipeline with new history data.
and according to the Dataset i has managed my Training Pipeline components.
and in the modelEvaluation my code will compare accuracy to the previous model. if my current model will be better to the previous model. then my current model will go to the production.

### Training Pipelines components.
<ul>
<li>Data ingestion.</li>
<li>Data validation.</li>
<li>Data cleaning.</li>
<li>Data Transformation.</li>
<li>ModelTrainer.</li>
<li>ModelEvaluation. </li>
<li>ModelPusher.</li>
</ul>

### To complete this project, i arranged chrome History from multiple chrome users.
<p><b>i arranged the chrome history from multiple of my collegeus and friends, because i wants to complete this project on the realworld dataset. when i was got the histories of the chrome users, there are very messy data was available, i did lot of text preprocessing on the chrome user histories. then i performed NLP operation on my dataset. </b></p>

<p><b>Code written in Modular fashion, i used Object oriented programing concepts<br>
of python, and followed the industry standard of project.</b></p>

### steps to execute this project
<ul>
<li>run the requirements.txt file</li>
<li>to train the pipeline run the train.py file</li>
<li>to test the model performance, run the test.py</li>
</ul>

### commands to execute the project
<ul>
<li>pip install -r requirements.txt</li>
<li>python train.py</li>
<li>python test.py</li>
</ul>

### key points

## train.py
<p>when you will execute the train.py file. it will start the training pipeline process and it will execute all the components one by one which i written in training pipeline, and execution report will be stored in the logs folder in the form of logs. and all the models and Data and other iformation will be stored in the artifact directory.
</p>

## test.py
when you will execute the test.py file. first it will start training of training pipeline. after execution of training pipeline it will ask to you,"do you want to test the performance of model" you can insert "Y" / 'N'.
it will guide you to during the testing. that,s why you can easily test this project.
because every lines of code i has customize, in this project. and this is my personal project. i written each and every lines of code and logics by itself to complete this project. i did not take any refferenced from youtube and any other resuarces to complete this project.

Thank You ❤🤍❤