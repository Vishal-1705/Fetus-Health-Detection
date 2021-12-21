# Fetus-Health-Detection

Every year around 3,00,000 prenatal deaths are occuring all around the world. To detect the problems of pregnancy, and to predict the appropriate method for delivery, it is useful to have regular checkups and examination of the reports. One way of prenatal examination is Cardiotocography (CTG). CTG is an electronic method which helps records the signals from the baby's heart rate and uterine contractions simultaneously along with many other details. Here, with many of the histrionic data regarding the fetal health, we can train a machine learning model to predict the health of a fetus as 'normal', 'suspect' or 'pathologic'. ML algorithms can be used in different applications within fetal medicine such as to predict preterm births etc. CTG traces are visually examined by clinicians and their interpretation is largely dependent on the clinician’s expertise, leading to high inter- and intra observer variability. Therefore, despite the existence of standardized guidelines, the accuracy and robustness of CTG to improve prenatal outcomes remain controversial. In many other medical fields, today, AL/ML domains are extensively used to take critical decisions, which also have performed very well and have stood out in all the tasks they have been assigned. This project solves the global issue and will address this critical problem.

Azure services are used to do this project. Machine learning model is build manually witht the help of VS Code in the 'Azure Machine Learning Studio'. All the files like the HTML file, CSS file, Procfile, requirements.txt file etc have been created in the VS Code.

![fsc](https://user-images.githubusercontent.com/64015389/146978945-804dd678-6de0-4f1c-a765-aec4263cbcd3.png)

'Load Balancing service' has also been used where we can control the network traffic and increase the number of computer instances according to the number of users.
'Azure App service' is used to deploy the website along the machine learning model using the flask framework.
