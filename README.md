Jebin Joseph (Undergraduate) and Ravi Parashar (Graduate)

First, you must move the data folder so that it is in the same directory as Project2. 

For all scala files, uncomment the desired optimizer to be used. 


Python/Keras
Some packages are necessary. These are: sklearn, matplotlib, and keras. Next, you can choose the desired models by setting the booleans at the top to true, and multiple models can be selected. Only one optimizer can be chosen at a time. Next, set the desired activation functions for each layer to true, and only one of these should be true per layer. Then, running the file will print out the graphs required by Dr. Miller. 
The main method calls runner, which will call multiple other methods. Runner first calls forward selection to choose the desired attributes. Then, runner will perform k-fold cross validation for a model and record r Sq CV as the average over the splits. Then a model will be created without cross validation and will be used to calculate r Sq and adjusted r Sq. 
Forward selection will usually choose all columns, so after all columns are chosen, the code will plot a graph of the three r Sq values vs. the number of parameters. Afterwards, the code will plot the number of epochs vs. loss for the model. The best model is included in the write-up. 
